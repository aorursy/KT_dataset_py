%pylab inline --no-import-all
import pandas as pd
import seaborn as sns
import tqdm
import ipywidgets as widgets
import gc
import pickle
import re
import os
DATA_PATH = "../input/data-conversion-to-speedup-future-imports/"
SAVE_PATH = "."
INTERACTIVE = False
with open(os.path.join(DATA_PATH, "train.pkl"), 'rb') as f:
    train_data = pickle.load(f)
print(f"loaded train data with {len(train_data)} rows")
gc.collect()
nas_df = pd.DataFrame([0 for col in train_data.columns], columns=["NA's count"], index=train_data.columns)
for col in train_data.columns:
    nas = train_data[col].isna().sum()
    nas_df.loc[col, ("NA's count")] = nas
    
nas_df = nas_df[nas_df["NA's count"] != 0]
nas_df.sort_values(by="NA's count", ascending=False, inplace=True)

nas_df["% of NA's"] = nas_df["NA's count"] * 100 / len(train_data)
nas_df
to_del = [
    'DefaultBrowsersIdentifier',
    'Census_ProcessorClass',
    'Census_InternalBatteryType',
    'Census_IsFlightingInternal',
    'Census_IsWIMBootEnabled'
]
train_data.drop(columns=to_del, inplace=True)
train_data.SmartScreen.describe()
train_data.OrganizationIdentifier.astype('category').describe()
def replace_nas(data: pd.DataFrame, col: str, type_: str):
    nas = data[col].isna().sum()
    if nas > 0:        
        if type_ == 'category' or type_ == 'bool':
            mode = data[col].mode().values[0]
            print(f"replacing NAs in {col} with {mode}")
            data.loc[:, col][data.loc[:, col].isna()] = mode
        else:
            median = data[col].median()
            print(f"replacing NAs in {col} with {median}")
            data.loc[:, col][data.loc[:, col].isna()] = median

def convert_types(data: pd.DataFrame):
    converter_rules = {
        ('category',)         : ['MachineIdentifier', 'OsBuildLab'],
        ('int32',)            : ['AVProductsInstalled', 'AVProductsEnabled', 'Census_ProcessorCoreCount'],
        ('bool',)             : ['(.*_)?Is.*', '(.*_)?Has.*', 'Firewall'],
        ('int32', 'category') : ['SMode', 'RtpStateBitfield', 'Census_OSBuildRevision', 
                                 'CityIdentifier', 'AVProductStatesIdentifier', 'UacLuaenable',
                                 'Census_OEMModelIdentifier', 'Census_FirmwareVersionIdentifier'],
        ('int16', 'category') : ['.*Identifier.*', 'OsBuild', 'OsSuite', 'Census_OSBuildNumber']
    }
    
    used_cols = set()
    for types, columns in converter_rules.items():
        for col_rule in columns:
            for col in data.columns:
                if col not in used_cols and re.match(col_rule, col):
                    replace_nas(data, col, types[-1])
                    for type_ in types:
                        data[col] = data[col].astype(type_)
                    used_cols.add(col)
    rest_cols = set(data.columns).difference(used_cols)
    for col in data.columns:
        replace_nas(data, col, data[col].dtype.name)
convert_types(train_data)
correlations = train_data.corr()['HasDetections'].abs().sort_values(0, False)

checkboxes = []
labels = []
values = []
disabled = [
    'Census_IsPortableOperatingSystem',
    'HasTpm',
    'Census_IsSecureBootEnabled',
    'Census_IsFlightingInternal',
    'Census_IsPenCapable',
    'Firewall',
    'Census_IsFlightsDisabled',
    'IsBeta',
    'Census_IsWIMBootEnabled',
]

if INTERACTIVE:
    for i, val in correlations.iteritems():
        value = (i not in disabled)

        checkboxes.append(widgets.Checkbox(value=value))
        labels.append(widgets.Label(f"{i} ({train_data[i].dtype})"))
        values.append(widgets.Label(str(val)))

    button = widgets.Button(description="Process")    
    def on_button_clicked(b):
        global train_data
        to_del = [col for box, col in zip(checkboxes, correlations.index) if box.value == False]
        train_data.drop(columns=to_del, inplace=True)

    button.on_click(on_button_clicked)
    
    w = widgets.VBox([
        widgets.HBox([widgets.VBox(w) for w in (checkboxes, labels, values)]),
        button
    ])
else:
    max_col_len = max(map(len, correlations.index)) + 15

    for i, val in correlations.iteritems():
        mark = "v" if i not in disabled else "x"
        key = f"{i} ({train_data[i].dtype})".ljust(max_col_len)
        print(f"{mark} {key} : {val}")
        
        if i in disabled:
            train_data.drop(columns=[str(i)], inplace=True)
    
    w = ""
    
w
values = []
disabled = [
    'Census_OSSkuName',
    'Census_MDC2FormFactor',
    'Census_OSEdition',
    'Census_OSBuildNumber',
    'Census_PowerPlatformRoleName',
    'Census_OSArchitecture'
]
N = 3


if INTERACTIVE:
    columns = [str(col) for col in train_data.columns if train_data[col].dtype.name]
    columns.remove('MachineIdentifier')

    boxes = [widgets.Checkbox(value=(c not in disabled)) for c in columns]
    keys = [widgets.Label(c) for c in columns]


    for i, (idx, row) in enumerate(train_data.head(3)[columns].iterrows()):
        values.append(widgets.VBox([widgets.Label(str(v)) for k, v in row.iteritems()]))
    

    button = widgets.Button(description="Process")
    def on_click(b):
        global train_data
        to_del = [col.value for box, col in zip(boxes, keys) if box.value == False]
        train_data.drop(columns=to_del, inplace=True)

    button.on_click(on_click)
    
    w = widgets.VBox([
        widgets.HBox([widgets.VBox(boxes), widgets.VBox(keys), *values]),
        button
    ])

else:
    columns = [str(col) for col in train_data.columns if train_data[col].dtype.name]
    columns.remove('MachineIdentifier')

    max_col_len = max(map(len, columns)) + 5
    
    val_lens = [[len(str(train_data.loc[i, c])) for c in columns] for i in range(N)]
    max_val_lens = list(map(max, val_lens))

    def description(c):
        key = c
        strip_idx = int(max_col_len * 0.5)
        if len(key) > strip_idx:
            key = key[:strip_idx] + "..."
            
        key = key.ljust(max_col_len, " ")
        
        values = ""
        for i, v in enumerate(train_data[c].head(3).tolist()):
            val = str(v)
            strip_idx = int(max_val_lens[i] * 0.5)
            if len(val) > strip_idx:
                val = val[:strip_idx] + "..."
                
            val = val.ljust(max_val_lens[i], " ")
            values = values + val
            
        return f"{key}: {values}"
    
    for c in columns:
        mark = "v" if c not in disabled else "x"
        print(f"{mark} {description(c)}")
        
    for c in disabled:
        train_data.drop(columns=[str(c)], inplace=True)
        
    w = ""
    
w
USED_COLS = train_data.columns.drop("HasDetections")

with open(os.path.join(SAVE_PATH, "train_clean.pkl"), 'wb') as f:
    pickle.dump(train_data, f)   

del(train_data)
gc.collect()
with open(os.path.join(DATA_PATH, "test.pkl"), 'rb') as f:
    test_data = pickle.load(f)
print(f"loaded test data with {len(test_data)} rows")

test_data = test_data.loc[:, USED_COLS]
gc.collect()

convert_types(test_data)
with open(os.path.join(SAVE_PATH, "test_clean.pkl"), 'wb') as f:
    pickle.dump(test_data, f)   