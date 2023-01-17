import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import soundfile as sf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, precision_recall_curve, plot_precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import keras

try:
    os.environ['KAGGLE_DATA_PROXY_TOKEN']
except KeyError:
    dir_out = "./"
    dir_path = "Respiratory_Sound_Database/Respiratory_Sound_Database/"
    fname_demo = dir_path + "demographic_info.txt"
else:
    dir_out = "/kaggle/working/"
    dir_path = "/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/"
    fname_demo = "/kaggle/input/respiratory-sound-database/" + "demographic_info.txt"
    
fname_diag = dir_path + "patient_diagnosis.csv"
dir_audio = dir_path + "audio_and_txt_files/"
group_pat_num = "([0-9]{3})"
group_rec_index = "([0-9][a-z][0-9])"
group_chest_loc = "(Tc|Al|Ar|Pl|Pr|Ll|Lr)"
group_acc_modes = "(sc|mc)"
group_equipments = "(AKGC417L|LittC2SE|Litt3200|Meditron)"

regex_info = re.compile("_".join([group_pat_num, group_rec_index, group_chest_loc, group_acc_modes, group_equipments]))

top = os.getcwd()
os.chdir(dir_audio)
fnames = glob.glob("*.txt")

### file name info, annotation, WAV recording

l_rec_info = []
num_cycles_sounds = []

max_cycles = 0

for fname in fnames:
    match_info = regex_info.match(fname)
    pat_num = int(match_info.group(1))
    rec_index = match_info.group(2)
    chest_loc = match_info.group(3)
    acc_mode = match_info.group(4)
    equipment = match_info.group(5)
     
    l_rec_info.append([pat_num, rec_index, chest_loc, acc_mode, equipment])
    
    with open(fname) as f_annot:
        lines = [ line.strip().split() for line in f_annot.readlines() ]
        lines = [ [ix_lines[0]] + ix_lines[1] for ix_lines in enumerate(lines) ]
        lines = [ [pat_num] + [rec_index] + [chest_loc] + line for line in lines ]
        
        num_cycles_sounds.extend(lines)
        
        if len(lines) > max_cycles:
            max_cycles = len(lines)


l_rec_info.sort(key=lambda subl: (subl[0], subl[1], subl[2], subl[3], subl[4]))
rec_info_cols = ["Patient number", "Recording index", "Chest location", "Acquisition mode", "Recording equipment"]
df_rec_info = pd.DataFrame(l_rec_info, columns=rec_info_cols)

annot_cols = ["Patient number", "Recording index", "Chest location", "Cycle number", "Cycle start", "Cycle end", "Crackles", "Wheezes"]
df_annotation = pd.DataFrame(num_cycles_sounds, columns=annot_cols)

os.chdir(top)

### create a simpler auxiliary DF / CSV for the annotations: one-hot-encoded crackles / wheezes per breath cycle:
### [cycle_0_crackles][cycle_0_wheezes][cycle_1_crackles][cycle_1_wheezes] etc.

df_tmp = df_annotation.set_index(["Patient number", "Recording index", "Chest location"])

aux = []

for ix in df_tmp.index.unique().sort_values():
    pat_num = ix[0]
    rec_index = ix[1]
    chest_loc = ix[2]
    subdf = df_tmp.loc[pat_num, rec_index, chest_loc]
    crackles_wheezes = [ yesno for c_w in zip( subdf["Crackles"], subdf["Wheezes"] ) for yesno in c_w ]
    len_cur = len(crackles_wheezes)
    row = [pat_num] + [rec_index] + [chest_loc] + crackles_wheezes + [0] * (max_cycles * 2 - len_cur)
    
    aux.append(row)

col_names = ["Patient number", "Recording index", "Chest location"]
col_names_c = ["Crackles_C{}".format(num_c) for num_c in range(max_cycles)]
col_names_w = ["Wheezes_C{}".format(num_w) for num_w in range(max_cycles)]

col_names_cw = [ name for tup in zip(col_names_c, col_names_w) for name in  tup]
col_names.extend(col_names_cw)

df_annot_aux = pd.DataFrame(aux, columns=col_names)

### diagnosis

diag = pd.read_csv(fname_diag, names=["Patient number", "Diagnosis"])
df_rec_info_diag = pd.merge(df_rec_info, diag)

### demographic info

with open(fname_demo) as f_demo:
    # skip single empty line at the beginning
    f_demo.readline()

    lines = [line.strip().split() for line in f_demo.readlines()]


for split in lines:
    split[0] = int(split[0])
    if split[1] != "NA":
        split[1] = float(split[1])
    else:
        split[1] = np.nan        
    if split[2] != "NA":
        pass
    else:
        split[2] = np.nan        
    if split[3] != "NA":
        split[3] = float(split[3])
    else:
        split[3] = np.nan        
    if split[4] != "NA":
        split[4] = float(split[4])
    else:
        split[4] = np.nan
    if split[5] != "NA":
        split[5] = float(split[5])
    else:
        split[5] = np.nan

df_demo = pd.DataFrame(lines, columns=["Patient number", "Age", "Sex", "Adult BMI", "Child weight kg", "Child height cm"])


df_full_info = pd.merge(df_rec_info_diag, df_demo, on="Patient number")
df_full_info.to_csv(dir_out + "full_info.csv", index = False)
df_annotation.to_csv(dir_out + "rec_annotation.csv", index = False)
df_annot_aux.to_csv(dir_out + "annot_aux.csv", index=False)
df_full_info = pd.read_csv(dir_out + "full_info.csv")
df_annotation = pd.read_csv(dir_out + "rec_annotation.csv")
df_annot_aux = pd.read_csv(dir_out + "annot_aux.csv")
df_full= pd.merge(df_full_info, df_annot_aux).set_index(["Patient number"])
df_full
diag.groupby(["Diagnosis"]).count()
cat_attrs = ["Chest location", "Recording equipment"]
num_attrs = ["Age"]

# one NaN there
df_full["Sex"]  = df_full["Sex"].fillna("F")
df_full["Age"]  = df_full["Age"].fillna(df_full.Age.mean())

col_tr = ColumnTransformer([
    ("one_hot", OneHotEncoder(), cat_attrs),
    ("standard", StandardScaler(), num_attrs)
], remainder="drop")

label_enc = LabelEncoder()
label_enc.fit(diag["Diagnosis"])

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
# for train_ix, test_ix in split.split(df_full, df_full["Diagnosis"]):
#     df_train = df_full[train_ix]
#     df_test = df_full[test_ix]

df_train, df_test = train_test_split(df_full, test_size=0.1, random_state=42)

labels_train = label_enc.transform(df_train["Diagnosis"])
labels_test = label_enc.transform(df_test["Diagnosis"])

df_train.drop(["Diagnosis"], axis=1, inplace=True)
df_test.drop(["Diagnosis"], axis=1, inplace=True)
train_trans = col_tr.fit_transform(df_train)
test_trans = col_tr.fit_transform(df_test)

dectree_clf = DecisionTreeClassifier()
dectree_clf.fit(train_trans, labels_train)
pred = dectree_clf.predict(train_trans)
accuracy_score(labels_train, pred)
confusion_matrix(labels_train, pred)
pred = dectree_clf.predict(test_trans)
accuracy_score(labels_test, pred)
confusion_matrix(labels_test, pred)