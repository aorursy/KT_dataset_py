# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np
data=pd.ExcelFile("/kaggle/input/multitabdata/reshaping_data.xlsx")
tab_names=data.sheet_names

tab_names
data.parse(sheetname=tab_names[0])
df=data.parse(sheetname=tab_names[0],skiprows=7)

df.head()
cols1=[str(x)[:4] for x in df.columns]

cols2=[str(x) for x in df.iloc[0,:]]

col_names=[x+"_"+y for x,y in zip(cols1,cols2)]
df.columns=col_names

df.head()
df.drop("Unna_nan", axis=1, inplace=True)
df=df.iloc[1:,:].rename(columns={

    'dist_nan':'district',

    'prov_nan':'province',

    'part_nan':'partner',

    'fund_nan':'funding_source'

})
df.head()
df['main_org']=tab_names[0].split("_")[0]+" "+tab_names[0].split("_")[1]
df.head()
df.info()
to_remove=[c for c in df.columns if "Total" in c]

to_change=[c for c in df.columns if "yrs" in c]

df.drop(to_remove, axis=1, inplace=True)

for c in to_change:

    df[c]=df[c].apply(lambda x:pd.to_numeric(x))

    
df.info()
df.head()
idx=['district','province','partner','funding_source','main_org']

multi_indexed_df=df.set_index(idx)

multi_indexed_df.head()
stacked_df=multi_indexed_df.stack(dropna=False)

stacked_df.head(20)
long_df=stacked_df.reset_index()

long_df.head()
col_str=long_df.level_5.str.split("_")
col_str
long_df["target_year"]=[x[0] for x in col_str]

long_df["target_age"]=[x[1] for x in col_str]
long_df['target_quantity']=long_df[0]

long_df.drop("level_5", axis=1, inplace=True)

long_df.drop(0, axis=1, inplace=True)
long_df
def ReshapeFunc(excel_obj, i):

    """ Takes in an excel file object with multiple tabs in a wide format, and a specified index of the tab to be parsed and reshaped. Returns a data frame of the specified tab reshaped to long format"""



    tabnames = data.sheet_names

    assert i < len(tabnames), "Your tab index exceeds the number of available tabs, try a lower number" 

    

    # parse and clean columns

    df = excel_obj.parse(sheetname=tabnames[i], skiprows=7)

    cols1 = [str(x)[:4] for x in list(df.columns)]

    cols2 = [str(x) for x in list(df.iloc[0,:])]

    cols = [x+"_"+y for x,y in zip(cols1,cols2)]

    df.columns = cols

    df = df.drop(["Unna_nan"], axis=1).iloc[1:,:].rename(columns={

        "dist_nan":"district",

        "prov_nan":"province",

        "part_nan":"partner",

        "fund_nan":"funding_source"

    })                            

    # new columns, drop some and change data type

    df['main_org'] = tabnames[i].split("_")[0] + " "+ tabnames[i].split("_")[1]

    df.drop([c for c in df.columns if "Total" in c], axis=1, inplace= True) 

    

    for c in [c for c in df.columns if "yrs" in c]:

        df[c] = df[c].apply(lambda x: pd.to_numeric(x))

    # reshape - indexing, pivoting and stacking

    idx = ['district','province', 'partner','funding_source'

,'main_org']

    multi_indexed_df = df.set_index(idx)

    stacked_df = multi_indexed_df.stack(dropna=False)

    long_df = stacked_df.reset_index()

    

    # clean up and finalize

    col_str = long_df.level_5.str.split("_") 

    long_df['target_year'] = [x[0] for x in col_str] 

    long_df['target_age'] = [x[1] for x in col_str]

    long_df['target_quantity'] = long_df[0] # rename this column

    df_final = long_df.drop(['level_5', 0], axis=1)

    return df_final
dfs_list=[ReshapeFunc(data,i) for i in range(4)]

concat_dfs=pd.concat(dfs_list)

concat_dfs.to_excel("Reshaped_data.xlsx")