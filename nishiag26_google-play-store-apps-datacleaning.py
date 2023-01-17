import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
gplay_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
gplay_df.info()
gplay_df.head()
gplay_df.shape
gplay_df.isna().sum()
duplicate_ser = gplay_df[gplay_df.duplicated()]
len(duplicate_ser)
gplay_df.drop_duplicates(inplace=True)
gplay_df.rename(columns={'Reviews':'ReviewCount','Size':'AppSize'},inplace=True)
def strip_cols(col_name):
    col_name=col_name.str.replace('$','')
    col_name=col_name.str.replace('+','')
    col_name=col_name.str.replace(',','')
    col_name=col_name.str.replace('M','e+6')
    col_name=col_name.str.replace('k','e+3')
    col_name=col_name.str.replace(' and up','')
    #col_name=col_name.str.strip('.GP','')
    #col_name=col_name.str.strip('W','')
    #col_name=col_name.str.strip('-prod','')
    
    return col_name
    
    
def change_dtype(col_name):
    col_name=col_name.astype('float')
    return col_name

def change_intdtype(col_name):
    col_name=col_name.astype('int64')
    return col_name

def replace_nan(col):
    col = col.replace('Varies with device',np.nan)
    return col
    
    
gplay_df.App.value_counts()
gplay_df.App.nunique()
gplay_df['Rating'].value_counts()
# taking the Rating as 1.9 instead of 19
gplay_df['Rating'].replace('19.0','1.9',inplace=True)
gplay_df.Price.value_counts().sort_index()
gplay_df.drop(gplay_df[gplay_df['Price']=='Everyone'].index,inplace=True)

gplay_df['Price'] = strip_cols(gplay_df['Price'])
gplay_df['Price'] = change_dtype(gplay_df['Price'])
gplay_df.Price.value_counts().sort_index()
gplay_df.AppSize.value_counts()
gplay_df['AppSize'] = replace_nan(gplay_df['AppSize'])
gplay_df['AppSize'] = strip_cols(gplay_df['AppSize'])
gplay_df['AppSize'] = change_dtype(gplay_df['AppSize'])
gplay_df['AppSize'].value_counts()
gplay_df['Installs'].value_counts()
gplay_df['Installs'] = strip_cols(gplay_df['Installs'])
gplay_df['Installs'] = change_intdtype(gplay_df['Installs'])
gplay_df['Installs'].value_counts().sort_index()
gplay_df.ReviewCount.value_counts()
gplay_df['ReviewCount'] = strip_cols(gplay_df['ReviewCount'])
gplay_df['ReviewCount'] = change_intdtype(gplay_df['ReviewCount'])
gplay_df.ReviewCount.value_counts().sort_index()
gplay_df['Genres'].value_counts().sort_values()
prim = gplay_df.Genres.apply(lambda x:x.split(';')[0])
gplay_df['Prim_Genre']=prim
gplay_df['Prim_Genre'].tail()
sec = gplay_df.Genres.apply(lambda x:x.split(';')[-1])
gplay_df['Sec_Genre']=sec
gplay_df['Sec_Genre'].tail()
group_gen=gplay_df.groupby(['Prim_Genre','Sec_Genre'])
group_gen.size().head(20)
gplay_df.drop(['Genres'],axis=1,inplace=True)
gplay_df['Last Updated'].value_counts().sort_values()
gplay_df['Last Updated'] = pd.to_datetime(gplay_df['Last Updated'])
gplay_df['Last Updated'].value_counts().sort_index()
gplay_df['Current Ver'].value_counts().sort_values()
gplay_df.drop(gplay_df[gplay_df['Current Ver']=='MONEY'].index,inplace=True)
gplay_df['Current Ver'] = replace_nan(gplay_df['Current Ver'])
gplay_df['Current Ver'].sample(20)
gplay_df['Android Ver'].value_counts().sort_values()
gplay_df['Android Ver'] = strip_cols(gplay_df['Android Ver'])
gplay_df['Android Ver'] = replace_nan(gplay_df['Android Ver'])
gplay_df['Android Ver'].replace('4.4W','4.4',inplace=True)
gplay_df['Android Ver'].value_counts().sort_values()
gplay_df['Category'].value_counts().sort_values()
gplay_df['Type'].value_counts() 
gplay_df['Content Rating'].value_counts() 
gplay_df.info()
# categorical and Numerical Values:
num_var = gplay_df.select_dtypes(include=['int64','float64']).columns
cat_var = gplay_df.select_dtypes(include=['object','datetime64','timedelta64']).columns
num_var,cat_var
gplay_df.isna().sum()
missing_perc = (gplay_df.isna().sum()*100)/len(gplay_df)
missing_df = pd.DataFrame({'columns':gplay_df.columns,'missing_percent':missing_perc})
missing_df
col_cat = ['Type','Current Ver','Android Ver'] #Categorical Var.
for col in col_cat:
    gplay_df[col].fillna(gplay_df[col].mode()[0],inplace=True)
    
col_num=['Rating','AppSize'] #Numerical Var.
for col in col_num:
    gplay_df[col].fillna(gplay_df[col].median(),inplace=True)
gplay_df.isna().sum()
clean_df=gplay_df
clean_df.to_csv('Clean_GplayApps.csv')