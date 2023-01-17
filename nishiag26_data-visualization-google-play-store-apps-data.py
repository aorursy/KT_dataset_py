import pandas as pd
import os
import numpy as np
import seaborn as sns
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
#### data is from year 2010,May to 2018,Aug
from datetime import datetime,date
gplay_df['Last_Updated_Days']=gplay_df['Last Updated'].apply(lambda x: date.today()-datetime.date(x))
gplay_df['Last_Updated_Days'].head()
gplay_df['Last_Updated_Days'] = gplay_df['Last_Updated_Days'].dt.days
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
gplay_df.info()

gplay_df.to_csv('Clean_GplayApps.csv',index=False)
# After Cleaning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
gplay_df = pd.read_csv('Clean_GplayApps.csv')
gplay_df.head()
num_var = gplay_df.select_dtypes(include=['int64','float64'])
col_num = num_var.columns
num_var
cat_var = gplay_df.select_dtypes(include=['object'])
col_cat = cat_var.columns
cat_var
num_var.hist(figsize=(9,9),bins=50);
sns.boxplot(gplay_df['Rating']);
sns.boxplot(gplay_df['ReviewCount']);
sns.boxplot(gplay_df['Installs']);
df_paid = gplay_df[gplay_df['Type']=='Paid']
df_paid
df_paid['Price'].skew()
sns.boxplot(df_paid['Price']);
sns.boxplot(gplay_df['AppSize']);
print('Skewness before Removing Outliers')
lis=list(col_num)
lis.remove('Price');
for col in lis:
    print(gplay_df[col].skew());
lis
print('Skewness before removing the outliers')
print(df_paid['Price'].skew())
per = df_paid['Price'].quantile([0.10,0.90]).values
df_paid['Price']=np.clip(df_paid['Price'],per[0],per[1])
print('Skewness after removing the outliers')
print(df_paid['Price'].skew())
sns.boxplot(df_paid['Price']);
#Skewness in betweek-1 and +1 is said normally distributed and any deviation indicates extreme values 
print('Skewness after cleaning outliers')
for col in lis:
    perc = gplay_df[col].quantile([0.10,0.90]).values
    gplay_df[col]=np.clip(gplay_df[col],perc[0],perc[1])
    print(gplay_df[col].skew())
sns.boxplot(gplay_df['Rating']);
sns.boxplot(gplay_df['ReviewCount']);
sns.boxplot(gplay_df['Installs']);
sns.boxplot(gplay_df['AppSize']);
sns.countplot(data=gplay_df,x='Type');
gplay_df.Category.value_counts().plot(kind='bar');
gplay_df['Content Rating'].value_counts().plot(kind='bar');
gplay_df.Prim_Genre.value_counts().plot(kind='bar');
gplay_df.Sec_Genre.value_counts().plot(kind='bar');
gplay_df['Last Updated'].value_counts().sort_index()
plt.figure(figsize=(8,8));
gplay_df["Last Updated"].value_counts().sort_index().plot(kind='line');
plt.ylim(bottom = 0);
plt.xlabel('Last Updated Date');
plt.ylabel('Count of Apps');
sns.lineplot(x='Rating',y='Installs',data=gplay_df);
sns.lineplot(x='AppSize',y='Installs',data=gplay_df);
sns.lineplot(x='Price',y='Installs',data=df_paid);
plt.xlabel('Price (Dollars)');
sns.lineplot(y='ReviewCount',x='Installs',data=gplay_df);
sns.lineplot(x='Rating',y='Price',data=df_paid);
sns.lineplot(x='Rating',y='AppSize',data=gplay_df);
sns.lineplot(x='Rating',y='ReviewCount',data=gplay_df);
sns.lineplot(x='AppSize',y='ReviewCount',data=gplay_df);
sns.barplot(x='Type',y='Installs',data=gplay_df);
sns.barplot(x='Type',y='ReviewCount',data=gplay_df);
plt.figure(figsize=(8,5));
sns.barplot(x='Content Rating',y='Price',data=df_paid);
plt.figure(figsize=(8,5));
sns.barplot(x='Content Rating',y='ReviewCount',data=gplay_df);
plt.figure(figsize=(8,5));
sns.barplot(x='Content Rating',y='Installs',data=gplay_df);
plt.figure(figsize=(30,5));
sns.barplot(x='Android Ver',y='ReviewCount',data=gplay_df);
plt.figure(figsize=(30,5));
sns.barplot(x='Android Ver',y='Price',data=df_paid);
plt.figure(figsize=(30,5));
sns.barplot(x='Android Ver',y='Installs',data=gplay_df);
plt.figure(figsize=(30,5));
sns.lineplot(x='Android Ver',y='Price',data=df_paid);
