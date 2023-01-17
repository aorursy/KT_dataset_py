import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import missingno as msno

%matplotlib inline

from ggplot import *



from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from tqdm import tqdm



from subprocess import check_output



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_members = pd.read_csv('../input/members.csv')

df_songs = pd.read_csv('../input/songs.csv')

#df_sample = pd.read_csv('../input/sample_submission.csv')



type(df_train.iloc[1])
print(df_train.shape)

df_train.head()
mem = df_train.memory_usage(index=True).sum()

print("Memory consumed by train dataframe : {} MB" .format(mem/ 1024**2)) 
df_train['target'] = df_train['target'].astype(np.int8)

df_test['id'] = df_test['id'].astype(np.int32)
print(df_members.shape)

df_members.head()
mem = df_members.memory_usage(index=True).sum()

print("Memory consumed by members dataframe : {} MB" .format(mem/ 1024**2))
df_members['city'] = df_members['city'].astype(np.int8)

df_members['bd'] = df_members['bd'].astype(np.int16)

df_members['registered_via'] = df_members['registered_via'].astype(np.int8)

df_members['registration_init_time'] = df_members['registration_init_time'].astype(np.int32)

df_members['expiration_date'] = df_members['expiration_date'].astype(np.int32)
print(df_songs.shape)

df_songs.head()
mem = df_songs.memory_usage(index=True).sum()

print("Memory consumed by songs dataframe : {} MB" .format(mem/ 1024**2))
df_songs['song_length'] = df_songs['song_length'].astype(np.int32)



#-- Since language column contains Nan values we will convert it to 0,

#-- After converting the type of the column we will revert it back to nan

df_songs['language'] = df_songs['language'].fillna(0)

df_songs['language'] = df_songs['language'].astype(np.int8)



df_songs['language'] = df_songs['language'].replace(0, np.nan)

df_train_members = pd.merge(df_train, df_members, on='msno', how='inner')

df_train_merged = pd.merge(df_train_members, df_songs, on='song_id', how='outer')

print(df_train_merged.head())

print(len(df_train_merged.columns))

print('\n')

#--- Performing the same for test set ---

df_test_members = pd.merge(df_test, df_members, on='msno', how='inner')

df_test_merged = pd.merge(df_test_members, df_songs, on='song_id', how='outer')

print(df_test_merged.head())

print(len(df_test_merged.columns))
print(len(df_train.columns))

print(len(df_songs.columns))

print(len(df_members.columns))

print(len(df_test.columns))



print(len(df_train_merged.columns))

print(len(df_test_merged.columns))
del df_train

del df_test

del df_songs

del df_members
print(df_train_merged.isnull().values.any())

print(df_test_merged.isnull().values.any())
print(df_train_merged.columns[df_train_merged.isnull().any()].tolist(), '\n')

print(df_test_merged.columns[df_test_merged.isnull().any()].tolist())
#--- Removing rows having missing values in msno and target ---

df_train_merged = df_train_merged[pd.notnull(df_train_merged['msno'])]

df_train_merged = df_train_merged[pd.notnull(df_train_merged['target'])]
msno.bar(df_train_merged[df_train_merged.columns[df_train_merged.isnull().any()].tolist()],figsize=(20,8),color="#32885e",fontsize=18,labels=True,)
msno.matrix(df_train_merged[df_train_merged.columns[df_train_merged.isnull().any()].tolist()],width_ratios=(10,1),\

            figsize=(20,8),color=(0.2,0.2,0.2),fontsize=18,sparkline=True,labels=True)

 
pd.value_counts(df_train_merged['target'])
ggplot(df_train_merged, aes(x='target')) + geom_bar()
plt.figure(figsize = (8, 6))

ax = sns.countplot(x = "source_type", data = df_train_merged)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
ct = pd.crosstab(df_train_merged.source_type, df_train_merged.target)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.show()
plt.figure(figsize = (8, 6))

ax = sns.countplot(y=df_train_merged['source_screen_name'], data=df_train_merged, facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

plt.show()
ct = pd.crosstab(df_train_merged.source_screen_name, df_train_merged.target)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.show()
plt.figure(figsize = (8, 6))

ax = sns.countplot(y = "source_system_tab", data = df_train_merged)

plt.show()

''' 

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="left")

plt.tight_layout()

plt.show()

'''
ct = pd.crosstab(df_train_merged.source_system_tab, df_train_merged.target)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.show()


plt.figure(figsize = (8, 8))

pp = pd.value_counts(df_train_merged.gender)

pp.plot.pie(startangle=90, autopct='%1.1f%%', shadow=False, explode=(0.05, 0.05))

plt.axis('equal')

plt.show()
ct = pd.crosstab(df_train_merged.gender, df_train_merged.target)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.show()
plt.figure(figsize = (8, 6))

ax = sns.countplot(x = "language", data = df_train_merged)

plt.show()
plt.figure(figsize = (8, 6))

ax = sns.countplot(y = "city", data = df_train_merged)

plt.show()
plt.figure(figsize = (8, 6))

ax = sns.countplot(x = "registered_via", data = df_train_merged)

plt.show()
'''

sns.factorplot(y="source_system_tab", hue="gender", data=df_train_merged,

                   size=6, kind="bar", palette="muted")

''' 

df_train_merged.columns
ax = sns.countplot(y = df_train_merged.dtypes, data = df_train_merged)
df_train_merged['target'] = df_train_merged['target'].astype(np.int8)

df_test_merged['id'] = df_test_merged['id'].astype(np.int32)



df_train_merged['city'] = df_train_merged['city'].astype(np.int8)

df_train_merged['bd'] = df_train_merged['bd'].astype(np.int16)

df_train_merged['registered_via'] = df_train_merged['registered_via'].astype(np.int8)

df_train_merged['registration_init_time'] = df_train_merged['registration_init_time'].astype(np.int32)

df_train_merged['expiration_date'] = df_train_merged['expiration_date'].astype(np.int32)



df_test_merged['city'] = df_test_merged['city'].astype(np.int8)

df_test_merged['bd'] = df_test_merged['bd'].astype(np.int16)

df_test_merged['registered_via'] = df_test_merged['registered_via'].astype(np.int8)

df_test_merged['registration_init_time'] = df_test_merged['registration_init_time'].astype(np.int32)

df_test_merged['expiration_date'] = df_test_merged['expiration_date'].astype(np.int32)



df_train_merged['song_length'] = df_train_merged['song_length'].astype(np.int32)

#-- Since language column contains Nan values we will convert it to 0,

#-- After converting the type of the column we will revert it back to nan

df_train_merged['language'] = df_train_merged['language'].fillna(0)

df_train_merged['language'] = df_train_merged['language'].astype(np.int8)

df_train_merged['language'] = df_train_merged['language'].replace(0, np.nan)



df_test_merged['song_length'] = df_test_merged['song_length'].astype(np.int32)

#-- Since language column contains Nan values we will convert it to 0,

#-- After converting the type of the column we will revert it back to nan

df_test_merged['language'] = df_test_merged['language'].fillna(0)

df_test_merged['language'] = df_test_merged['language'].astype(np.int8)

df_test_merged['language'] = df_test_merged['language'].replace(0, np.nan)



df_test_merged.columns
ax = sns.countplot(y = df_test_merged.dtypes, data = df_train_merged)
date_cols = ['registration_init_time', 'expiration_date']

for col in date_cols:

    df_train_merged[col] = pd.to_datetime(df_train_merged[col])

    df_test_merged[col] = pd.to_datetime(df_test_merged[col])
print(len(df_train_merged))

print(df_train_merged['msno'].nunique())

#--- Function to check if missing values are present and if so print the columns having them ---

def check_missing_values(df):

    print (df.isnull().values.any())

    if (df.isnull().values.any() == True):

        columns_with_Nan = df.columns[df.isnull().any()].tolist()

    print(columns_with_Nan)

    for col in columns_with_Nan:

        print("%s : %d" % (col, df[col].isnull().sum()))

    

check_missing_values(df_train_merged)

check_missing_values(df_test_merged)
#--- Function to replace Nan values in columns of type float with -5 ---

def replace_Nan_non_object(df):

    object_cols = list(df.select_dtypes(include=['float']).columns)

    for col in object_cols:

        df[col]=df[col].fillna(np.int(-5))

       

replace_Nan_non_object(df_train_merged) 

replace_Nan_non_object(df_test_merged)  
#--- Function to replace Nan values in columns of type object with 'Others' ---

def replace_Nan_object(df):

    object_cols = list(df.select_dtypes(include=['object']).columns)

    for col in object_cols:

        df[col]=df[col].fillna(' ')

    print (object_cols)



replace_Nan_object(df_train_merged)  

replace_Nan_object(df_test_merged)  

#check_missing_values(cop)

#print(object_cols)
corr_matrix = df_train_merged.corr()
''' 

for col in tqdm(cols):

    if df_train_merged[col].dtype == 'object':

        df_train_merged[col] = df_train_merged[col].apply(str)

        df_test_merged[col] = df_test_merged[col].apply(str)



        le = LabelEncoder()

        train_vals = list(df_train_merged[col].unique())

        test_vals = list(df_test_merged[col].unique())

        le.fit(train_vals + test_vals)

        df_train_merged[col] = le.transform(df_train_merged[col])

        df_test_merged[col] = le.transform(df_test_merged[col])

'''        