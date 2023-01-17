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
df = pd.read_csv('../input/ckdisease/kidney_disease.csv')
df.shape
df.info()
df.head().T
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
cols_names={"bp":"blood_pressure",

          "sg":"specific_gravity",

          "al":"albumin",

          "su":"sugar",

          "rbc":"red_blood_cells",

          "pc":"pus_cell",

          "pcc":"pus_cell_clumps",

          "ba":"bacteria",

          "bgr":"blood_glucose_random",

          "bu":"blood_urea",

          "sc":"serum_creatinine",

          "sod":"sodium",

          "pot":"potassium",

          "hemo":"haemoglobin",

          "pcv":"packed_cell_volume",

          "wc":"white_blood_cell_count",

          "rc":"red_blood_cell_count",

          "htn":"hypertension",

          "dm":"diabetes_mellitus",

          "cad":"coronary_artery_disease",

          "appet":"appetite",

          "pe":"pedal_edema",

          "ane":"anemia"}



df.rename(columns=cols_names, inplace=True)
print(f"So we have {df.shape[1]} columns and {df.shape[0]} instances")
df.head().T
df.info()
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')

df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')

df.drop(["id"],axis=1,inplace=True) 
df.isnull().sum().sort_values(ascending=False)
((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False).plot(kind='bar', figsize=(10,10))
for i in df.columns:

    print(f'{i} : {df[i].nunique()} values')
numerical_features = []

categorical_features = []



for i in df.columns:

    if df[i].nunique()>7:

        numerical_features.append(i)

    else:

        categorical_features.append(i)
print('Numerical features: ', numerical_features)

print('\nCategorical features: ', categorical_features)
for feats in categorical_features:

    print(f'{feats} has {df[feats].unique()} categories.\n')
#Replace incorrect values

df['diabetes_mellitus'] = df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'})

df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')

df['classification'] = df['classification'].replace(to_replace = 'ckd\t', value = 'ckd')
for feats in categorical_features:

    print(f'{feats} has {df[feats].unique()} categories.\n')
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,15))

fig.subplots_adjust(hspace=0.5)

fig.suptitle('Distributions of numerical Features')





for ax, feats in zip(axes.flatten(), numerical_features):

    sns.distplot(a=df[feats], ax=ax)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15,15))

fig.subplots_adjust(hspace=0.5)

fig.suptitle('Distributions of categorical Features')





for ax, feats in zip(axes.flatten(), categorical_features):

    sns.countplot(df[feats], ax=ax)
sns.countplot(x='classification',data=df)

plt.xlabel("classification")

plt.ylabel("Count")

plt.title("target Class")

plt.show()

print('Percent of chronic kidney disease sample: ',round(len(df[df['classification']=='ckd'])/len(df['classification'])*100,2),"%")

print('Percent of not a chronic kidney disease sample: ',round(len(df[df['classification']=='notckd'])/len(df['classification'])*100,2),"%")
corr_df = df.corr()

f,ax=plt.subplots(figsize=(15,15))

mask = np.zeros_like(corr_df)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_df,annot=True,fmt=".2f",ax=ax,linewidths=0.5,linecolor="orange", mask = mask, square=True)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.title('Correlations between different predictors')

plt.show()
import plotly.express as px
# Defining violin and scatter plot functions

def violin(col):

    fig = px.violin(df, y=col, x="classification", color="classification", box=True, points="all", hover_data=df.columns)

    return fig.show()



def scatters(col1,col2):

    fig = px.scatter(df, x=col1, y=col2, color="classification")

    fig.show()
scatters('red_blood_cell_count', 'packed_cell_volume')
scatters('red_blood_cell_count', 'haemoglobin')
scatters('haemoglobin','packed_cell_volume')
violin('red_blood_cell_count')
violin('packed_cell_volume')
violin('haemoglobin')
violin('serum_creatinine')
scatters('red_blood_cell_count','albumin')
scatters('packed_cell_volume','blood_urea')
scatters('haemoglobin','blood_urea')
scatters('red_blood_cell_count','packed_cell_volume')
fig = px.bar(df, x="specific_gravity", y="packed_cell_volume",

             color='classification', barmode='group',

             height=400)

fig.show()
scatters('serum_creatinine','sodium')
import missingno as msno
%matplotlib inline
fig = msno.matrix(df)
msno.heatmap(df[numerical_features])
msno.bar(df)