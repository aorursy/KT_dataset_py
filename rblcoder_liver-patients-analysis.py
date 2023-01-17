import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

sns.set()
df_liver = pd.read_csv('../input/indian_liver_patient.csv')

print(df_liver.info())

print(df_liver.isnull().sum())

print(df_liver.head().T)
_=sns.pairplot(df_liver, hue="Dataset")
# print(df_liver.Dataset.value_counts())

# df_counts = pd.DataFrame(df_liver.Dataset.replace([1, 2], ['Yes', 'No']).value_counts()).reset_index()

# df_counts.columns = ['With Liver Disease', 'Count']

# fig = px.bar(df_counts, x='With Liver Disease', y='Count')

# fig.show()

df_liver['With Liver Disease']=df_liver.Dataset.replace([1, 2], ['Yes', 'No'])

_=sns.countplot(y=df_liver['With Liver Disease'])
# df_counts = pd.DataFrame(df_liver.Gender.value_counts()).reset_index()

# df_counts.columns = ['Gender', 'Count']

# fig = px.bar(df_counts, x='Gender', y='Count')

# fig.show()



_=sns.countplot(y=df_liver['Gender'])
_=sns.countplot(y=df_liver['With Liver Disease'], hue=df_liver['Gender'])
df_liver.groupby('Dataset').mean()
df_liver.columns
for col in ['Total_Bilirubin', 'Direct_Bilirubin',

       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',

       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',

       'Albumin_and_Globulin_Ratio']:

    

    fig = px.scatter(df_liver, x="Age", y=col, facet_col="Dataset", facet_row="Gender")

    fig.update_xaxes(showgrid=False, zeroline=False)

    fig.update_yaxes(showgrid=False, zeroline=False)

    fig.show()

for col in ['Total_Bilirubin', 'Direct_Bilirubin',

       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',

       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',

       'Albumin_and_Globulin_Ratio']:

    fig = px.histogram(df_liver, x=col, color="Dataset", marginal="rug", hover_data=df_liver.columns)

    fig.show()
for col in ['Total_Bilirubin', 'Direct_Bilirubin',

       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',

       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',

       'Albumin_and_Globulin_Ratio','Age']:

    fig = px.strip(df_liver, x=col, y="Gender", orientation="h", color="Dataset")

    fig.show()
fig = px.histogram(df_liver, x="Gender", y="Age", histfunc='avg', facet_col='Dataset'

                   ,marginal="histogram")

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))



fig.show()
_=df_liver.groupby('Dataset').Gender.value_counts().plot(kind='bar')
_=df_liver.groupby('Dataset').mean().T.plot(kind='bar')
ax = sns.boxplot(data=df_liver[['Age','Total_Bilirubin', 'Direct_Bilirubin',

       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',

       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',

       'Albumin_and_Globulin_Ratio']], orient="h")

ax.set_xscale('log')
#https://stackoverflow.com/questions/37191983/python-side-by-side-box-plots-on-same-figure

fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))

_=df_liver[['Dataset', 'Age','Total_Bilirubin', 'Direct_Bilirubin',

       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',

       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',

       'Albumin_and_Globulin_Ratio']].query("Dataset in [1, 2]").boxplot(by='Dataset', return_type='axes', ax=axes)