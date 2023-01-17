import pandas as pd

import numpy as np

import plotly.graph_objects as go

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

import sklearn as skl

from sklearn import preprocessing as prep

from sklearn import datasets, neighbors

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
df_adult = pd.read_csv('../input/adult-pmr3508/train_data.csv', index_col='Id', na_values = '?')
display(df_adult.head(11))

print(df_adult.shape)

display(df_adult.describe())

print(df_adult.info())
print('Zero values percentage in capital.gain:',

      df_adult[df_adult['capital.gain']==0].shape[0]/df_adult.shape[0])

print('Zero values percentage in capital.loss:',

      df_adult[df_adult['capital.loss']==0].shape[0]/df_adult.shape[0])
fig1 = go.Figure()

# Add the first plot, regarding only low income

fig1.add_trace(go.Histogram(

x=df_adult[(df_adult['income'] == '<=50K') & (df_adult['capital.loss'] != 0)]['capital.loss'],

marker_color='#FF8C00',

name = '<=50K'

))

# Add the second plot, regarding only high income

fig1.add_trace(go.Histogram(

x=df_adult[(df_adult['income'] == '>50K') & (df_adult['capital.loss'] != 0)]['capital.loss'],

marker_color='#9400D3',

name = '>50K'

))

#Just some visual plotly config

fig1.update_layout(

    title_text='capital.loss'.title(),

    autosize=False,

    width=700,

    height=400)

fig1.show()



fig2 = go.Figure()

# Add the first plot, regarding only low income

fig2.add_trace(go.Histogram(

x=df_adult[(df_adult['income'] == '<=50K') & (df_adult['capital.gain'] != 0)]['capital.gain'],

marker_color='#FF8C00',

name = '<=50K'

))

# Add the second plot, regarding only high income

fig2.add_trace(go.Histogram(

x=df_adult[(df_adult['income'] == '>50K') & (df_adult['capital.gain'] != 0)]['capital.gain'],

marker_color='#9400D3',

name = '>50K'

))

#Just some visual plotly config

fig2.update_layout(

    title_text='capital.gain'.title(),

    autosize=False,

    width=700,

    height=400)

fig2.show()
# More sophisticated info on the data: histogram plots on relevant columns

for column in ['age','workclass',

 'education', 'marital.status', 'occupation',

 'relationship', 'race', 'sex', 'capital.gain',

 'capital.loss', 'native.country',]:

    print('\n',column.upper())

    # Create a plotly figure for each graph

    fig = go.Figure()

    # Add the first plot, regarding only low income

    fig.add_trace(go.Histogram(

    x=df_adult[df_adult['income'] == '<=50K'][column],

    marker_color='#FF8C00',

    name = '<=50K'

    ))

    # Add the second plot, regarding only high income

    fig.add_trace(go.Histogram(

    x=df_adult[df_adult['income'] == '>50K'][column],

    marker_color='#9400D3',

    name = '>50K'

    ))

    #Just some visual plotly config

    fig.update_layout(

        title_text=column.title(),

        autosize=False,

        width=700,

        height=400)

    fig.show()
df_adult.isna().sum()
df_adult = df_adult.dropna(axis=0)

df_adult = df_adult.drop_duplicates()
for i in sorted(df_adult['education.num'].unique().tolist()):

    print('\nNumber: ', str(i), '\nLabel: ', df_adult[df_adult['education.num'] == i]['education'].unique())
df_adult = df_adult.drop(labels = 'education', axis = 1)
df_adult.columns
df_adult['foreigner'] = (df_adult['native.country'] != 'United-States').astype('int')

df_adult = df_adult.drop(labels = 'native.country', axis = 1)

df_adult['male'] = (df_adult['sex'] == 'Male').astype('int')

df_adult = df_adult.drop(labels = 'sex', axis = 1)
df_adult['high.income'] = (df_adult['income'] == '>50K').astype('int')
df_adult = df_adult.drop(labels = 'fnlwgt', axis = 1)

df_adult.shape
df_adult = df_adult.reset_index()

df_class = df_adult[['workclass','marital.status','occupation',

                     'relationship','race','male','foreigner']].copy()

df_numerical_normal = df_adult[['age']].copy()

df_numerical = df_adult[[ 'education.num', 'hours.per.week']].copy()

df_robust =  df_adult[['capital.gain', 'capital.loss']].copy()

df_target = df_adult[['high.income']].copy()
df_class = pd.get_dummies(df_class)



df_numerical_normal = prep.StandardScaler().fit_transform(df_numerical_normal)

df_numerical_normal = prep.MinMaxScaler().fit_transform(df_numerical_normal)

df_numerical_normal = pd.DataFrame({'age' : df_numerical_normal[:, 0]})



df_numerical = prep.MinMaxScaler().fit_transform(df_numerical)

df_numerical = pd.DataFrame({'education.num' : df_numerical[:, 0],

                                 'hours.per.week' : df_numerical[:, 1]})



df_robust = prep.RobustScaler().fit_transform(df_robust)

df_robust = pd.DataFrame({'capital.gain' : df_robust[:, 0],

                       'capital.loss' : df_robust[:, 1]})



target = df_target['high.income'].to_numpy()
DFs = [df_numerical_normal,df_numerical,df_robust]

df_master = df_class.copy()

for df in DFs:

    for column in df.columns.tolist():

        df_master[column] = df[column]
df_master.shape
md_scores = []

for k in range(10,40):

    print('\n Running for k = ', k)

    classifier = KNeighborsClassifier(n_neighbors=k)

    score = cross_val_score(classifier, df_master, target, scoring="accuracy").mean()

    print('Accuracy: ', score)

    md_scores.append(score)
# Importing

df_test = pd.read_csv('../input/adult-pmr3508/test_data.csv', index_col='Id', na_values = '?')

# Basic drops



# df_test = df_test.drop_duplicates()

df_test = df_test.drop(labels = 'education', axis = 1)

# Columns direct changes

df_test['foreigner'] = (df_test['native.country'] != 'United-States').astype('int')

df_test = df_test.drop(labels = 'native.country', axis = 1)

df_test['male'] = (df_test['sex'] == 'Male').astype('int')

df_test = df_test.drop(labels = 'sex', axis = 1)

df_test = df_test.drop(labels = 'fnlwgt', axis = 1)

df_test = df_test.reset_index()

# Creation of sub data frames

df_class = df_test[['workclass','marital.status','occupation',

                     'relationship','race','male','foreigner']].copy()

df_numerical_normal = df_test[['age']].copy()

df_numerical = df_test[[ 'education.num', 'hours.per.week']].copy()

df_robust =  df_test[['capital.gain', 'capital.loss']].copy()

# Sub data frames treatment

df_class = pd.get_dummies(df_class)

df_numerical_normal = prep.StandardScaler().fit_transform(df_numerical_normal)

df_numerical_normal = prep.MinMaxScaler().fit_transform(df_numerical_normal)

df_numerical_normal = pd.DataFrame({'age' : df_numerical_normal[:, 0]})

df_numerical = prep.MinMaxScaler().fit_transform(df_numerical)

df_numerical = pd.DataFrame({'education.num' : df_numerical[:, 0],

                                 'hours.per.week' : df_numerical[:, 1]})

df_robust = prep.RobustScaler().fit_transform(df_robust)

df_robust = pd.DataFrame({'capital.gain' : df_robust[:, 0],

                       'capital.loss' : df_robust[:, 1]})

# Master Table assembly

DFs = [df_numerical_normal,df_numerical,df_robust]

df_master_test = df_class.copy()

for df in DFs:

    for column in df.columns.tolist():

        df_master_test[column] = df[column]

df_master_test = df_master_test.drop(labels='workclass_Never-worked',axis=1)

classifier = KNeighborsClassifier(n_neighbors=22)

classifier.fit(df_master,target)

results = classifier.predict(df_master_test)
export = pd.DataFrame()

export['id'] = df_master_test.index

export['income'] = results

export = export.set_index('id')
export[export['income'] == 0] = '<=50K'

export[export['income'] == 1] = '>50K'

export
export.to_csv("submission.csv",index_label = "id")