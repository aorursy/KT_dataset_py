#!pip install seaborn==0.9.0
import pandas as pd
import numpy as np
import os 

import plotly.plotly as py
import plotly.graph_objs as go

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')

# Private DataSet path: ../input/kddbr2018dataset/kddbr-2018-dataset/dataset. This dataset is the same of competitions
#
path = '../input/kddbr2018dataset/kddbr-2018-dataset/dataset/'
print(os.listdir(path))
df_train = pd.read_csv(path+'train.csv')
df_test  = pd.read_csv(path+'test.csv')
df_all   = pd.concat([df_train, df_test])

print(df_train.shape, df_test.shape, df_all.shape)
df_all.head(3)
def to_date(df):
    return pd.to_datetime((df.harvest_year*10000+df.harvest_month*100+1)\
                                  .apply(str),format='%Y%m%d')
# Add date variable 
for d in [df_train, df_test, df_all]:
    d['date'] = to_date(d)
sns.distplot(df_all.production, hist=False, color="g",rug=True, kde_kws={"shade": True})
#sns.boxplot(x="production", y="field", data=df_train, palette="vlag")
print("Mean: ", df_all.production.mean())
#df_all = df_all[(df_all.production < (df_all.production.mean()+df_all.production.std()*4)) | (df_all.production.isna())]
#data = [go.Scatter(x=df_all.date, y=df_all.production)]
#py.iplot(data)

f, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='date', y="production", data=df_all, palette="tab10", linewidth=2.5)

sns.despine(left=True)
#df_all = df_all[df_all.harvest_year >= 2006]
df_all.shape
f, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='harvest_month', y="production", data=df_all)

sns.despine(left=True)
mean_production = df_all.groupby(['harvest_month']).mean()['production'].reset_index()
mean_production.columns = ['harvest_month', 'production_mean']
mean_production

print(mean_production.shape)
mean_production.head(2)
f, ax = plt.subplots(figsize=(12, 6))
#sns.lineplot(x='age', y="production", data=df_group, palette="tab10", linewidth=2.5)
sns.lineplot(x='age', y="production", data=df_all, palette="tab10", linewidth=2.5)

sns.despine(left=True)
df_all.type.unique()
f, ax = plt.subplots(figsize=(12, 6))
#print(df_all.groupby(['type'])['date'].count())
sns.countplot(x="type", palette="tab10", data=df_all)
ordered_days = df_all.type.value_counts().index
g = sns.FacetGrid(df_all, row="type", row_order=ordered_days, height=1.7, aspect=4,)
g.map(sns.distplot, "production", hist=False, rug=True, kde_kws={"shade": True});
type_prod = df_all.groupby(['date', 'type']).mean()['production'].reset_index()

print(mean_production.shape)
mean_production.head(2)

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(type_prod, col="type", hue="type", palette="tab10",
                     col_wrap=3, height=2.5)

grid.map(plt.plot, "date", "production")

grid.fig.tight_layout(w_pad=1)
sns.despine(left=True)
f, ax = plt.subplots(figsize=(12, 6))
#sns.lineplot(x='age', y="production", data=df_group, palette="tab10", linewidth=2.5)
sns.lineplot(x='age', y="production", data=df_all, hue='type', palette="tab10", linewidth=2.5, legend='full')

sns.despine(left=True)
df_all.field.unique()
f, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x="field", data=df_all)
fields    = [0, 9, 27] 
df_filter = df_all[df_all.field.isin(fields)]

f, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='date', y="production", data=df_filter, hue='field', palette="tab10", linewidth=2.5, legend='full')

sns.despine(left=True)
field_prod = df_all.groupby(['date', 'field']).mean()['production'].reset_index()

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(field_prod, col="field", hue="field", palette="tab10",
                     col_wrap=6, height=2.5)

grid.map(plt.plot, "date", "production")

grid.fig.tight_layout(w_pad=1)
sns.despine(left=True)

f, ax = plt.subplots(figsize=(12, 6))

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="field", y="production", data=df_all)
sns.despine(offset=10, trim=True)
# read
df_field = pd.read_csv(path+'field-0.csv')
df_field['field'] = 0
for i in range(1, 28):
    _df_field = pd.read_csv(path+'field-{}.csv'.format(i))
    _df_field['field'] = i
    df_field = pd.concat([df_field, _df_field])

# remove duplicates
df_field = df_field.drop_duplicates()

# Group 
df_field = df_field.groupby(['month', 'year', 'field']).mean().reset_index()
print(df_field.shape)
df_field.head()
# df_all
df_all   = pd.merge(df_all, df_field, left_on=['harvest_year', 'harvest_month','field'], 
                    right_on=['year', 'month', 'field'], how='inner').reset_index()

print(df_all.shape)
df_all.head()
df_all.columns
f, ax = plt.subplots(figsize=(10, 10))
features  = ['temperature', 'dewpoint',
               'windspeed', 'Soilwater_L1', 'Soilwater_L2', 'Soilwater_L3',
               'Soilwater_L4', 'Precipitation']
corr = df_all[features+['production']].corr()

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, linewidths=.5,cmap="YlGnBu")
# Features i will duplicate with the past months
features  = ['temperature', 'dewpoint', 'windspeed', 'Precipitation', 'Soilwater_L1']

df_all    = df_all.drop(columns=['Soilwater_L2', 'Soilwater_L3','Soilwater_L4'])
#df_all2 = df_all.copy()
df_all.head()
df_group = df_all.groupby(['field', 'date']).mean().reset_index()[['field', 'date', 'production'] + features ]
df_group = df_group.sort_values(['field', 'date'])
print(df_group.shape)
df_group.head()
# Collect shift values of variables in all features time
period = 2

new_features = {}
for f in features:
    new_features[f] = []
    for i in range(1, period):
        new_features[f].append('{}_{}'.format(f, i))
        df_group['{}_{}'.format(f, i)] = df_group[f].shift(i).fillna(df_group[f].mean())
        #df_group['{}_{}'.format(f, i)] = df_group[f].rolling(i, min_periods=1).mean().fillna(df_group.temperature.mean())
fig = plt.figure(figsize=(18, 8))

for i in range(1, len(features)+1):
    ax1 = fig.add_subplot(240+i)
    
    f = features[i-1]
    f_filter = [f] + new_features[f]+['production']
    corr    = df_group[f_filter].corr()
    g = sns.barplot(x=[i-1 for i in range(1, period+1)], y=corr['production'].values[:-1], palette="YlGnBu", ax=ax1)
    plt.title(f)
    plt.xticks(rotation=45)
plt.show() #corr['production'].keys().values[:-1]
df_all.head()
df_group= df_group.drop(features+['production'], axis=1)
df_group.head()
df_all = df_all.drop(['index', 'month', 'year'], axis=1)
df_all = pd.merge(df_all, df_group, left_on=['field', 'date'], right_on=['field','date'], how='inner').reset_index()

print(df_all.shape)
df_all.head()
df_soil = pd.read_csv(path+'soil_data.csv')
print(df_soil.shape)
df_soil.head()
# Join datasets
df_all_soil = pd.merge(df_all, df_soil, on='field', how='inner')
print(df_all_soil.shape)
df_all_soil.head()
f, ax = plt.subplots(figsize=(10, 10))
features  = list(df_soil.columns.values) + ['production']
corr      = df_all_soil[features].corr()

#Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, linewidths=.5,cmap="YlGnBu")
df_all.columns
## Import the random forest model.
from sklearn.ensemble import RandomForestRegressor

## This line instantiates the model. 
rf = RandomForestRegressor() 

# data
df      = df_all_soil[~df_all_soil.production.isna()]
X_train = df.drop(['production', 'date', 'Id', 'index'], axis=1)
y_train = df.production.values

## Fit the model on your training data.
rf.fit(X_train, y_train) 
# feature_importances
feature_importances = pd.DataFrame(rf.feature_importances_, 
                                   index = X_train.columns, 
                                   columns=['importance']).sort_values('importance', ascending=False).reset_index()
feature_importances.head(3)
size = 50
f, ax = plt.subplots(figsize=(10, 10))
sns.barplot(x="importance", y='index', data=feature_importances.head(size), palette="rocket")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
# Load Dataset
# y~X
df_train = df_all_soil[~df_all_soil.production.isna()]
X        = df_train.drop(['production', 'date', 'Id'], axis=1)

#Filter importance
features = list(feature_importances['index'].values)[:15]
X        = X[features]
# y
y        = df.production.values

# normalize
scaler = StandardScaler()
norm_X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(norm_X, y, test_size=0.2, random_state=1)
(X_train.shape, X_test.shape)
base_model = RandomForestRegressor()
base_model.fit(X_train, y_train)
y_hat = base_model.predict(X_test)

score_mae = sklearn.metrics.mean_absolute_error(y_test, y_hat)
r2        = sklearn.metrics.r2_score(y_test, y_hat)

#MAE score 0.04333937538802412
print("MAE score", score_mae)
sns.jointplot(x=y_test, y=y_hat, kind="reg", color="m", height=7)
#sns.scatterplot(y=(y_test-y_hat), x=[i for i in range(len(y_test))], ax=ax2)
#df_all.to_csv('dataset/df_all.csv')
df_test = df_all[df_all.production.isna()]
print(df_test.shape)
df_test.tail(2)
#Filter importance
X  = df_test[features]
X  = scaler.transform(X) # normalize

# y
y = df.production.values
prod = base_model.predict(X) 
prod[:10]
## create a submission.csv
import math
f = open('submission.csv', 'w')
f.write("Id,production\n")
for i in range(len(df_test.Id.values)):
    _id = df_test.Id.values[i]
    p   = math.fabs(prod[i])
    f.write("{},{}\n".format(_id, p))
f.close()

