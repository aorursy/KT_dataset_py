import numpy as np 

import pandas as pd 

import plotly.express as px

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, plot_confusion_matrix

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn import preprocessing

import optuna

from optuna.samplers import TPESampler
train = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/train_data.csv')

test = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/test_data.csv')

sub = pd.read_csv('/kaggle/input/av-healthcare-analytics-ii/healthcare/sample_sub.csv')



train = train.drop(['case_id'], axis=1)

test = test.drop(['case_id'], axis=1)

train['dataset'] = 'train'

test['dataset'] = 'test'



df = pd.concat([train, test])
df
ds = df.groupby(['Hospital_code', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['hospital', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='hospital', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Cases per hospital distribution', 

    width=900,

    height=700

)

fig.show()
ds = df.groupby(['Hospital_type_code', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['hospital', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='hospital', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Cases hospital type distribution', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Hospital_region_code', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['hospital', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='hospital', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Cases hospital region distribution', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Department', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['department', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='department', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Department distribution', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Ward_Type', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['Ward_Type', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='Ward_Type', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Ward Type distribution', 

    width=900,

    height=600

)

fig.show()
ds = ds[ds['dataset']=='train']

fig = px.pie(

    ds, 

    names='Ward_Type', 

    values="count", 

    title='Ward type pie chart for train set', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Ward_Facility_Code', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['Ward_Facility_Code', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='Ward_Facility_Code', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Ward Facility Code distribution', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Bed Grade', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['bed_grade', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='bed_grade', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Bed_grade distribution', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Age', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['age', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='age', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Age distribution', 

    width=900,

    height=600

)

fig.show()


ds = df.groupby(['Type of Admission', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['admission', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='admission', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Admission type distribution', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Severity of Illness', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['Severity of Illness', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='Severity of Illness', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Severity of Illness type distribution', 

    width=900,

    height=600

)

fig.show()
ds = df.groupby(['Stay', 'dataset'])['patientid'].count().reset_index()

ds.columns = ['Stay', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='Stay', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='Stay length distribution', 

    width=900,

    height=600

)

fig.show()
data = df['patientid'].value_counts().reset_index()

data.columns = ['patientid', 'cases']

data['patientid'] = 'patient ' + data['patientid'].astype(str)

data = data.sort_values('cases')

fig = px.bar(

    data.tail(50), 

    x="cases", 

    y="patientid", 

    orientation='h', 

    title='Top 50 patients',

    width=800,

    height=900

)

fig.show()
fig = px.histogram(

    df, 

    "City_Code_Patient", 

    nbins=40, 

    color = 'dataset',

    barmode='group',

    title='City_Code_Patient', 

    width=700,

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "Visitors with Patient", 

    nbins=40, 

    color = 'dataset',

    barmode='group',

    title='Visitors with Patient', 

    width=700,

    height=600

)

fig.show()
fig = px.histogram(

    df, 

    "Admission_Deposit", 

    nbins=50, 

    color = 'dataset',

    barmode='group',

    title='Admission Deposit destribution', 

    width=700,

    height=600

)

fig.show()
df.loc[df['Stay'] == '0-10', 'Stay'] = 0

df.loc[df['Stay'] == '11-20', 'Stay'] = 1

df.loc[df['Stay'] == '21-30', 'Stay'] = 2

df.loc[df['Stay'] == '31-40', 'Stay'] = 3

df.loc[df['Stay'] == '41-50', 'Stay'] = 4

df.loc[df['Stay'] == '51-60', 'Stay'] = 5

df.loc[df['Stay'] == '61-70', 'Stay'] = 6

df.loc[df['Stay'] == '71-80', 'Stay'] = 7

df.loc[df['Stay'] == '81-90', 'Stay'] = 8

df.loc[df['Stay'] == '91-100', 'Stay'] = 9

df.loc[df['Stay'] == 'More than 100 Days', 'Stay'] = 10
train = df[df['dataset']=='train']

test = df[df['dataset']=='test']



target = train['Stay']



features = ['Available Extra Rooms in Hospital', 'Bed Grade', 'Visitors with Patient', 'Admission_Deposit']



train = train[features]

train = train.fillna(0)

test = test[features]
X, X_val, y, y_val = train_test_split(train, target, random_state=0, test_size=0.2, shuffle=True)

y=y.astype('int')

y_val=y_val.astype('int')
model = LogisticRegression(random_state=666)

model.fit(X, y)

preds = model.predict(X_val)

print('Baseline accuracy: ', accuracy_score(y_val, preds)*100, '%')
fig, ax = plt.subplots(figsize=(10, 10))

plot_confusion_matrix(model, X_val, y_val, ax=ax)
need_to_encode = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness']

for column in need_to_encode:

    le = preprocessing.LabelEncoder()

    le.fit(df[column])

    df[column] = le.transform(df[column])
df.loc[df['Age'] == '0-10', 'Age'] = 0

df.loc[df['Age'] == '11-20', 'Age'] = 1

df.loc[df['Age'] == '21-30', 'Age'] = 2

df.loc[df['Age'] == '31-40', 'Age'] = 3

df.loc[df['Age'] == '41-50', 'Age'] = 4

df.loc[df['Age'] == '51-60', 'Age'] = 5

df.loc[df['Age'] == '61-70', 'Age'] = 6

df.loc[df['Age'] == '71-80', 'Age'] = 7

df.loc[df['Age'] == '81-90', 'Age'] = 8

df.loc[df['Age'] == '91-100', 'Age'] = 9
categorical = ['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 

              'City_Code_Patient', 'Type of Admission', 'Severity of Illness']
train = df[df['dataset']=='train']

test = df[df['dataset']=='test']



target = train['Stay']

train = train.fillna(0)

test = test.fillna(0)

train = train.drop(['patientid', 'dataset', 'Stay'], axis=1)

test = test.drop(['patientid', 'dataset'], axis=1)

train
X, X_val, y, y_val = train_test_split(train, target, random_state=0, test_size=0.2, shuffle=True)

y=y.astype('int')

y_val=y_val.astype('int')
model = LGBMClassifier(random_state=666)

model.fit(X, y, categorical_feature=categorical)

preds = model.predict(X_val)

print('LGBM accuracy: ', accuracy_score(y_val, preds)*100, '%')
fig, ax = plt.subplots(figsize=(10, 10))

plot_confusion_matrix(model, X_val, y_val, ax=ax)
sampler = TPESampler(seed=0)

def create_model(trial):

    max_depth = trial.suggest_int("max_depth", 2, 30)

    n_estimators = trial.suggest_int("n_estimators", 1, 500)

    learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)

    num_leaves = trial.suggest_int("num_leaves", 2, 5000)

    min_child_samples = trial.suggest_int('min_child_samples', 3, 200)

    model = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves, min_child_samples=min_child_samples,

                           random_state=0)

    return model



def objective(trial):

    model = create_model(trial)

    model.fit(X, y)

    preds = model.predict(X_val)

    return accuracy_score(y_val, preds)



def optimize():

    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=20)

    return study.best_params



params = optimize()
params['random_state'] = 666

model = LGBMClassifier(**params)

model.fit(X, y, categorical_feature=categorical)

preds = model.predict(X_val)

print('LGBM accuracy: ', accuracy_score(y_val, preds)*100, '%')
fig, ax = plt.subplots(figsize=(10, 10))

plot_confusion_matrix(model, X_val, y_val, ax=ax)
X.columns
test.columns
preds = model.predict(test.drop('Stay',axis=1))
sub['Stay']=preds
sub.loc[sub['Stay'] == 0, 'Stay'] = '0-10'

sub.loc[sub['Stay'] == 1, 'Stay'] = '11-20'

sub.loc[sub['Stay'] == 2, 'Stay'] = '21-30'

sub.loc[sub['Stay'] == 3, 'Stay'] = '31-40'

sub.loc[sub['Stay'] == 4, 'Stay'] = '41-50'

sub.loc[sub['Stay'] == 5, 'Stay'] = '51-60'

sub.loc[sub['Stay'] == 6, 'Stay'] = '61-70'

sub.loc[sub['Stay'] == 7, 'Stay'] = '71-80'

sub.loc[sub['Stay'] == 8, 'Stay'] = '81-90'

sub.loc[sub['Stay'] == 9, 'Stay'] = '91-100'

sub.loc[sub['Stay'] == 10, 'Stay'] = 'More than 100 Days'
sub.to_csv('lgbm.csv',index=False)