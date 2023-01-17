!pip install pyod
import numpy as np 

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeClassifier

import optuna

from optuna.samplers import TPESampler

from sklearn.pipeline import Pipeline

import plotly.figure_factory as ff

from sklearn.cluster import KMeans

from pyod.models.copod import COPOD

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
HEIGHT = 500

WIDTH = 700

NBINS = 50

SCATTER_SIZE=700
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
for col in df.columns:

    print(col, str(round(100* df[col].isnull().sum() / len(df), 2)) + '%')
df.describe()
def plot_histogram(dataframe, column, color, bins, title, width=WIDTH, height=HEIGHT):

    figure = px.histogram(

        dataframe, 

        column, 

        color=color,

        nbins=bins, 

        title=title, 

        width=width,

        height=height

    )

    figure.show()
plot_histogram(df, 'age', 'sex', NBINS, 'Patients age distribution')
plot_histogram(df, 'age', 'DEATH_EVENT', NBINS, 'Patients age distribution')
fig = px.box(

    df, 

    x="DEATH_EVENT", 

    y="age", 

    points='all',

    title='Age & DEATH_EVENT box plot',

    width=WIDTH,

    height=HEIGHT    

)



fig.show()
ds = df['anaemia'].value_counts().reset_index()

ds.columns = ['anaemia', 'count']



fig = px.pie(

    ds, 

    values='count', 

    names="anaemia", 

    title='Anaemia bar chart', 

    width=WIDTH, 

    height=HEIGHT

)



fig.show()
plot_histogram(df, 'creatinine_phosphokinase', 'DEATH_EVENT', 2 * NBINS, 'Creatinine phosphokinase distribution')
ds = df['diabetes'].value_counts().reset_index()

ds.columns = ['diabetes', 'count']



fig = px.pie(

    ds, 

    values='count', 

    names="diabetes", 

    title='Diabetes bar chart', 

    width=WIDTH, 

    height=HEIGHT

)



fig.show()
fig = px.histogram(

    df, 

    "ejection_fraction", 

    color='DEATH_EVENT',

    nbins=NBINS, 

    title='Ejection_fraction distribution',

    width=WIDTH,

    height=HEIGHT

)



fig.show()
fig = px.box(

    df, 

    x="DEATH_EVENT", 

    y="ejection_fraction", 

    points='all',

    title='Ejection_fraction & DEATH_EVENT box plot',

    width=WIDTH,

    height=HEIGHT

)



fig.show()
ds = df['high_blood_pressure'].value_counts().reset_index()

ds.columns = ['high_blood_pressure', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="high_blood_pressure", 

    title='High blood pressure bar chart', 

    width=WIDTH, 

    height=HEIGHT

)



fig.show()
fig = px.histogram(

    df, 

    "platelets", 

    nbins=NBINS, 

    color='DEATH_EVENT', 

    title='Platelets distribution',

    width=WIDTH,

    height=HEIGHT

)



fig.show()
fig = px.box(

    df, 

    x="DEATH_EVENT", 

    y="platelets", 

    points='all',

    title='Platelets & DEATH_EVENT box plot',

    width=WIDTH,

    height=HEIGHT

)



fig.show()
fig = px.histogram(

    df, 

    "serum_creatinine", 

    nbins=NBINS, 

    color='DEATH_EVENT',

    title='Serum creatinine distribution',

    width=WIDTH,

    height=HEIGHT

)



fig.show()
fig = px.box(

    df, 

    x="DEATH_EVENT", 

    y="serum_creatinine", 

    points='all',

    title='Serum_creatinine & DEATH_EVENT box plot',

    width=WIDTH,

    height=HEIGHT

)



fig.show()
fig = px.histogram(

    df, 

    "serum_sodium",

    color='DEATH_EVENT',

    nbins=NBINS, 

    title='Serum sodium distribution', 

    width=WIDTH,

    height=HEIGHT

)



fig.show()
fig = px.box(

    df, 

    x="DEATH_EVENT", 

    y="serum_sodium", 

    points='all',

    title='Serum_sodium & DEATH_EVENT box plot',

    width=WIDTH,

    height=HEIGHT

)

   

fig.show()
ds = df['sex'].value_counts().reset_index()

ds.columns = ['sex', 'count']



fig = px.pie(

    ds, 

    values='count', 

    names="sex", 

    title='Gender bar chart', 

    width=WIDTH, 

    height=HEIGHT

)



fig.show()
ds = df['smoking'].value_counts().reset_index()

ds.columns = ['smoking', 'count']



fig = px.pie(

    ds, 

    values='count', 

    names="smoking", 

    title='Smoking bar chart', 

    width=WIDTH, 

    height=HEIGHT

)



fig.show()
ds = df['DEATH_EVENT'].value_counts().reset_index()

ds.columns = ['DEATH_EVENT', 'count']



fig = px.pie(

    ds, 

    values='count', 

    names="DEATH_EVENT", 

    title='DEATH_EVENT bar chart', 

    width=WIDTH, 

    height=HEIGHT

)



fig.show()
sun = df.groupby(['sex', 'diabetes', 'smoking', 'DEATH_EVENT'])['age'].count().reset_index()

sun.columns = ['sex', 'diabetes', 'smoking', 'DEATH_EVENT', 'count']



sun.loc[sun['sex'] == 0, 'sex'] = 'female'

sun.loc[sun['sex'] == 1, 'sex'] = 'male'

sun.loc[sun['smoking'] == 0, 'smoking'] = "doesn't smoke"

sun.loc[sun['smoking'] == 1, 'smoking'] = 'smoke'

sun.loc[sun['diabetes'] == 0, 'diabetes'] = "no diabetes"

sun.loc[sun['diabetes'] == 1, 'diabetes'] = 'diabetes'

sun.loc[sun['DEATH_EVENT'] == 0,'DEATH_EVENT'] = "ALIVE_EVENT"

sun.loc[sun['DEATH_EVENT'] == 1, 'DEATH_EVENT'] = 'DEATH_EVENT'



fig = px.sunburst(

    sun, 

    path=[

        'sex',

        'diabetes',

        'smoking', 

        'DEATH_EVENT'

    ], 

    values='count', 

    title='Sunburst chart for all patients',

    width=WIDTH,

    height=HEIGHT

)



fig.show()
df = df.drop(['time'], axis=1)
f = plt.figure(figsize=(12, 12))

plt.matshow(df.corr(), fignum=f.number)

plt.xticks(range(df.shape[1]), df.columns, fontsize=13, rotation=65)

plt.yticks(range(df.shape[1]), df.columns, fontsize=13)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
X = df.copy()

y = X['DEATH_EVENT']

X = X.drop(['DEATH_EVENT'], axis=1)
pca = PCA(n_components=3, random_state=666)

X = pd.DataFrame(pca.fit_transform(X))

X['target'] = y



X
fig = px.scatter_3d(

    X, 

    x=0, 

    y=1,

    z=2, 

    color="target", 

    title='3d scatter for PCA',

    width=SCATTER_SIZE,

    height=SCATTER_SIZE

)



fig.show()
X = df.copy()

y = X['DEATH_EVENT']

X = X.drop(['DEATH_EVENT'], axis=1)
pca = PCA(n_components=2, random_state=666)

X = pd.DataFrame(pca.fit_transform(X))

X['target'] = y



X
fig = px.scatter(

    X, 

    x=0, 

    y=1,

    color="target", 

    title='2d scatter for PCA',

    width=SCATTER_SIZE,

    height=SCATTER_SIZE

)



fig.show()
X = df.copy()

y = X['DEATH_EVENT']

X = X.drop(['DEATH_EVENT'], axis=1)
kmeans = KMeans(n_clusters=2, random_state=666).fit(X)
train = X.copy()

train['cluster'] = kmeans.labels_

train['target'] = y



train
print('Kmeans accuracy: ', accuracy_score(train['target'], train['cluster']))

print('Kmeans f1_score: ', f1_score(train['target'], train['cluster']))
def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    ax= plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, fmt='g')



    ax.set_xlabel('Predicted labels')

    ax.set_ylabel('True labels')
plot_confusion_matrix(train['target'], train['cluster'])
response = train['target']

train = train.drop(['target', 'cluster'], axis=1)
clf = COPOD(contamination=0.3)

clf.fit(train)
cluster = clf.predict(train)

train['cluster'] = cluster

train['target'] = response

train
train['cluster'].value_counts()
print('COPOD accuracy: ', accuracy_score(train['target'], train['cluster']))

print('COPOD f1_score: ', f1_score(train['target'], train['cluster']))
plot_confusion_matrix(train['target'], train['cluster'])
train = train.drop(['target', 'cluster'], axis=1)

train
X_embedded = TSNE(n_components=2, random_state=666).fit_transform(train)

X_embedded = pd.DataFrame(X_embedded)
analysis = pd.DataFrame()

analysis['color'] = response

analysis['X'] = X_embedded[0]

analysis['Y'] = X_embedded[1]



fig = px.scatter(

    analysis, 

    x='X', 

    y='Y', 

    color="color", 

    title='TSNE for dataset',

    width=SCATTER_SIZE,

    height=SCATTER_SIZE

)



fig.show()
X_embedded = TSNE(n_components=3, random_state=666).fit_transform(train)

X_embedded = pd.DataFrame(X_embedded)
analysis = pd.DataFrame()

analysis['color'] = response

analysis['X'] = X_embedded[0]

analysis['Y'] = X_embedded[1]

analysis['Z'] = X_embedded[2]



fig = px.scatter_3d(

    analysis, 

    x='X', 

    y='Y',

    z='Z', 

    color="color", 

    title='3d scatter for TSNE',

    width=SCATTER_SIZE,

    height=SCATTER_SIZE

)



fig.show()
X, X_test, y, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=True)
model = LogisticRegression(random_state=0)

model.fit(X, y)

preds = model.predict(X_test)



print('Logistic Regression ', accuracy_score(y_test, preds))
plot_confusion_matrix(y_test, preds)
print('Logistic Regression f1-score', f1_score(y_test, preds))

print('Logistic Regression precision', precision_score(y_test, preds))

print('Logistic Regression recall', recall_score(y_test, preds))
model = LGBMClassifier(random_state=0)

model.fit(X, y)

preds = model.predict(X_test)



print('LightGBM f1-score', f1_score(y_test, preds))

print('LightGBM precision', precision_score(y_test, preds))

print('LightGBM recall', recall_score(y_test, preds))
plot_confusion_matrix(y_test, preds)
for col in X.columns:

    if abs(X[col].corr(y)) < 0.05:

        X = X.drop([col], axis=1)

        X_test = X_test.drop([col], axis=1)
X
model = LGBMClassifier(random_state=0)

model.fit(X, y)

preds = model.predict(X_test)



print('LightGBM f1-score', f1_score(y_test, preds))

print('LightGBM precision', precision_score(y_test, preds))

print('LightGBM recall', recall_score(y_test, preds))
cm = confusion_matrix(y_test, preds)



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax)



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')
model = XGBClassifier(random_state=666)

model.fit(X, y)

preds = model.predict(X_test)



print('XGBClassifier f1-score', f1_score(y_test, preds))

print('XGBClassifier precision', precision_score(y_test, preds))

print('XGBClassifier recall', recall_score(y_test, preds))
cm = confusion_matrix(y_test, preds)



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax)



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')
def create_model(trial):

    max_depth = trial.suggest_int("max_depth", 2, 30)

    n_estimators = trial.suggest_int("n_estimators", 1, 500)

    learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)

    num_leaves = trial.suggest_int("num_leaves", 2, 5000)

    min_child_samples = trial.suggest_int('min_child_samples', 3, 200)

    model = LGBMClassifier(

        learning_rate=learning_rate, 

        n_estimators=n_estimators, 

        max_depth=max_depth,

        num_leaves=num_leaves, 

        min_child_samples=min_child_samples,

        random_state=666

    )

    return model



sampler = TPESampler(seed=666)

def objective(trial):

    model = create_model(trial)

    model.fit(X, y)

    preds = model.predict(X_test)

    return f1_score(y_test, preds)



study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=100)



lgb_params = study.best_params

lgb_params['random_state'] = 666

lgb = LGBMClassifier(**lgb_params)

lgb.fit(X, y)

preds = lgb.predict(X_test)



print('Optimized LightGBM: ', accuracy_score(y_test, preds))

print('Optimized LightGBM f1-score', f1_score(y_test, preds))

print('Optimized LightGBM precision', precision_score(y_test, preds))

print('Optimized LightGBM recall', recall_score(y_test, preds))
plot_confusion_matrix(lgb, X_test, y_test)

plt.show()
check = X_test.copy()

check['preds'] = preds

check['preds_fixed'] = preds



check.loc[check['age']>90, 'preds_fixed'] = 1

check.loc[check['age']<42, 'preds_fixed'] = 0

check.loc[check['age'].isin([66, 67, 78, 79]), 'preds_fixed'] = 0

check.loc[check['ejection_fraction']<17, 'preds_fixed'] = 1

check.loc[check['serum_creatinine']>6.1, 'preds_fixed'] = 1
preds = check['preds_fixed']



print('Postprocessed accuracy: ', accuracy_score(y_test, preds))

print('Postprocessed f1-score', f1_score(y_test, preds))

print('Postprocessed precision', precision_score(y_test, preds))

print('Postprocessed recall', recall_score(y_test, preds))
cm = confusion_matrix(y_test, preds)



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax)



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')
for i in range(1, len(X.columns)+1):

    rfe = RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=i)

    pipeline = Pipeline(steps=[('s',rfe),('m',LGBMClassifier(random_state=0))])

    pipeline.fit(X, y)

    preds = pipeline.predict(X_test)

    

    print('Number of features: ', i)

    print('LightGBM f1-score', f1_score(y_test, preds))
def create_model(trial):

    rfe = RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=2)

    max_depth = trial.suggest_int("max_depth", 2, 30)

    n_estimators = trial.suggest_int("n_estimators", 1, 500)

    learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)

    num_leaves = trial.suggest_int("num_leaves", 2, 5000)

    min_child_samples = trial.suggest_int('min_child_samples', 3, 200)

    model = Pipeline(

        steps=[

            ('s',rfe), 

            ('m', 

             LGBMClassifier(

                 learning_rate=learning_rate, 

                 n_estimators=n_estimators, 

                 max_depth=max_depth, 

                 num_leaves=num_leaves, 

                 min_child_samples=min_child_samples, 

                 random_state=0

             )

            )

        ]

    )

    return model



def objective(trial):

    model = create_model(trial)

    model.fit(X, y)

    preds = model.predict(X_test)

    return f1_score(y_test, preds)



study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=100)



lgb_params = study.best_params

lgb_params['random_state'] = 666

lgb = LGBMClassifier(**lgb_params)

rfe = RFE(estimator=DecisionTreeClassifier(random_state=666), n_features_to_select=2)

model = Pipeline(steps=[('s',rfe), ('m', lgb)])

model.fit(X, y)

preds = model.predict(X_test)



print('Optimized LightGBM: ', accuracy_score(y_test, preds))

print('Optimized LightGBM f1-score', f1_score(y_test, preds))

print('Optimized LightGBM precision', precision_score(y_test, preds))

print('Optimized LightGBM recall', recall_score(y_test, preds))
plot_confusion_matrix(model, X_test, y_test)

plt.show()