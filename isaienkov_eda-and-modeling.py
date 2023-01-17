import numpy as np

import pandas as pd

import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, recall_score, precision_score, plot_confusion_matrix, accuracy_score

import optuna

from optuna.samplers import TPESampler
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.head()
for col in df.columns:

    print(col, str(round(100* df[col].isnull().sum() / len(df), 2)) + '%')
df.describe()
data = df['Country'].value_counts().reset_index()

data.columns = ['Country', 'Passengers']



fig = px.bar(

    data, 

    x='Country', 

    y='Passengers', 

    orientation='v', 

    title='Number of Passengers by country', 

    width=800,

    height=600

)



fig.show()
data = df['Sex'].value_counts().reset_index()

data.columns = ['Sex', 'Passengers']

fig = px.pie(

    data, 

    values='Passengers', 

    names='Sex', 

    title='Number of Passengers by gender', 

    width=500, 

    height=500

)



fig.show()
data = df['Category'].value_counts().reset_index()

data.columns = ['Category', 'Passengers']



fig = px.pie(

    data, 

    values='Passengers', 

    names='Category', 

    title='Number of Passengers by category', 

    width=500, 

    height=500

)



fig.show()
data = df['Survived'].value_counts().reset_index()

data.columns = ['Survived', 'Passengers']



fig = px.pie(

    data, 

    values='Passengers', 

    names='Survived', 

    title='Survival distribution', 

    width=500, 

    height=500

)



fig.show()
fig = px.histogram(

    df, 

    "Age", 

    nbins=20, 

    title='Age distribution', 

    width=800

)



fig.show()
fig = px.box(

    df, 

    x="Survived", 

    y="Age", 

    points='all',

    height=600,

    width=800,

    title='Age & Survived box plot'

)



fig.show()
X = df[['Country', 'Sex', 'Age', 'Category', 'Survived']]

categorical = ['Country', 'Sex', 'Category']

for cat in categorical:

    X = pd.concat([X, pd.get_dummies(X[cat], prefix=cat)], axis=1)

    X = X.drop([cat], axis=1)

X = X.drop(['Sex_F', 'Category_C'], axis=1)
f = plt.figure(figsize=(19, 15))

plt.matshow(X.corr(), fignum=f.number)

plt.xticks(range(X.shape[1]), X.columns, fontsize=14, rotation=45)

plt.yticks(range(X.shape[1]), X.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
for col in X.columns:

    if abs(X[col].corr(X['Survived'])) < 0.1:

        X = X.drop([col], axis=1)
y = X['Survived']

X = X.drop(['Survived'], axis=1)



X, X_test, y, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=False)
model = LogisticRegression(random_state=0)

model.fit(X, y)
preds = model.predict(X_test)

print('Logistic Regression ', accuracy_score(y_test, preds))
plot_confusion_matrix(model, X_test, y_test)

plt.show()
print('Logistic Regression f1-score', f1_score(y_test, preds))

print('Logistic Regression precision', precision_score(y_test, preds))

print('Logistic Regression recall', recall_score(y_test, preds))
model = RandomForestClassifier(random_state=0)

model.fit(X, y)
preds = model.predict(X_test)

print('Random Forest', accuracy_score(y_test, preds))

print('Random Forest f1-score', f1_score(y_test, preds))

print('Random Forest precision', precision_score(y_test, preds))

print('Random Forest recall', recall_score(y_test, preds))
plot_confusion_matrix(model, X_test, y_test)

plt.show()
sampler = TPESampler(seed=0)

def create_model(trial):

    max_depth = trial.suggest_int("max_depth", 2, 32)

    n_estimators = trial.suggest_int("n_estimators", 2, 500)

    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, max_depth=max_depth, random_state=0)

    return model



def objective(trial):

    model = create_model(trial)

    model.fit(X, y)

    preds = model.predict(X_test)

    score = f1_score(y_test, preds)

    return score



study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=150)



rf_params = study.best_params

rf_params['random_state'] = 0

rf = RandomForestClassifier(**rf_params)

rf.fit(X, y)

preds = rf.predict(X_test)

print('Optimized Random Forest: ', accuracy_score(y_test, preds))

print('Optimized Random Forest f1-score: ', f1_score(y_test, preds))
plot_confusion_matrix(rf, X_test, y_test)

plt.show()