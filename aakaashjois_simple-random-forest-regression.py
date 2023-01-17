import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

import dill
data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv').set_index('Serial No.')

data = data.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'})
print('The dataset has {} rows.'.format(len(data)))

print('The dataset has {} columns'.format(data.columns))
print(data.info())
print(data.isnull().any())
sns.heatmap(data.corr(), annot=True)
sns.pairplot(data.drop(columns='Research'))
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='Chance of Admit'), data['Chance of Admit'], test_size=0.2)
X_train.describe()
dfm = X_train.melt(var_name='columns')

g = sns.FacetGrid(dfm, col='columns')

g = (g.map(sns.distplot, 'value'))
scaler = StandardScaler().fit(X_train)

X_train_norm = scaler.transform(X_train)

X_test_norm = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_norm, columns=X_train.columns)

X_test = pd.DataFrame(X_test_norm, columns=X_test.columns)
dfm = X_train.melt(var_name='columns')

g = sns.FacetGrid(dfm, col='columns')

g = (g.map(sns.distplot, 'value'))
gridsearch = GridSearchCV(estimator=RandomForestRegressor(),

                          param_grid={

                              'n_estimators': [50, 100, 250, 300],

                              'max_depth': [None, 100, 200, 300, 400]

                          },

                          cv=3,

                          return_train_score=False,

                          scoring='r2')

gridsearch.fit(X=X_train, y=y_train)

pd.DataFrame(gridsearch.cv_results_).set_index('rank_test_score').sort_index()
pipe = make_pipeline(scaler, gridsearch)
print('Original model: ' + str(gridsearch.predict(X=scaler.transform(data.drop(columns='Chance of Admit').iloc[0].values.reshape(1, -1)))[0]))

print('Pipeline model: ' + str(pipe.predict(X=data.drop(columns='Chance of Admit').iloc[0].values.reshape(1, -1))[0]))
with open('rf_v1.pkl', 'wb') as f:

    dill.dump(pipe, f)
with open('rf_v1.pkl', 'rb') as f:

    model = dill.load(f)

    print(model.predict(X=data.drop(columns='Chance of Admit').iloc[0].values.reshape(1, -1))[0])