import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/adult-dataset/train_data.csv', 
                         index_col='Id', 
                         na_values='?',
                         engine='python', 
                         sep=r'\s*,\s*')
train_data.head()
train_data.describe()
train_data.dtypes
train_data.shape
train_data.isnull().any()
train_data = train_data.dropna()
train_data.shape
notnum = train_data[['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income']]
notnum.nunique()
plt.figure()
x = train_data.loc[train_data['sex']=='Male']
x["age"].value_counts().sort_index().plot(kind="bar")
x = train_data.loc[train_data['sex']=='Female']
x["age"].value_counts().sort_index().plot(kind="bar")
train_data["sex"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["sex"].value_counts().plot(kind="bar")
train_data["workclass"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["workclass"].value_counts().plot(kind="bar")
train_data["education"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["education"].value_counts().plot(kind="bar")
train_data["marital.status"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["marital.status"].value_counts().plot(kind="bar")
train_data["occupation"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["occupation"].value_counts().plot(kind="bar")
train_data["relationship"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["relationship"].value_counts().plot(kind="bar")
train_data["race"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["race"].value_counts().plot(kind="bar")
train_data["native.country"].value_counts().plot(kind="bar")
y = train_data.loc[train_data["income"]=='>50K']
y["native.country"].value_counts().plot(kind="bar")
train_data_teste = pd.get_dummies(train_data)
analise = train_data_teste.corr().loc[:,'income_>50K'].sort_values(ascending=True)
analise
analise.plot(kind='bar')
analise = train_data_teste.corr().loc[:,'income_>50K'].sort_values(ascending=True).where(lambda x : abs(x) > 0.15).dropna()
analise
test_data = pd.read_csv('../input/adult-dataset/test_data.csv', 
                         index_col='Id', 
                         na_values='?', 
                         engine='python', 
                         sep=r'\s*,\s*')
from sklearn import preprocessing
test_data = test_data.dropna()
test_data.head()
Xtrain_data = train_data[["age","education","education.num","marital.status","occupation","relationship","capital.gain","capital.loss","hours.per.week","race","sex","workclass"]]
Ytrain_data = train_data.income
Xtest_data = test_data[["age","education","education.num","marital.status","occupation","relationship","capital.gain","capital.loss","hours.per.week","race","sex","workclass"]]
Xtrain_data = pd.get_dummies(Xtrain_data)
Xtest_data = pd.get_dummies(Xtest_data)
Xtrain_data = Xtrain_data[[ 'marital.status_Never-married',
                            'relationship_Own-child',
                            'sex_Female',
                            'relationship_Not-in-family',
                            'occupation_Other-service',
                            'capital.loss',
                            'education_Prof-school',
                            'education_Masters',
                            'education_Bachelors',
                            'occupation_Prof-specialty',
                            'occupation_Exec-managerial',
                            'sex_Male',
                            'capital.gain',
                            'hours.per.week',
                            'age',
                            'education.num',
                            'relationship_Husband',
                            'marital.status_Married-civ-spouse']]
Xtest_data = Xtest_data[['marital.status_Never-married',
                        'relationship_Own-child',
                        'sex_Female',
                        'relationship_Not-in-family',
                        'occupation_Other-service',
                        'capital.loss',
                        'education_Prof-school',
                        'education_Masters',
                        'education_Bachelors',
                        'occupation_Prof-specialty',
                        'occupation_Exec-managerial',
                        'sex_Male',
                        'capital.gain',
                        'hours.per.week',
                        'age',
                        'education.num',
                        'relationship_Husband',
                        'marital.status_Married-civ-spouse']]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xtrain_data, Ytrain_data, cv=10)
scores
scores.mean()
from sklearn.model_selection import GridSearchCV

k_range = list(range(1,35))
weights = ['uniform', 'distance']
p_range = list(range(1,3))
param = dict(n_neighbors=k_range, p=p_range)

knn = KNeighborsClassifier(n_neighbors=3)
grid = GridSearchCV(knn, param, cv=10, scoring='accuracy', n_jobs = -2)
grid.fit(Xtrain_data, Ytrain_data)
print(grid.best_estimator_)
print(grid.best_score_)
knn_final = grid.best_estimator_
knn_final.fit(Xtrain_data,Ytrain_data)
Ytest_data = knn_final.predict(Xtest_data)
submission = pd.DataFrame(Ytest_data)
submission.to_csv("submission.csv")