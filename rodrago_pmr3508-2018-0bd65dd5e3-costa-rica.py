import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv', 
                         index_col='Id', 
                         na_values='NaN', 
                         engine='python', 
                         sep=r'\s*,\s*')
train_data.head()
train_data.describe()
train_data.dtypes
train_data.shape
train_data.isnull().any()
train_data.dropna().shape
train_data = train_data.fillna(value=0)
test_data = pd.read_csv('../input/test.csv', 
                         index_col='Id', 
                         na_values='NaN', 
                         engine='python', 
                         sep=r'\s*,\s*')
test_data.shape
test_data.dropna().shape
test_data = test_data.fillna(value=0)
train_data.head()
analise = train_data.corr().loc[:,'Target'].sort_values(ascending=True)
analise
analise.plot(kind='bar')
analise = train_data.corr().loc[:,'Target'].sort_values(ascending=True).where(lambda x : abs(x) > 0.10).dropna()
analise
campos = analise.keys().tolist()
campos.remove('Target')
Xtrain_data = train_data[campos]
Ytrain_data = train_data.Target
Xtest_data = test_data[campos]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xtrain_data, Ytrain_data, cv=10)
scores.mean()
from sklearn.model_selection import GridSearchCV

k_range = list(range(1,50))
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
prediction = Ytest_data
prediction
Id = test_data.index.values
Id
s = { 'Id' : Id, 'Target' : prediction.astype(int) }
submission = pd.DataFrame(s)
submission
submission.to_csv("submission.csv")