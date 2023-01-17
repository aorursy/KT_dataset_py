import pandas as pd
import sklearn
import matplotlib.pyplot as plt
raw_train_base = pd.read_csv("../input/train.csv")
raw_train_base.info()
clean = raw_train_base
null_columns = raw_train_base.columns[raw_train_base.isnull().any()]
raw_train_base[null_columns].isnull().sum()
#removendo atributos com muitos dados faltatantes
clean = clean.drop(columns=['v2a1','v18q1','rez_esc'])
clean = clean.dropna()
clean.head()
clean.loc[:,'tamhog'].value_counts().plot(kind='bar')
clean.loc[:,'escolari'].value_counts().plot(kind='bar')
clean.loc[:,'bedrooms'].value_counts().plot(kind='bar')
clean.loc[:,'overcrowding'].value_counts().plot(kind='bar')
clean.loc[:,'Target'].value_counts().plot(kind='bar')
plt.matshow(clean.corr())
anl0 = clean.corr().loc[:,'Target'].sort_values(ascending=True)
anl0
clean = clean.drop(columns = 'elimbasu5')
clean.select_dtypes(include= 'object').head()
clean.loc[:,'idhogar'].nunique()

train_clean = clean.replace(to_replace=['yes','no'], value = [1,0])
train_clean = train_clean.drop(columns='idhogar')
train_clean.loc[:,['edjefe','edjefa','dependency']] = train_clean.loc[:,['edjefe','edjefa','dependency']].astype('float64')
index = anl0.where(lambda x : abs(x) > 0.2).dropna().index[1:-1]
X_train = train_clean.loc[:,index]
Y_train = train_clean.iloc[:,-1]
raw_test_base = pd.read_csv('../input/test.csv')
test_clean = raw_test_base.drop(columns=['v2a1','v18q1','rez_esc','elimbasu5'])
test_clean = test_clean.replace(to_replace=['yes','no'], value = [1,0])
test_clean = test_clean.drop(columns='idhogar')
test_clean.loc[:,['edjefe','edjefa','dependency']] = test_clean.loc[:,['edjefe','edjefa','dependency']].astype('float64')
test_clean = test_clean.fillna(0.0)
X_test = test_clean.loc[:,index]
X_test.info()
X_train.info()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
#usarei gridsearch para verificar os melhores hiperparâmetros do kNN.
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
p_options = list(range(1,3))
param_grid = dict(n_neighbors=k_range, p=p_options)#, p=p_options
#inicializando o classificador a ser verificado no GridSearchCV
knn = KNeighborsClassifier(n_neighbors=5)

from sklearn.model_selection import cross_val_score
#inicializando o GridSearchCV
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs = -2)  
grid.fit(X_train, Y_train)
print(grid.best_estimator_)
print(grid.best_score_)
f_kNN = grid.best_estimator_
f_kNN.fit(X_train,Y_train)
Y_test = f_kNN.predict(X_test)
output = pd.DataFrame({'Id':test_clean.loc[:,'Id'],'Target':Y_test})
output.to_csv("submitHouseholdIncome5.csv", index = False)
