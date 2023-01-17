import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
ntrain = train.dropna()
print(ntrain.shape)
print(train.shape) #nao e possivel retirar todos os valores nan da tabela pois ficaria com muitos poucos dados
print(train[train['parentesco1'] == 1].shape)
train_p = train[train['parentesco1'] == 1] #seleciono apenas os membros com parentesco = 1
train_p.isnull().sum() #funcao que verifica quais colunas possuem mais valores NaN, vou retirar essas colunas do meu dataframe
train_p.shape
train_p.replace(to_replace = 'yes', value = 1, inplace = True)
train_p.replace(to_replace = 'no', value = 0, inplace = True)
train_drop = train_p.drop(labels = ['Id', 'v2a1', 'v18q1', 'rez_esc', 'dependency', 'idhogar'], axis = 1)
train_drop.dropna().shape
ntrain_drop = train_drop.dropna()
ntrain_drop.head()
X_train = ntrain_drop.iloc[:, :136]
Y_train = ntrain_drop[['Target']]
#processando o banco de testes
test_p = test[test['parentesco1'] == 1]

Id_p = test_p[['Id']]
Id = test[['Id']]
test_p.replace(to_replace = 'yes', value = 1, inplace = True)
test_p.replace(to_replace = 'no', value = 0, inplace = True)
test.replace(to_replace = 'yes', value = 1, inplace = True)
test.replace(to_replace = 'no', value = 0, inplace = True)

test_d = test.drop(labels = ['v2a1', 'v18q1', 'rez_esc', 'Id', 'dependency', 'idhogar'], axis = 1)
test_drop = test_p.drop(labels = ['v2a1', 'v18q1', 'rez_esc', 'Id', 'dependency', 'idhogar'], axis = 1)


X_test = test_drop.iloc[:, :136]
Xtest = test_d.iloc[:, :136]
#Y_test = ntest_drop[['Target']]
print(test_d.shape)

for column in X_train:
    print(column, X_train[column].dtype)
X_train['r4h1'].dtype
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
Y_train.values.ravel()
for i in range(5, 30):
    knn = knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, X_train, Y_train.values.ravel(), cv = 5)
    print(i, np.mean(scores))
knn = KNeighborsClassifier(n_neighbors = 25)
knn.fit(X_train, Y_train.values.ravel())
Xtest.fillna(0, inplace = True)
#Y_pred = knn.predict(X_test)
Ypred = knn.predict(Xtest)
#print(Y_pred)
print(Ypred)
#print(Id_p)
print(Id)
#df_Y_pred = pd.DataFrame(data = Y_pred)
df_Ypred = pd.DataFrame(data = Ypred)
#Id_p['Target'] = df_Y_pred
Id['Target'] = df_Ypred
#Id_p.to_csv('submission_p.csv', index = False)
Id.to_csv('submission.csv', index = False)
print(test.shape)
print(Ypred.shape)
Id.shape