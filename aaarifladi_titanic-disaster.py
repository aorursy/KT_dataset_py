import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

train_data= pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()

print(test_data.shape)


train_data.describe()
train_data=train_data[['Survived','Pclass','Sex','Age','Fare','SibSp','Parch']] #4 colonnes nous interessent

train_data.dropna(axis=0,inplace=True)#permet de supprimer les lignes qui ont des cases vident

train_data.describe()

train_data['Sex'].replace(['male','female'],[0,1],inplace=True)

train_data.head()




from sklearn.model_selection import train_test_split

y=train_data['Survived']

X=train_data.drop(['Survived'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=5)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()

model.fit(X_train,y_train)

print('Train set:', model.score(X_train,y_train))
from sklearn.model_selection import cross_val_score

cross_val_score(KNeighborsClassifier(),X_train,y_train,cv=5,scoring='accuracy')
from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt
model=KNeighborsClassifier()

k=np.arange(1,50)

train_score, val_score=validation_curve(model,X_train,y_train,'n_neighbors',k,cv=5) 

#cv = nombre de decoupe qu'on veut avoir dans cross validation

#n_neighbors=nom de l'hyper parametre qu'on desire reglé

#val_score

#val_score.mean(axis=1)

plt.plot(k,val_score.mean(axis=1), label='validation')

plt.plot(k,train_score.mean(axis=1), label='train')

plt.xlabel('n_neighbors')

plt.ylabel('score')

plt.legend()
from sklearn.model_selection import GridSearchCV
#on cree un dictionnaire avec les hyperparametres qu'on desire utilisé

param_grid={'n_neighbors':np.arange(1,20),

            'metric':['euclidean','manhathan','minkowski','canberra'],'weights':['uniform','distance']}

#on crée une grille avec un avec notre model, dictionnaire et cv qu'on veut

grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)

#on entraine notre model avec les données d'entrainnements

grid.fit(X_train,y_train)
grid.best_params_
grid.best_score_
model=grid.best_estimator_
model.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,model.predict(X_test))
from sklearn.model_selection import learning_curve

learning_curve(model,X_train,y_train,train_sizes=np.linspace(0.1,1,10),cv=5)
N,train_score,val_score=learning_curve(model,X_train,y_train,train_sizes=np.linspace(0.1,1,10),cv=5)

print(N)

plt.plot(N,train_score.mean(axis=1),label='train')

plt.plot(N,val_score.mean(axis=1),label='val')

plt.xlabel('train_sizes')

plt.legend()
test_data.head()

test_data.dropna(axis=0,inplace=True)
test_data.head()
X_submission = test_data

X_submission.replace(['male','female'],[0,1],inplace=True)

print(X_submission)
test_predictions = model.predict(X_test)



print(confusion_matrix(y_test, test_predictions))

X_submission.columns
y_submission = model.predict(X_submission[['Age', 'Sex', 'Age','Fare','SibSp','Parch']])

output = pd.DataFrame({'PassengerId': X_submission.PassengerId, 'Survived': y_submission})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
y_submission