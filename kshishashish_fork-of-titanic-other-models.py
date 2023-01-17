import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

trainset=pd.read_csv('../input/titanic/train.csv')
testset=pd.read_csv('../input/titanic/test.csv')
sns.pairplot(trainset)
corr_matrix=trainset.corr()
plt.figure(figsize=(12,12))

sns.heatmap(corr_matrix,annot=True)
plt.show()
from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()


trainset['Name'] = enc.fit_transform(trainset['Name'])
testset['Name'] = enc.fit_transform(testset['Name'])

trainset['Sex'] = enc.fit_transform(trainset['Sex'])
testset['Sex'] = enc.fit_transform(testset['Sex'])


trainset['Cabin'] = enc.fit_transform(trainset['Cabin'].astype('str'))
testset['Cabin'] = enc.fit_transform(testset['Cabin'].astype('str'))

trainset['Embarked'] = enc.fit_transform(trainset['Embarked'].astype('str'))
testset['Embarked'] = enc.fit_transform(testset['Embarked'].astype('str'))

trainset['Ticket'] = enc.fit_transform(trainset['Ticket'].astype('category'))
testset['Ticket'] = enc.fit_transform(testset['Ticket'].astype('category'))

y_train=trainset['Survived']

X_train=trainset
X_test=testset
X_train.drop(['Survived'],axis=1,inplace=True)
X_train.head()
X_test.head()
PID=X_test['PassengerId']
PID.head()
X_train.set_index(['PassengerId'],inplace = True)
X_test.set_index(['PassengerId'],inplace = True)

X_train.isnull().sum()
X_test.isnull().sum()
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
X_test.isnull().sum()
from sklearn.model_selection import train_test_split

X_train_1, X_CV, y_train_1, y_CV= train_test_split(X_train,y_train, test_size=0.25, random_state=1)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train_1= sc.fit_transform(X_train_1)
X_CV= sc.transform(X_CV)
X_test=sc.transform(X_test)
X_train
import tensorflow as tf
import keras
from keras.layers import Dense,Activation,Dropout

ANN_model=keras.Sequential()
ANN_model.add(Dense(50,input_dim=10))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
from keras.optimizers import Adam
ANN_model.compile(loss='mse',optimizer='adam')
from sklearn.metrics import mean_squared_error
ANN_model.compile(optimizer='adam',loss='mean_squared_error')

epoch_hist=ANN_model.fit(X_train_1,y_train_1,epochs=100,batch_size=20,validation_split=0.2)
result= ANN_model.evaluate(X_CV,y_CV)
accuracy_ANN=1-result
accuracy_ANN
from sklearn.svm import LinearSVC
clf1=LinearSVC().fit(X_train_1,y_train_1)
y_predict=clf.predict(X_CV)
#clf.score(X_train_1,y_train_1)
clf.score(X_CV,y_CV)
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(X_train_1,y_train_1)
RF.predict(X_CV)
round(RF.score(X_CV,y_CV), 4)
import sklearn as sk
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(X_train_1, y_train_1)
NN.predict(X_CV)
round(NN.score(X_CV,y_CV), 4)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_CV,y_predict)
cm
cm_df = pd.DataFrame(cm,
                     index = ['yes','no'], 
                     columns = ['yes','no'])

plt.figure(figsize=(6,6))
sns.heatmap(cm_df, annot=True)

plt.title('Logistic Regression for Survival')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


y_test=clf.predict(X_test)
y_test=y_test.reshape(-1,1)
y_test.shape
df=pd.DataFrame({'Survived':y_test[:,0]})
df.head()
df.shape
df['PassengerId']=PID
df=df[["PassengerId","Survived"]]
df.head()
df.to_csv('results.csv',index = False)
