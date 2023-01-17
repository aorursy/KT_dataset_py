import numpy as np
import pandas as pd
import tensorflow as tf
import keras
dataset_train=pd.read_csv('/kaggle/input/titanic/train.csv')
dataset_test=pd.read_csv('/kaggle/input/titanic/test.csv')
dataset_merged=pd.concat([dataset_train,dataset_test],axis=0)

print("Training Set")
dataset_train.head(n=10)
print("Test Set")
dataset_test.head(n=10)
X=dataset_merged[['Embarked','Pclass','Sex','Age','SibSp','Parch','Fare']]
y=dataset_merged[['Survived']]
print("Independent Variables")
X.head(n=10)
print("Dependent Variables")
y.head(n=10)
X.info()
pd.options.mode.chained_assignment=None
X['Embarked']=X['Embarked'].fillna(X['Embarked'].mode()[0])
X['Age']=X['Age'].fillna(X['Age'].mean())
X['Fare']=X['Fare'].fillna(X['Fare'].mean())
X.info()
X=X.to_numpy()
y=y.to_numpy()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

le=LabelEncoder()
X[:,5]=le.fit_transform(X[:,5])
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[5])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

print(X)
print("Shape of X is ",X.shape)
from sklearn.model_selection import train_test_split
X_train,X_dev,y_train,y_dev=train_test_split(X[0:891,:],y[0:891],test_size=0.2,random_state=0)
X_dev=X_dev.astype('float64')
X_train=X_train.astype('float64')
y_train=y_train.reshape(-1,1).astype('float64')
y_dev=y_dev.reshape(-1,1).astype('float64')
X_test=X[891:,:].astype('float64')
print("Shape of Training Set is ",X_train.shape)
print("Shape of Dev Set is ",X_dev.shape)
print("Shape of Test Set is ",X_test.shape)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
X_dev=sc_X.transform(X_dev)
print("Training set now looks like this\n",X_train)
print("Shape of Training set is",X_train.shape)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train.reshape(-1,))

from sklearn.metrics import confusion_matrix,accuracy_score
y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

from sklearn.metrics import confusion_matrix,accuracy_score
y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score=np.zeros((10,2))
score[0,:]=[accuracy_train,accuracy_dev]
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
classifier.fit(X_train,y_train.reshape(-1,))

y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[1,:]=[accuracy_train,accuracy_dev]
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train.reshape(-1,))

from sklearn.metrics import confusion_matrix,accuracy_score
y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

from sklearn.metrics import confusion_matrix,accuracy_score
y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[2,:]=[accuracy_train,accuracy_dev]
from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train.reshape(-1,))

y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[3,:]=[accuracy_train,accuracy_dev]
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',C=0.5,gamma=0.2)
classifier.fit(X_train,y_train.reshape(-1,))

y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[4,:]=[accuracy_train,accuracy_dev]
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=3)
classifier.fit(X_train,y_train.reshape(-1,))

y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[5,:]=[accuracy_train,accuracy_dev]
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=3)
classifier.fit(X_train,y_train.reshape(-1,))

y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[6,:]=[accuracy_train,accuracy_dev]
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train.reshape(-1,))

y_train_pred=classifier.predict(X_train).reshape(-1,1)
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=classifier.predict(X_dev).reshape(-1,1)
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[7,:]=[accuracy_train,accuracy_dev]
classifier=tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=12,activation='relu'))
classifier.add(tf.keras.layers.Dense(units=12,activation='relu'))
classifier.add(tf.keras.layers.Dense(units=12,activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

opt=tf.keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=32,epochs=1000,verbose=0)

model_json=classifier.to_json()
with open("ANN_model_dev.json","w") as json_file:
    json_file.write(model_json)
classifier.save_weights("ANN_model_weights_dev.h5")
print("Saved model to disk")

y_train_pred=np.round(classifier.predict(X_train).reshape(-1,1))
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=np.round(classifier.predict(X_dev).reshape(-1,1))
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[8,:]=[accuracy_train,accuracy_dev]
classifier=tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=12,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)))
classifier.add(tf.keras.layers.Dense(units=12,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)))
classifier.add(tf.keras.layers.Dense(units=12,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)))
classifier.add(tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)))

opt=tf.keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=32,epochs=1000,verbose=0)

model_json=classifier.to_json()
with open("ANN_model_regularised_dev.json","w") as json_file:
    json_file.write(model_json)
classifier.save_weights("ANN_model_regularised_weights_dev.h5")
print("Saved model to disk")

y_train_pred=np.round(classifier.predict(X_train).reshape(-1,1))
cm_train=confusion_matrix(y_train,y_train_pred)
accuracy_train=accuracy_score(y_train,y_train_pred)
print("Training Accuracy score is: ",accuracy_train)
print("Training Confusion matrix: ",cm_train)

y_dev_pred=np.round(classifier.predict(X_dev).reshape(-1,1))
cm_dev=confusion_matrix(y_dev,y_dev_pred)
accuracy_dev=accuracy_score(y_dev,y_dev_pred)
print("Dev Accuracy score is: ",accuracy_dev)
print("Dev Confusion matrix: ",cm_dev)

score[9,:]=[accuracy_train,accuracy_dev]
names=['Logistic Regression','KNN','Naive-Bayes','Linear SVM','Kernel SVM','Decision Trees','Random Forest','XGBoost','4-layer ANN','4-layer ANN with L2 regularisation']
names=np.array(names).reshape(-1,1)
score_all=np.append(names,score,axis=1)
score_all=pd.DataFrame(data=score_all,columns=['Model','Train Set Accuracy','Dev Set Accuracy'],index=None)
score_all.head(n=10)
classifier1=LogisticRegression()
classifier2=GaussianNB()
classifier3=SVC(kernel='linear')
classifier4=SVC(kernel='rbf')
json_file=open('ANN_model_regularised_dev.json','r')
loaded_model_json=json_file.read()
json_file.close()
classifier5=tf.keras.models.model_from_json(loaded_model_json)

def ensemble_classifier(X):
    y_pred1=np.round(classifier1.predict(X).reshape(-1,1))
    y_pred2=np.round(classifier2.predict(X).reshape(-1,1))
    y_pred3=np.round(classifier3.predict(X).reshape(-1,1))
    y_pred4=np.round(classifier4.predict(X).reshape(-1,1))
    y_pred5=np.round(classifier5.predict(X).reshape(-1,1))
    y_pred=np.round(np.mean([y_pred1,y_pred2,y_pred3,y_pred4,y_pred5],axis=0).reshape(-1,1))
    return y_pred

X_test=X[891:,:].astype('float64')
X_train=X[0:891,:].astype('float64')
y_train=y[0:891].reshape(-1,).astype('float64')
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


classifier1.fit(X_train,y_train)
classifier2.fit(X_train,y_train)
classifier3.fit(X_train,y_train)
classifier4.fit(X_train,y_train)
classifier5.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
classifier5.fit(X_train,y_train,batch_size=32,epochs=1000,verbose=0)

model_json=classifier.to_json()
with open("ANN_model_regularised_actual.json","w") as json_file:
    json_file.write(model_json)
classifier.save_weights("ANN_model_regularised_weights_actual.h5")
print("Saved model to disk")
y_train_pred1=classifier1.predict(X_train).reshape(-1,1)
y_train_pred2=classifier2.predict(X_train).reshape(-1,1)
y_train_pred3=classifier3.predict(X_train).reshape(-1,1)
y_train_pred4=classifier4.predict(X_train).reshape(-1,1)
y_train_pred_prob5=classifier5.predict(X_train).reshape(-1,1)
y_train_pred5=np.zeros(y_train_pred_prob5.shape)

for i in range(y_train.shape[0]):
    if y_train_pred_prob5[i]>0.5:
        y_train_pred5[i]=1
    else:
        y_train_pred5[i]=0

cm1=confusion_matrix(y_train,y_train_pred1)
accuracy1=accuracy_score(y_train,y_train_pred1)
cm2=confusion_matrix(y_train,y_train_pred2)
accuracy2=accuracy_score(y_train,y_train_pred2)
cm3=confusion_matrix(y_train,y_train_pred3)
accuracy3=accuracy_score(y_train,y_train_pred3)
cm4=confusion_matrix(y_train,y_train_pred4)
accuracy4=accuracy_score(y_train,y_train_pred4)
cm5=confusion_matrix(y_train,y_train_pred5)
accuracy5=accuracy_score(y_train,y_train_pred5)

print("Train Accuracy score 1 is: ",accuracy1)
print("Train Confusion 1 matrix: \n",cm1)
print("Train Accuracy 2 score is: ",accuracy2)
print("Train Confusion 2 matrix: \n",cm2)
print("Train Accuracy 3 score is: ",accuracy3)
print("Train Confusion 3 matrix: \n",cm3)
print("Train Accuracy 4 score is: ",accuracy4)
print("Train Confusion 4 matrix: \n",cm4)
print("Train Accuracy 5 score is: ",accuracy5)
print("Train Confusion 5 matrix: \n",cm5)
y_test_pred=ensemble_classifier(X_test)
dataset2=pd.read_csv('../input/titanic/test.csv')
dataset2=dataset2['PassengerId']
output=dataset2.to_numpy()
output=output.astype('float64').reshape(-1,1)
output=np.concatenate([output,y_test_pred],axis=1).astype(int)
output=pd.DataFrame(output,index=None,columns=["PassengerId","Survived"])
output.to_csv('output.csv',index=False)
output.head(n=20)