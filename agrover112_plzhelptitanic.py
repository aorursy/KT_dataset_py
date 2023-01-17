import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import os

print(os.listdir("../input/titanic"))

train=pd.read_csv("../input/titanic/train.csv").drop(['PassengerId','Cabin','Ticket'],axis=1)

test=pd.read_csv("../input/titanic/test.csv").drop(['Cabin','Ticket'],axis=1)

passid=pd.DataFrame(test['PassengerId'])

test=test.drop(['PassengerId'],axis=1)
#c_t= train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

train['Title']=train['Name'].str.extract('([A-Za-z]+)\.')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')

train['Title']=train['Title'].replace(['Capt','Col','Dr','Major','Rev'],'Special')

train['Title']=train['Title'].replace(['Lady','Countess','Don','Sir','Jonkheer','Dona'],'Royalty')



train['Title']=train['Title'].replace(['Royalty','Special','Miss','Mrs','Mr','Master'],[0,1,2,2,3,4])



from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()

title=train['Title'].values

title=pd.DataFrame(OneHotEncoder().fit_transform(title.reshape(-1,1)).toarray())

title.columns=['Title1','Title2','Title3','Title4','Title5']

title=title.iloc[:,:-1]



train=train.drop(['Title','Name'],axis=1)

train=pd.concat([train,title],axis=1)

test['Title']=test['Name'].str.extract('([A-Za-z]+)\.')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')

test['Title']=test['Title'].replace(['Capt','Col','Dr','Major','Rev'],'Special')

test['Title']=test['Title'].replace(['Lady','Countess','Don','Sir','Jonkheer','Dona'],'Royalty')



test['Title']=test['Title'].replace(['Royalty','Special','Miss','Mrs','Mr','Master'],[0,1,2,2,3,4])



from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()

title=test['Title'].values

title=pd.DataFrame(OneHotEncoder().fit_transform(title.reshape(-1,1)).toarray())

title.columns=['Title1','Title2','Title3','Title4','Title5']

title=title.iloc[:,:-1]



test=test.drop(['Title','Name'],axis=1)

test=pd.concat([test,title],axis=1)

train.head()
emb=pd.DataFrame(train['Embarked'])#Port from which they embarked

sex=pd.DataFrame(train['Sex'])#The GENDER

y=pd.DataFrame(train['Survived'])

train_2=train.drop(['Embarked','Sex','Survived'],axis=1)

train_2.columns=['Pclass','Age','SibSp','Parch','Fare','Title1','Title2','Title3','Title4']
family_train= train_2[["Parch", "SibSp"]].sum(axis=1)

#train_2.(4)
from sklearn.impute import SimpleImputer

imp=SimpleImputer(strategy="most_frequent")

train_2=pd.DataFrame(imp.fit_transform(train_2))
from sklearn.impute import SimpleImputer

imp=SimpleImputer(strategy="most_frequent")

emb=pd.DataFrame(imp.fit_transform(emb))

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder

#emb=pd.DataFrame(LabelEncoder().fit_transform(emb.values.ravel()))

emb=pd.DataFrame(OneHotEncoder().fit_transform(emb).toarray())

emb=emb.iloc[:,:-1]

emb.columns=['Emb1','Emb2']
from sklearn.preprocessing import LabelEncoder

sex=pd.DataFrame(LabelEncoder().fit_transform(sex))

sex.columns=['Sex']
#train_2.reset_index(drop=True, inplace=True)

#sex.reset_index(drop=True, inplace=True)



train_2=pd.concat([train_2,sex],axis=1)

train_2=pd.concat([train_2,emb],axis=1)

"""ADDING FAMILY SIZE IF IT WORKS"""

train_2=pd.concat([train_2,family_train],axis=1)

train_2.columns=['Pclass','Age','SibSp','Parch','Fare','Sex','Title1','Title2','Title3','Title4','Emb1','Emb2','FamilySize']

#train_2.columns=['Pclass','Age','SibSp','Parch','Fare','Sex','Emb1','Emb2']

from sklearn.preprocessing import MinMaxScaler

sc1=MinMaxScaler()

sc2=MinMaxScaler()

#train_2['Age']=sc1.fit_transform(train_2['Age'].values.reshape(-1,1))

train_2['Age']=pd.qcut(train_2['Age'].values,4,labels=False)

train_2['Fare']=sc2.fit_transform(train_2['Fare'].values.reshape(-1,1))
emb=pd.DataFrame(test['Embarked'])

sex=pd.DataFrame(test['Sex'])

y=pd.DataFrame(train['Survived'])

test_2=test.drop(['Embarked','Sex'],axis=1)

test_2.columns=['Pclass','Age','SibSp','Parch','Fare','Title1','Title2','Title3','Title4']

family_test= test_2[["Parch", "SibSp"]].sum(axis=1)

from sklearn.impute import SimpleImputer

imp=SimpleImputer(strategy="most_frequent")

test_2=pd.DataFrame(imp.fit_transform(test_2))

from sklearn.impute import SimpleImputer

imp=SimpleImputer(strategy="most_frequent")

emb=pd.DataFrame(imp.fit_transform(emb))

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder

#emb=pd.DataFrame(LabelEncoder().fit_transform(emb.values.ravel()))

emb=pd.DataFrame(OneHotEncoder().fit_transform(emb).toarray())

emb=emb.iloc[:,:-1]

emb.columns=['Emb1','Emb2']

from sklearn.preprocessing import LabelEncoder

sex=pd.DataFrame(LabelEncoder().fit_transform(sex))

sex.columns=['Sex']

#train_2.reset_index(drop=True, inplace=True)

#sex.reset_index(drop=True, inplace=True)

test_2=pd.concat([test_2,sex],axis=1)

test_2=pd.concat([test_2,emb],axis=1)

test_2=pd.concat([test_2,family_test],axis=1)

test_2.columns=['Pclass','Age','SibSp','Parch','Fare','Sex','Title1','Title2','Title3','Title4','Emb1','Emb2',"FamilySize"]

from sklearn.preprocessing import MinMaxScaler

sc1=MinMaxScaler()

sc2=MinMaxScaler()

#test_2['Age']=sc1.fit_transform(test_2['Age'].values.reshape(-1,1))

test_2['Age']=pd.qcut(test_2['Age'].values,4,labels=False)

test_2['Fare']=sc2.fit_transform(test_2['Fare'].values.reshape(-1,1))
"""from mlxtend.classifier import StackingClassifier

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

l=LogisticRegression(max_iter=500,solver='lbfgs')

l.fit(X_train,y_train.values.ravel())

from sklearn.svm import SVC

clf2=SVC(kernel="linear")

clf2.fit(X_train,y_train.values.ravel())

from xgboost import XGBClassifier

xr=XGBClassifier(n_estimators=1000)

xr.fit(X_train,y_train.values.ravel())

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=500,solver='lbfgs')

sclf = StackingClassifier(classifiers=[lr, clf2,xr], 

                          meta_classifier=lm)

sclf.fit(X_train,y_train)"""
from xgboost import XGBClassifier

xr=XGBClassifier(n_estimators=1000)

xr.fit(train_2,y.values.ravel())
len(train_2.T)
import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))



# Adding the second hidden layer

classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'relu'))



#Adding a dropout layer

classifier.add(Dropout(rate=0.2))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(train_2, y, batch_size = 16, epochs = 300)



# Part 3 - Making predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(test_2)

y_pred = (y_pred > 0.5)
submission = pd.DataFrame(

    { 

        'PassengerId': passid['PassengerId'].astype(int),

        'Survived': y_pred.ravel().astype(int)

    }

)

submission.to_csv("submission_final.csv", index=False)
#pd.Series(clf.feature_importances_,train_2.columns).sort_values().plot(kind="barh")
#pd.Series(clf2.feature_importances_,train_2.columns).sort_values().plot(kind="barh")
#pd.Series(xr.feature_importances_,train_2.columns).sort_values().plot(kind="barh")
#import pandas_profiling as pp

#pp.ProfileReport(train)
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(train_2, y, test_size=0.33, random_state=42)
#from sklearn.metrics import confusion_matrix,classification_report

#cm=confusion_matrix(y_test, xr.predict(X_test).astype(int))
""""from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(xr,X_train,np.array(y_train).ravel(),cv=10)

ACCURACIES=accuracies.mean()

STD=accuracies.std()

print("Accuracy Cross Val:",ACCURACIES*100)

print("Std :",STD*100)"""
"""acc=(cm[0][0]+cm[1][1])/(cm[1][0]+cm[0][1]+cm[0][0]+cm[1][1])*100

print("Test acc:",acc)"""
"""from sklearn import metrics

y_pred_proba=l.predict_proba(X_test)[::,1]

fpr,tpr,_=metrics.roc_curve(y_test,y_pred_proba)

auc=metrics.roc_auc_score(y_test,y_pred_proba)

plt.legend(loc=4)

plt.plot(fpr,tpr,label="auc="+str(auc))

plt.show()"""
#print(auc)
