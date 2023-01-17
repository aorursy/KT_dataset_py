import time

import numpy as np

import pandas as pd

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import cross_validation

from sklearn.preprocessing import MinMaxScaler



class Titanic(object):

    

    def __init__(self):

        self.trainset=[]

        self.trainlabel=[]

        self.testset=[]

        self.testlabel=[]

        

    def loaddataset(self,path,module):                

       df=pd.read_csv(path)

       subdf = df[['PassengerId','Pclass','Sex','Age','Embarked','Fare','SibSp','Parch']]

       SibSp=subdf['SibSp']

       Parch=subdf['Parch']

#      supplement Age

       Age=subdf['Age'].fillna(value=subdf.Age.mean())

             

       Fare=subdf['Fare'].fillna(value=subdf.Fare.mean())

       

       dummies_Sex=pd.get_dummies(subdf['Sex'],prefix='Sex')

       

       dummies_Embarked = pd.get_dummies(subdf['Embarked'], prefix= 'Embarked')     

       

       dummies_Pclass = pd.get_dummies(subdf['Pclass'], prefix= 'Pclass')

       

       PassengerId=subdf['PassengerId']

       

#      Age&Fare to Scaler

       scaler=MinMaxScaler()

       age_scaled=scaler.fit_transform(Age.values)

       fare_scaled=scaler.fit_transform(Fare.values)

       

       Age_Scaled=pd.DataFrame(age_scaled,columns=['Age_Scaled'])

       Fare_Scaled=pd.DataFrame(fare_scaled,columns=['Fare_Scaled'])

       

       if module=='train':

          self.trainlabel=df.Survived

          self.trainset=pd.concat([dummies_Pclass,dummies_Sex,dummies_Embarked,Age_Scaled,Fare_Scaled,SibSp,Parch],axis=1)

       elif module=='test':

          self.testset=pd.concat([PassengerId,dummies_Pclass,dummies_Sex,dummies_Embarked,Age_Scaled,Fare_Scaled,SibSp,Parch],axis=1)

    

    def train_LR(self):

        samples=self.trainset.values

        target=self.trainlabel.values

        classifier_LR=LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

        classifier_LR.fit(samples,target)

        

        return classifier_LR

    

    def train_RF(self):

        samples=self.trainset.values

        target=self.trainlabel.values

        classifier_RF=RandomForestClassifier(n_estimators=1000)

        classifier_RF.fit(samples,target)        

        

        return classifier_RF

    

    def train_GBDT(self):

        samples=self.trainset.values

        target=self.trainlabel.values

        classifier_GB=GradientBoostingClassifier(n_estimators=1000)

        classifier_GB.fit(samples,target)

        

        return classifier_GB

    

#   Evaluation on Trainset: overfitting/underfitting

    def evaluate(self,y_pred):

        y_true=self.trainlabel

        m=np.shape(y_true)[0]

        count=0        

        for i in range(m):

            if y_pred[i]==y_true[i]:

                count=count+1

        

        accuracy=float(count)/m

        

        target_names = ['survived 0', 'survived 1']  

        print(classification_report(y_true, y_pred, target_names=target_names))

        

        return accuracy

    

    def cross_validation(self,clf,k):

        

        scores=cross_validation.cross_val_score(clf,self.trainset.values,self.trainlabel.values,cv=k)

        print(scores)

        return scores

    

    def toCSV(self,y_pred):

        

        ID=self.testset['PassengerId']

        Survived=pd.DataFrame(y_pred,columns=['Survived'])

        result=pd.concat([ID,Survived],axis=1)

        

        result.to_csv('sample_submission.csv',index=False)

        

    def get_corrf(self):

        size=self.trainset.columns.get_values().size

        corrdata=np.zeros((size,1))

        for i in range(size):

            corr=self.trainset[self.trainset.columns[i]].corr(self.trainlabel)

            corrdata[i]=corr

        corrdf=pd.DataFrame(corrdata,index=self.trainset.columns.get_values(),columns=['corr'])

        

        return corrdf

        

#==============================================================================

#Load Dataset

#==============================================================================



time_start=time.time()



path_train='../input/train.csv'

path_test='../input/test.csv'



titanic=Titanic()



titanic.loaddataset(path_train,'train')

titanic.loaddataset(path_test,'test')



trainset=titanic.trainset.values

trainlabel=titanic.trainlabel.values



testset=titanic.testset.values



inX=testset[:,1:]



#==============================================================================

# corr to feature enginering

#==============================================================================



#corrdf=titanic.get_corrf()



#==============================================================================

# Logistic Regression

#==============================================================================



classifier_LR=titanic.train_LR()



print('Precision-Recall-LR-Train')



y_train_pred_LR=classifier_LR.predict(trainset)



accuracy_train_LR=titanic.evaluate(y_train_pred_LR)



y_test_LR=classifier_LR.predict(inX)



print('Cross-Validation : Logistic Regression')



scores_LR=titanic.cross_validation(classifier_LR,5)



#titanic.toCSV(y_test_LR)



print('<------------------------------------------------->')



#==============================================================================

# Random Forest

#==============================================================================



classifier_RF=titanic.train_RF()



print('Precision-Recall-RF-Train')



y_train_pred_RF=classifier_RF.predict(trainset)



accuracy_train_RF=titanic.evaluate(y_train_pred_RF)



y_test_RF=classifier_RF.predict(inX)



print('Cross-Validation : Random Forest')



scores_RF=titanic.cross_validation(classifier_RF,5)

#

titanic.toCSV(y_test_RF)



print('<------------------------------------------------->')



#==============================================================================

# Gadient Boosting

#==============================================================================



classifier_GB=titanic.train_GBDT()



print('Precision-Recall-GB-Train')



y_train_pred_GB=classifier_GB.predict(trainset)



accuracy_train_GB=titanic.evaluate(y_train_pred_GB)



y_test_GB=classifier_GB.predict(inX)



print('Cross-Validation : Gradient Boosting')



scores_GB=titanic.cross_validation(classifier_GB,5)



#titanic.toCSV(y_test_GB)



print('<------------------------------------------------->')



#==============================================================================

# Evaluation

#==============================================================================





#==============================================================================

# Time_during

#==============================================================================



time_end=time.time()

time_during=time_end-time_start