import pandas as pd

#import numpy as np

#import random as rnd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import svm

from sklearn import linear_model

print("Libraries imported")

#Importing libraries
#importing training and testing  data

train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

print("Training and testing data imported")

train_df.shape, test_df.shape
train_df.columns
TotalRecsTrain=train_df['PassengerId'].count() #total number of records in training data

TotalRecsTest=test_df['PassengerId'].count()# total number of records in testing data
#Checking for nulls in training data

for column in train_df:

    if(train_df[column].isnull().any()):

        print("Column Name:",column,"|Column Data type:",train_df[column].dtype,",|Not null count:",train_df[column].count(),'Total count:',TotalRecsTrain,"|Null Values",TotalRecsTrain-train_df[column].count())
#Checking for nulls in testing data

for column in test_df:

    if(test_df[column].isnull().any()):

        print("Column Name:",column,"|Column Data type:",test_df[column].dtype,",|Not null count:",test_df[column].count(),'Total count:',TotalRecsTest,"|Null Values",TotalRecsTest-test_df[column].count())
freq_age_train=train_df.Age.dropna().mode()[0]

freq_embarked=train_df.Embarked.dropna().mode()[0]

freq_age_test=test_df.Age.dropna().mode()[0]

freq_fare_test=test_df.Fare.dropna().mode()[0]

train_df['Age']=train_df['Age'].fillna(freq_age_train)

train_df['Embarked']=train_df['Embarked'].fillna(freq_embarked)

test_df['Age']=test_df['Age'].fillna(freq_age_test)

test_df['Fare']=test_df['Fare'].fillna(freq_fare_test)

#After filling null values, lets drop Cabin column from test and train data.

train_df=train_df.drop(['Cabin'],axis=1)

test_df=test_df.drop(['Cabin'],axis=1)

# To be on safe side, whenever a column is dropped then check shape of data set

train_df.shape,test_df.shape
train_df.columns
train_df=train_df.drop(['Ticket'],axis=1)

test_df=test_df.drop(['Ticket'],axis=1)

train_df.shape,test_df.shape
train_df_grp_Pclass=train_df.groupby(['Survived','Pclass']).size()

train_df_grp_Pclass
#now lets check the gender now

train_df_grp_gender=train_df.groupby(['Survived','Sex']).size()

train_df_grp_gender
train_df_grp_embarked=train_df.groupby(['Survived','Embarked']).size()

train_df_grp_embarked
train_df_grp_emb_gen=train_df.groupby(['Survived','Embarked','Sex']).size()

train_df_grp_emb_gen
# Total number of people boared from each port

PortCppl=train_df['Embarked'].loc[train_df['Embarked']=='C'].count()

PortSppl=train_df['Embarked'].loc[train_df['Embarked']=='S'].count()

PortQppl=train_df['Embarked'].loc[train_df['Embarked']=='Q'].count()



#Females survived on each port

FemalesSuvPortC=train_df['Survived'].loc[train_df['Embarked']=='C'].loc[train_df['Sex']=='female'].loc[train_df['Survived']==1].count()

FemalesSuvPortS=train_df['Survived'].loc[train_df['Embarked']=='S'].loc[train_df['Sex']=='female'].loc[train_df['Survived']==1].count()

FemalesSuvPortQ=train_df['Survived'].loc[train_df['Embarked']=='Q'].loc[train_df['Sex']=='female'].loc[train_df['Survived']==1].count()

#Females died on each port

FemalesDiedPortC=train_df['Survived'].loc[train_df['Embarked']=='C'].loc[train_df['Sex']=='female'].loc[train_df['Survived']==0].count()

FemalesDiedPortS=train_df['Survived'].loc[train_df['Embarked']=='S'].loc[train_df['Sex']=='female'].loc[train_df['Survived']==0].count()

FemalesDiedPortQ=train_df['Survived'].loc[train_df['Embarked']=='Q'].loc[train_df['Sex']=='female'].loc[train_df['Survived']==0].count()

#Males survived on each port

MalesSuvPortC=train_df['Survived'].loc[train_df['Embarked']=='C'].loc[train_df['Sex']=='male'].loc[train_df['Survived']==1].count()

MalesSuvPortS=train_df['Survived'].loc[train_df['Embarked']=='S'].loc[train_df['Sex']=='male'].loc[train_df['Survived']==1].count()

MalesSuvPortQ=train_df['Survived'].loc[train_df['Embarked']=='Q'].loc[train_df['Sex']=='male'].loc[train_df['Survived']==1].count()

#Males died on each port

MalesDiedPortC=train_df['Survived'].loc[train_df['Embarked']=='C'].loc[train_df['Sex']=='male'].loc[train_df['Survived']==0].count()

MalesDiedPortS=train_df['Survived'].loc[train_df['Embarked']=='S'].loc[train_df['Sex']=='male'].loc[train_df['Survived']==0].count()

MalesDiedPortQ=train_df['Survived'].loc[train_df['Embarked']=='Q'].loc[train_df['Sex']=='male'].loc[train_df['Survived']==0].count()

#Lets create a dataframe to visualize all this now

embarked_df=pd.DataFrame({"1_Total People Embarked":[PortCppl,PortSppl,PortQppl],"2_Males Survived" : [MalesSuvPortC,MalesSuvPortS,MalesSuvPortQ],"2_Males Died" : [MalesDiedPortC,MalesDiedPortS,MalesDiedPortQ],"3_Females Survived":[FemalesSuvPortC,FemalesSuvPortS,FemalesSuvPortQ],"3_Females Died":[FemalesDiedPortC,FemalesDiedPortS,FemalesDiedPortQ],

                          "4_Overall Survival Rate":[round(((MalesSuvPortC+FemalesSuvPortC)/PortCppl)*100,2),round(((MalesSuvPortS+FemalesSuvPortS)/PortSppl)*100,2),round(((MalesSuvPortQ+FemalesSuvPortQ)/PortQppl)*100,2)]

                          

                         ,'5_Female Survival Rate':[round((FemalesSuvPortC/(FemalesDiedPortC+FemalesSuvPortC))*100,2),round((FemalesSuvPortS/(FemalesDiedPortS+FemalesSuvPortS))*100,2),round((FemalesSuvPortQ/(FemalesDiedPortQ+FemalesSuvPortQ))*100,2)]

                         

                         ,'6_Male Survival Rate':[round((MalesSuvPortC/(MalesDiedPortC+MalesSuvPortC))*100,2),round((MalesSuvPortS/(MalesDiedPortS+MalesSuvPortS))*100,2),round((MalesSuvPortQ/(MalesDiedPortQ+MalesSuvPortQ))*100,2)]

                         

                         },index=["C","S","Q"])

embarked_df.index.name="Port"

embarked_df





#Name doest seem to have much of an impact but it has titles so will extract titles

train_df['Name'].str.split(',', expand = True)[1].str.split('.',expand=True)[0]

train_df['Title']=train_df['Name'].str.split(',', expand = True)[1].str.split('.',expand=True)[0].str.strip()

test_df['Title']=test_df['Name'].str.split(',', expand = True)[1].str.split('.',expand=True)[0].str.strip()



#Dropping names column

train_df=train_df.drop(['Name'],axis=1)

test_df=test_df.drop(['Name'],axis=1)

train_df.shape,test_df.shape
train_df_grp_title=train_df.groupby(['Survived','Sex','Title']).size()

train_df_grp_title
train_df.groupby(['Title']).groups.keys()
#We are going to map these titles to more generic ones in training and testing data

train_df['Title']=train_df['Title'].map({'Miss':'Miss','Lady':'Miss','Mlle':'Miss','Ms':'Miss','the Countess':'Miss','Mme':'Miss','Mrs':'Mrs','Mr':'Mr','Master':'Master','Capt':'Others','Col':'Others','Don':'Others','Dr':'Others','Jonkheer':'Others','Major':'Others','Rev':'Others','Sir':'Others'})

test_df['Title']=test_df['Title'].map({'Miss':'Miss','Lady':'Miss','Mlle':'Miss','Ms':'Miss','the Countess':'Miss','Mme':'Miss','Mrs':'Mrs','Mr':'Mr','Master':'Master','Capt':'Others','Col':'Others','Don':'Others','Dr':'Others','Jonkheer':'Others','Major':'Others','Rev':'Others','Sir':'Others'})
train_df_grp_title=train_df.groupby(['Survived','Sex','Title']).size()

train_df_grp_title
train_df.groupby(['SibSp']).groups.keys()
alonePassengers=len(train_df.groupby(['SibSp']).groups[0])

famMem1=len(train_df.groupby(['SibSp']).groups[1])

famMem2=len(train_df.groupby(['SibSp']).groups[2])

famMem3=len(train_df.groupby(['SibSp']).groups[3])

famMem4=len(train_df.groupby(['SibSp']).groups[4])

famMem5=len(train_df.groupby(['SibSp']).groups[5])

famMem8=len(train_df.groupby(['SibSp']).groups[8])
AloneSurvived=train_df['Survived'].loc[train_df['SibSp']==0].loc[train_df['Survived']==1].count()

print('Passengers travelling alone:',alonePassengers,'| Passengers survived:',AloneSurvived,'|Survival %:',round((AloneSurvived/alonePassengers)*100,2))

With1Person=train_df['Survived'].loc[train_df['SibSp']==1].loc[train_df['Survived']==1].count()

print('Passengers travelling with one spouse or family member:',famMem1,'| Passengers survived:',With1Person,'|Survival %:',round((With1Person/famMem1)*100,2))

With2Person=train_df['Survived'].loc[train_df['SibSp']==2].loc[train_df['Survived']==1].count()

print('Passengers travelling with spouse or family member(2):',famMem2,'|Passengers survived:',With2Person,'|Survival %:',round((With2Person/famMem2)*100,2))

With3Person=train_df['Survived'].loc[train_df['SibSp']==3].loc[train_df['Survived']==1].count()

print('Passengers travelling with spouse or family member(3):',famMem3,'|Passengers survived:',With3Person,'|Survival %:',round((With3Person/famMem3)*100,2))

With4Person=train_df['Survived'].loc[train_df['SibSp']==4].loc[train_df['Survived']==1].count()

print('Passengers travelling with spouse or family member(4):',famMem4,'|Passengers survived:',With4Person,'|Survival %:',round((With4Person/famMem4)*100,2))

With5Person=train_df['Survived'].loc[train_df['SibSp']==5].loc[train_df['Survived']==1].count()

print('Passengers travelling with spouse or family member(5):',famMem5,'|Passengers survived:',With5Person,'|Survival %:',round((With5Person/famMem5)*100,2))

With8Person=train_df['Survived'].loc[train_df['SibSp']==8].loc[train_df['Survived']==1].count()

print('Passengers travelling with spouse or family member(8):',famMem8,'|Passenfers survived:',With8Person,'|Survival %:',round((With8Person/famMem8)*100,2))
train_df['SibSp']=train_df['SibSp'].map({0:1,1:1,2:1,3:2,4:2,5:2,8:2})

test_df['SibSp']=test_df['SibSp'].map({0:1,1:1,2:1,3:2,4:2,5:2,8:2})
train_df.groupby(['Parch']).groups.keys()
AloneP=len(train_df.groupby(['Parch']).groups[0])

PMem1=len(train_df.groupby(['Parch']).groups[1])

PMem2=len(train_df.groupby(['Parch']).groups[2])

PMem3=len(train_df.groupby(['Parch']).groups[3])

PMem4=len(train_df.groupby(['Parch']).groups[4])

PMem5=len(train_df.groupby(['Parch']).groups[5])

PMem6=len(train_df.groupby(['Parch']).groups[6])



AlonePS=train_df['Survived'].loc[train_df['Parch']==0].loc[train_df['Survived']==1].count()

print('Passengers travelling alone:',AloneP,'| Passengers survived:',AlonePS,'|Survival %:',round((AlonePS/AloneP)*100,2))

With1Parent=train_df['Survived'].loc[train_df['Parch']==1].loc[train_df['Survived']==1].count()

print('Passengers travelling parent or child:',PMem1,'| Passengers survived:',With1Parent,'|Survival %:',round((With1Parent/PMem1)*100,2))

With2Parent=train_df['Survived'].loc[train_df['Parch']==2].loc[train_df['Survived']==1].count()

print('Passengers travelling with parent or child(2):',PMem2,'|Passengers survived:',With2Parent,'|Survival %:',round((With2Parent/PMem2)*100,2))

With3Parent=train_df['Survived'].loc[train_df['Parch']==3].loc[train_df['Survived']==1].count()

print('Passengers travelling with parent or child(3):',PMem3,'|Passengers survived:',With3Parent,'|Survival %:',round((With3Parent/PMem4)*100,2))

With4Parent=train_df['Survived'].loc[train_df['Parch']==4].loc[train_df['Survived']==1].count()

print('Passengers travelling with parent or child(4):',PMem4,'|Passengers survived:',With4Parent,'|Survival %:',round((With4Parent/PMem4)*100,2))

With5Parent=train_df['Survived'].loc[train_df['Parch']==5].loc[train_df['Survived']==1].count()

print('Passengers travelling with parent or child(5):',PMem5,'|Passengers survived:',With5Parent,'|Survival %:',round((With5Parent/PMem5)*100,2))

With6Parent=train_df['Survived'].loc[train_df['Parch']==6].loc[train_df['Survived']==1].count()

print('Passengers travelling with spouse or family member(8):',PMem6,'|Passenfers survived:',With6Parent,'|Survival %:',round((With6Parent/PMem6)*100,2))
train_df['Parch']=train_df['Parch'].map({0:1,1:1,2:1,3:1,4:2,5:2,6:2})

test_df['Parch']=test_df['Parch'].map({0:1,1:1,2:1,3:1,4:2,5:2,6:2,9:2})
#Total number of passengers travelling in each class

class1=train_df['Pclass'].loc[train_df['Pclass']==1].count()

class2=train_df['Pclass'].loc[train_df['Pclass']==2].count()

class3=train_df['Pclass'].loc[train_df['Pclass']==3].count()



#Total number of passengers survived in each class

class1Sur=train_df['Pclass'].loc[train_df['Pclass']==1].loc[train_df['Survived']==1].count()

class2Sur=train_df['Pclass'].loc[train_df['Pclass']==2].loc[train_df['Survived']==1].count()

class3Sur=train_df['Pclass'].loc[train_df['Pclass']==3].loc[train_df['Survived']==1].count()



# Creating dataset for a better understanding

SurDF=pd.DataFrame({'0_Class':[1,2,3],'1_Total passengers':[class1,class2,class3],'2_Survived':[class1Sur,class2Sur,class3Sur],'3_Not Survived':[class1-class1Sur,class2-class2Sur,class3-class3Sur],'4_Survival Percent':[round((class1Sur/class1)*100,2),round((class2Sur/class2)*100,2),round((class3Sur/class3)*100,2)]})

SurDF.sort_values(by='4_Survival Percent',ascending=False)

#Now comes the fare part

print('Average fare in 1st class: ',train_df['Fare'].loc[train_df['Pclass']==1].mean())

print('Average fare in 2nd class: ',train_df['Fare'].loc[train_df['Pclass']==2].mean())

print('Average fare in 3rd class: ',train_df['Fare'].loc[train_df['Pclass']==3].mean())
train_df=train_df.drop(['Fare'],axis=1)

test_df=test_df.drop(['Fare'],axis=1)

train_df.shape,test_df.shape
#Now lets analyze age

minAge=train_df['Age'].min()

maxAge=train_df['Age'].max()

print('Age Range:',minAge,' to ',maxAge)
index1=train_df['Age'].loc[(train_df['Age']>=0) & (train_df['Age']<10)].index

index2=train_df['Age'].loc[(train_df['Age']>=10) & (train_df['Age']<20)].index

index3=train_df['Age'].loc[(train_df['Age']>=20) & (train_df['Age']<30)].index

index4=train_df['Age'].loc[(train_df['Age']>=30) & (train_df['Age']<40)].index

index5=train_df['Age'].loc[(train_df['Age']>=40) & (train_df['Age']<50)].index

index6=train_df['Age'].loc[(train_df['Age']>=50) & (train_df['Age']<60)].index

index7=train_df['Age'].loc[(train_df['Age']>=60)].index

for idx in index1:

    train_df.loc[idx,'Age']=1

for idx in index2:

    train_df.loc[idx,'Age']=2

for idx in index3:

    train_df.loc[idx,'Age']=3

for idx in index4:

    train_df.loc[idx,'Age']=4

for idx in index5:

    train_df.loc[idx,'Age']=5

for idx in index6:

    train_df.loc[idx,'Age']=6

for idx in index7:

    train_df.loc[idx,'Age']=7
#Converting to integer type

train_df['Age']=train_df['Age'].astype(int)
Between_1_10=train_df[['Age','Survived']].loc[(train_df['Age']==1) &(train_df['Survived']==1)].size

Between_10_20=train_df[['Age','Survived']].loc[(train_df['Age']==2) &(train_df['Survived']==1)].size

Between_20_30=train_df[['Age','Survived']].loc[(train_df['Age']==3) &(train_df['Survived']==1)].size

Between_30_40=train_df[['Age','Survived']].loc[(train_df['Age']==4) &(train_df['Survived']==1)].size

Between_40_50=train_df[['Age','Survived']].loc[(train_df['Age']==5) &(train_df['Survived']==1)].size

Between_50_60=train_df[['Age','Survived']].loc[(train_df['Age']==6) &(train_df['Survived']==1)].size

Above_60=train_df[['Age','Survived']].loc[(train_df['Age']==7) &(train_df['Survived']==1)].size

Total_Between_1_10=train_df[['Age','Survived']].loc[(train_df['Age']==1)].size

Total_Between_10_20=train_df[['Age','Survived']].loc[(train_df['Age']==2) ].size

Total_Between_20_30=train_df[['Age','Survived']].loc[(train_df['Age']==3) ].size

Total_Between_30_40=train_df[['Age','Survived']].loc[(train_df['Age']==4) ].size

Total_Between_40_50=train_df[['Age','Survived']].loc[(train_df['Age']==5) ].size

Total_Between_50_60=train_df[['Age','Survived']].loc[(train_df['Age']==6) ].size

Total_Above_60=train_df[['Age','Survived']].loc[(train_df['Age']==7) ].size



# Making data set to present survival data in tabular form

Age_Sur_df=pd.DataFrame({'Total':[Total_Between_1_10,Total_Between_10_20,Total_Between_20_30,Total_Between_30_40,Total_Between_40_50,Total_Between_50_60,Total_Above_60],'Survived':[Between_1_10,Between_10_20,Between_20_30,Between_30_40,Between_40_50,Between_50_60,Above_60],'Age':['Child<10 years','Between 10 and 20','Between 20 and 30','Between 30 and 40','Between 40 and 50','Between 50 and 60','Above 60'],

                         'Survival %':[round((Between_1_10/Total_Between_1_10)*100,2),round((Between_10_20/Total_Between_10_20)*100,2),round((Between_20_30/Total_Between_20_30)*100,2),round((Between_30_40/Total_Between_30_40)*100,2),round((Between_40_50/Total_Between_40_50)*100,2),round((Between_50_60/Total_Between_50_60)*100,2),round((Above_60/Total_Above_60)*100,2)]

                        

                        

                        },index={1,2,3,4,5,6,7})



Age_Sur_df
index1_test=test_df['Age'].loc[(test_df['Age']>=0) & (test_df['Age']<10)].index

index2_test=test_df['Age'].loc[(test_df['Age']>=10) & (test_df['Age']<20)].index

index3_test=test_df['Age'].loc[(test_df['Age']>=20) & (test_df['Age']<30)].index

index4_test=test_df['Age'].loc[(test_df['Age']>=30) & (test_df['Age']<40)].index

index5_test=test_df['Age'].loc[(test_df['Age']>=40) & (test_df['Age']<50)].index

index6_test=test_df['Age'].loc[(test_df['Age']>=50) & (test_df['Age']<60)].index

index7_test=test_df['Age'].loc[(test_df['Age']>=60)].index

for idx in index1_test:

    test_df.loc[idx,'Age']=1

for idx in index2_test:

    test_df.loc[idx,'Age']=2

for idx in index3_test:

    test_df.loc[idx,'Age']=3

for idx in index4_test:

    test_df.loc[idx,'Age']=4

for idx in index5_test:

    test_df.loc[idx,'Age']=5

for idx in index6_test:

    test_df.loc[idx,'Age']=6

for idx in index7_test:

    test_df.loc[idx,'Age']=7
#converting test data age column to int

test_df['Age']=test_df['Age'].astype(int)
def EncodeColumn(cats,ColName,df):

    dummiesTrain=pd.get_dummies(df[ColName],prefix=ColName,prefix_sep='_')

    dummiesTrain=dummiesTrain.T.reindex(cats).T.fillna(0).astype(int)

    df=pd.concat([df,dummiesTrain],axis=1)

    df.drop([ColName],axis=1,inplace=True)

    return df
train_df=EncodeColumn(['Sex_male','Sex_female'],'Sex',train_df)

test_df=EncodeColumn(['Sex_male','Sex_female'],'Sex',test_df)
#Coverting embarked column

cats_embarked=['Embarked_S','Embarked_C','Embarked_Q']

train_df=EncodeColumn(cats_embarked,'Embarked',train_df)

test_df=EncodeColumn(cats_embarked,'Embarked',test_df)
test_df.isnull().any()
train_df.isnull().any()
test_df[pd.isnull(test_df).any(axis=1)]
test_df.loc[414,'Title']='Miss'

test_df['Title'].isnull().any()
#Now lets encode title column



titleCats=['Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Others']

train_df=EncodeColumn(titleCats,'Title',train_df)

test_df=EncodeColumn(titleCats,'Title',test_df)

y=train_df['Survived']

X=train_df.drop(['Survived','PassengerId'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.7)
#Random Forest

clf=RandomForestClassifier(n_estimators=300)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

accuracy_randF=round(metrics.accuracy_score(y_test,y_pred),2)
#SVC with linear kernel

clf_SVMLin = svm.SVC(kernel='linear', C = 1)

clf_SVMLin.fit(X_train,y_train)

y_pred_lin=clf_SVMLin.predict(X_test)

accuracy_SVCLin=round(metrics.accuracy_score(y_test,y_pred_lin),2)

#SVC with default DBF kernel

clf_SVM_RBF = svm.SVC(kernel='rbf', C = 1.0)

clf_SVM_RBF.fit(X_train,y_train)

y_pred_rbf=clf_SVM_RBF.predict(X_test)

accuracy_svm_rbf=round(metrics.accuracy_score(y_test,y_pred_rbf),2)

Acc_dataSet=pd.DataFrame({'Classifier Name':['Random Forest','Linear SVC','RBF SVC'],

                          'Accuracy':[accuracy_randF,accuracy_SVCLin,accuracy_svm_rbf]

    

})

Acc_dataSet.sort_values(by='Accuracy',ascending=False)
X_test_final=test_df.drop(['PassengerId'],axis=1)

y_pred_final=clf_SVMLin.predict(X_test_final)

submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_pred_final})

#submission.to_csv('TitanicPred.csv',index=False)