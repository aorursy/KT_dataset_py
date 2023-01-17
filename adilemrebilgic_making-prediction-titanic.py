# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.metrics import confusion_matrix,precision_score

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn import svm

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier,VotingClassifier 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
trainData=pd.read_csv('../input/train.csv', index_col=0)

testData=pd.read_csv('../input/test.csv', index_col=0)
#Shape of data

print('Number of obs: ',trainData.shape[0])

print('Number of features: ',trainData.shape[1])
trainData.head().T
trainData.info()
trainData.describe(include='all')
#Visualize Numeric Vals

##Age

fig=plt.figure()



###

ax1=fig.add_subplot(4,3,1)

ax1=trainData['Age'].plot(kind='hist',

         title='Age',

        

         grid=True,

        figsize=(25,20))



###

ax2=fig.add_subplot(4,3,2)

ax2=trainData['Age'].plot(kind='line',

         title='Age',

         style='o',

         grid=False,

         alpha=0.5,figsize=(25,20)

         )

ax2=plt.plot([trainData['Age'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)

ax2=plt.plot([trainData['Age'].quantile(q=0.01)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)

ax2=plt.plot([trainData['Age'].quantile(q=0.99)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)



###

ax3=fig.add_subplot(4,3,3)

ax3=trainData['Age'].plot(kind='box',figsize=(26,18))



##Fare

###

ax4=fig.add_subplot(4,3,4)

ax4=trainData['Fare'].plot(kind='hist',

         title='Fare',

         grid=True,

        figsize=(25,20))

###

ax5=fig.add_subplot(4,3,5)

ax5=trainData['Fare'].plot(kind='line',

         title='Fare',

         style='o',

         grid=False,

         alpha=0.5,figsize=(25,20)

         )

ax5=plt.plot([trainData['Fare'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)

ax5=plt.plot([trainData['Fare'].quantile(q=0.01)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)

ax5=plt.plot([trainData['Fare'].quantile(q=0.99)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)



###

ax6=fig.add_subplot(4,3,6)

ax6=trainData['Fare'].plot(kind='box')



##Sibsp

###



ax7=fig.add_subplot(4,3,7)

ax7=trainData['SibSp'].plot(kind='hist',

         title='SibSp',

         grid=True,

        figsize=(16,8))

###

ax8=fig.add_subplot(4,3,8)

ax8=trainData['SibSp'].plot(kind='line',

         title='SibSp',

         style='o',

         grid=False,

         alpha=0.5,figsize=(25,20)

         )

ax8=plt.plot([trainData['SibSp'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)

ax8=plt.plot([trainData['SibSp'].quantile(q=0.01)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)





ax8=plt.plot([trainData['SibSp'].quantile(q=0.99)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)



###

ax9=fig.add_subplot(4,3,9)

ax9=trainData['SibSp'].plot(kind='box')





##Parch

###



ax10=fig.add_subplot(4,3,10)

ax10=trainData['Parch'].plot(kind='hist',

         title='Parch',

         grid=True,

        figsize=(25,20))

###

ax11=fig.add_subplot(4,3,11)

ax11=trainData['Parch'].plot(kind='line',

         title='Parch',

         style='o',

         grid=False,

         alpha=0.5,figsize=(25,20)

         )

ax11=plt.plot([trainData['Parch'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)

ax11=plt.plot([trainData['Parch'].quantile(q=0.01)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)





ax11=plt.plot([trainData['Parch'].quantile(q=0.99)]*trainData.shape[0],

          linestyle='--',

          color='red',

          alpha=0.5)



###

ax12=fig.add_subplot(4,3,12)

ax12=trainData['Parch'].plot(kind='box')



plt.tight_layout()
feature_list=['Survived','Pclass','Sex','Embarked']

fig_row_size=len(feature_list)

fig_col_size=2

fig=plt.figure()

## Survived

for i in range(len(feature_list)):

    feature=feature_list[i]

    fig_name=('ax'+str(i))

    fig_name=fig.add_subplot(fig_row_size,fig_col_size,int(i+1))

    trainData[feature].value_counts().plot.pie(startangle=90,

                                         autopct='%1.0f%%',

                                         figsize=(8,14),

                                         colormap='Pastel1',

                                         ax=fig_name

                                              )

    fig_name.set_ylabel(feature)

    

plt.tight_layout()
#Features and survival rates

def ageGroup(age):

    if age > 100: return '100+'

    elif age > 90: return '90-99'

    elif age > 80: return '80-89'

    elif age > 70: return '70-79'

    elif age > 60: return '60-69'

    elif age > 50: return '50-59'

    elif age > 40: return '40-49'

    elif age > 30: return '30-39'

    elif age > 20: return '20-29'

    elif age > 10: return '10-19'

    else: return '10-'



trainData['Age_Group']=trainData['Age'].map(ageGroup)

feature_list=['Pclass','Age_Group','Sex','SibSp','Parch','Embarked']

fig_row_size=len(feature_list)

fig_col_size=1

fig=plt.figure()

for i in range(len(feature_list)):

    feature=feature_list[i]

    fig_name=fig.add_subplot(fig_row_size,fig_col_size,int(i+1))

    df_for_plotting=trainData.groupby(feature).agg(['sum','count'])['Survived']

    df_for_plotting['SurvivalRate']=df_for_plotting['sum']/df_for_plotting['count']

    plot_title=('Survival Rate - '+feature)

    df_for_plotting.SurvivalRate.plot(kind='bar',title=plot_title,ax=fig_name,figsize=(16,28))

    fig_name.set_ylabel('Survival Rate')



plt.tight_layout()

##Create a function to find variables with missing value

def anyMissingValueInDataFrame(df):

    for column in df.columns:

        if df[column].isnull().any():

            numberOfMissingVals=df[column].isnull().sum()

            print(' {} feature has {} missing values'.format(column,numberOfMissingVals));



anyMissingValueInDataFrame(trainData)  
##You should also check the test data for missing values.

anyMissingValueInDataFrame(testData)  
#Replacing missing values

##age

print('<--Missing Value Replacement-Age-->')

print(trainData.groupby(['Sex','Pclass']).agg(['mean','max','min','median'])['Age'])

print('Age feature differentiates between PClass and Sex groups. It is better to replace missing values with group median')

trainData['Age']=trainData.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

testData['Age']=testData.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

print('Missing {} values are replaced!!!'.format('Age'))

print('')

##Cabin

print('<--Missing Value Replacement-Cabin-->')

print('Most of cabin values are missing. Droping this feature is better')

trainData=trainData.drop(labels='Cabin', axis=1,errors='ignore')

testData=testData.drop(labels='Cabin', axis=1,errors='ignore')

print('{} feature is dropped!!!'.format('Cabin'))

print('')

##Embarked

print('<--Missing Value Replacement-Embarked-->')

trainData['Embarked']=trainData['Embarked'].fillna(trainData['Embarked'].value_counts().idxmax())

testData['Embarked']=testData['Embarked'].fillna(testData['Embarked'].value_counts().idxmax())

print('Missing {} values are replaced!!!'.format('Embarked'))

print('')

##Fare

print('<--Missing Value Replacement-Fare-->')

trainData['Fare']=trainData.groupby(['Pclass','Sex'])['Fare'].transform(lambda x: x.fillna(x.median()))

testData['Fare']=testData.groupby(['Pclass','Sex'])['Fare'].transform(lambda x: x.fillna(x.median()))

print('Missing {} values are replaced!!!'.format('Fare'))
trainData=pd.get_dummies(trainData, columns=['Pclass','Sex','Embarked'])

testData=pd.get_dummies(testData, columns=['Pclass','Sex','Embarked'])
trainDataTarget=trainData['Survived']

trainDataFeatures=trainData.drop(labels=['Survived','Name','Ticket','Age_Group'],axis=1)

testDataFeatures=testData.drop(labels=['Name','Ticket'],axis=1)
## first create a scaler

scaler=MinMaxScaler()

scaler.fit(trainDataFeatures)

## apply the scaler to the data

trainDataFeatures=pd.DataFrame(scaler.transform(trainDataFeatures), index=trainDataFeatures.index,columns=trainDataFeatures.columns)

testDataFeatures=pd.DataFrame(scaler.transform(testDataFeatures), index=testDataFeatures.index,columns=testDataFeatures.columns)
models_columns=['Model_Name','Parameter','Model_Trained','Accuracy_Precision','Sensitivity','Lift']

models=pd.DataFrame(columns=models_columns)
for d in np.arange(2,15,1):

    #Create classifier

    dt=DecisionTreeClassifier(criterion='entropy',

                           max_depth=d)

    #do cross validation in 5 folds

    scores = cross_val_score(dt, trainDataFeatures, trainDataTarget, cv=5)

    print('Cross Val Scores-{:.2f}:'.format(d),scores)

    print('Mean Accuracy-{}: {:.2f}'.format(d,scores.mean()))

    

    dt_model=dt.fit(trainDataFeatures,trainDataTarget)

    dt_pred=dt_model.predict(trainDataFeatures)

    

    tn, fp, fn, tp=confusion_matrix(y_true=trainDataTarget, 

                                    y_pred= dt_pred

                                   ).ravel()

    print('tn: {}, fp:{}, fn:{}, tp:{}'.format(tn, fp, fn, tp))

    

    Sensitivity=tp/(tp+fn)

    print('Sensitivity-{} : {:.2f}'.format(d,Sensitivity))

    

    Accuracy_Precision=precision_score(y_true=trainDataTarget, y_pred=dt_pred,average='binary')

    print('Precision-{} :{:.2f}'.format(d,Accuracy_Precision))

    

    model_success=tp/(tp+fp)

    random_selection=(tp+fn)/(tp+fp+tn+fn)

    lift=model_success/random_selection

    print('Lift-{} :{:.2f}'.format(d,lift))

    

    print('-'*50)

    Model='dt'+str(d)

    Parameter=d

    Model_Trained=dt_model

    lift=lift

    new_row=[Model,Parameter,Model_Trained,Accuracy_Precision,Sensitivity,lift]

    models.loc[-1]=new_row

    models.index=models.index+1



%matplotlib inline

models.plot(x='Parameter',y=['Sensitivity','Accuracy_Precision','Lift'])

plt.show();
##knn

for n_neighbor in np.arange(1,10,1):

    knn=KNeighborsClassifier(n_neighbors=n_neighbor,

                    weights='uniform',

                    algorithm='auto', 

                    )

    print('-'*10,'Number Of Neighbors:{}'.format(n_neighbor),'-'*10)

    scores_knn=cross_val_score(knn,trainDataFeatures,trainDataTarget,cv=5)

    print('{} accuracy is:{:.2f}'.format('KNN',scores_knn.mean()))

    

    knn_model=knn.fit(trainDataFeatures,trainDataTarget)

    trainDataTarget_pred=knn.predict(trainDataFeatures)

    

    tn, fp, fn, tp=confusion_matrix(y_true=trainDataTarget, y_pred=trainDataTarget_pred).ravel()

    print('tn: {}, fp:{}, fn:{}, tp:{}'.format(tn, fp, fn, tp)) 

    

    Sensitivity=tp/(tp+fn)

    print('Sensitivity: {:.2f}'.format(Sensitivity))



    Accuracy_Precision=precision_score(y_true=trainDataTarget, y_pred=trainDataTarget_pred,average='binary')

    print('Precision:{:.2f}'.format(Accuracy_Precision))

    plt.plot(n_neighbor,Sensitivity,'bo')



    model_success=tp/(tp+fp)

    random_selection=(tp+fn)/(tp+fp+tn+fn)

    lift=model_success/random_selection

    print('Lift:{:.2f}'.format(lift))

    

    Model='knn'+str(n_neighbor)

    Parameter=n_neighbor

    Model_Trained=knn_model

    lift=lift

    new_row=[Model,Parameter,Model_Trained,Accuracy_Precision,Sensitivity,lift]

    models.loc[-1]=new_row

    models.index=models.index+1

    
##log reg



#log_reg=LogisticRegression(solver='liblinear',max_iter=100) 

log_reg=LogisticRegression()

scores_log_reg=cross_val_score(log_reg,

                               trainDataFeatures,

                               trainDataTarget,

                               cv=5

                              )

print('{} accuracy is:{:.2f}'.format('Logistic Regression',

                                     scores_log_reg.mean()

                                    )

     )



log_reg_model=log_reg.fit(trainDataFeatures,

            trainDataTarget

           )

log_reg_pred=log_reg.predict(trainDataFeatures

                            )



tn, fp, fn, tp=confusion_matrix(y_true=trainDataTarget, 

                                y_pred=log_reg_pred

                               ).ravel()

print('tn: {}, fp:{}, fn:{}, tp:{}'.format(tn, 

                                           fp, 

                                           fn, 

                                           tp)

     ) 



Sensitivity=tp/(tp+fn)

print('Sensivity: {:.2f}'.format(Sensitivity))



Accuracy_Precision=precision_score(y_true=trainDataTarget, 

                                   y_pred=log_reg_pred,

                                   average='binary'

                                  )

print('Precision:{:.2f}'.format(Accuracy_Precision)

     )



model_success=tp/(tp+fp)

random_selection=(tp+fn)/(tp+fp+tn+fn)

lift=model_success/random_selection

print('Lift:{:.2f}'.format(lift))





Model='logReg'+str(1)

Parameter=0

Model_Trained=log_reg_model

lift=lift

new_row=[Model,Parameter,Model_Trained,Accuracy_Precision,Sensitivity,lift]

models.loc[-1]=new_row

models.index=models.index+1



RandomForest=RandomForestClassifier(n_estimators=10, #you can change this parameter to change number of tree in your forest.

                                    criterion='gini' 

                                    )





scores_RandomForest=cross_val_score(RandomForest,

                             trainDataFeatures,

                             trainDataTarget,cv=5)









print('{} accuracy is:{:.2f}'.format('Random Forest',scores_RandomForest.mean()))



RandomForest_model=RandomForest.fit(trainDataFeatures,trainDataTarget)

RandomForest_pred=RandomForest.predict(trainDataFeatures)



tn, fp, fn, tp=confusion_matrix(y_true=trainDataTarget, y_pred=RandomForest_pred).ravel()

print('tn: {}, fp:{}, fn:{}, tp:{}'.format(tn, fp, fn, tp)) 



Sensitivity=tp/(tp+fn)

print('Sensitivity: {:.2f}'.format(Sensitivity))



Accuracy_Precision=precision_score(y_true=trainDataTarget, y_pred=RandomForest_pred,average='binary')

print('Precision:{:.2f}'.format(Accuracy_Precision))



model_success=tp/(tp+fp)

random_selection=(tp+fn)/(tp+fp+tn+fn)

lift=model_success/random_selection

print('Lift:{:.2f}'.format(lift))



#Decision tree

dt=DecisionTreeClassifier(criterion='entropy',

                           max_depth=6)

#K Nearest Neighbours

knn=KNeighborsClassifier(n_neighbors=1,

                weights='uniform',

                algorithm='auto', 

                )

#Logistic Regression

log_reg=LogisticRegression()



#Random Forest

RandomForest=RandomForestClassifier(n_estimators=10, 

                                    criterion='gini' 

                                    )







clf1 = dt

clf2 = knn

clf3 = log_reg

clf4 =RandomForest



#Create your ensembled classifier

Voting_Classifier = VotingClassifier(estimators=[('DecisionTree', clf1),

                                    ('KNN_1', clf2), 

                                    ('LogReg', clf3),

                                   ('RandomForest', clf4)

                                   ], 

                        voting='hard')
scores_Voting_Classifier=cross_val_score(Voting_Classifier,

                             trainDataFeatures,

                             trainDataTarget,cv=5)









print('{} accuracy is:{:.2f}'.format('Voting Classifier',scores_Voting_Classifier.mean()))



Voting_Classifier_model=Voting_Classifier.fit(trainDataFeatures,trainDataTarget)

Voting_Classifier_pred=Voting_Classifier.predict(trainDataFeatures)



tn, fp, fn, tp=confusion_matrix(y_true=trainDataTarget, y_pred=Voting_Classifier_pred).ravel()

print('tn: {}, fp:{}, fn:{}, tp:{}'.format(tn, fp, fn, tp)) 



Sensitivity=tp/(tp+fn)

print('Sensivity: {:.2f}'.format(Sensitivity))



Accuracy_Precision=precision_score(y_true=trainDataTarget, y_pred=Voting_Classifier_pred,average='binary')

print('Precision:{:.2f}'.format(Accuracy_Precision))



model_success=tp/(tp+fp)

random_selection=(tp+fn)/(tp+fp+tn+fn)

lift=model_success/random_selection

print('Lift:{:.2f}'.format(lift))
#Predict test data and create a new column, Survived

testDataFeatures['Survived']=Voting_Classifier.predict(testDataFeatures)
#Check your prediction

testDataFeatures['Survived'].head(15)
#Write your prediction to gender_submission.csv

testDataFeatures['Survived'].to_csv('gender_submission.csv')