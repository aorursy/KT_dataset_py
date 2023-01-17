# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 



# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

import numpy as np

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, VotingClassifier

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import plot_roc_curve



train_main_df = pd.read_csv("../input/titanic/train.csv")

test_main_df = pd.read_csv("../input/titanic/test.csv")



train_df=train_main_df

test_df=test_main_df
def AgetoNum(df):

    AgeChange={'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

    for key, item in AgeChange.items():

        row_indexes=df[df['Age']==key].index

        df.loc[row_indexes,'NewAge']=item

    df.Age=df.NewAge

    df.drop(['NewAge'], axis=1, inplace=True)

    df.Age=df.Age.astype(int)

    

##################

#from Scikit-Learn ML from Start to Finish Kernel for titanic: Machine Learning From Disaster



def simplify_ages(df):

    

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df

################



def feature_importance(estimator, ax, title):

    

    feat_importances = pd.Series(estimator.feature_importances_, index=X.columns)

    feat_importances.nlargest(20).plot(kind='barh', ax=ax, title=title)

    plt.title(title)
train_df.info()
test_df.info()
#Deck column code is from : Kaggle Notbook: Titanic - Advanced Feature Engineering TutorialDeck



#Deck column

train_df['Deck'] = train_df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

test_df['Deck'] = test_df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

####################################################################################

#New feature: Class_Deck

train_df['Class_Deck']=train_df['Pclass'].astype(str) + train_df['Deck']

test_df['Class_Deck']=test_df['Pclass'].astype(str) + test_df['Deck']



####################################################################################

#Once we have used Cabin to extract Deck, Cabin can be dropped. 



train_df.drop(['Cabin'], axis=1, inplace=True)

test_df.drop(['Cabin'], axis=1, inplace=True)



####################################################################################

#Passenger ID could be treated as index as they all have unique numbers. Will have no bearing on prediction so dropping it. 

#Name also has no influence on the prediction model. Droping Name too. Same goes for Ticket



train_df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)

test_df.drop(['Name','Ticket'], axis=1, inplace=True)



####################################################################################



#Embarked

#Droping these two Null entried from train_df. 



train_df.dropna(subset=['Embarked'], inplace=True)



##################################################################################



#Age

train_df[train_df.Age.isnull()]

test_df[test_df.Age.isnull()]



#Replacing Age with categories ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']



train_df.Age.fillna(- 0.5, inplace=True)

test_df.Age.fillna(- 0.5, inplace=True)



train_df=simplify_ages(train_df)

test_df=simplify_ages(test_df)



#################################################################################

#Lets see how fare is distributed:



sns.distplot(train_df.Fare, kde=False, rug=True);

plt.xlabel("Fare")

plt.ylabel("Frequency")

#Fare is left skewed, with majority of the data is between 0 and 100.  
#box plot of Pclass and Fare for train_df and test_df

fig, axes=plt.subplots(1,2, figsize=(10,6))

sns.boxplot(x='Pclass',y='Fare',data=train_df, ax=axes[0])

sns.boxplot(x='Pclass',y='Fare',data=test_df, ax=axes[1])





#Looks like there are a few outliers in the fare. Those might be the reason to skew data a bit. BUT I am going to keep them for now as I think those

#might be helpful in not overfitting the model on the training data 
#How is class related to Fare and did males paid different in each class from woman 

plt.figure(figsize=(8,6))

sns.boxplot(x='Pclass',y='Fare', hue='Sex',data=train_df)



#Females paid more in fare in Pclass1 then males. The median price females paid in Pclass1 is around 90 and 

# median price for males in the same class is around 50



#Visualizing pairwise relationship between variables in train_df



sns.pairplot(train_df, height=2)
#MORE SNS PLOTS
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = train_df)

#As seen before, Females in all three class have survived more than man. The most survived people were in Class 1. 
sns.barplot(y = 'Class_Deck', x = 'Survived', order=['1A','1B','1C','1D','1E','1M','2D','2E','2F','2M', '3E','3F','3G','3M','1T'], ci=None, data = train_df)

#How many males and females were on each Deck

temp=train_df[['Sex','Deck', 'Pclass']]



fig, ax = plt.subplots(figsize=(5,5))





#temp.groupby(['Deck','Sex']).count()['Pclass'].unstack()

temp=temp.groupby(['Deck','Sex']).count()['Pclass'].unstack().plot(ax=ax, kind='bar', colormap='RdBu',alpha=0.7)



#The following graph shows that the missing information on the deck might be very crucial in our prediction model since many people are missing that data

#More of male data for the deck information is missing than for the females. 

#How many total people per class survived or died?

temp=train_df[['Deck', 'Pclass', 'Sex','Survived']]



#temp.groupby(['Deck','Pclass']).count().unstack()

fig, ax = plt.subplots(figsize=(20,10))





#temp.groupby(['Deck','Sex']).count()['Pclass'].unstack()

temp.groupby(['Deck','Pclass','Survived'])['Sex'].count().unstack(['Deck','Survived']).plot(ax=ax, kind='bar')

plt.ylim([0,180]) 

#chaning ylim to see more clearly where majority of the people were in different classes.



#The most people died are from class3 (M,0). (M,1) looks large however, combining all survivors in class 1 makes it largest class to survive. 

#This is proven in next graph



temp=train_df[['Pclass', 'Sex','Survived']]



#temp.groupby(['Deck','Pclass']).count().unstack()

fig, ax = plt.subplots(figsize=(8,8))





#temp.groupby(['Deck','Sex']).count()['Pclass'].unstack()

temp.groupby(['Pclass','Survived'])['Sex'].count().unstack(['Survived']).plot(ax=ax, kind='bar')



#Most survival was in class 1

#Most deaths were in class 3
train_df.shape, test_df.shape
##################### Preprocessing 



enc = preprocessing.LabelEncoder()

df_combined=pd.concat([train_df, test_df])

enc = enc.fit(df_combined['Sex'])

train_df['Sex'] = enc.transform(train_df['Sex'])

test_df['Sex'] = enc.transform(test_df['Sex'])

#Changing the Age Labels (Categories) to values as the Classifiers can't work with categorical values

AgetoNum(train_df)

AgetoNum(test_df)



#########################

enc = preprocessing.LabelEncoder()

df_combined=pd.concat([train_df, test_df])

enc = enc.fit(df_combined['Class_Deck'])

train_df['Class_Deck'] = enc.transform(train_df['Class_Deck'])

test_df['Class_Deck'] = enc.transform(test_df['Class_Deck'])
enc = preprocessing.LabelEncoder()

df_combined=pd.concat([train_df, test_df])

enc = enc.fit(df_combined['Embarked'])

train_df['Embarked'] = enc.transform(train_df['Embarked'])

test_df['Embarked'] = enc.transform(test_df['Embarked'])
#Since EDA is done I can get rid of Deck here as well.

train_df.drop(['Deck'], axis=1, inplace=True)

test_df.drop(['Deck'], axis=1, inplace=True)
##### Modeling ######
X=train_df.drop(['Survived'], axis=1)

y=train_df['Survived']



X_hold=test_df.drop(['PassengerId'], axis=1)

#y_hold -> predict model for this??



X.shape, X_hold.shape, y.shape
X_hold['Fare'].fillna(method='ffill', inplace=True)

X_hold.info()
### K Nearest Neighbors Classifiers



param_grid={'n_neighbors': np.arange(1,50)}



knn=KNeighborsClassifier()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)



knn_cv=GridSearchCV(knn,param_grid)



knn_cv.fit(X_train, y_train)

y_pred=knn_cv.predict(X_test)



knn_score_test=knn_cv.score(X_test, y_test)

#knn_score_hold=knn_cv.score(X_hold, y_pred_hold)

knn_score_test
knn_cv.best_score_, knn_cv.best_params_, knn_cv.best_estimator_
#LogisticRegression





param_grid={'penalty':['l1','l2'], 

            'C': np.arange(0.1,10),

           'solver' : ['liblinear']}

logReg=LogisticRegression()







logReg_cv=GridSearchCV(logReg,param_grid,cv=5)



logReg_cv.fit(X_train, y_train)

y_pred=logReg_cv.predict(X_test)



logReg_score=logReg_cv.score(X_test, y_test)
logReg_cv.best_params_, logReg_cv.best_score_, logReg_cv.best_estimator_
#Random Forest Classifier

param_grid = [

    {

    'n_estimators' : list(range(10,101,10)),

    'max_features': list(range(6,8,2)),

    'max_depth'    : [2, 3, 5, 10], 

    } 

]



RF=RandomForestClassifier()

RF_cv=GridSearchCV(RF, param_grid, cv=5)

RF_cv.fit(X_train, y_train)

y_pred=RF_cv.predict(X_test)



RF_score=RF_cv.score(X_test, y_test)

RF_score, RF_cv.best_score_, RF_cv.best_estimator_, RF_cv.best_params_,
#Support Vector Machine Classifier





param_grid = {'C': [0.1,1,10, 100], 

              'gamma': [1,0.1,0.01,0.001]}



svm_clf=svm.SVC()

svm_clf_cv=GridSearchCV(svm_clf,param_grid)



svm_clf_cv.fit(X_train, y_train)

y_pred=svm_clf_cv.predict(X_test)



svm_score=svm_clf_cv.score(X_test, y_test)
svm_clf_cv.score(X_test, y_test), svm_clf_cv.best_params_
#Gradient Boosting Classifier 





param_grid=[{'n_estimators':list(range(20,100,10))}]

gb=GradientBoostingClassifier()



gb_cv=GridSearchCV(gb,param_grid)

gb_cv.fit(X_train, y_train)

y_pred=gb_cv.predict(X_test)

gb_cv_score=gb_cv.score(X_test, y_test)



gb_cv_score,gb_cv.best_params_, gb_cv.best_score_
#AdaBoost Classifier



from sklearn.ensemble import AdaBoostClassifier



param_grid=[{'n_estimators':list(range(20,100,10))}]





ab=AdaBoostClassifier()

ab_cv=GridSearchCV(ab,param_grid)

ab_cv.fit(X_train, y_train)

y_pred=ab_cv.predict(X_test)



ab_cv.best_score_, ab_cv.best_estimator_
#SVM classifier with scaled values







steps=[('scalar', StandardScaler()),

      ('svm', svm.SVC())]



parameters={'svm__kernel':['rbf', 'linear'],

           'svm__C':[1,10,100,1000],

           'svm__gamma':[1e-3,1e-4]}



pipeline=Pipeline(steps)



svm_tuned_cv=GridSearchCV(pipeline, param_grid=parameters)



svm_tuned_cv.fit(X_train, y_train)

y_pred=svm_tuned_cv.predict(X_test)

svm_tuned_score=svm_tuned_cv.score(X_test,y_test)

svm_tuned_score, svm_tuned_cv.best_score_, svm_tuned_cv.best_params_,
#Ensemble VotingClassifer - Not sure what this will do but trying here as it is supposedly better option :) 



#vclass=VotingClassifier(estimators=[('rf',RF_cv), ('ab',ab_cv)], voting='hard') #0.805

#vclass=VotingClassifier(estimators=[('knn',knn_cv), ('ab',ab_cv), ('rf',RF_cv)], voting='soft', weights=[4,1,5])

#vclass=VotingClassifier(estimators=[('gb',gb_cv),('rf',RF_cv), ], voting='soft', weights=[1,2,1])







#Several combinations between estimators were chosen and ran with voting classifier, this combination gave best result

vclass=VotingClassifier(estimators=[('rf',RF_cv), ('gb',gb_cv)], voting='hard') #.808

vclass.fit(X_train,y_train)

y_pred=vclass.predict(X_test)
vclass.score(X_test, y_test)
#ROC Curves

# I am not including votingclassifier results here as they are clearly not greater than RF, or GB or AB



knn_disp=plot_roc_curve(knn_cv, X_test, y_test, label="KNN")

logreg_disp=plot_roc_curve(logReg_cv, X_test, y_test, ax=knn_disp.ax_, label="LogReg")

rfc_disp= plot_roc_curve(RF_cv, X_test, y_test, ax=knn_disp.ax_, label="RandomForest")

svm_disp= plot_roc_curve(svm_clf_cv, X_test, y_test, ax=knn_disp.ax_, label="SVM")

gb_disp=plot_roc_curve(gb_cv, X_test, y_test,ax=knn_disp.ax_, label='GradientBoost')

ab_disp=plot_roc_curve(ab_cv, X_test, y_test,ax=knn_disp.ax_, label='AdaBoost')

svm_tuned_disp=plot_roc_curve(svm_tuned_cv, X_test, y_test,ax=knn_disp.ax_, label='SVM - ScaledValues')





svm_tuned_disp.figure_.suptitle("ROC curve comparison")
#Accuracy Scores



my_dict={'KNN':'{:.2f}'.format(knn_cv.best_score_*100), 

         'LogReg':'{:.2f}'.format(logReg_cv.best_score_*100),

         'RandomForest': '{:.2f}'.format(RF_cv.best_score_*100), 

         'SVM':'{:.2f}'.format(svm_clf_cv.best_score_*100), 

         'Gradient Boosting': '{:.2f}'.format(gb_cv.best_score_*100),

         'AdaBoost': '{:.2f}'.format(ab_cv.best_score_*100),

         'SVM - Scaled': '{:.2f}'.format(svm_tuned_cv.best_score_*100)

         

        }

score_df=pd.DataFrame(list(my_dict.items()),

                      columns=['Model','Best_Score'])



score_df.sort_values(by='Best_Score', ascending=False)
#Feature Importances of RF, AB and GB. 



fig, axes=plt.subplots(3,1, figsize=(8,7))

feature_importance(RF_cv.best_estimator_, axes[0], 'RF_cv')

feature_importance(ab_cv.best_estimator_, axes[1], 'AB_cv')

feature_importance(gb_cv.best_estimator_, axes[2], 'GB_cv')

plt.tight_layout()









#Predictor 'Sex' has most importance in RF and GB. AB gives most importance to Fare (perhaps bias towards continous variable?)
#Decided to use the best scored classifier i.e. Random Forest. (RF_cv)



y_pred_RF=RF_cv.predict(X_hold)





Passengerid = test_df['PassengerId']



final = pd.DataFrame(y_pred_RF,columns=['Survived'])



submit = pd.concat([Passengerid, final], axis=1, sort=False)

submit.to_csv('sub1.csv', header=True, index=False)

submit

#Second Submission with GB



y_pred_GB=gb_cv.predict(X_hold)





Passengerid = test_df['PassengerId']



final2 = pd.DataFrame(y_pred_GB,columns=['Survived'])



submit2 = pd.concat([Passengerid, final2], axis=1, sort=False)

submit2.to_csv('sub2.csv', header=True, index=False)



submit2