#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as r
#set pandas warning off
pd.set_option('mode.chained_assignment', None)
#function to generate random colors
def randomColor(n):
    color = []
    colorArr = ['00','11','22','33','44','55','66','77','88','99','AA','BB','CC','DD','EE','FF']
    for _ in range(n):
        color.append('#' + colorArr[r.randint(0,15)] + colorArr[r.randint(0,15)] + colorArr[r.randint(0,15)])
    return color
#get datasets
traindf = pd.read_csv('/kaggle/input/titanic/train.csv')
display(traindf.head())
print(traindf.shape)
print(traindf.info())
display(traindf.describe())
testdf = pd.read_csv('/kaggle/input/titanic/test.csv')
display(testdf.head())
print(testdf.shape)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (18,14))

#Gender vs Survuval Ratio
gender_survived = traindf[['Sex', 'Survived']].groupby('Sex').mean()
gender_survived.plot(kind='barh', 
                     color = [randomColor(len(gender_survived))], 
                     ax = axes[0,0],
                     title = 'Gender vs Survival Ratio',
                     xlim = (0,1),
                     grid = True,)

#Pclass vs Survival Ratio
pclass_survived = traindf[['Pclass', 'Survived']].groupby('Pclass').mean()
pclass_survived.plot(kind='bar', 
                     color = [randomColor(len(pclass_survived))], 
                     ax=axes[0,1],
                     title = 'Pclass vs Survival Ratio',
                     ylim = (0,1),
                     grid = True,)

#Embarked vs Survival Ratio
embarked_survived = traindf[['Embarked', 'Survived']].groupby('Embarked').mean()
embarked_survived.plot(kind='bar', 
                       color = [randomColor(len(embarked_survived))], 
                       ax=axes[0,2],
                       title = 'Embarked vs Survival Ratio',
                       ylim = (0,1),
                       grid = True,)

#Parch vs Survival Ratio
parch_survived = traindf[['Parch', 'Survived']].groupby('Parch').mean()
parch_survived.plot(kind='bar', 
                    color = [randomColor(len(parch_survived))], 
                    ax=axes[1,0], 
                    title = 'Parch vs Survival Ratio',
                    ylim = (0,1),
                    grid = True,)

#SibSp vs Survival Ratio
sibsp_survived = traindf[['SibSp', 'Survived']].groupby('SibSp').mean()
sibsp_survived.plot(kind='bar', 
                    color = [randomColor(len(sibsp_survived))], 
                    ax=axes[1,1],
                    title = 'SibSp vs Survival Ratio',
                    ylim = (0,1),
                    grid = True,)

#Age Group (without missing values) vs Survival Ratio
agegrp = {'child':(0,13), 'teen':(13,20), 'young_adult':(20,35), 'middle_adult':(35,45), 'old_adult':(45,60), 'senior_citizen':(60,100)}
age_survival = traindf[['Age','Survived']].dropna().reset_index(drop=True)
age_survival['age_grp'] = None

for i in range(len(age_survival)):
    for grp in agegrp:
        temp = agegrp[grp]
        if age_survival.loc[i,'Age'] in range(temp[0],temp[1]):
            age_survival.loc[i,'age_grp'] = grp
            break
            
age_survivalgrouped = age_survival.drop(columns=['Age']).groupby('age_grp').mean()

age_survivalgrouped.plot(kind= 'barh', 
                         color = [randomColor(6)], 
                         legend=False, ax=axes[1,2],
                         title = 'Age group vs Survival Ratio',
                         xlim = (0,1),
                         grid = True,)

#Fare (without missing values) vs survival Ratio
faregrp = [(i,50*i,50*(i+1)) for i in range(20)]
fare_survival = traindf[['Fare','Survived']].dropna().reset_index(drop=True)
fare_survival['fare_group'] = None

for i in range(len(fare_survival)):
    for grp in faregrp:
        if fare_survival.loc[i,'Fare'] >= grp[1] and fare_survival.loc[i,'Fare'] < grp[2]:
            fare_survival.loc[i,'fare_group'] = grp[0]
            break

fare_survivalgrouped = fare_survival.drop(columns=['Fare']).groupby('fare_group').mean()

fare_survivalgrouped.plot(kind= 'barh',
                          color = [randomColor(len(fare_survival))],
                          legend = False, ax=axes[2,0],
                          title = 'Fare group vs Survival Ratio',
                          xlim = (0,1),
                          grid = True,)

fig.tight_layout()
fig.show()
#managing missing values
traindf2 = traindf[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
missingdf = traindf2.transpose()
missingdf['missing values'] = missingdf.apply(lambda x: len(traindf)-x.count(), axis=1)
missingdf = missingdf[['missing values']]
missingdf
traindf2.drop(columns=['Age'], inplace=True)
traindf2['Embarked'].fillna(traindf2['Embarked'].mode()[0], inplace=True)
display(traindf2.head())
#selecting features
y = traindf2["Survived"]

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
X = pd.concat([pd.get_dummies(traindf2[features[0]]),pd.get_dummies(traindf2[features[1:]])], axis = 1, sort = False)
display(X.head())
print(X.shape)
#importing ML packages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#creating model
params_dict = {'criterion':['gini','entropy'], 
               'splitter':['random'], 
               'max_depth':[3,4,5,6,7,8]}

model = GridSearchCV(estimator=DecisionTreeClassifier(),
                     param_grid=params_dict,
                     scoring='accuracy', 
                     cv=6, 
                     verbose = 0, 
                     refit=True)

hist = model.fit(X,y)
print('Best Parameters : ',model.best_params_)
print('Best Score : ', model.best_score_)

resdf = pd.DataFrame(model.cv_results_).drop(columns=['params','mean_fit_time','std_fit_time','mean_score_time','std_score_time'])
display(resdf)
print('Missing Values in test data : ',len(testdf[features])-len(testdf[features].dropna()))
#transforming test dataset
Xfinaltest = pd.concat([pd.get_dummies(testdf[features[0]]),pd.get_dummies(testdf[features[1:]])], axis = 1, sort = False)
display(Xfinaltest.head())
#make prediction on test dataset
Ypred = model.predict(Xfinaltest)
#save the results
resultdf = pd.DataFrame({'PassengerId': testdf['PassengerId'], 'Survived': Ypred})
display(resultdf.head())
resultdf.to_csv('my_submission2.csv', index=False)
print("Your submission was successfully saved!")