import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats





from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.svm import SVC

from sklearn import metrics



dftrain = pd.read_csv('../input/train.csv')

dftest = pd.read_csv('../input/test.csv')
# this is for checking null values for datasets

def checknull(x):

    return x.apply(lambda x: sum(x.isnull()))



# this is to generate random int numbers to fill missing values

def genRandom(x, columns):

    mean = x[columns].mean()

    std = x[columns].std()

    null_count =x[columns].isnull().sum()

    return np.random.randint(mean-std, mean+std, size = null_count)



# this is a function to apply data, predictors, target_column on a picked model.

def classification(model, data, predictors, outcome):

    print('Training below model with features stated: \n\r %s' % model)

    error = []

    model.fit(data[predictors], data[outcome])

    predictions = model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions, data[outcome])

    print('Accuracy is %s' % '{0:.3%}'.format(accuracy))

    

    # KFOld the train set

    

    kf = KFold(data.shape[0], n_folds=5)

    

    # cross-validation on the model

    for train, test in kf:

        train_predictors = data[predictors].iloc[train,:]

        train_target = data[outcome].iloc[train]

        

        model.fit(train_predictors, train_target)

        

        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

    print('Cross-Validation Score: %s \n\r\n\r' % '{0:.3%}'.format(np.mean(error)))

    
dftrain.head(10)
checknull(dftrain)
checknull(dftest)
# drop unnecessary data.

dftrain = dftrain.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

dftest = dftest.drop(['PassengerId','Name', 'Ticket','Cabin'],axis=1)
# apply one hot encoding on Embarked.

dftrain= pd.concat([dftrain, pd.get_dummies(dftrain.Embarked)], axis=1)
# Majority of paasenger embarked from S so we just assume these missing values are 'S' as well. 

dftrain.Embarked.fillna('S', inplace =True)
# Combine Siblin and parents since they are all considered families. 

dftrain['Family'] = dftrain['Parch'] + dftrain['SibSp']

# if the individual does not have family on board, it is 0, if he/she does, it is 1

dftrain.loc[dftrain.Family>0, 'Family'] = 1 
sns.factorplot(x='Sex', hue='Embarked', y='Survived', data=dftrain, col='Pclass')

sns.factorplot('Pclass', 'Survived', order=[1,2,3], data=dftrain, size=3)
# take a closer look at embarked column

fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(20,5))



sns.countplot(x='Embarked', data=dftrain, ax=axis1)

axis1.set_title('Total count by embarked')



sns.countplot(x='Survived', hue='Embarked', data=dftrain,ax=axis2)

axis2.set_title('Total count of survival by embarked')



embarkedpec=dftrain[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data =embarkedpec, order=['S','C','Q'],ax=axis3)

axis3.set_title('Suvival rate by embarked')
# There are only very few missing values so we just fill them with median.

dftest.Fare.fillna(np.median(dftest.Fare), inplace=True)



# devide fare column into two categories

fare_survived = dftrain['Fare'][dftrain.Survived==1]

fare_nsurvived = dftrain['Fare'][dftrain.Survived==0]



dftrain.boxplot(column='Fare',by='Survived', sym='k.')
dftest.boxplot(column='Fare', sym='k.')
# drop the outliers

dftrain = dftrain.loc[dftrain.Fare <500]

dftest = dftest.loc[dftest.Fare <500]





fig2, (axis5, axis6) = plt.subplots(1,2,figsize=(15,5))



pd.DataFrame.hist(data=dftrain, column='Age',ax=axis5,bins=70)

dftrain.loc[np.isnan(dftrain.Age), 'Age'] = genRandom(dftrain, 'Age')

pd.DataFrame.hist(data=dftrain, column='Age',ax=axis6,bins=70)



dftest.loc[np.isnan(dftest.Age), 'Age'] = genRandom(dftest, 'Age')
Agerange = pd.cut(dftrain.Age,[0,16,25,60,85])

dftrain['AgeRange'] = Agerange



fig3, axis7 = plt.subplots(1,1,figsize=(5,4))

average_age = dftrain[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean()

sns.barplot(x='AgeRange',y='Survived', data=average_age, ax=axis7)

dftrain.AgeRange.value_counts()



axis7.set_ylabel('Survival Rate')
familypec = dftrain[['Family','Survived']].groupby(['Family'], as_index=False).mean()

parentpec = dftrain[['Parch','Survived']].groupby(['Parch'], as_index=False).mean()

sibpec = dftrain[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean()



fig5, (axis12, axis13, axis14) = plt.subplots(1,3, figsize = (10,5))

sns.barplot(x='Family', y ='Survived',data =familypec, order=[1,0], ax =axis12)

sns.barplot(x='Parch', y ='Survived',data =parentpec, ax =axis13)

sns.barplot(x='SibSp', y ='Survived',data =sibpec, ax =axis14)
def get_person(passenger):

    age, sex = passenger

    return 'child' if age<16 else sex



dftrain['Person'] = dftrain[['Age', 'Sex']].apply(get_person, axis=1)

dftest['Person'] = dftest[['Age', 'Sex']].apply(get_person, axis=1)



person_dummies_titanic  = pd.get_dummies(dftrain['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']



person_dummies_test  = pd.get_dummies(dftest['Person'])

person_dummies_test.columns = ['Child','Female','Male']



dftrain = dftrain.join(person_dummies_titanic)

dftest = dftest.join(person_dummies_test)
fig4, ((axis8,axis9), (axis10, axis11)) = plt.subplots(2,2,figsize=(10,10))

axis9.set_ylim(0,1)

axis10.set_ylim(0,1)

axis11.set_ylim(0,1)



sns.countplot(x='Person', data=dftrain, ax=axis8)

axis8.set_title('Total Count by Gender')

person_perc = dftrain[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis9, order=['male','female','child'])

axis9.set_title('Total Survival rate by Gender')



high_person_perc = dftrain.loc[dftrain.Pclass<3,['Person','Survived'] ].groupby(['Person'],as_index=False).mean()

low_person_perc = dftrain.loc[dftrain.Pclass==3,['Person','Survived'] ].groupby(['Person'],as_index=False).mean()



sns.barplot(x='Person', y='Survived', data=high_person_perc, ax=axis10, order=['male','female','child'])

axis10.set_title('High Class survival rate by Gender')



sns.barplot(x='Person', y='Survived', data=low_person_perc, ax=axis11, order=['male','female','child'])

axis11.set_title('Low Class survival rate by Gender')
np.corrcoef(dftrain.Pclass, dftrain.Fare)
pclass_dummies_titanic  = pd.get_dummies(dftrain['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']





pclass_dummies_test  = pd.get_dummies(dftest['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']





titanic_df = dftrain.join(pclass_dummies_titanic)

test_df    = dftest.join(pclass_dummies_test)
outcome_var = 'Survived'



train_var1 = ['Pclass', 'Age', 'Family', 'C','Q','S', 'Child','Female', 'Male']
#build four models for comparison

model1 = DecisionTreeClassifier()



model2=RandomForestClassifier()



model3=SVC()



model4=KNeighborsClassifier()
classification(model1, dftrain, train_var1,outcome_var)

classification(model2, dftrain, train_var1,outcome_var)

classification(model3, dftrain, train_var1,outcome_var)

classification(model4, dftrain, train_var1,outcome_var)
featimp = pd.Series(model2.feature_importances_, index =train_var1).sort_values(ascending=False)

featimp
train_var2=['Age','Female','Male','Pclass']
classification(model1, dftrain, train_var2, outcome_var)

classification(model2, dftrain, train_var2, outcome_var)

classification(model3, dftrain, train_var2, outcome_var)

classification(model4, dftrain, train_var2, outcome_var)
# I did not use Kneighbor Model because it has lowest score.

dftest['Model1'] = model1.predict(dftest[train_var2])

dftest['Model2'] = model2.predict(dftest[train_var2])

dftest['Model3'] = model3.predict(dftest[train_var2])
dftest.head()
dftest['Survived']=dftest.loc[:,('Model1', 'Model2','Model3')].mode(axis=1)
dftest = dftest.drop(['Model1', 'Model2','Model3'], axis=1)
dftest.head()