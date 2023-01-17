%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import sklearn

sns.set_style('whitegrid')
titanic_df = pd.read_csv('../input/train.csv')

print(titanic_df.info())

titanic_df.head()
# Plotting by Pclass

def plot_bar(varname):

    fig,(ax1, ax2)=plt.subplots(1,2, figsize=(15,5))

    group_mean = titanic_df.groupby(varname, as_index=False).mean()

    sns.barplot(group_mean[varname], group_mean.Survived, ax=ax1)

    #group_count = titanic_df.groupby(varname, as_index=False).count()

    sns.countplot(titanic_df[varname], ax=ax2)

    

def plot_prob(varname):

    fig= plt.figure()

    sns.boxplot(titanic_df.Survived, titanic_df[varname])
# Passenger Class - Pclass

plot_bar('Pclass')
#varnames=['Pclass', 'Sex',  'SibSp', 'Parch','Embarked'] 

plot_bar('Sex')

# Females much more likely to survive than males
plot_bar('SibSp')

# If siblings are present your chances of survival were higher

#titanic_df[titanic_df['SibSp']>=3]

titanic_df['Siblings_present']=1*(titanic_df['SibSp']>0)

plot_bar('Siblings_present')
plot_bar('Parch')

# Parent or Child Present

# If parents/children are present your chances of survival were higher

#titanic_df[titanic_df['SibSp']>=3]

titanic_df['Parch_present']=1*(titanic_df['Parch']>0)

plot_bar('Parch_present')
plot_bar('Embarked')

sns.factorplot(x='Embarked', y='Survived', hue='Sex', kind='bar', data=titanic_df)
facet = sns.FacetGrid(titanic_df, hue="Survived", aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, 100))

facet.add_legend()
import re

titanic_df.Name

def title_map(name):

    if re.findall(r'[a-zA-Z]+\.', name) is not []:

        title=re.findall(r'[a-zA-Z]+\.', name)[0][:-1]

        return title if title in ['Mr', 'Mrs', 'Miss', 'Master' ] else 'Other'

    else:

        return 'Other'



titanic_df['titles']=titanic_df.Name.map(title_map)

sns.countplot(titanic_df['titles'])

sns.factorplot(x='titles', y='Age', data=titanic_df, kind='bar', aspect=4)

sns.factorplot(x='titles', y='Survived', data=titanic_df, kind='bar', aspect=4)

#titanic_df[titanic_df.titles=='Lady']
plot_prob('Fare')

# This is probably a predictor.
#titanic_df
#Compare survival odds for persons with missing age

print('For age missing mean survived = ', titanic_df[titanic_df.Age.isnull()].Survived.mean())

print('For age present mean survived = ', titanic_df[titanic_df.Age.notnull()].Survived.mean())

titanic_df['Age_present']=1

titanic_df.ix[titanic_df.Age.isnull(),'Age_present']=0

sns.factorplot(x='titles', y='Survived', hue='Age_present', data=titanic_df, kind='bar', aspect=4)
def compare_missing(varname):

    sns.factorplot(x=varname, y='Survived', hue='Age_present', data=titanic_df, kind='bar')

varnames=['Pclass', 'Sex',  'SibSp', 'Parch','Embarked'] 

for var in varnames:

    compare_missing(var)
# Cleaning up

# Fill in missing values for age with median age

def data_cleanup(data):

    data.Embarked=data.Embarked.fillna('S')



    #Convert Embarked categorical variable into a dummy variable.

    # http://stackoverflow.com/questions/24715230/random-forest-with-categorical-features-in-sklearn

    data = data.join(pd.get_dummies(data.Embarked).applymap(lambda x: int(x)))

    

    #Convert Sex to a number

    data.Sex=data.Sex.map(lambda x: 1 if x=='female' else 0)

    return data



titanic_df = data_cleanup(titanic_df)

titanic_df.head()
# If age not present, then choose randomly with replacement from persons with the same title.

age_bootstrap = lambda x : np.random.choice(titanic_df[(titanic_df.titles == x) & (titanic_df.Age_present==1)].Age)

titanic_df.ix[titanic_df.Age_present==0, 'Age']=titanic_df.ix[titanic_df.Age_present==0, 'titles'].map(age_bootstrap)

    

sns.boxplot(titanic_df['Age_present'], titanic_df['Age'])
predictors=['Pclass', 'Sex', 'Age', 'Siblings_present', 'Parch_present','Fare','C','Q']

# Dropped 'S' because it is correlated to 'C' and 'Q'

out=['Survived']
def predict_accuracy(_test, prediction):

    return (_test==prediction).mean()
from sklearn.metrics import confusion_matrix

train,test =  sklearn.cross_validation.train_test_split(titanic_df, train_size= 0.6)

print (train.shape, test.shape)

prediction=titanic_df['Sex']

print('Prediction based on Sex - Accuracy', predict_accuracy(titanic_df.Survived, prediction))

print (confusion_matrix(titanic_df.Survived, prediction))
from sklearn.linear_model import LogisticRegression





for _c in [1e-3, 1e-2, 0.1, 1, 10, 100]:

    train,test =  sklearn.cross_validation.train_test_split(titanic_df, train_size= 0.9)

    xtrain=train[predictors]

    ytrain=np.ravel(train[out])

    xtest=test[predictors]

    ytest=np.ravel(test[out])

    clr = LogisticRegression().set_params(C=_c)

    clr.fit(xtrain, ytrain)

    #y_pred=clr.predict(xtest)

    training_accuracy = clr.score(xtrain, ytrain)

    test_accuracy = clr.score(xtest, ytest)

    print ("############# Logistic Regression","C:",_c," ############")

    print ("Accuracy on training data: %0.2f" % (training_accuracy))

    print ("Accuracy on test data:     %0.2f" % (test_accuracy))

    print (confusion_matrix(ytest, clr.predict(xtest)))

    print ("########################################################")
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

# use a full grid over all parameters

clf = RandomForestClassifier(n_estimators=20)

param_grid = {"max_depth": [3, None],

              "max_features": [2, 2, 8],

              "min_samples_split": [2, 2, 8],

              "min_samples_leaf": [2, 2, 8],

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}



# run grid search

grid_search = GridSearchCV(clf, param_grid=param_grid)
training_accuracy=test_accuracy=dataset_accuracy=0

for i in range(10):

    train,test =  sklearn.cross_validation.train_test_split(titanic_df, train_size= 0.9)

    xtrain=train[predictors]

    ytrain=np.ravel(train[out])

    xtest=test[predictors]

    ytest=np.ravel(test[out])

    grid_search.fit(xtrain, ytrain)

    training_accuracy+=predict_accuracy(ytrain, grid_search.predict(xtrain))/10.

    test_accuracy += predict_accuracy(ytest, grid_search.predict(xtest))/10.

    dataset_accuracy += predict_accuracy(np.ravel(titanic_df[out]), grid_search.predict(titanic_df[predictors]))/10.

    

print ("############# Random Forest  ################")

print ("Accuracy on training data: %0.2f" % (training_accuracy))

print ("Accuracy on test data:     %0.2f" % (test_accuracy))

print ("Accuracy on all data:     %0.2f" % (dataset_accuracy))

print (confusion_matrix(ytest, grid_search.predict(xtest)))

print ("########################################################")
# Finally fit on all training data

grid_search.fit(titanic_df[predictors], np.ravel(titanic_df[out]))
# Work on test data

test_df=pd.read_csv('../input/test.csv')

test_df.head()
#final_test_cleaned[predictors]

#pd.set_option(mode.use_inf_as_null)

#xtest_final.Fare.describe()

test_df['Siblings_present']=1*(test_df['SibSp']>0)

test_df['Parch_present']=1*(test_df['Parch']>0)

test_df['titles']=test_df.Name.map(title_map)



test_df=test_df.join(pd.get_dummies(test_df.Embarked).applymap(lambda x: int(x)))

test_df.Sex=test_df.Sex.map(lambda x: 1 if x=='female' else 0)



test_df['Age_present']=1

test_df.ix[test_df.Age.isnull(),'Age_present']=0

test_df.ix[test_df.Age_present==0, 'Age']=test_df.ix[test_df.Age_present==0, 'titles'].map(age_bootstrap)

test_df.ix[test_df.Fare.isnull(),'Fare'] = test_df.Fare.median()

test_df.head()

test_df['Survived']=grid_search.predict(test_df[predictors])

print( 'Of', test_df.Survived.count(), 'passengers:')

print('Predicted Male Survivors:',test_df[(test_df.Sex==0) & (test_df.Survived==1)].Survived.sum())

print('Predicted Female Survivors:',test_df[(test_df.Sex==1) & (test_df.Survived==1)].Survived.sum())
#Submission

test_df[['PassengerId','Survived']].to_csv('titanic_prediction.csv',index=False)