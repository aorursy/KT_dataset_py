import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn import svm





# Import training data and test data from the location

df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
df_train.head(5) 

# .head(n) shows  first n rows of the data frame, df_train. 
df_train.info()
# Figures inline and set visualization style

% matplotlib inline

import seaborn as sns

sns.set()

sns.countplot(x='Survived', data=df_train);
sns.countplot(x='Sex', data=df_train);
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);
df_train.groupby(['Sex']).Survived.sum()
# count total passangers and groupby sex, total males and total female passangers in data set

df_train.groupby(['Sex']).count()
print('Total no of female survived:',df_train[df_train.Sex == 'female'].Survived.sum())

print('Total no of female passangers:',df_train[df_train.Sex == 'female'].Survived.count())

print('Percentage of female survived',df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female']

      .Survived.count())



print('Total no of male survived:',df_train[df_train.Sex == 'male'].Survived.sum())

print('Total no of male passangers:',df_train[df_train.Sex == 'male'].Survived.count())

print('Percentage of male survived:',df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].

      Survived.count())
# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the 

#feature 'Pclass'

sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);

# Use seaborn to plot a histogram of the 'Age' column of df_train. You'll need to drop null values before doing so

df_train_drop = df_train.dropna()

sns.distplot(df_train_drop.Age, kde=False);
df_train_drop = df_train.dropna()

sns.distplot(df_train_drop.Age, kde=True);
# Plot a strip plot & a swarm plot of 'Fare' with 'Survived' on the x-axis

sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.5, jitter=True);
sns.swarmplot(x='Survived', y='Fare', data=df_train);
# Use seaborn to plot a scatter plot of 'Age' against 'Fare', colored by 'Survived'

sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.7});
df_train.describe()
df_train.median()
df_train['Age'] = df_train.Age.fillna(df_train.Age.median())

df_train['Fare'] = df_train.Fare.fillna(df_train.Fare.median())



# Check out info of data

df_train.info()
df_train = pd.get_dummies(df_train, columns=['Sex'], drop_first=True)
df_train.head()
df_train = pd.get_dummies(df_train, columns=['Embarked'], drop_first=False)
df_train.head(5)
# Select columns and view head, those columns will be the features for machine learning algorithm

training_vectors = df_train[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp','Embarked_C','Embarked_Q',

                             'Embarked_S']]

training_vectors.head()
target_vectors=df_train['Survived']
target_vectors.head()
# Support Vector Classifier

classifier_svm= svm.SVC()

classifier_svm.fit(training_vectors,target_vectors)
# Again edit NaN of Numerial variables of test data as well, fill Nan with their median value, it 

#introduces less error in system

df_test['Age'] = df_test.Age.fillna(df_test.Age.median())

df_test['Fare'] = df_test.Fare.fillna(df_test.Fare.median())

#fill dummies in 'Sex' and 'Embarked' columnts with method get_dummies()

df_test = pd.get_dummies(df_test, columns=['Sex'], drop_first=True)

df_test = pd.get_dummies(df_test, columns=['Embarked'], drop_first=False)
df_test_test= df_train[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp','Embarked_C','Embarked_Q',

                             'Embarked_S']]
df_test_test.head()
# take 1st row of the df_test_test and pass it to our classifier_svm to predict the survival of that 

#passanger

#df_test_test.iloc[0] -> first row extracted

print('Survived:',classifier_svm.predict([df_test_test.iloc[0]]))
# predict the survival of 101 th row of df_test_test

print('Survived:',classifier_svm.predict([df_test_test.iloc[100]]))
#we can also calculate the accuracy of your classifier model. 

classifier_svm.score(training_vectors,target_vectors)
# Lets use another ML algorithm to build our classifier model. RandomForest !!

from sklearn.ensemble import RandomForestClassifier



classifier_randomforest = RandomForestClassifier(n_estimators=100)

classifier_randomforest.fit(training_vectors,target_vectors)
#we can also calculate the accuracy of our second classifier model,classifier_randomforest 

classifier_randomforest.score(training_vectors,target_vectors)