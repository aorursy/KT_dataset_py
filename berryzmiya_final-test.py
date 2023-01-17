# Import all the libraries I need

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.decomposition import PCA

# ignore Deprecation Warning

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 



from sklearn.ensemble import RandomForestRegressor

#from sklearn.ensemble import RandomForestClassifier

#from xgboost import XGBClassifier

#from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import GridSearchCV



import keras 

from keras.models import Sequential # intitialize the ANN

from keras.layers import Dense      # create layers



# load the data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df = df_train.append(df_test , ignore_index = True)



# some quick inspections

df_train.shape, df_test.shape, df_train.columns.values
# check if there is any NAN

df['Pclass'].isnull().sum(axis=0)
# inspect the correlation between Pclass and Survived

df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
df.Name.head(10)
df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())



# inspect the amount of people for each title

df['Title'].value_counts()
df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 

             & (df.Title !=  'Mrs')] = 'Others'



# inspect the correlation between Title and Survived

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# inspect the amount of people for each title

df['Title'].value_counts()
df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)
# check if there is any NAN

df.Sex.isnull().sum(axis=0)
# inspect the correlation between Sex and Survived

df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
# map the two genders to 0 and 1

df.Sex = df.Sex.map({'male':0, 'female':1})
# check if there is any NAN

df.Age.isnull().sum(axis=0)
# check if there is any NAN

df.SibSp.isnull().sum(axis=0), df.Parch.isnull().sum(axis=0)
# create a new feature "Family"

df['Family'] = df['SibSp'] + df['Parch'] + 1



# inspect the correlation between Family and Survived

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
# inspect the amount of people for each Family size

df['Family'].value_counts()
df.Family = df.Family.map(lambda x: 0 if x > 4 else x)

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
df['Family'].value_counts()
# check if there is any NAN

df.Ticket.isnull().sum(axis=0)
df.Ticket.head(20)
df.Ticket = df.Ticket.map(lambda x: x[0])



# inspect the correlation between Ticket and Survived

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
# inspect the amount of people for each type of tickets

df['Ticket'].value_counts()
df[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).mean()
df[['Ticket', 'Pclass']].groupby(['Ticket'], as_index=False).mean()
# check if there is any NAN

df.Fare.isnull().sum(axis=0)

df.Ticket[df.Fare.isnull()]
df.Pclass[df.Fare.isnull()]
df.Cabin[df.Fare.isnull()]
df.Embarked[df.Fare.isnull()]
# use boxplot to visualize the distribution of Fare for each Pclass

sns.boxplot('Pclass','Fare',data=df)

plt.ylim(0, 300) # ignore one data point with Fare > 500

plt.show()
# inspect the correlation between Pclass and Fare

df[['Pclass', 'Fare']].groupby(['Pclass']).mean()
# use boxplot to visualize the distribution of Fare for each Ticket

sns.boxplot('Ticket','Fare',data=df)

plt.ylim(0, 300) # ignore one data point with Fare > 500

plt.show()
# inspect the correlation between Ticket and Fare 

# (we saw this earlier)

df[['Ticket', 'Fare']].groupby(['Ticket']).mean()
# use boxplot to visualize the distribution of Fare for each Embarked

sns.boxplot('Embarked','Fare',data=df)

plt.ylim(0, 300) # ignore one data point with Fare > 500

plt.show()
# inspect the correlation between Embarked and Fare

df[['Embarked', 'Fare']].groupby(['Embarked']).mean()
guess_Fare = df.Fare.loc[ (df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()

df.Fare.fillna(guess_Fare , inplace=True)



# inspect the mean Fare values for people who died and survived

df[['Fare', 'Survived']].groupby(['Survived'],as_index=False).mean()
# visualize the distribution of Fare for people who survived and died

grid = sns.FacetGrid(df, hue='Survived', size=4, aspect=1.5)

grid.map(plt.hist, 'Fare', alpha=.5, bins=range(0,210,10))

grid.add_legend()

plt.show()
# visualize the correlation between Fare and Survived using a scatter plot

df[['Fare', 'Survived']].groupby(['Fare'],as_index=False).mean().plot.scatter('Fare','Survived')

plt.show()
# bin Fare into five intervals with equal amount of people

df['Fare-bin'] = pd.qcut(df.Fare,5,labels=[1,2,3,4,5]).astype(int)



# inspect the correlation between Fare-bin and Survived

df[['Fare-bin', 'Survived']].groupby(['Fare-bin'], as_index=False).mean()
# check if there is any NAN

df.Cabin.isnull().sum(axis=0)
df = df.drop(labels=['Cabin'], axis=1)
# check if there is any NAN

df.Embarked.isnull().sum(axis=0)
df.describe(include=['O']) # S is the most common
# fill the NAN

df.Embarked.fillna('S' , inplace=True )
# inspect the correlation between Embarked and Survived as well as some other features

df[['Embarked', 'Survived','Pclass','Fare', 'Age', 'Sex']].groupby(['Embarked'], as_index=False).mean()
df = df.drop(labels='Embarked', axis=1)
# visualize the correlation between Title and Age

grid = sns.FacetGrid(df, col='Title', size=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# inspect the mean Age for each Title

df[['Title', 'Age']].groupby(['Title']).mean()
# visualize the correlation between Fare-bin and Age

grid = sns.FacetGrid(df, col='Fare-bin', size=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# inspect the mean Age for each Fare-bin

df[['Fare-bin', 'Age']].groupby(['Fare-bin']).mean()
# visualize the correlation between SibSp and Age

grid = sns.FacetGrid(df, col='SibSp', col_wrap=4, size=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# inspect the mean Age for each SibSp

df[['SibSp', 'Age']].groupby(['SibSp']).mean()
# visualize the correlation between Parch and Age

grid = sns.FacetGrid(df, col='Parch', col_wrap=4, size=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
# inspect the mean Age for each Parch

df[['Parch', 'Age']].groupby(['Parch']).mean()
# notice that instead of using Title, we should use its corresponding dummy variables 

df_sub = df[['Age','Master','Miss','Mr','Mrs','Others','Fare-bin','SibSp']]





X_train  = df_sub.dropna().drop('Age', axis=1)

y_train  = df['Age'].dropna()

X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)

pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



regressor = RandomForestRegressor(n_estimators = 300)

regressor.fit(X_train, y_train)

y_pred = np.round(regressor.predict(X_test),1)

df.Age.loc[df.Age.isnull()] = y_pred



df.Age.isnull().sum(axis=0) # no more NAN now
bins = [ 0, 4, 12, 18, 30, 50, 65, 100] 

age_index = (1,2,3,4,5,6,7)

#('baby','child','teenager','young','mid-age','over-50','senior')

df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)



df[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df['Ticket'].value_counts()
df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')



# check the correlation again

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
# dummy encoding

df = pd.get_dummies(df,columns=['Ticket'])
df = df.drop(labels=['SibSp','Parch','Age','Fare','Title'], axis=1)

y_train = df[0:891]['Survived'].values

X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values

X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values
# Initialising the NN

model = Sequential()



# layers

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Train the ANN

model.fit(X_train, y_train, batch_size = 32, epochs = 200)
y_pred = model.predict(X_test)

y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction-ann.csv', index=False)