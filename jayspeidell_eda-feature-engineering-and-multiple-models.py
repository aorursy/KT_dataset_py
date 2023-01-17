import pandas as pd

import numpy as np

import os

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt







os.chdir('../input')

df_train = pd.read_csv('train.csv', header=0, index_col=0, sep=',')

df_test = pd.read_csv('test.csv', header=0, index_col=0, sep=',')



df_train['train'] = 1

df_test['train'] = 0

df = pd.concat([df_train, df_test], ignore_index=False, axis=0) 



columns = df.columns



y_targets = df_train.iloc[:,0].values
# A function for checking null values in each columns



def null_percentage(column):

    df_name = column.name

    nans = np.count_nonzero(column.isnull().values)

    total = column.size

    frac = nans / total

    perc = int(frac * 100)

    print('%d%% of values or %d missing from %s column.' % (perc, nans, df_name))



def check_null(df, columns):

    for col in columns:

        null_percentage(df[col])
# Convert sex from strings to binary values



map_sex = {'male': 1, 'female': 0}

df.Sex = df.Sex.replace(map_sex)
# Create variables that identify known and unknown variables for columns 

# missing lots of data 



df['AgeKnown'] = df['Age']

df.loc[df.AgeKnown.isnull(), 'AgeKnown'] = 0

df.loc[df.AgeKnown != 0, 'AgeKnown'] = 1

df.AgeKnown = df.AgeKnown.astype('int')



df['CabinKnown'] = df['Cabin']

df.loc[df.CabinKnown.notnull(), 'CabinKnown'] = 1 

df.loc[df.CabinKnown.isnull(), 'CabinKnown'] = 0

df.CabinKnown = df.CabinKnown.astype('int')
def heatmap(df):

    plt.figure('heatmap')

    df_corr = df.drop(['train'], axis=1)

    df_corr = df_corr.corr()

    sns.heatmap(df_corr, vmax=0.6, square=True, annot=True)

    plt.yticks(rotation = 45)

    plt.xticks(rotation = 45)

    plt.show()



heatmap(df)
# Class shows a high correlation, let's look at that first 



tab = pd.crosstab(df.Survived, df.Pclass)

tab_perc = pd.crosstab(df.Survived, df.Pclass).apply(lambda r: r/r.sum(), axis=1)

print(tab)

tab_perc.plot(kind='bar', stacked='true', color=['darkgreen', 'green', 'lightgreen'], grid=False)

plt.show()

# Sex shows highest correlation with survival



tab = pd.crosstab(df_train['Sex'], df_train['Survived'])

tab_perc = pd.crosstab(df_train['Sex'], df_train['Survived']).apply(lambda r: r/r.sum(), axis=1)

print('Sex vs Survival')

print(tab)

tab.plot(kind='bar', stacked='true', color=['red','blue'], grid=False)

plt.show()



print('By percentage:')

tab_perc.plot(kind='bar', stacked='true', color=['red','blue'], grid=False)

print(tab_perc)

plt.show()

# Age shows a very weak correlation with survival, but this doesn't make sense 

# Plot histograms of ages and survival by age 



plt.figure('age distribution', figsize=(18,12))

plt.subplot(411)

ax = sns.distplot(df.Age.dropna().values, bins=range(0,81,1), kde=False, 

                  axlabel='Age')

plt.subplot(412)

sns.distplot(df[df['Survived'] == 1].Age.dropna().values, 

             bins = range(0, 81, 1), color='blue', kde=False)

sns.distplot(df[df['Survived'] == 0].Age.dropna().values, 

             bins = range(0, 81, 1), color='red', kde=False, 

             axlabel='All survivors by age')

plt.subplot(413)

sns.distplot(df[(df['Sex']==0) & (df['Survived'] == 1)].Age.dropna().values, 

                    bins = range(0, 81, 1), color='blue', kde=False)

sns.distplot(df[(df['Sex']==0) & (df['Survived'] == 0)].Age.dropna().values, 

                    bins = range(0, 81, 1), color='red', kde=False, 

                    axlabel='Female survivors by age')

plt.subplot(414)

sns.distplot(df[(df['Sex']==1) & (df['Survived'] == 1)].Age.dropna().values, 

                    bins = range(0, 81, 1), color='blue', kde=False)

sns.distplot(df[(df['Sex']==1) & (df['Survived'] == 0)].Age.dropna().values, 

                    bins = range(0, 81, 1), color='red', kde=False, 

                    axlabel='Male survivors by age')

plt.show()

# There aren't many children compared to the other passengers, but they do 

# have a higher survival rate. Let's make a columns 'Child'



df['Child'] = df['Age']

df.loc[df.Age < 10, 'Child'] = 1

df.loc[df.Age >= 10, 'Child'] = 0





# lots of nans in age, though... 

# Kids usually have the title "Master" or "Miss" and travel with parents

# It gets messy here, so I think it's too hard to predict children based 

# on other attributes



df.loc[df.Child.isnull(), 'Child'] = 0
# Let's look at families

df['Family'] = df['SibSp'] + df['Parch']

heatmap(df)



print('Survival by Family Size:')

tab = pd.crosstab(df.Survived, df.Family)

print(tab)

sns.distplot(df[df['Survived'] == 1].Family.dropna().values, 

             bins = range(-1, 11, 1), color='blue', kde=False)

sns.distplot(df[df['Survived'] == 0].Family.dropna().values, 

             bins = range(-1, 11, 1), color='red', kde=False, 

             axlabel='Survival by Family Size')

plt.show()
# Those traveling with 2-4 other family members were likely to survive

# Let's make a category for that 



df['MedFamily'] = np.nan

df.loc[(df.Family >= 2) & (df.Family <= 4), 'MedFamily'] = 1

df.loc[df.MedFamily.isnull(), 'MedFamily'] = 0

# Let's look at couples 



df['Couple'] = np.nan

df.loc[(df['Parch'] == 0) & (df['SibSp'] == 1), 'Couple'] = 1

df.loc[df.Couple.isnull(), 'Couple'] = 0



tab = pd.crosstab(df.Survived, df.Couple)

tab_perc = pd.crosstab(df.Couple, df.Survived).apply(lambda r: r/r.sum(), axis=1)

tab_perc.plot(kind='bar', stacked='true', color=['red','blue'], grid=False)

print(tab, tab_perc)

plt.show()
# Let's look at the port of departure: 



tab = pd.crosstab(df.Embarked, df.Survived)

print(tab)

tab.plot(kind='bar', stacked='true', color=['red', 'blue', 'teal'], grid=False)

plt.show()

# C has a higher chance of surviving



tab = pd.crosstab(df.Embarked, df.Pclass)

print(tab)

tab.plot(kind='bar', stacked='true', color=['red', 'blue', 'teal'], grid=False)

plt.show()
# I want to try creating some more specific features due to poor performance 

# with the previously engineered features. 



df['C1_male'] = np.nan

df['C1_female'] = np.nan

df['C2_male'] = np.nan

df['C2_female'] = np.nan

df['C3_male'] = np.nan

df['C3_female'] = np.nan



df.loc[(df.Sex == 1) & (df.Pclass == 1), 'C1_male'] = 1

df.loc[df.C1_male.isnull(), 'C1_male'] = 0

df.loc[(df.Sex == 1) & (df.Pclass == 2), 'C2_male'] = 1

df.loc[df.C2_male.isnull(), 'C2_male'] = 0

df.loc[(df.Sex == 1) & (df.Pclass == 3), 'C3_male'] = 1

df.loc[df.C3_male.isnull(), 'C3_male'] = 0



df.loc[(df.Sex == 1) & (df.Pclass == 1), 'C1_female'] = 1

df.loc[df.C1_female.isnull(), 'C1_female'] = 0

df.loc[(df.Sex == 1) & (df.Pclass == 2), 'C2_female'] = 1

df.loc[df.C2_female.isnull(), 'C2_female'] = 0

df.loc[(df.Sex == 1) & (df.Pclass == 3), 'C3_female'] = 1

df.loc[df.C3_female.isnull(), 'C3_female'] = 0
# Deal with missing values



df.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'

df['Fare'][df['Name'] == 'Storey, Mr. Thomas'] = df['Fare'][df['Pclass'] == 3].median()



check_null(df, df.columns)
# Select columns to use 

keep_cols = [#'Embarked', 

             #'Fare', 

             'Pclass', 

             'Sex',

             'AgeKnown', 

             'CabinKnown',

             'Child', 

             #'Family',

             'MedFamily', 

             #'Couple', 

             'train',

             #'C1_male', 

             #'C2_male', 

             #'C3_male', 

             #'C1_female', 

             #'C3_female', 

             #'C2_female'

             ]



df = df[keep_cols]

# Create dummies for the categorical Embarked feature

# Comment out if the feature isn't going to be used! 

#df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# split into training and test sets



df_train = df.loc[df['train'] == 1].drop(['train'], axis=1).copy()

df_test = df.loc[df['train'] == 0].drop(['train'], axis=1).copy()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train, y_targets, test_size=0.3)
# Model evaluation function 

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.metrics import log_loss



def evaluate_model(y_test, y_pred):

        print('Confusion matrix: tp, fp, fn, tn')

        print(confusion_matrix(y_test, y_pred))

        print('Cross val score')

        print(cross_val_score(classifier, X_test, y_test, cv=10).mean())

        print('Log Loss:')

        print(log_loss(y_test, y_pred))
# Kernel Support Vector Machine



from sklearn.svm import SVC

classifier = SVC()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

evaluate_model(y_test, y_pred)
# Random Forest



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

evaluate_model(y_test, y_pred)
# Kernel Support Vector Machine



from sklearn.svm import SVC

classifier = SVC()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

evaluate_model(y_test, y_pred)

# Extra Trees

from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier(n_estimators=300)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

evaluate_model(y_test, y_pred)

# Extra Trees with more estimators and bootstrap sampling

from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier(n_estimators=30, bootstrap=True)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

evaluate_model(y_test, y_pred)
# Extra Trees

from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier(n_estimators=500)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

evaluate_model(y_test, y_pred)