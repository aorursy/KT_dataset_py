# Essentials

import pandas as pd  

import numpy as np

import math





# Visuals

import seaborn as sns



# Data Preprocessing

from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler



# Model Selection Tools

from sklearn.model_selection import KFold, train_test_split, GridSearchCV



# Models

from sklearn.svm import SVR, SVC

from sklearn.dummy import DummyClassifier



df = pd.read_csv('../input/train.csv')

RawX = df.iloc[:, 2:].values  # for train, use 2: //  for test, use 1:

y = df.iloc[:, 1].values
sns.countplot(x = df.SibSp, hue = df.Survived)
sns.countplot(x = df.Pclass, hue = df.Survived)
df = df.drop(['Ticket'], axis = 1)
df['NameLen'] = df.Name.apply(lambda x : len(x)) 

for i in range(np.shape(RawX)[0]): RawX[i,1] = RawX[i,1][RawX[i,1].find(', ')+2:RawX[i,1].find('. ')]

np.unique(RawX[:,1])  # How many unique values? Let's plot these to see how many of each title



# Clean it up a bit. No need for all these fancy titles, let's break it down

def replace_titles(x):

        title=RawX[x,1]

        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Master',]:

            return 'Mr'

        elif title in ['Countess', 'Mme','Dona']:

            return 'Mrs'

        elif title in ['Mlle', 'Ms', 'Lady']:

            return 'Miss'

        elif title in ['Dr','Master','Col','Rev','Capt','the Countess']:

            return 'Special'

        else:

            return title



for i in range(np.shape(RawX)[0]):

    RawX[i,1] = replace_titles(i)



# Let's look at the data.

plot = sns.countplot(RawX[:,1])



df['Name'] = RawX[:,1] # Insert back into our df
# Average age of Mr

mravg = RawX[np.where(RawX[:,1] == 'Mr')][:,3]

mravg = round(np.nanmean(mravg, dtype = np.dtype(np.float)))

# Average age of Mrs

mrsavg = RawX[np.where(RawX[:,1] == 'Mrs')][:,3]

mrsavg = round(np.nanmean(mrsavg, dtype = np.dtype(np.float))) 

# Average age of Miss

msavg = RawX[np.where(RawX[:,1] == 'Miss')][:,3]

msavg = round(np.nanmean(msavg, dtype = np.dtype(np.float)))

# Average of every person (for special category)

avg = RawX[np.where(RawX[:,1] == 'Special')][:,3]

avg = round(np.nanmean(avg, dtype = np.dtype(np.float)))



# Below, we will define a function that iterates through the entire age set, looking for nan and replacing

def replace_age(x):

    for i in range(np.shape(x)[0]):

        if math.isnan(x[i,3]) == True:

            if x[i,1] == 'Mr':

                x[i,3] = mravg

            if x[i,1] == 'Mrs':

                x[i,3] = mrsavg

            if x[i,1] == 'Miss':

                x[i,3] = msavg

            if x[i,1] == 'Special':

                x[i,3] = avg

                

            

replace_age(RawX)

df['Age'] = RawX[:,3] # Insert back into our df

df['Age'] = df.Age.astype(int)
# Now let's clean up NA's in Embarked

df['Embarked'] = df['Embarked'].fillna(list(df['Embarked'].value_counts().index)[0]) 
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



df['Alone']=0

df.loc[(df.FamilySize==1),'Alone'] = 1
sns.countplot(df.Age, hue = df.Survived)
est = KBinsDiscretizer(n_bins = 5, encode = 'ordinal', strategy = 'quantile')

est.fit(df[['Age']])  # use me for train data

new = est.transform(df[['Age']]) # Use me for train data

df['AgeBin'] = new



sns.countplot(df.AgeBin, hue = df.Survived)



df = df.drop('AgeBin', axis = 1)
ohe = OneHotEncoder()



# For the Name/Title column:

X = ohe.fit_transform(df.Name.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ["Title_"+str(int(i)) for i in range(X.shape[1])])

df = pd.concat([df, dfOneHot], axis=1)



# For embarked column

X = ohe.fit_transform(df.Embarked.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ["Embarked_"+str(int(i)) for i in range(X.shape[1])])

df = pd.concat([df, dfOneHot], axis=1)
df = df.drop('Cabin', axis = 1)

df = df.drop('Embarked', axis = 1)

df = df.drop('Name', axis = 1)

df = df.drop('Sex', axis = 1)

df = df.drop('Age', axis = 1)



RawX = df.iloc[:, 2:].values
scaler = MinMaxScaler()

Scaled_X = scaler.fit_transform(RawX)
#%% Dummy with cross validation

scores_dummy = []

clf = DummyClassifier()

cv = KFold(n_splits = 10, random_state = 42, shuffle = False)



for train_index, test_index in cv.split(Scaled_X):

    X_train, X_test, y_train, y_test = Scaled_X[train_index], Scaled_X[test_index], y[train_index], y[test_index]

    clf.fit(X_train, y_train)

    scores_dummy.append(clf.score(X_test, y_test))

print(scores_dummy)
"""

# Grid search to determine parameters

X_train, X_test, y_train, y_test = train_test_split(Scaled_X, y, test_size = 0.2, random_state = 0)



def svc_param_selection(X, y, nfolds):

    Cs = [0.001, 0.01, 0.1, 1, 10, 100]

    gammas = [0.001, 0.01, 0.1, 1, 10]

    degrees = list(range(1,9))

    param_grid = { 'kernel' : ('linear', 'rbf'), 

                  'C': Cs, 

                  'gamma' : gammas, 

                  'degree' : degrees}

    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search.best_params_



# Find best parameters

svc_param_selection(X_train, y_train, 5)

"""
#%% SVC w/ CV

scores_SVC = []

cv = KFold(n_splits = 10, random_state = 42, shuffle = False)

clf = SVC(kernel = 'rbf', C = 0.1, degree = 1, gamma = 1)



for train_index, test_index in cv.split(Scaled_X):

    X_train, X_test, y_train, y_test = Scaled_X[train_index], Scaled_X[test_index], y[train_index], y[test_index]

    clf.fit(X_train, y_train)

    scores_SVC.append(clf.score(X_test, y_test))

print(scores_SVC)
df = pd.read_csv('../input/test.csv')

RawX = df.iloc[:, 1:].values



df = df.drop(['Ticket'], axis = 1)





df['NameLen'] = df.Name.apply(lambda x : len(x)) 

    

    

for i in range(np.shape(RawX)[0]): RawX[i,1] = RawX[i,1][RawX[i,1].find(', ')+2:RawX[i,1].find('. ')]

np.unique(RawX[:,1])  # How many unique values? Let's plot these to see how many of each title



# Clean it up a bit. No need for all these fancy titles, let's break it down

def replace_titles(x):

        title=RawX[x,1]

        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Master',]:

            return 'Mr'

        elif title in ['Countess', 'Mme','Dona']:

            return 'Mrs'

        elif title in ['Mlle', 'Ms', 'Lady']:

            return 'Miss'

        elif title in ['Dr','Master','Col','Rev','Capt','the Countess']:

            return 'Special'

        else:

            return title



for i in range(np.shape(RawX)[0]):

    RawX[i,1] = replace_titles(i)



# Let's look at the data.

plot = sns.countplot(RawX[:,1])



df['Name'] = RawX[:,1] # Insert back into our df





# Average age of Mr

mravg = RawX[np.where(RawX[:,1] == 'Mr')][:,3]

mravg = round(np.nanmean(mravg, dtype = np.dtype(np.float)))

# Average age of Mrs

mrsavg = RawX[np.where(RawX[:,1] == 'Mrs')][:,3]

mrsavg = round(np.nanmean(mrsavg, dtype = np.dtype(np.float))) 

# Average age of Miss

msavg = RawX[np.where(RawX[:,1] == 'Miss')][:,3]

msavg = round(np.nanmean(msavg, dtype = np.dtype(np.float)))

# Average of every person (for special category)

avg = RawX[np.where(RawX[:,1] == 'Special')][:,3]

avg = round(np.nanmean(avg, dtype = np.dtype(np.float)))



# Below, we will define a function that iterates through the entire age set, looking for nan and replacing

def replace_age(x):

    for i in range(np.shape(x)[0]):

        if math.isnan(x[i,3]) == True:

            if x[i,1] == 'Mr':

                x[i,3] = mravg

            if x[i,1] == 'Mrs':

                x[i,3] = mrsavg

            if x[i,1] == 'Miss':

                x[i,3] = msavg

            if x[i,1] == 'Special':

                x[i,3] = avg

                

            

replace_age(RawX)

df['Age'] = RawX[:,3] # Insert back into our df

df['Age'] = df.Age.astype(int)





# Now let's clean up NA's in Embarked

df['Embarked'] = df['Embarked'].fillna(list(df['Embarked'].value_counts().index)[0])





df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



df['Alone']=0

df.loc[(df.FamilySize==1),'Alone'] = 1





ohe = OneHotEncoder()



# For the Name/Title column:

X = ohe.fit_transform(df.Name.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ["Title_"+str(int(i)) for i in range(X.shape[1])])

df = pd.concat([df, dfOneHot], axis=1)



# For embarked column

X = ohe.fit_transform(df.Embarked.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ["Embarked_"+str(int(i)) for i in range(X.shape[1])])

df = pd.concat([df, dfOneHot], axis=1)





df = df.drop('Cabin', axis = 1)

df = df.drop('Embarked', axis = 1)

df = df.drop('Name', axis = 1)

df = df.drop('Sex', axis = 1)

df = df.drop('Age', axis = 1)

df = df.fillna(0)

RawX = df.iloc[:, 1:].values



# Scale based on train data

Scaled_X = scaler.transform(RawX)







y_pred = clf.predict(Scaled_X)
new = pd.DataFrame({'Survived':y_pred[:]})

output = pd.concat([df['PassengerId'],new], axis=1)



output.to_csv('SVC_Unbinned_HW.csv',index = False)