import pandas as pd

import numpy as np

datapath = "../input/train.csv"

df = pd.read_csv(datapath)



# print column names

print(df.columns.values)



# look at the first 5 examples + column names

df.head()


#print the data types and quantities in each column:

df.info()
# `describe` reveals some statistics about our numeric columns.

df.describe()
# the 'O' option gives us some important info about nominal columns.

df.describe(include='O')
# remove columns we won't use

df = df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)

df.head()
# first things first:  pairwise correlation of the numeric fields in df

df.corr()
# What is the sample likelihood of survival for different passenger classes

df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt



# what are the frequencies of Death and Survival given Passenger Class?

g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Pclass')
# what are the frequencies of Death and Survival given Age?

g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
df['AgeRange'] = pd.cut(df['Age'], [0, 18, 35, 50,100], labels=[1, 2, 3, 4], include_lowest=True, right=True).astype(np.float)

df.head(10)
# What is the sample likelihood of survival for different Age Ranges

df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# what are the frequencies of Death and Survival given Age range?

g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'AgeRange')
# What is the sample likelihood of survival for different AgeRange?

df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# What is the sample likelihood of survival for different SibSp?

df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# what are the frequencies of Death and Survival given SibSp?

g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'SibSp')
# What is the sample likelihood of survival for different Parch?

df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# what are the frequencies of Death and Survival given Parch?

g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Parch')
# what are the frequencies of Death and Survival given Fare?

g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Fare', bins=20)
g = sns.FacetGrid(df, col='Pclass', row='Survived')

g.map(plt.hist, 'Fare', )
# Bin the Fare into a FareClass column

df['FareClass'] = pd.cut(df['Fare'], [0, 50, 150, 275,1000], labels=[1, 2, 3, 4], include_lowest=True, right=True).astype(np.int8)

df.head()
# What is the sample likelihood of survival for different Parch?

df[['FareClass', 'Survived']].groupby(['FareClass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# What is the sample likelihood of survival for different Sex?

df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# what are the frequencies of Death and Survival given Sex and Pclass?

g = sns.FacetGrid(df, col='Pclass', row='Sex')

g.map(plt.hist, 'Survived')
# what are the frequencies of Death and Survival given Fare?

g = sns.FacetGrid(df, col='Survived', row='Sex')

g.map(plt.hist, 'Survived', bins=20)
# How does Embarkation and Sex compare to Survival?

g = sns.FacetGrid(df, col='Embarked', row='Sex')

g.map(plt.hist, 'Survived')
# How does Embarkation and FareClass compare to Survival?

g = sns.FacetGrid(df, col='FareClass', row='Embarked')

g.map(plt.hist, 'Survived')
# Extract title from name using the fact that titles end with period.

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



# lets see if we have one for every row (there are 891 rows).

df['Title'].shape
# how many different values are there?

# crosstab the result with the Sex column to see how they are realated

titles = pd.crosstab(df['Title'], df['Sex'])

print(titles.shape)

titles
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df.head()
# What is the sample likelihood of survival for different Family Sizes?

df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import numpy as np

df['Gender'] = df['Sex'].map({'male':0, 'female':1}).astype(np.uint8)

df['Depart'] = df['Embarked'].map({'S':1, 'C':2, 'Q':3}, na_action='ignore').astype(np.float)

titleArr = df['Title'].unique()

mapping = {v: k for k, v in dict(enumerate(titleArr)).items()}

df['NamePrefix'] = df['Title'].map(mapping, na_action='ignore').astype(np.uint8)

df.head(10)
df1 = df.drop(['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title'], axis=1)

df1.head(10)
df1.info()
import numpy as np

# code to support knn imputation

def euclidean_distance(a, b):

    c = a-b

    return np.sqrt(c.dot(c))



# test

row1 = df1.iloc[0]

row2 = df1.iloc[1]

print(euclidean_distance(row1, row2))

print(euclidean_distance(row1, row1))



# test

# apply euclidean distance to every pair of rows

# return the (dis)similarity matrix with the distances between each pair of rows

def distance_matrix(df, distance_measure = euclidean_distance):

    x = df1.drop(['Survived'], axis=1).fillna(0).values

    print(x.shape)

    m = np.matrix(np.zeros(shape=(x.shape[0],x.shape[0])))

    for i in range(x.shape[0]):

        for j in range(i, x.shape[0]):

            m[i,j] = euclidean_distance(x[i], x[j])

    return m + m.T



# m is the (dis)similarity matrix with the distances between each pair of rows

m = distance_matrix(df1)

#print(m[:10,:10])



# remove all indexes that are nan on the field we want

AgeRange_nan_indexes = df1[df1['AgeRange'].isnull()].index.values

#print(AgeRange_nan_indexes)

m1 = np.delete(m, AgeRange_nan_indexes, axis=0)

print(m1.shape)

#print(np.sort(m1[:,5])[:5,0])



def get_k_nn_indexes(m, df_row, k):

    idxs = np.argsort(m[:,df_row], axis = 0)

    return idxs[:k,0]



# test

#print('get_k_nn_indexes(m1, 5, 9)')

#print(get_k_nn_indexes(m1, 5, 9))



def get_k_nn_values(df, idxs, col):

    icol = df.columns.get_loc(col)

    return df.values[idxs, icol]



# test

idxs = get_k_nn_indexes(m1, 5, 15)

#print('get_k_nn_values(df1, idxs, "AgeRange")')

#print(get_k_nn_values(df1, idxs, 'AgeRange'))



# select the most common value for the missing data among the top k samples

def select_best(knns):

    dfknn = pd.DataFrame(knns, columns=['values'])

    return dfknn['values'].value_counts().index[0]  # pick the top count of knns



# test

knns = get_k_nn_values(df1, idxs, 'AgeRange')

#print('select_best(knns)')

#print(select_best(knns))



# replace nan values in a dataframe

def replace_value(df, col, indexes, values):

    d = dict(zip(indexes, values))

    return df[col].fillna(d)



# test

# impute nans for a column

values = []

for idx in AgeRange_nan_indexes:

    impute_idxs = get_k_nn_indexes(m1, idx, 7)

    knns = get_k_nn_values(df1, impute_idxs, 'AgeRange')

    best = select_best(knns)

    values.append(best)

    

#print(values)

df2 = df1

df2['AgeRange'] = replace_value(df1, 'AgeRange', AgeRange_nan_indexes, values)

df2.info()

#df2[df2.isnull()].shape

df2



def impute_knn(m, df, column, nan_indexes, k):

    values = []

    for idx in nan_indexes:

        impute_idxs = get_k_nn_indexes(m, idx, k)

        knns = get_k_nn_values(df, impute_idxs, column)

        best = select_best(knns)

        values.append(best)

    return values



#test

Depart_nan_indexes = df1[df1['Depart'].isnull()].index.values

v = impute_knn(m1, df1, 'Depart', Depart_nan_indexes ,11)

df2['Depart'] = replace_value(df1, 'Depart', Depart_nan_indexes, v)

df2.info()
# To find the knn of a row, pick the column of m corresponding to a row with a missing value, 

#  sort the values of that vector ascending, then pick the top k values from the list.

#  Note that if the row index < k, you should pick k+1 rows and throw out the top value

#  since it will correspond to the row being selected.  Next pick the indexes of the top k 

#  values and get those rows from the dataframe.  Use the values in the columns in 

#  question to impute the missing values.

def impute_nans(df, cols, k=9):

    m = distance_matrix(df)

    for col in cols:

        # get the indexes to rows with nan entries in this column

        nan_indexes = df[df[col].isnull()].index.values

        #remove those rows from the m1 matrix

        m1 = np.delete(m, nan_indexes, axis=0)

        nan_values = impute_knn(m1, df, col, nan_indexes, k)

        df[col] = replace_value(df, col, nan_indexes, nan_values)

        

# test

impute_nans(df1, ['AgeRange', 'Depart'], k=11)

df1.head()
g = sns.FacetGrid(df, row='FamilySize')

g.map(plt.hist, 'Survived')
g = sns.FacetGrid(df, row='AgeRange')

g.map(plt.hist, 'Survived')
g = sns.FacetGrid(df, row='NamePrefix')

g.map(plt.hist, 'Survived')
# show a scattermatrix and a correlation matrix

df1.corr()
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

scatter_matrix(df1, figsize=(15,15), diagonal='kde')
# train and test - we will use scikit-learn

from sklearn.linear_model import LogisticRegression

from patsy import dmatrices

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score



y,X = dmatrices('Survived ~ Pclass + AgeRange + FareClass + FamilySize + Gender + Depart + NamePrefix', 

                df1, return_type="dataframe")

y = np.ravel(y)



# in this case we hold out 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)



# create a logistic regression model

model_lr = LogisticRegression()

model_lr.fit(X_train,y_train)

model_lr.score(X_test, y_test) # check accuracy against the test data

# Lets try Support Vector Machine.

from sklearn import svm



model_svc = svm.SVC(kernel='rbf', C=1)

model_svc.fit(X_train, y_train)

model_svc.score(X_test, y_test)
# Now for a Random Forest

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()

model_rf.fit(X_train, y_train)

model_rf.score(X_test, y_test)
# logistic regression

scores_lm = cross_val_score(model_lr, X, y, cv=5)

print(scores_lm)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lm.mean(), scores_lm.std() * 2))
# Support vector machine

scores_svc = cross_val_score(model_svc, X, y, cv=5)

print(scores_svc)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svc.mean(), scores_svc.std() * 2))
# Random Forest

scores_rf = cross_val_score(model_rf, X, y, cv=5)

print(scores_rf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))