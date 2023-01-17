import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

import itertools as it

%matplotlib inline
mushrooms_full = pd.read_csv('../input/mushrooms.csv')

mushrooms_full.describe()
from sklearn.model_selection import train_test_split

train, test = train_test_split(mushrooms_full, test_size=0.2)
# we add in a step to capture any values we have that are not just standard letters

# using the union of the alphabet and the values in the table

allin = set(np.unique(train.values.ravel()))

letters = 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'

letters = set(letters)

additional_letters = allin - letters

letters = list(letters) + list(additional_letters)



train1 = train.copy()

for i in range(len(letters)):

    train1 = train1.replace({letters[i]: i},regex=False)



train1_corr = train1.corr()

print(train1_corr['class'])

sns.heatmap(train1_corr, square=True)
letters.reverse()

train2 = train.copy()

for i in range(26):

   train2=train2.replace({letters[i]: i}, regex=False)



train2_corr = train2.corr()

print(train2_corr['class'])

sns.heatmap(train2_corr, square=True)
train['veil-type'].unique()
train = train.drop('veil-type',1)
train1_corr = train1_corr.abs()

train2_corr = train2_corr.abs()

train1_corr.sort_values('class', ascending=0).head(5)
train3 = train.copy()

num_cols = len(train.columns)

encode_dictionary = []

for i in range(num_cols):

    train3_unique = train3.iloc[:,i].unique()

    encode_dictionary.append({'Variable Order' : list(train3_unique) })

    for j in range(len(train3_unique)):

        train3.iloc[:,i] = train3.iloc[:,i].replace({train3_unique[j]: j})
train3.describe()
train3_corr = train3.corr()

train3_corr = train3_corr.abs()

print(train3_corr.sort_values('class', ascending=0)['class'])

sns.heatmap(train3_corr, square=True)
print(train1_corr['class'].sum()) #a=0, b=1 ...

print(train2_corr['class'].sum()) #a=25, b=24, ...

print(train3_corr['class'].sum()) #each unique value assigned to 0,1,2,3, ...
train_unique = train['cap-surface'].unique()

mush_perm = list(it.permutations(train_unique))
train5 = train[['class','cap-surface']].copy()

train5.iloc[:,0] = train5.iloc[:,0].replace({'p': 1})

train5.iloc[:,0] = train5.iloc[:,0].replace({'e': 0})
corr_compare = []

for i in range(24):

    perm = list(mush_perm[i])

    for j in range(4):

        train5.iloc[:,1] = train5.iloc[:,1].replace({perm[j]: j})

    corr_compare.append({'List': perm, 'Corr': np.absolute(train5.corr().iloc[1,0])})

    train5 = train[['class','cap-surface']].copy()

    train5.iloc[:,0] = train5.iloc[:,0].replace({'p': 1})

    train5.iloc[:,0] = train5.iloc[:,0].replace({'e': 0})
pd.DataFrame(corr_compare).sort_values('Corr',ascending=0).head()
train6 = train.copy()

encoding_index = []

encoding_index.append({'Order': ['e','p'] , 'Corr to Class': 1 })

num_cols = len(train6.columns)

train6.iloc[:,0] = train6.iloc[:,0].replace({'p': 1})

train6.iloc[:,0] = train6.iloc[:,0].replace({'e': 0})

for i in range(1,num_cols):

    train6_unique = train6.iloc[:,i].unique()

    if len(train6_unique) < 3 or len(train6_unique) > 6:

        for j in range(len(train6_unique)):

            train6.iloc[:,i] = train6.iloc[:,i].replace({train6_unique[j] : j})

        encoding_index.append({'Order': list(train6_unique) , 'Corr to Class': np.absolute(train6.iloc[:,[0,i]].corr().iloc[1,0]) })

    else:

            corr_max = 0

            mush_perm = list(it.permutations(train6_unique))

            for s in range(int(np.floor(len(mush_perm)/2))):

                a = list(mush_perm[s])

                a.reverse()

                a = tuple(a)

                mush_perm.remove(a)

            for k in range(len(mush_perm)):

                train_temp = train6.iloc[:,[0,i]].copy()

                perm = list(mush_perm[k])

                for j in range(len(train6_unique)):

                    train_temp.iloc[:,1] = train_temp.iloc[:,1].replace({perm[j]: j})

                if corr_max < np.absolute(train_temp.corr().iloc[1,0]):

                    corr_max = np.absolute(train_temp.corr().iloc[1,0])

                    best_perm = perm

            encoding_index.append({'Order': list(best_perm) , 'Corr to Class': corr_max })

            for q in range(len(train6_unique)):

                train6.iloc[:,i] = train6.iloc[:,i].replace({best_perm[q] : q})   

train6.corr().abs()['class'].sum()
sorted(train1['class'].unique())
#scale so that the values in each column are between 0 and 1

t1_dividers = []

t1_dividers.append(train1.iloc[:,0].max())

for i in range(1,len(train1.columns)):

    t1_dividers.append(train1.iloc[:,i].max())

    train1.iloc[:,i] = train1.iloc[:,i]/train1.iloc[:,i].max()

train1 = train1.drop('veil-type',1) #to match 3 and 6 we remove the veil type column

train1['class'] = train1['class'].replace({min(train1['class'].unique()) : 0}, regex = False) #to make the class column zeroes and ones

train1['class'] = train1['class'].replace({max(train1['class'].unique()) : 1}, regex = False) #to make the class column zeroes and ones

t3_dividers = []

for i in range(len(train3.columns)):

    t3_dividers.append(train3.iloc[:,i].max())

    train3.iloc[:,i] = train3.iloc[:,i]/train3.iloc[:,i].max()



t6_dividers = []    

for i in range(len(train6.columns)):

    t6_dividers.append(train6.iloc[:,i].max())

    train6.iloc[:,i] = train6.iloc[:,i]/train6.iloc[:,i].max()
from sklearn import neighbors

x1 = train1.drop('class',1).values

y1 = train1['class'].values

knn1 = neighbors.KNeighborsClassifier(n_neighbors=9)

#Train the classifier

knn1.fit(x1,y1)

#Compute the prediction according to the model

knn1.score(x1,y1)
x3 = train3.drop('class',1).values

y3 = train3['class'].values

knn3 = neighbors.KNeighborsClassifier(n_neighbors=9)

#Train the classifier

knn3.fit(x3,y3)

#Compute the prediction according to the model

knn3.score(x3,y3)
x6 = train6.drop('class',1).values

y6 = train6['class'].values

knn6 = neighbors.KNeighborsClassifier(n_neighbors=9)

#Train the classifier

knn6.fit(x6,y6)

#Compute the prediction according to the model

knn6.score(x6,y6)
encode_dictionary = pd.DataFrame(encode_dictionary) #this contains the lists with the order we assigned the variables to train3

encoding_index = pd.DataFrame(encoding_index)   #this contains the lists from when we used the optimum correlation

letters.reverse()
test1 = test.copy()

test1 = test1.drop('veil-type',1)

num_cols = len(test1.columns)

for i in range(len(letters)):

    test1 = test1.replace({letters[i]: i},regex=False)

np.unique(test1.values.ravel())  #to check we had no letters in test that we didn't have in train
test3 = test.copy()

test3 = test3.drop('veil-type',1)

num_cols = len(test3.columns)

for i in range(num_cols):

    test3_unique = encode_dictionary.iloc[i,0]

    test3_unique2 = test3.iloc[:,i].unique()

    for j in range(len(test3_unique)):

        if test3_unique[j] in test3_unique2:

            test3.iloc[:,i] = test3.iloc[:,i].replace({test3_unique[j]: j})

np.unique(test3.values.ravel())  #to check we had no letters in test that we didn't have in train
test6 = test.copy()

test6 = test6.drop('veil-type',1)

num_cols = len(test6.columns)

for i in range(num_cols):

    test6_unique = encoding_index.iloc[i,1]

    test6_unique2 = test6.iloc[:,i].unique()

    for j in range(len(test6_unique)):

        if test6_unique[j] in test6_unique2:

            test6.iloc[:,i] = test6.iloc[:,i].replace({test6_unique[j]: j})

np.unique(test6.values.ravel())  #to check we had no letters in test that we didn't have in train
#scale the test values in the same way as train

for i in range(len(test1.columns)):

    test1.iloc[:,i] = test1.iloc[:,i]/t1_dividers[i]

test1['class'] = test1['class'].replace({min(test1['class'].unique()) : 0}, regex = False)

test1['class'] = test1['class'].replace({max(test1['class'].unique()) : 1}, regex = False)#to make the class column zeroes and ones

for i in range(len(test3.columns)):

    test3.iloc[:,i] = test3.iloc[:,i]/t3_dividers[i]

for i in range(len(test6.columns)):

    test6.iloc[:,i] = test6.iloc[:,i]/t6_dividers[i]

x1test = test1.drop('class',1).values

y1test = test1['class'].values

x3test = test3.drop('class',1).values

y3test = test3['class'].values  

x6test = test6.drop('class',1).values

y6test = test6['class'].values
from sklearn import metrics

y1hat=knn1.predict(x1test)

print ("TESTING STATS - Alphabetic:")

print ("classification accuracy:", metrics.accuracy_score(y1hat.astype(int), y1test.astype(int)))

print ("confusion matrix: \n"+ str(metrics.confusion_matrix(y1hat,y1test)))

y3hat=knn3.predict(x3test)

print ("\n TESTING STATS - Unique:")

print ("classification accuracy:", metrics.accuracy_score(y3hat, y3test))

print ("confusion matrix: \n"+ str(metrics.confusion_matrix(y3hat,y3test)))

y6hat=knn6.predict(x6test)

print ("\n TESTING STATS - Correlation:")

print ("classification accuracy:", metrics.accuracy_score(y6hat, y6test))

print ("confusion matrix: \n"+ str(metrics.confusion_matrix(y6hat,y6test)))