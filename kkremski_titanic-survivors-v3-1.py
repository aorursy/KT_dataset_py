2


%matplotlib inline  



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_inp = pd.read_csv('../input/train.csv')

#train_inp[(train_inp['Cabin'].notnull()) & (train_inp['Survived'] == 0)]

#(train_inp[train_inp['Cabin'].isnull()].size - train_inp[(train_inp['Cabin'].isnull()) & (train_inp['Survived'] == 0)].size)/train_inp[train_inp['Cabin'].isnull()].size

train = train_inp
test_inp = pd.read_csv('../input/test.csv')

test = test_inp
def fillnan_mult(col, df, mult, *dne_action):

#method to fill NaN values of input dataframe

#col is the column name in a Dataframe in which the user wants to fill in NANs

#df is the input Dataframe

#mult is a multiindexed Series, where the relevant columns in the original dataframe have been grouped

#and aggregated appropriately.

#the optional dne_action signifies what is to be done if the attributes in the row which has an NAN

#that the user is trying to fill do not lead to a valid entry in the grouped multiindexed Series.

#if left empty, or set with any value other than 'mc', all the entries of that tier will be averaged

#if dne_action is assigned as 'mc', the most common value of all the elements at that tier will be used



    dout = df.copy()

    orignan_str = col+'_origna'

    fill_str = col+'_fill'

    dout[orignan_str] = 0

    dout[fill_str] = df[col]

    mult_index_names = list(mult.index.names)

    print(len(mult_index_names))

    print(mult_index_names)

    for i in df[np.isnan(df[col])].index.tolist():

        dout.set_value(i, orignan_str, 1)

        dtemp = mult

        for j in range(len(mult_index_names)):

            if df[mult_index_names[j]].loc[i] in dtemp:

                dtemp = dtemp[df[mult_index_names[j]].loc[i]]

            elif dne_action == 'mc':

                dtemp = dtemp.value_counts().idxmax()

                break

            else:

                dtemp = dtemp.mean()

                break

        dout.set_value(i, fill_str, dtemp)

        

    return dout

   
def df_feature_eng(df):

    #method to augment and adjust input train or test dataframe

    

    # adjust gender to be binary values

    df['Sex_Bin'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    #split fare values into bins of width $50.

    #This will be used later to fill in age gaps, using bins as categories.



    binsize = 50

    bins = [x for x in range(int(df['Fare'].min()), int(df['Fare'].max()) + binsize, binsize)]

    label = [x for x in range(1, len(bins))]

    df['Fare_Cat'] = pd.cut(df['Fare'],bins,labels = label)



    df['Cabin_isnan'] = 1

    df['Cabin_isnan'] = np.where(df['Cabin'].notnull(), 0, df['Cabin_isnan'])

    

    #Values are give for the embarkation points, ignoring, and thus perpetuating any NANs

    df['Embarked_Num'] = df['Embarked'].map( { 'C': 1, 'Q': 2, 'S': 3} )

    

    #The multiindexed Series is constructed using values which are not NANs in the newly formed Embarked_Num column

    tg = df[df['Embarked_Num'] > 0].groupby(['Pclass', 'Sex_Bin'])['Embarked_Num'].agg(lambda x:x.value_counts().index[0]).astype(int)

    

    #Use fillna_mult to fill in NaN entries in the numerated Embarked field

    df = fillnan_mult('Embarked_Num', df, tg)

    

    #The multiindexed Series is constructed using values which are not NaNs in the Age column

    tg = df[np.isfinite(train['Age'])].groupby(['Pclass', 'Sex_Bin','Embarked_Num_fill'])['Age'].agg('mean')

    df = fillnan_mult('Age', df, tg)



    #Create a binned value for ages, with a bin size of 5, to use as a feature for filling in NaN fare values

    binsize = 5

    bins = [x for x in range(0, int(df['Age_fill'].max()) + binsize, binsize)]

    label = [x for x in range(1, len(bins))]

    df['Age_Cat'] = pd.cut(df['Age_fill'],bins,labels = label)



    tg = df[np.isfinite(df['Fare'])].groupby(['Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_Cat'])['Fare'].agg('mean')

    df = fillnan_mult('Fare', df, tg)

    

    #Add features of whether last digit of (possible list of) cabin number, when not NaN, is even or odd

    df['Odd_Cabin'] = df['Cabin'].map(lambda x: int(x[-1])%2 if '0' <= str(x)[-1] <= '9' else 0, na_action=None)

    df['Even_Cabin'] = df['Cabin'].map(lambda x: (int(x[-1])+1)%2 if '0' <= str(x)[-1] <= '9' else 0, na_action=None)

    

    #Add features for special-ness of title of passenger

    df['Rev'] = df['Name'].map(lambda x: x.split(', ')[1].split('.')[0] ==  'Rev')

    df['Special'] = df['Name'].map(lambda x: (x.split(', ')[1].split('.')[0] !=  'Mr') &

                                            (x.split(', ')[1].split('.')[0] !=  'Ms') &

                                            (x.split(', ')[1].split('.')[0] !=  'Miss') &

                                            (x.split(', ')[1].split('.')[0] !=  'Mrs') & 

                                            (x.split(', ')[1].split('.')[0] !=  'Rev') &

                                            (x.split(', ')[1].split('.')[0] !=  'Capt'))

    df['Ms'] = df['Name'].map(lambda x: (x.split(', ')[1].split('.')[0] ==  'Miss') | (x.split(', ')[1].split('.')[0] ==  'Ms'))

    

    #Add feature for complexity of name (number of listed names past first and last)

    df['Name_len'] = df['Name'].map(lambda x: -3 + len(x.split()) if ")" not in x else

                                              -3 + len(re.sub(r"\..+\(", ". ", x).split()))

    

    #Add feature to classify a mother, female gender, over the age of 16 and with at least 2 children on board    

    df['Mother'] = 0

    df.ix[df[(df['Sex_Bin'] == 0) & (df['Parch'] >=2) & (df['Age_fill'] > 16)].index.values, 'Mother'] = 1

    

    #Add feature to classify attended children, as being under the age of 12 with a parent on board

    df['Child_Attended'] = 0

    df.ix[df[(df['Parch'] > 1) & (df['Age_fill'] < 12)].index.values,'Child_Attended'] = 1

    

    return df
train = df_feature_eng(train)

test = df_feature_eng(test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier
features_list = ['Pclass', 'SibSp',

       'Parch', 'Sex_Bin',

       'Cabin_isnan', 'Embarked_Num_origna',

       'Embarked_Num_fill', 'Age_origna', 'Age_fill', 'Fare_origna',

       'Fare_fill', 'Odd_Cabin', 'Even_Cabin', 'Rev', 'Special', 'Ms',

       'Name_len', 'Mother', 'Child_Attended']
from sklearn.model_selection import train_test_split



#split the training data into a train/test subset

X_train, X_test, y_train, y_test = train_test_split(train[features_list],

                                                   train['Survived'],

                                                   test_size=0.1,

                                                    random_state=0)



X = train[features_list]

Y = train['Survived']
#Create a classifier model, based on random forests and train it using the training portion of the data from train.csv



#clf = RandomForestClassifier(n_estimators=100, random_state=7)

clf = ExtraTreesClassifier(n_estimators=250, random_state=7, min_samples_split=5)

#clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5,

#                                 max_depth=5, random_state=7)

clf = clf.fit(X_train, y_train)
clf.predict(X_test)
#Validate this model using the testing portion of the train.csv data

clf.score(X_test, y_test)
#test['Survived'] = clf.predict(test[['Cabin_isnan', 'Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_fill', 'SibSp', 'Parch', 'Fare_fill', 'Fare_origna', 'Age_origna', 'Embarked_Num_origna']].as_matrix())

#test.reindex_axis(sorted(test.columns), axis=1)

#test = test.reindex_axis(['Survived'] + list([a for a in test.columns if a != 'Survived']), axis=1)

#test[['Survived', 'Pclass', 'Sex', 'Age_fill', 'Fare_fill', 'Embarked_Num_fill', 'Cabin_isnan', 'Age_origna']].sort('Fare_fill', ascending=False).head(50)
#Determine the feature importances and plot the weights in a bar graph



importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in np.squeeze(clf.estimators_)],

             axis=0)

indices = np.argsort(importances)[::-1]

feature_rank = [features_list[x] for x in indices]

# Print the feature ranking

print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, feature_rank[f], importances[indices[f]]))



plt.figure()

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), feature_rank, rotation='vertical')

plt.xlim([-1, X_train.shape[1]])

plt.show()
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_test, y_test, cv=3)

scores
clf2 = ExtraTreesClassifier(n_estimators=250, random_state=7, min_samples_split=5)

clf2 = clf2.fit(train[features_list],

              train['Survived'])

test['Survived'] = clf2.predict(test[features_list])
results = test[['PassengerId', 'Survived']].copy()
results.to_csv('./results.csv', index=False)