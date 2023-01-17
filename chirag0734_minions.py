# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.decomposition import FastICA, PCA

from sklearn.cross_validation import train_test_split

#import matplotlib.pyplot as plt

import seaborn as sns

# Any results you write to the current directory are saved as output.
def intersect(a, b):

    return list(set(a) & set(b))



def get_features(train, test):

    intersecting_features = intersect(train.columns, test.columns)

    intersecting_features.remove('people_id')

    intersecting_features.remove('activity_id')

    return sorted(intersecting_features)



def process_date(table):

        table['year'] = table['date'].dt.year

        table['month'] = table['date'].dt.month

        table['day'] = table['date'].dt.day

        table.drop('date', axis=1, inplace=True)

        return table
colors = ['r','b']

def plotFeatureCount(x, featureColumnName, targetColumnName):

    uniqueTargets = x[targetColumnName].unique()

    tbl = x[[featureColumnName, targetColumnName]]

    width = 0.75

    length = 10

    for v in uniqueTargets:

        tblV = tbl[tbl[targetColumnName] == v]

        group = tblV.groupby([featureColumnName]).agg(['count'])

        bottom = np.zeros(len(group))

        for key,gr in group:

            plt.bar(group[key].iloc[:,0].axes[0],group[key].iloc[:,0].values, width, 

                    color=colors[v], bottom=bottom, label=v)

            bottom = bottom + group[key].iloc[:,0].values



    

    plt.ylabel('Counts')

    plt.xlabel(featureColumnName)

    plt.title('Counts by feature ' + featureColumnName)

    #plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))

    #plt.yticks(np.arange(0, 81, 10))

    plt.legend()



    plt.show()
def read_test_train():

    train = pd.read_csv("../input/act_train.csv",

                           dtype={'people_id': np.str,

                                  'activity_id': np.str,

                                  'outcome': np.int8},

                           parse_dates=['date'])

    test = pd.read_csv("../input/act_test.csv",

                          dtype={'people_id': np.str,

                                 'activity_id': np.str},

                          parse_dates=['date'])

    people = pd.read_csv("../input/people.csv",

                        dtype={'people_id': np.str,

                               'activity_id': np.str,

                               'char_38': np.int32},

                        parse_dates=['date'])



    for table in [train,test]:

        process_date(table)

        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)

       

        for i in range(1, 11):

            table['char_' + str(i)].fillna('type -999', inplace=True)

            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

        

    process_date(people)

    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)

    for i in range(1, 9):

            people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    for i in range(10, 38):

            people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    

    train = pd.merge(train, people, how='left', on='people_id', left_index=True)

    test = pd.merge(test, people, how='left', on='people_id', left_index=True)



    features = get_features(train, test)

    return train, test, features
def training_testing(x_train, x_test, y_train, y_test, testing):

    cv=KFold(len(y_train), n_folds=8)

    fold=1

    for train, test in cv:

        training=RandomForestClassifier(n_estimators=20).fit(x_train.iloc[train], y_train.iloc[train])

        prediction=training.predict_proba(x_train.iloc[test])

        fpr, tpr, thresholds = roc_curve(y_train.iloc[test], prediction[:, 1])

        roc_auc = auc(fpr, tpr)

        pl.plot(fpr, tpr, label='ROC curve (fold %1i) (area = %0.4f)' % (fold, roc_auc))

        print(fold, roc_auc)

        print (training.feature_importances_)

        fold+=1



    final=training.predict_proba(x_test)

    fpr,tpr, thresholds=roc_curve(y_test, final[:,1])

    roc_auc=auc(fpr, tpr)

    print(roc_auc)

    pl.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % (roc_auc))

    return training.predict(testing)
train, test, features = read_test_train()



#Separate label and data

outcome = train['outcome']

train = train.drop(['outcome'], axis=1)

train = train.iloc[:,1:]



testing_data=features[features.outcome.isnull()]



x_train, x_test, y_train, y_test=train_test_split(features, outcome, test_size=0.20, random_state=42)

prediction=training_testing(x_train, x_test, y_train, y_test, testing_data.drop(['outcome'], axis=1))
#print(train.columns)

#for y in train.columns:

    #print(train[y].dtype))

#print(train['char_1_y'])

#print(train['activity_category'])



#ica = FastICA(n_components=8)

#sel_features = ica.fit(m_train)

#print(sel_features)



#print(sorted(train))

#print(train['char_10_x'])

#print(train['char_10_y'])

#print(train['date_x'])

#print(train['date_y'])

#print(train['days'])





    #print('Start of ' + d + ': ' + str(train[d].min().date()))

    #print('  End of ' + d + ': ' + str(train[d].max().date()))

#print(str(train['date_x'] - train['date_y']) + '\n')



#for t in train:

   # d = t['date_x'].date() - t['date_y'].date()

    #print(d)





#plotFeatureCount(train, 'char_38','outcome')

#plotFeatureCount(train, 'activity_category','outcome')

#plotFeatureCount(train, 'char_10_x','outcome')



#sns.countplot(x='outcome',data=train)

#sns.plt.show()



#sns.countplot(x='activity_category',data=test)

#sns.plt.show()



#sns.countplot(x='char_38',data=test)

#sns.plt.show()



#sns.countplot(x='activity_category',data=train,hue='outcome')

#sns.plt.show()



#fig, ax = plt.subplots()

#fig.set_size_inches(30, 20)

#sns.countplot(x='char_38',data=train,hue='outcome', palette="Set2", ax=ax)

#sns.plt.show()



#sns.countplot(x='char_10_x',data=train,hue='outcome', palette="Set2", ax=ax)

#sns.plt.show()



#sns.axes_style()



#print('Length of train: ', len(train))

#print('Length of test: ', len(test))

#print('Features [{}]: {}'.format(len(features), sorted(features)))



#print(train)

#print(train['char_10_y'])
#components = 8

#pca = PCA(n_components=components).fit(train)

#print(pca)

#ica = FastICA(n_components=8)

#sel_features = ica.fit(train)

#print(sel_features)
#print(train)

#print(test)

#print(people)

#print(features)

#print(sel_features)