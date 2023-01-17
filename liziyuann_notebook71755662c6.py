import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

% pylab inline

% matplotlib inline
df = pd.read_csv('../input/adult.csv')

data = pd.read_csv('../input/adult.csv')
# size of data set

df.shape
df.head()
# data type of every feature

df.dtypes
# 0. income

df.income.value_counts()
# create a new feature for income

df['income>50K'] = np.where(df.income == '<=50K', 0, 1)    # 0: less than 50K, 1: more than 50K

df['income>50K'].value_counts()
df.age.describe()
# cut them into "pieces"

ageSlice = np.arange(10, 100, 10)

binned = pd.cut(x=df.age, bins=ageSlice) 

binned_df = pd.DataFrame(binned)

binned_df.columns = ['age_binned']

df = pd.concat([df, binned_df],axis=1)
# define a function to build dataframe which can be used to compare probability of earning more or less than 50K

def make_df_simple(df, feature):

    df_less50K = df.loc[df['income>50K']==0, feature].value_counts().sort_index()

    df_more50K = df.loc[df['income>50K']==1, feature].value_counts().sort_index()

    df_sep = pd.concat([df_less50K, df_more50K], axis=1)

    df_sep.columns = ['<50K', '>50K']

    df_sep = df_sep.fillna(value=0)

    return df_sep
df_age = make_df_simple(df, 'age_binned')
df_age.plot(kind='bar', stacked=True, figsize=(5,4), rot=45, color=['#FDB927','#552582'], fontsize=10)

plt.ylabel('number of samples', fontsize=10)

plt.xlabel('age', fontsize=10)
# define a function to change categorical variables into numeric variables

def make_map(feature):

    mapList = np.sort(feature.unique())

    mapDict = {}

    for i in range(len(mapList)):

        mapDict[mapList[i]] = i

    return mapDict
df['age_num'] = df.age_binned.map(make_map(df.age_binned))

feature_processed = []

feature_processed.append('age_num')
df.workclass.unique()
# change "?" to "Unknown"

df.loc[df.workclass=='?', 'workclass'] = 'Unknown'

unknownList = []

unknownList.append('workclass')
# a function similar to make_df_simple, but with one more column, which is the ratio of '>50K'

def make_df(df, feature):

    df_less50K = df.loc[df['income>50K']==0, feature].value_counts().sort_index()

    df_more50K = df.loc[df['income>50K']==1, feature].value_counts().sort_index()

    df_sep = pd.concat([df_less50K, df_more50K], axis=1)

    df_sep.columns = ['<50K', '>50K']

    df_sep = df_sep.fillna(value=0)

    df_sep['total'] = df_sep.sum(axis=1)

    df_sep['percent of >50K'] = df_sep['>50K'] / df_sep['total']

    df_sep.sort_values(by=['percent of >50K'], ascending=False, inplace=True)

    return df_sep
df_workclass = make_df(df, 'workclass')
df_workclass[['<50K', '>50K']].plot(kind='barh', stacked=True, figsize=(6,4), \

                                    color=['#FDB927','#552582'], fontsize=10)

plt.xlabel('number of samples', fontsize=10)

plt.ylabel('workclass', fontsize=10)
df['workclass_num'] = df.workclass.map(make_map(df.workclass))

feature_processed.append('workclass_num')
df.fnlwgt.describe()
df.education.unique()
df_ed = make_df(df, 'education')

df_ed
df_ed_temp = df_ed.loc['Assoc-voc':'Preschool', :].copy()

df_ed_temp2 = df_ed.loc['Doctorate':'Bachelors', :].copy()

df_ed_new = pd.concat([df_ed_temp2.T, df_ed_temp.sum(axis=0)], axis=1)

df_ed_new.columns = ['Doctorate', 'Prof-school', 'Masters', 'Bachelors', 'Other']

df_ed_new = df_ed_new.T

df_ed_new['percent of >50K'] = df_ed_new['>50K'] / (df_ed_new['<50K'] + df_ed_new['>50K']) 
plt.figure(figsize=(18, 3.2))

for i in range(5):

    labels = "<50K", ">50K"

    explode = (0.1, 0)

    toPie = [df_ed_new.iloc[i, 0], df_ed_new.iloc[i, 1]]

    plt.subplot(1, 5, i+1)

    plt.pie(toPie, explode=explode, labels=labels, colors=['#552582', '#FDB927'],\

        autopct='%1.1f%%', shadow=False, startangle=140);

    plt.title(df_ed_new.index[i], fontsize=15)
mapDegree = {x:0 for x in df_ed_temp.index}

mapDegree.update({'Doctorate':2, 'Prof-school':2, 'Masters':1, 'Bachelors':1})

df['education_num'] = df.education.map(mapDegree)

feature_processed.append('education_num')
df_edNum_new = make_df(df, 'education.num')

df_edNum_new.sort_index(ascending=True, inplace=True)
df_edNum_new[['<50K', '>50K']].plot(kind='bar', stacked=True, color=['#FDB927','#552582'],\

                                   figsize=(6,4), rot=0, fontsize=10)

plt.xlabel('years of education', fontsize=10)

plt.ylabel('number of samples', fontsize=10)
df_edNum_new['percent of >50K'].plot(kind='line', figsize=(6, 4), rot=0, fontsize=10, marker='o')

plt.xlabel('years of education', fontsize=10)

plt.ylabel('percent of >50K', fontsize=10)
dropList = []

feature_processed.append('education.num')
df['marital.status'].unique()
df_marry = make_df(df, 'marital.status')

df_marry
mapDict_marry = {'Married-civ-spouse':0, 'Married-AF-spouse':0, 'Divorced':1, 'Widowed':1,

           'Married-spouse-absent': 1, 'Separated':1, 'Never-married':1}

df['marital.status_num'] = df['marital.status'].map(mapDict_marry)

feature_processed.append('marital.status_num')
grouped = df.groupby('marital.status_num')

married = [grouped['income>50K'].mean()[0], 1-grouped['income>50K'].mean()[0]]

unmarried = [grouped['income>50K'].mean()[1], 1-grouped['income>50K'].mean()[1]]

labels = ">50K", "<50K"

explode = (0.1, 0)



plt.figure(figsize=(9, 4));

plt.subplot(121);

plt.pie(married, explode=explode, labels=labels, colors=['#FDB927','#552582'],\

        autopct='%1.1f%%', shadow=True, startangle=140);

plt.title('married', fontsize=10)

plt.subplot(122);

plt.pie(unmarried, explode=explode, labels=labels, colors=['#FDB927','#552582'],\

        autopct='%1.1f%%', shadow=True, startangle=140);

plt.title('unmarried', fontsize=10)
df.occupation.unique()
df.loc[df.occupation=='?', 'occupation'] = 'Unknown'

unknownList.append('occupation')
df_occupation = make_df(df, 'occupation')

df_occupation
df['occupation_num'] = df.occupation.map(make_map(df.occupation))

feature_processed.append('occupation_num')
df.relationship.unique()
df_relation = make_df(df, 'relationship')

df_relation
# again, it seems wise to divide whether they have spouse...

mapDict_relation = {'Wife':0, 'Husband':0, 'Not-in-family':1, 'Unmarried':1, 'Other-relative':1, 'Own-child':1}

df['relation_num'] = df['relationship'].map(mapDict_relation)
# but we suspect that it is correlated with the feature 'marital.status'

df[['marital.status_num', 'relation_num']].corr()
dropList.append('relationship')
df.race.unique()
df_race = make_df(df, 'race')

df_race
df['race_num'] = df.race.map(make_map(df.race))

feature_processed.append('race_num')
plt.figure(figsize=(18, 3.2))

for i in range(5):

    labels = ">50K", "<50K"

    explode = (0.1, 0)

    toPie = [df_race.iloc[i, 0], df_race.iloc[i, 1]]

    plt.subplot(1, 5, i+1)

    plt.pie(toPie, explode=explode, labels=labels, colors=['#552582', '#FDB927'],\

        autopct='%1.1f%%', shadow=False, startangle=140);

    plt.title(df_race.index[i], fontsize=15)
df.sex.unique()
df_sex = make_df(df, 'sex')

df_sex
toPie = [df_sex.iloc[0,1], df_sex.iloc[0,0], df_sex.iloc[1,0], df_sex.iloc[1,1]]

labels = "Male >50K: 30.5%", "Male <50K", "Female <50K", "Female >50K: 10.9%"

explode = (0.05, 0, 0, 0.15)



plt.figure(figsize=(5, 5));

plt.pie(toPie, explode=explode, labels=labels, colors=['#FDB927', '#FDB927', '#552582', '#552582'],\

        autopct='%1.1f%%', shadow=False, startangle=140);

plt.title('Male VS Female', fontsize=10)
df['sex_num'] = df.sex.map(make_map(df.sex))

feature_processed.append('sex_num')
df['capital.gain'].describe()
gainSlice = [0, 1, 2500, 5000, 10000, 100000]

gain_binned = pd.cut(x=df['capital.gain'], bins=gainSlice, include_lowest=True)

df_gain_binned = pd.DataFrame(gain_binned)

df_gain_binned.columns = ['gain_binned']

df = pd.concat([df, df_gain_binned], axis=1)
df_gain = make_df(df, 'gain_binned').sort_index()
df_gain
df['capital.gain_num'] = df.gain_binned.map(make_map(df.gain_binned))
df['capital.loss'].describe()
lossSlice = [0, 1, 1500, 1750, 2000, 5000]

loss_binned = pd.cut(x=df['capital.loss'], bins=lossSlice, include_lowest=True)

df_loss_binned = pd.DataFrame(loss_binned)

df_loss_binned.columns = ['loss_binned']

df = pd.concat([df, df_loss_binned], axis=1)
df_loss = make_df(df, 'loss_binned').sort_index()
df_loss
df['capital.loss_num'] = df.loss_binned.map(make_map(df.loss_binned))
plt.figure(figsize(16, 4))

ax = plt.subplot(1,2,1)

df_gain.plot(y='percent of >50K', marker='o', fontsize=10, rot=45, ax=ax)

plt.xlabel('capital gain category', fontsize=15)

plt.ylabel('percent of >50K', fontsize=15)



ax = plt.subplot(1,2,2)

df_loss.plot(y='percent of >50K', marker='o', fontsize=10, rot=45, ax=ax)

plt.xlabel('capital loss category', fontsize=15)

plt.ylabel('percent of >50K', fontsize=15)
df[['capital.gain', 'capital.loss']].corr()
df['capital.net'] = df['capital.gain'] - df['capital.loss']

df['capital.net'].describe()
netSlice = [-5000, -1, 0, 2500, 5000, 10000, 100000]

net_binned = pd.cut(x=df['capital.net'], bins=netSlice, include_lowest=True)

df_net_binned = pd.DataFrame(net_binned)

df_net_binned.columns = ['net_binned']

df = pd.concat([df, df_net_binned], axis=1)
df_capitalNet = make_df(df, 'net_binned').sort_index()
df_capitalNet
df_capitalNet.plot(y='percent of >50K', figsize=(6, 4), marker='o', fontsize=10, rot=45)

plt.xlabel('capital net category', fontsize=10)

plt.ylabel('percent of >50K', fontsize=10)
df['capital.net_num'] = df.net_binned.map(make_map(df.net_binned))

feature_processed.append('capital.net_num')
dropList.append('capital.gain')

dropList.append('capital.loss')
df['hours.per.week'].describe()
hourSlice = range(0, 110, 10)

hour_binned = pd.cut(x=df['hours.per.week'], bins=hourSlice, include_lowest=True)

df_hour_binned = pd.DataFrame(hour_binned)

df_hour_binned.columns = ['hours_binned']

df = pd.concat([df, df_hour_binned], axis=1)
df_hour = make_df(df, 'hours_binned').sort_index()

df_hour[['<50K', '>50K']].plot(kind='bar', stacked=True, figsize=(6, 4), rot=45,\

                              color=['#552582', '#FDB927'], fontsize=10)

plt.ylabel('number of samples', fontsize=10)

plt.xlabel('work hours per week', fontsize=10)
df_hour['percent of >50K'].plot(kind='line', figsize=(6, 4), rot=45, fontsize=10, marker='o')

plt.xlabel('work hours per week', fontsize=10)

plt.ylabel('percent of >50K', fontsize=10)
df['hours_num'] = df.hours_binned.map(make_map(df.hours_binned))

feature_processed.append('hours_num')
df['native.country'].unique()
df.loc[df['native.country']=='?', 'native.country'] = 'Unknown'

df_native = make_df(df, 'native.country')
df_native
unknownList.append('native.country')
df_native['total'] = df_native['<50K'] + df_native['>50K']

df_native.sort_values(by='total', ascending=True, inplace=True)
plt.figure(figsize=(8, 6))

plt.scatter(range(df_native.shape[0]), df_native['percent of >50K'], s=50, c=df_native['percent of >50K'],\

            cmap=cm.rainbow)

plt.xlabel('countries sorted by sample sizes', fontsize=10)

plt.ylabel('percent of > 50K', fontsize=10)



plt.text(18-2, df_native.loc['Iran', 'percent of >50K']+0.01, 'Iran', fontsize=10)

plt.text(41, df_native.loc['United-States', 'percent of >50K']+0.01, 'US', fontsize=10)

plt.text(28-3, df_native.loc['China', 'percent of >50K']+0.01, 'China', fontsize=10)

plt.text(0-9, df_native.loc['Holand-Netherlands', 'percent of >50K']-0.03, 'Holand-Netherlands', fontsize=10)

plt.colorbar()
df['native_num'] = df['native.country'].map(make_map(df['native.country']))

feature_processed.append('native_num')
feature_processed
# a function to create heatmap

def make_heatmap(df, feature1, feature2):

    grouped = df.groupby([feature1, feature2])['income>50K'].mean()

    df_grouped = grouped.unstack().fillna(0)

    plt.figure(figsize=(6,4))

    sns.set(font_scale=1)

    ax = sns.heatmap(df_grouped)

    return df_grouped
grouped = df.groupby('occupation')['education.num'].mean()

grouped.plot(kind='barh', figsize=(6,4), color=['#552582'], fontsize=10)

plt.xlabel('education years', fontsize=10)
make_heatmap(df, 'occupation', 'education.num');
grouped = df.groupby(['occupation'])['sex'].value_counts().unstack().fillna(0)

grouped.plot(kind='barh', stacked=False, figsize=(6,8), color=['#552582', '#FDB927'], fontsize=10)

plt.xlabel('number of sample', fontsize=10)
make_heatmap(df, 'occupation', 'sex');
df['marriage'] = df['marital.status_num'].map({0:'married', 1:'unmarried'})

grouped = df.groupby(['marriage'])['sex'].value_counts().unstack().fillna(0)

grouped.plot(kind='bar', stacked=False, figsize=(4,4), color=['#552582', '#FDB927'], fontsize=10, rot=0)

plt.ylabel('number of sample', fontsize=15)
hello = make_heatmap(df, 'marriage', 'sex')
hello
grouped = df.groupby('race')['education.num'].mean()

grouped.plot(kind='barh', figsize=(5,3), color=['#552582'], fontsize=10)

plt.xlabel('education years', fontsize=10)
grouped = df.groupby('race')

grouped.boxplot(column='education.num', figsize=(12, 4), layout=(1,5), fontsize=10);
make_heatmap(df, 'race', 'education.num');
grouped = df.groupby('race')

grouped.boxplot(column='hours.per.week', figsize=(12, 4), layout=(1,5), fontsize=10);
make_heatmap(df, 'race', 'hours_binned');
grouped = df.groupby('sex')

grouped.boxplot(column='hours.per.week', figsize=(5, 5), layout=(1,2), fontsize=10);
make_heatmap(df, 'hours_binned', 'sex');
feature_processed
dropList
unknownList
feature_processed.append('income>50K')

df_new = df[feature_processed].copy()
df_new.isnull().sum()
for x in unknownList:

    print('unknown of %s: '%x, df[df[x]=='Unknown'].shape)
from sklearn.cross_validation import train_test_split

X = df_new[feature_processed].drop('income>50K', axis=1).copy()

y = df_new['income>50K']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2016)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier as RFC

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy of decision tree: ', float(sum(np.array(y_test) == y_pred))/float(len(y_pred)))



clf = RFC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy of random forest: ', float(sum(np.array(y_test) == y_pred))/float(len(y_pred)))
from sklearn.cross_validation import cross_val_score

clf = DecisionTreeClassifier()

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

print('mean accuracy of decision tree: ', scores.mean())



clf = RFC()

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

print('mean accuracy of random forest: ', scores.mean())
clf = RFC()

feature_importance = pd.Series((clf.fit(X, y).feature_importances_), index=feature_processed[:-1])

feature_importance.sort_values(ascending=True).plot(kind='barh', figsize=(5,6), color=['#552582'],\

                                                   fontsize=10)

plt.xlabel('feature importance from random forest')
features_ranked = [x for x in feature_importance.sort_values(ascending=False).index]

acc = []

for i in range(1, len(features_ranked)+1):

    feature_selected = features_ranked[:i]

    X = df_new[feature_selected].copy()

    y = df_new['income>50K']

    clf = RFC()

    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

    acc.append(scores.mean())

plt.figure(figsize=(5, 4))

plt.plot(range(1, len(features_ranked)+1), acc)

plt.xlabel('number of features')

plt.ylabel('mean classification accuracy')
features_ranked = [x for x in feature_importance.sort_values(ascending=False).index if x != 'capital.net_num']

acc = []

for i in range(1, len(features_ranked)+1):

    feature_selected = features_ranked[:i]

    X = df_new[feature_selected].copy()

    y = df_new['income>50K']

    clf = RFC()

    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

    acc.append(scores.mean())

plt.figure(figsize=(5,4))

plt.plot(range(1, len(features_ranked)+1), acc)

plt.xlabel('number of features')

plt.ylabel('mean classification accuracy')
feature_test = features_ranked[:4]

X = df_new[feature_test].copy()

y = df_new['income>50K']

clf = RFC()

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

print('mean accuracy: ', scores.mean())
import time
# n_estimators

start_time = time.time()



acc = []

n_estimators = [10, 50, 100, 250, 500, 750, 1000]

for n in n_estimators:

    clf = RFC(n_estimators = n)

    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

    acc.append(scores.mean())

plt.figure(figsize=(5, 4))

plt.plot(n_estimators, acc)

plt.xlabel('n_estimators')

plt.ylabel('accuracy')



print("--- %s seconds ---" % (time.time() - start_time))
# max_features

start_time = time.time()



acc = []

n_features = list(range(1, X.shape[1]+1))

for n in n_features:

    clf = RFC(max_features = n)

    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

    acc.append(scores.mean())

plt.figure(figsize=(5, 4))

plt.plot(n_features, acc)

plt.xlabel('max_features')

plt.ylabel('accuracy')



print("--- %s seconds ---" % (time.time() - start_time))
# max_depth

start_time = time.time()



acc = []

n_depth = list(range(1, 16))

for n in n_depth:

    clf = RFC(max_depth = n)

    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

    acc.append(scores.mean())

plt.figure(figsize=(5, 4))

plt.plot(n_depth, acc)

plt.xlabel('max_depth')

plt.ylabel('accuracy')



print("--- %s seconds ---" % (time.time() - start_time))
# Let's try the so-called "best" parameters with the four selected features

start_time = time.time()



clf = RFC(n_estimators=500, max_depth=7)

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

print('mean accuracy: ', scores.mean())



print("--- %s seconds ---" % (time.time() - start_time))
# first we try all features in feature_processed

df_new_str = df_new.drop('income>50K', axis=1).copy()

df_str = df_new_str.applymap(str)

dummies = pd.get_dummies(df_str)

dummies.head()
X = dummies

y = df_new['income>50K']
# KNN

start_time = time.time()



from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 10)

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

print('mean accuracy: ', scores.mean())



print("--- %s seconds ---" % (time.time() - start_time))
# let's try the "selected" features from the frandom forest model

feature_test
df_new_str = df_new.drop('income>50K', axis=1).copy()

df_str = df_new_str[feature_test].applymap(str)

dummies = pd.get_dummies(df_str)

X = dummies

y = df_new['income>50K']
start_time = time.time()



clf = KNeighborsClassifier(n_neighbors = 10)

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

print('mean accuracy: ', scores.mean())



print("--- %s seconds ---" % (time.time() - start_time))
# Let's try different n_neighbors

start_time = time.time()



from sklearn.neighbors import KNeighborsClassifier

acc = []

n_neighbors = list(range(2, 32, 2))

for n in n_neighbors:

    clf = KNeighborsClassifier(n_neighbors = n)

    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

    acc.append(scores.mean())

plt.figure(figsize=(5, 4))

plt.plot(n_neighbors, acc)



print("--- %s seconds ---" % (time.time() - start_time))