# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

from scipy import stats

from scipy.stats import gaussian_kde

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import sklearn.metrics
warnings.filterwarnings("ignore")
def samplestrat(df, stratifying_column_name, num_to_sample, maxrows_to_est = 10000, bw_per_range = 50, eval_points = 1000): 

    '''

    Take a sample of dataframe df stratified by stratifying_column_name 

    ''' 

    strat_col_values = df[stratifying_column_name].values 

    samplcol = (df.sample(maxrows_to_est) if df.shape[0] > maxrows_to_est else df )[stratifying_column_name].values 

    vmin, vmax = min(samplcol), max(samplcol) 

    pts = np.linspace(vmin,vmax ,eval_points) 

    kernel = gaussian_kde(samplcol , bw_method = float( (vmax - vmin)/bw_per_range ) ) 

    density_estim_full = np.interp(strat_col_values, pts , kernel.evaluate(pts)) 

    return df.sample(n=num_to_sample, weights = 1/(density_estim_full)) 
def explore_outliers(dataframe):

    outliers = []

    threshold = 3

    z = np.abs(stats.zscore(dataframe))

    row = np.where(z>3)[0].tolist()

    col = np.where(z>3)[1].tolist()

    i = 0;

    while i<len(row):

        outlier = [row[i],col[i]]

        outliers.append(outlier)

        i+=1

    if len(outliers) == 0:

        print("There aren't any outliers")

    else:

        print("The outliers are found using zscore.")

        print("There are " + str(len(outliers)) + " outliers.")
def explore_missing_values(dataframe):

    if dataframe.isnull().values.any():

        print("There're missing values")

        dataframe.isnull().sum()

    else:

        print("There aren't any missing value")
def typicalSampling(group, typicalFracDict):

    name = group.name

    n = typicalFracDict[name]

    return group.sample(n=n)
def build_eval_clf(clf,x_train,y_train,x_test,y_test):

    clf.fit(x_train,y_train)

    #accuracy

    y_predict = clf.predict(x_test)

    acc = sklearn.metrics.accuracy_score(y_test,y_predict)

    #f1_score

    p = sklearn.metrics.precision_score(y_test,y_predict,average='weighted')

    f1 = sklearn.metrics.f1_score(y_test,y_predict,average='weighted')

    #precision

    print('accuracy score is ' + str(acc))

    print('f1 score is ' + str(f1))

    print('precision score is ' + str(p))
# train

train_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

# test

test_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
train_df.head()
test_df.head()
train_df['matchType'].unique()
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

train_df['matchType'] = train_df['matchType'].apply(mapper)

#train_df.groupby('matchId')['matchType'].first().value_counts().plot.bar()

train_df['matchType'].value_counts()
num_of_squad = train_df[train_df.matchType == 'squad'].shape[0]

num_of_duo = train_df[train_df.matchType == 'duo'].shape[0]

num_of_solo = train_df[train_df.matchType == 'solo'].shape[0]

num_of_rows = train_df.shape[0]
typicalFracDict = {

    'squad': int(round(num_of_squad/num_of_rows*8000)), 

    'duo': int(round(num_of_duo/num_of_rows*8000)), 

    'solo': int(round(num_of_solo/num_of_rows*8000))

}
sample_train_df = train_df.groupby('matchType',group_keys = False).apply(typicalSampling,typicalFracDict)
sample_train_df['matchType'].value_counts()
# random sampling for test set

sample_test_df = test_df.sample(1000)
sample_train_df.info()
sample_train_df.describe().T
explore_missing_values(sample_train_df)

explore_missing_values(sample_test_df)
# get rid of object type values for further evaluating outliers

col = list(sample_train_df.columns[3:])

col = [x for x in col if x != "matchType"]

sample_train_df_flt = sample_train_df[col].astype(float)
'''

should explain why there are so many outliers and why we do not want to remove them

'''

explore_outliers(sample_train_df_flt)
sns.set(rc={'figure.figsize':(41.7,15.27)})

ax = sns.boxplot(data=sample_train_df[col].astype(float))

ax.set_title('outliers for train data')
# sample train set

fig, qaxis = plt.subplots(figsize=(20,20))

corr = sample_train_df.corr()

g = sns.heatmap(corr,cmap="Reds",vmin = 0, vmax = 1,annot=True, linecolor ='white', square = True)

g.set_title('Correlation Heatmap of sample train set')
columns = np.full((corr.shape[0],), True, dtype=bool).tolist()
for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False



columns.insert(0,True)

columns.insert(0,True)

columns.insert(0,True)

columns.insert(17,True)
selected_columns = sample_train_df.columns[columns]

selected_columns
selected_feature_train_df = sample_train_df[selected_columns]
# Correlation Heatmap after feature selection

fig, qaxis = plt.subplots()

corr = selected_feature_train_df.corr()

g = sns.heatmap(corr,cmap="Reds",vmin = 0, vmax = 1,annot=True, linecolor ='white', square = True)

g.set_title('Correlation Heatmap after feature selection')
columns = ['assists', 'boosts', 'damageDealt', 'DBNOs',

       'headshotKills', 'heals', 'killPlace', 'killPoints', 'killStreaks',

       'longestKill', 'matchDuration', 'numGroups', 'rankPoints',

       'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills',

       'vehicleDestroys', 'walkDistance', 'weaponsAcquired','winPlacePerc']
solo_df = selected_feature_train_df[selected_feature_train_df.matchType == 'solo']

col = list(solo_df.columns[3:])

col = [x for x in col if x != "matchType"]

solo_df_flt = solo_df[col].reset_index()

for i in range(len(solo_df_flt.index)):

    if solo_df_flt.iloc[i]['winPlacePerc'] < 0.2:

        solo_df_flt.ix[i,'winPlacePerc'] = 0

    elif solo_df_flt.iloc[i]['winPlacePerc'] >= 0.2 and solo_df_flt.iloc[i]['winPlacePerc'] < 0.4:

        solo_df_flt.ix[i,'winPlacePerc'] = 1

    elif solo_df_flt.iloc[i]['winPlacePerc'] >= 0.4 and solo_df_flt.iloc[i]['winPlacePerc'] < 0.6:

        solo_df_flt.ix[i,'winPlacePerc'] = 2

    elif solo_df_flt.iloc[i]['winPlacePerc'] >= 0.6 and solo_df_flt.iloc[i]['winPlacePerc'] < 0.8:

        solo_df_flt.ix[i,'winPlacePerc'] = 3

    elif solo_df_flt.iloc[i]['winPlacePerc'] >= 0.8:

        solo_df_flt.ix[i,'winPlacePerc']= 4
duo_df = selected_feature_train_df[selected_feature_train_df.matchType == 'duo']

col = list(duo_df.columns[3:])

col = [x for x in col if x != "matchType"]

duo_df_flt = duo_df[col].reset_index()

for i in range(len(duo_df_flt.index)):

    if duo_df_flt.iloc[i]['winPlacePerc'] < 0.2:

        duo_df_flt.ix[i,'winPlacePerc'] = 0

    elif duo_df_flt.iloc[i]['winPlacePerc'] >= 0.2 and duo_df_flt.iloc[i]['winPlacePerc'] < 0.4:

        duo_df_flt.ix[i,'winPlacePerc'] = 1

    elif duo_df_flt.iloc[i]['winPlacePerc'] >= 0.4 and duo_df_flt.iloc[i]['winPlacePerc'] < 0.6:

        duo_df_flt.ix[i,'winPlacePerc'] = 2

    elif duo_df_flt.iloc[i]['winPlacePerc'] >= 0.6 and duo_df_flt.iloc[i]['winPlacePerc'] < 0.8:

        duo_df_flt.ix[i,'winPlacePerc'] = 3

    elif duo_df_flt.iloc[i]['winPlacePerc'] >= 0.8:

        duo_df_flt.ix[i,'winPlacePerc']= 4
squad_df = selected_feature_train_df[selected_feature_train_df.matchType == 'squad']

col = list(squad_df.columns[3:])

col = [x for x in col if x != "matchType"]

squad_df_flt = squad_df[col].reset_index()

for i in range(len(squad_df_flt.index)):

    if squad_df_flt.iloc[i]['winPlacePerc'] < 0.2:

        squad_df_flt.ix[i,'winPlacePerc'] = 0

    elif squad_df_flt.iloc[i]['winPlacePerc'] >= 0.2 and squad_df_flt.iloc[i]['winPlacePerc'] < 0.4:

        squad_df_flt.ix[i,'winPlacePerc'] = 1

    elif squad_df_flt.iloc[i]['winPlacePerc'] >= 0.4 and squad_df_flt.iloc[i]['winPlacePerc'] < 0.6:

        squad_df_flt.ix[i,'winPlacePerc'] = 2

    elif squad_df_flt.iloc[i]['winPlacePerc'] >= 0.6 and squad_df_flt.iloc[i]['winPlacePerc'] < 0.8:

        squad_df_flt.ix[i,'winPlacePerc'] = 3

    elif squad_df_flt.iloc[i]['winPlacePerc'] >= 0.8:

        squad_df_flt.ix[i,'winPlacePerc']= 4
sns.countplot(x='winPlacePerc',data = squad_df_flt)

plt.title('Win Place Count', fontsize = 20)

plt.show()
sns.countplot(x='matchType',data = selected_feature_train_df)

plt.title('Match Type Count', fontsize = 20)

plt.show()
squad_df_flt.describe().T
pair = selected_feature_train_df[['assists','boosts','kills','revives','DBNOs','winPlacePerc','matchType']]

sns.pairplot(pair, kind="scatter", hue="matchType", markers=["o", "s", "D"], palette="Set2")
fig, (maxis1, maxis2, maxis3) = plt.subplots(1, 3,figsize=(18,14))

sns.pointplot(x="boosts", y="winPlacePerc", hue="matchType",markers=["o", "s", "D"], palette="Set2",data=selected_feature_train_df, ax = maxis1)

maxis1.set_title('Boosts vs Match Type Win Place Point Plot')

sns.pointplot(x="weaponsAcquired", y="winPlacePerc", hue="matchType",markers=["o", "s", "D"], palette="Set2",data=selected_feature_train_df, ax = maxis2)

maxis2.set_title('Weapons Acquired vs Match Type Win Place Point Plot')

sns.pointplot(x="kills", y="winPlacePerc", hue="matchType",markers=["o", "s", "D"], palette="Set2",data=selected_feature_train_df, ax = maxis3)

maxis3.set_title('Kills vs Match Type Win Place Point Plot')
fig, (maxis1, maxis2, maxis3) = plt.subplots(1, 3,figsize=(18,14))

sns.barplot(x="winPlacePerc", y="swimDistance",data=squad_df_flt, ax = maxis1)

maxis1.set_title('Swim Distance vs Win Place Comparison')

sns.barplot(x="winPlacePerc", y="rideDistance",data=squad_df_flt, ax = maxis2)

maxis2.set_title('Ride Distance vs Win Place Comparison')

sns.barplot(x="winPlacePerc", y="walkDistance",data=squad_df_flt, ax = maxis3)

maxis3.set_title('Walk Distance vs Win Place Comparison')
fig, (maxis1, maxis2, maxis3, maxis4) = plt.subplots(1, 4,figsize=(18,14))

sns.pointplot(x="headshotKills", y="winPlacePerc",data=squad_df_flt, ax = maxis1)

maxis1.set_title('Headshot Kills vs Win Place Comparison')

sns.pointplot(x="killStreaks", y="winPlacePerc",data=squad_df_flt, ax = maxis2)

maxis2.set_title('Kill Streaks vs Win Place Comparison')

sns.pointplot(x="roadKills", y="winPlacePerc",data=squad_df_flt, ax = maxis3)

maxis3.set_title('Road Kills vs Win Place Comparison')

sns.pointplot(x="teamKills", y="winPlacePerc",data=squad_df_flt, ax = maxis4)

maxis4.set_title('Team Kills vs Win Place Comparison')
'''

data preprocessing need to handle winplaceperc to class label,

不然classifier准确度很低

'''
x_col = ['assists', 'boosts', 'damageDealt', 'DBNOs',

       'headshotKills', 'heals', 'killPlace', 'killPoints', 'killStreaks',

       'longestKill', 'matchDuration', 'numGroups', 'rankPoints',

       'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills',

       'vehicleDestroys', 'walkDistance', 'weaponsAcquired']

y_col = 'winPlacePerc'
solo_x = solo_df_flt[x_col]

solo_y = solo_df_flt[y_col]

duo_x = duo_df_flt[x_col]

duo_y = duo_df_flt[y_col]

squad_x = squad_df_flt[x_col]

squad_y = squad_df_flt[y_col]
solo_x_train, solo_x_test, solo_y_train, solo_y_test = train_test_split(solo_x,solo_y,test_size = 0.2)

duo_x_train, duo_x_test, duo_y_train, duo_y_test = train_test_split(duo_x,duo_y,test_size = 0.2)

squad_x_train, squad_x_test, squad_y_train, squad_y_test = train_test_split(squad_x,squad_y,test_size = 0.2)
scaler = preprocessing.Normalizer().fit(solo_x_train)

solo_x_train = scaler.transform(solo_x_train)

solo_x_test = scaler.transform(solo_x_test)
scaler = preprocessing.Normalizer().fit(duo_x_train)

duo_x_train = scaler.transform(duo_x_train)

duo_x_test = scaler.transform(duo_x_test)
scaler = preprocessing.Normalizer().fit(squad_x_train)

squad_x_train = scaler.transform(squad_x_train)

squad_x_test = scaler.transform(squad_x_test)
from sklearn.linear_model import LogisticRegression
clf_lg = LogisticRegression()
build_eval_clf(clf_lg,solo_x_train,solo_y_train,solo_x_test,solo_y_test)
build_eval_clf(clf_lg,duo_x_train,duo_y_train,duo_x_test,duo_y_test)
build_eval_clf(clf_lg,squad_x_train,squad_y_train,squad_x_test,squad_y_test)
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier()
build_eval_clf(clf_knn,solo_x_train,solo_y_train,solo_x_test,solo_y_test)
build_eval_clf(clf_knn,duo_x_train,duo_y_train,duo_x_test,duo_y_test)
build_eval_clf(clf_knn,squad_x_train,squad_y_train,squad_x_test,squad_y_test)
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
build_eval_clf(clf_dt,solo_x_train,solo_y_train,solo_x_test,solo_y_test)
build_eval_clf(clf_dt,duo_x_train,duo_y_train,duo_x_test,duo_y_test)
build_eval_clf(clf_dt,squad_x_train,squad_y_train,squad_x_test,squad_y_test)
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
build_eval_clf(clf_nb,solo_x_train,solo_y_train,solo_x_test,solo_y_test)
build_eval_clf(clf_nb,duo_x_train,duo_y_train,duo_x_test,duo_y_test)
build_eval_clf(clf_nb,squad_x_train,squad_y_train,squad_x_test,squad_y_test)
from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
build_eval_clf(clf_svm,solo_x_train,solo_y_train,solo_x_test,solo_y_test)
build_eval_clf(clf_svm,duo_x_train,duo_y_train,duo_x_test,duo_y_test)
build_eval_clf(clf_svm,squad_x_train,squad_y_train,squad_x_test,squad_y_test)
from sklearn.ensemble import RandomForestClassifier
clf_ec = RandomForestClassifier()
build_eval_clf(clf_ec,solo_x_train,solo_y_train,solo_x_test,solo_y_test)
build_eval_clf(clf_ec,duo_x_train,duo_y_train,duo_x_test,duo_y_test)
build_eval_clf(clf_ec,squad_x_train,squad_y_train,squad_x_test,squad_y_test)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

rf_solo = RandomForestClassifier()

rf_solo.fit(solo_x_train, solo_y_train)
rf_duo = RandomForestClassifier()

rf_duo.fit(duo_x_train, duo_y_train)
rf_squad = RandomForestClassifier()

rf_squad.fit(squad_x_train, squad_y_train)
from sklearn.model_selection import GridSearchCV

rf_solo = RandomForestClassifier()

param_grid = { 

    'n_estimators': [10, 120],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8,9,10],

    'criterion' :['gini', 'entropy'],

    'max_depth':[1,10]

}

gsearch1 = GridSearchCV(rf_solo,param_grid = param_grid, cv=5)

gsearch1.fit(solo_x_train, solo_y_train)
gsearch1.best_params_
solo_model=RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 120, max_depth=10, criterion='gini')
solo_model.fit(solo_x_train, solo_y_train)
pred_solo = solo_model.predict(solo_x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(solo_y_test,pred_solo))
rf_duo = RandomForestClassifier()

gsearch2 = GridSearchCV(rf_duo,param_grid = param_grid, cv=5)

gsearch2.fit(duo_x_train, duo_y_train)
gsearch2.best_params_
duo_model=RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 120, max_depth=10, criterion='gini')
duo_model.fit(duo_x_train, duo_y_train)
pred_duo = duo_model.predict(duo_x_test)

print(classification_report(duo_y_test,pred_duo))
rf_squad = RandomForestClassifier()

gsearch3 = GridSearchCV(rf_squad,param_grid = param_grid, cv=5)

gsearch3.fit(squad_x_train, squad_y_train)
gsearch3.best_params_
squad_model=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 120, max_depth=10, criterion='gini')
squad_model.fit(squad_x_train, squad_y_train)
pred_squad = squad_model.predict(squad_x_test)

print(classification_report(squad_y_test,pred_squad))