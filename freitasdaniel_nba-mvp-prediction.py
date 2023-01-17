from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) #do not miss this line





import plotly as py

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import plotly.graph_objs as go



from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model



import os

print(os.listdir("../input"))
import warnings

warnings.simplefilter(action='ignore', category=Warning)
%matplotlib inline
player_data = pd.read_csv('../input/nba-players-stats/player_data.csv')

Players = pd.read_csv('../input/nba-players-stats/Players.csv')

ss = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')

mvp = pd.read_csv('../input/mvp-voting/mvp_table.csv')
player_data.columns
Players.columns
ss.columns
ss.dropna(subset=['Year'], inplace=True)

ss.drop(['blanl','blank2'], axis=1, inplace=True) # empty columns
ss_train = ss[(ss['Year'].astype(int) > 1985) & (ss['Year'].astype(int) < 2006)]

ss_val = ss[(ss['Year'].astype(int) > 2005) & (ss['Year'].astype(int) < 2012)]

ss_test = ss[(ss['Year'].astype(int) > 2012)]
"""

a = []

for i in ss.columns:

    a.append(ss[i].isna().sum())

a = [x / ss.shape[0] for x in a] * 100

"""

a = ss.isna().sum()/len(ss)
data = [go.Bar(

    x=a.index,

    y=a

)]



py.offline.iplot(data, filename='percentage-of-nan')
ss.Pos[ss['3P'].isna()].groupby(ss.Pos).value_counts()
ss_3pt = ss[(ss['Year'].astype(int) > 1979)]

a = ss_3pt.isna().sum()/len(ss_3pt)



data = [go.Bar(

    x=a.index,

    y=a

)]

py.offline.iplot(data)
ss = ss[ss['Year'] > 1985]
# Create column mvp_votes and initialize with zeros

ss['mvp_votes'] = np.zeros(len(ss));
for i in mvp.index:

    try:

        a = ss[((ss['Player'] == str(mvp['Player'][i]))

                |(ss['Player'] == mvp['Player'][i]+'*') 

                & (ss['Year'] == mvp['Year'][i]))]

        if(a.index != None):

            ss['mvp_votes'][a.index] = mvp['mvp_score'][i]



    except:

        pass
fig, ax = plt.subplots(figsize=(10,10))

corr = ss.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
ss_fd = ss.drop(['Player','Year','BPM','VORP','Tm','Pos'], axis=1)
ss_ = ss_fd[ss_fd['mvp_votes'] != 0]

fig, ax = plt.subplots(figsize=(10,10))

corr = ss_.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            ax=ax);
corr.mvp_votes
print("Correlation lower thas 0.1 : ")

print(ss_.columns[corr.mvp_votes < 0.1])

ss_.drop(ss_.columns[corr.mvp_votes < 0.1], axis=1, inplace=True)
# Functions that will be in use from now on

def plot_tst_pred(pred, test):

    fig, ax = plt.subplots(figsize=(10,10))

    plt.plot(pred, color='darkblue', label='y pred')

    plt.plot(test.reset_index(drop=True), color='green', label='y_test')

    plt.ylabel('mvp votes')

    plt.xlabel('samples')

    plt.legend()

    plt.show()

    

def mean_error(pred, value):

    a = 0

    for i in range(len(pred)):

        a += abs(pred[i] - value[i])

    return a    



def correct_pred(pred, value):

    a = 0

    for i in range(len(pred)):

        if((pred[i] <= value[i] + 10) and (pred[i] >= value[i] - 10) and (pred[i] >= 1)): # will assume a range as right

            a += 1

    return a    



def fi(clf):

    feature_importance = clf.feature_importances_

    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)

    sorted_idx = sorted_idx[len(feature_importance) - 30:]

    pos = np.arange(sorted_idx.shape[0]) + .5



    plt.figure(figsize=(12,8))

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, X_train.columns[sorted_idx])

    plt.xlabel('Relative Importance')

    plt.title('Variable Importance')

    plt.show()



def check_nan(in_):

    print("Remaining NaN's:" + str(in_.isna().sum().sum()))

    if(in_.isna().sum().sum() > 0): 

        in_.fillna(0, inplace=True) # Avoid NaN in regression





def xy_sets(train, test):

    X_train = train.iloc[ : , 1:train.shape[1] - 1]

    y_train = train.iloc[:, train.shape[1] - 1]



    X_test = test.iloc[ : , 1:test.shape[1] - 1]

    y_test = test.iloc[ : , test.shape[1] - 1]

    return X_train, X_test, y_train, y_test



def results(pred, y):

    pos = 0

    f_pos = 0

    f_neg = 0

    for i in range(len(y)):

        if((pred[i] == 1 and y[i] == 1)): #or (pred[i] == 0 and y[i] == 0)):

            pos += 1

        elif(pred[i] == 1 and y[i] == 0):

            f_pos += 1

        elif(pred[i] == 0 and y[i] == 1):

            f_neg += 1

            

    return pos, f_pos, f_neg
check_nan(ss_)
train, test = train_test_split(ss_, test_size=0.25, shuffle=True )
test.shape
X_train, X_test, y_train, y_test = xy_sets(train, test)
clf = tree.DecisionTreeRegressor(max_depth=5)

clf = clf.fit(X_train, y_train)

pred = clf.predict(X_test)
plot_tst_pred(pred,y_test)
fi(clf)
print(mean_error(y_test.reset_index(drop=True), pred))

print ("Correct prediction:" + str(correct_pred(pred,y_test.values)) + " Total :" + str(y_test.shape[0]))
regr = RandomForestRegressor(max_depth=5, random_state=0,

                             n_estimators=100)

regr.fit(X_train, y_train)

pred = regr.predict(X_test)

print(pred)

print(y_test)
fi(regr)
plot_tst_pred(pred, y_test)
ss_all = ss.drop(['Player','Year','BPM','VORP','Tm','Pos'], axis=1)

train, test = train_test_split(ss_all, test_size=0.2, shuffle=True )
check_nan(train)

check_nan(test)
X_train, X_test, y_train, y_test = xy_sets(train,test)
regr = RandomForestRegressor(max_depth=5, random_state=0,

                             n_estimators=100)

regr.fit(X_train, y_train)

pred = regr.predict(X_test)

plot_tst_pred(pred, y_test)
fi(regr)
X_train, X_test, y_train, y_test = xy_sets(train,test)

check_nan(train)

check_nan(test)
reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,

                 normalize=False)

reg.fit(X_train, y_train)

pred = reg.predict(X_test)
plot_tst_pred(pred, y_test)
# This cell is here only for debug purpose

ss_all = ss.drop(['Player','Year','BPM','VORP','Tm','Pos'], axis=1)
ss_all['mvp_votes'][ss_all['mvp_votes'] < 10] = 0 # This row should come before. This is important

ss_all['mvp_votes'][ss_all['mvp_votes'] >= 10] = 1 
ss_all['mvp_votes'].value_counts()
train, test = train_test_split(ss_all, test_size=0.25, shuffle=True )

check_nan(train)

check_nan(test)

X_train, X_test, y_train, y_test = xy_sets(train,test)
y_test.value_counts()
log_reg = linear_model.LogisticRegression(random_state=0, solver='sag',

                             multi_class='multinomial').fit(X_train, y_train)

pred = log_reg.predict(X_test)
pos, f_pos, f_neg = results(pred,y_test.values)

print("Positives:" + str(pos) + " False positives: " + str(f_pos) + " False negatives: " + str(f_neg) )
plt.plot(y_test.values);
plt.plot(pred, color='green');