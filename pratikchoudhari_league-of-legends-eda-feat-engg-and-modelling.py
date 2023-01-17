import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler,MinMaxScaler



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

import tensorflow as tf



from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve



sns.set(rc={'figure.figsize':(9.7,7.27)})

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

#check for null values

print("Null values summary:\n{}\n".format(df.isna().sum()))

print("No null values found, data clean!" if (not df.isna().sum().all()) else "Data needs cleaning..." )
#categorical variable for winning team name

df['team_won'] = np.where(df['blueWins']==1,'blue','red')
#quick look at dtypes

df.dtypes
class_sym = df['blueWins'].value_counts()

count_plot = sns.countplot(data=df,  x='blueWins')

for ind, count in class_sym.items():

    count_plot.text(ind,count,count,ha='center',fontsize=17)
ind_list = []

unique_count = []

for col in df.columns:

    ind_list.append(col)

    unique_count.append(len(df[col].unique()))

final_series = pd.DataFrame({'Feature':ind_list,'Unique values':unique_count}).sort_values(by='Unique values',ascending=False).reset_index(drop=True).style.bar()

final_series
vars=['blueAvgLevel','redAvgLevel','blueExperienceDiff','redExperienceDiff','blueDeaths','redDeaths']

plt.figure(figsize=(15,15))

sns.pairplot(df,vars=vars)

plt.show()
plt.figure(figsize=(20,20))

sns.heatmap(df.drop('gameId',axis=1).corr(),annot=False,square=True)

plt.show()
def findTeam(x,blue):

    if x.team.iloc[0]==blue:

        if x['count'].iloc[0]>x['count'].iloc[1]:

            return 'blue'

        else:

            return 'red'

    else:

        if x['count'].iloc[0]>x['count'].iloc[1]:

            return 'red'

        else:

            return 'blue'

def feature_report(win_team,red,blue):

    temp_df = pd.melt(df[df.team_won==win_team],id_vars='gameId', value_vars=[red,blue],var_name='team',value_name='count').sort_values(by='gameId')

    return temp_df.groupby('gameId').apply(findTeam,blue).value_counts()
res = feature_report('red','redWardsPlaced','blueWardsPlaced')

print(res,"\n")

print("When red team wins, {} team has most wards placed in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
res = feature_report('blue','redWardsPlaced','blueWardsPlaced')

print(res,"\n")

print("When blue team wins, {} team has most wards placed in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
res = feature_report('red','redKills','blueKills')

print(res,"\n")

print("When red team wins, {} team has most kills in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
res = feature_report('blue','redKills','blueKills')

print(res,"\n")

print("When blue team wins, {} team has most kills in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
res = feature_report('red','redTowersDestroyed','blueTowersDestroyed')

print(res,"\n")

print("When red team wins, {} team has most towers destroyed in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
res = feature_report('blue','redTowersDestroyed','blueTowersDestroyed')

print(res,"\n")

print("When blue team wins, {} team has most towers destroyed in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
res = feature_report('red','redDragons','blueDragons')

print(res,"\n")

print("When red team wins, {} team has most dragons killed in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
res = feature_report('blue','redDragons','blueDragons')

print(res,"\n")

print("When blue team wins, {} team has most towers destroyed in game, almost {:.3f} times more".

      format(res[res==res.max()].index[0], res[0]/res[1]))
def getData():

    df2 = pd.DataFrame()

    df2['kill'] = df['blueKills'] - df['redKills']

    df2['totalGold'] = df['blueTotalGold'] - df['redTotalGold']

    df2['avgLevel'] = df['blueAvgLevel'] - df['redAvgLevel']

    df2['goldDiff'] = df['blueGoldDiff'] - df['redGoldDiff']

    df2['expDiff'] = df['blueExperienceDiff'] - df['redExperienceDiff']

    df2['goldPerMin'] = df['blueGoldPerMin'] - df['redGoldPerMin']

    df2['totalExp'] = df['blueTotalExperience'] - df['redTotalExperience']

    

    df2['target'] = df['blueWins']

    return df2



def scale_data(dfs):

    scaler = StandardScaler()

    scaler.fit(df2.drop('target',axis=1))

    out = []

    out.append(scaler)

    for df in dfs:

        out.append(scaler.transform(df))

    return out

    

def splitData(df2):

    X = df2.iloc[:,:-1]

    y = df2.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=111)

    return X_train, X_test, y_train, y_test



def get_score(clf,X_train,y_train,X_test,y_test):

    clf.fit(X_train,y_train)

    y_pred_proba = clf.predict_proba(X_test)

    y_pred= clf.predict(X_test)

    return y_pred_proba, f1_score(y_test, y_pred)



def plotPRCurve(y_test, y_pred_proba):

    lr_precision, lr_recall, _ = precision_recall_curve(y_test,y_pred_proba[:, 1])

    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    return None
sns.set(rc={'figure.figsize':(7,5)})
df2 = getData()

X_train, X_test, y_train, y_test = splitData(df2)

scaler, X_train, X_test = scale_data([X_train, X_test])
#Logistic Regression

y_pred_proba, f1_sc = get_score(LogisticRegression(), X_train,y_train, X_test,y_test)

print('f1 score for Logistic Regression:{:.3f}'.format(f1_sc))

plotPRCurve(y_test,y_pred_proba)
#Random Forest

y_pred_proba, f1_sc = get_score(RandomForestClassifier(random_state=111), X_train,y_train, X_test,y_test)

print('f1 score for Random Forest Classifier:{:.3f}'.format(f1_sc))

plotPRCurve(y_test,y_pred_proba)
#Adaboost

abc = AdaBoostClassifier(DecisionTreeClassifier(random_state=111))

y_pred_proba, f1_sc = get_score(abc, X_train,y_train, X_test,y_test)

print('f1 score for AdaBoost Classifier:{:.3f}'.format(f1_sc))

plotPRCurve(y_test,y_pred_proba)
#SVM classifier

y_pred_proba, f1_sc=get_score(SVC(probability=True),X_train,y_train, X_test,y_test)

print('f1 score for SVM Classifier:{:.3f}'.format(f1_sc))

plotPRCurve(y_test,y_pred_proba)
params={'C':[0.001,0.01,1,10,100],

       'fit_intercept':[True,False],

       'multi_class':['ovr'],

       'solver':['newton-cg','sag','saga','lbfgs']}

gcv = GridSearchCV(LogisticRegression(random_state=111),param_grid=params,cv=10,scoring='f1')

gcv.fit(X_train,y_train)

print("Best parameters found: \n{}\nbest score: \n{:.3f}".format(gcv.best_params_,gcv.best_score_))
y_pred_proba, f1_sc = get_score(LogisticRegression(C=0.001,fit_intercept=False,multi_class='ovr',solver='newton-cg',random_state=111), X_train,y_train, X_test,y_test)

print('f1 score for Logistic Regression:{:.3f}'.format(f1_sc))

plotPRCurve(y_test,y_pred_proba)