# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



games = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")

games.head()
games.isnull().sum()
games[games==np.inf]=np.nan

games.isnull().sum()
games.dtypes
games_blue=games.iloc[:,1:21] #drop ID and red team feature

games_blue.head()
f, ax=plt.subplots(figsize=(18,8))

sns.heatmap(games_blue.corr(), annot=True, linewidth=0.5, fmt='.1f', ax=ax,cmap="YlGnBu")
#adding "relative" features

games_df = games_blue

blueVision = games['blueWardsPlaced']-games['redWardsDestroyed']

redVision = games['redWardsPlaced']-games['blueWardsDestroyed']

blueKdRatio = games_df['blueKills']/games_df['blueDeaths']

redKdRatio = games['redKills']/games['redDeaths']

games_df['blueRedKdDiff']= blueKdRatio - redKdRatio

games_df['blueVisionDiff']= blueVision - redVision
x=games_df.drop(['blueWins','blueWardsPlaced','blueWardsDestroyed','blueKills','blueDeaths','blueGoldPerMin',

                'blueTotalExperience','blueTotalGold'],axis=1)

y=games_df.blueWins
#Move absolute of neagative values to a new features

x['rGD_n']=0

x['rGD_p']=0

for i in range(9879):

    if(x.loc[i,'blueGoldDiff']<0):

        x.loc[i,'rGD_n']=abs(x.loc[i,'blueGoldDiff'])

    elif(x.loc[i,'blueGoldDiff']>=0):

            x.loc[i,'rGD_p']=x.loc[i,'blueGoldDiff']

            



x['blueKdDiff_p']=0

x['blueKdDiff_n']=0



for i in range(9879):

    if(x.loc[i,'blueRedKdDiff']<0):

        x.loc[i,'blueKdDiff_n']=abs(x.loc[i,'blueRedKdDiff'])

    elif(x.loc[i,'blueRedKdDiff']>=0):

            x.loc[i,'blueKdDiff_p']=x.loc[i,'blueRedKdDiff']



x['blueVD_n']=0

x['blueVD_p']=0



for i in range(9879):

    if(x.loc[i,'blueVisionDiff']<0):

        x.loc[i,'blueVD_n']=abs(x.loc[i,'blueVisionDiff'])

    elif(x.loc[i,'blueVisionDiff']>=0):

            x.loc[i,'blueVD_p']=x.loc[i,'blueVisionDiff']



x=x.drop(['blueGoldDiff','blueExperienceDiff','blueRedKdDiff','blueVisionDiff'], axis=1)



#check data

x.head()
#check infinity value

x[x==np.inf]=np.nan

x.isnull().sum()
#fill nan with max value in there column

x.fillna(x.max(axis=0), inplace=True)

x.isnull().sum()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size = 0.2, random_state=1)
#for train/test split

x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))

x_test = (x_test - x_test.min(axis=0)) / (x_test.max(axis=0) - x_test.min(axis=0))

#for cross fold validation

X = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
#used to store accuracies of each algorithms

acc={}
#naive bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train, y_train)

acc_gnb1=gnb.score(x_test,y_test)*100

acc['GaussianNB_brfore_tuning'] = acc_gnb1

print('Accuracy of GNB:{:.2f}%'.format(acc_gnb1))
from sklearn.model_selection import GridSearchCV



np.random.seed(999)



nb_classifier = GaussianNB()



params_NB = {'var_smoothing': np.logspace(0,-9, num=150)}



gs_NB = GridSearchCV(estimator=nb_classifier, 

                     param_grid=params_NB, 

                     cv=3,

                     verbose=1, 

                     scoring='accuracy')



gs_NB.fit(X, y);
gs_NB.best_params_
gs_NB.best_score_
results_NB = pd.DataFrame(gs_NB.cv_results_['params'])

results_NB['test_score'] = gs_NB.cv_results_['mean_test_score']



plt.plot(results_NB['var_smoothing'], results_NB['test_score'], marker = '.')    

plt.xlabel('Var. Smoothing')

plt.ylabel("Mean CV Score")

plt.title("NB Performance Comparison")

plt.show()
gnb2 = GaussianNB(var_smoothing = 0.003338027673990301)

gnb2.fit(x_train, y_train)

acc_gnb2=gnb2.score(x_test,y_test)*100

acc['GaussianNB_after_tuning'] = acc_gnb2

print('Accuracy of GNB after tuning:{:.2f}%'.format(acc_gnb2))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, p=2)



knn.fit(x_train, y_train)

acc_knn1 = knn.score(x_test, y_test)*100

acc['KNN_before_tuning'] = acc_knn1

print('Accuracy of knn:{:.2f}%'.format(acc_knn1))
params_KNN = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7], 

              'p': [1, 2, 5]}



from sklearn.model_selection import GridSearchCV



gs_KNN = GridSearchCV(estimator=KNeighborsClassifier(), 

                      param_grid=params_KNN, 

                      cv=3,

                      verbose=1,  # verbose: the higher, the more messages

                      scoring='accuracy', 

                      return_train_score=True)
gs_KNN.fit(X,y)
gs_KNN.best_params_
gs_KNN.best_score_
gs_KNN.cv_results_['mean_test_score']
results_KNN = pd.DataFrame(gs_KNN.cv_results_['params'])

results_KNN['test_score'] = gs_KNN.cv_results_['mean_test_score']
results_KNN['metric'] = results_KNN['p'].replace([1,2,5], ["Manhattan", "Euclidean", "Minkowski"])

results_KNN
%config InlineBackend.figure_format = 'retina'

plt.style.use("ggplot")



for i in ["Manhattan", "Euclidean", "Minkowski"]:

    temp = results_KNN[results_KNN['metric'] == i]

    plt.plot(temp['n_neighbors'], temp['test_score'], marker = '.', label = i)

    

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel("Mean CV Score")

plt.title("KNN Performance Comparison")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn2 = KNeighborsClassifier(n_neighbors=7, p=1)



knn2.fit(x_train, y_train)

acc_knn2 = knn2.score(x_test, y_test)*100

acc['KNN_after_tuning'] = acc_knn2

print('Accuracy of knn after tuning:{:.2f}%'.format(acc_knn2))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

acc_dt1 = dt.score(x_test,y_test)*100

acc['Decision_tree_before_tuning'] = acc_dt1

print('Accuracy of Decision tree before tuning:{:.2f}%'.format(acc_dt1))
df_classifier = DecisionTreeClassifier(random_state=999)



params_DT = {'criterion': ['gini', 'entropy'],

             'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],

             'min_samples_split': [2, 3]}



gs_DT = GridSearchCV(estimator=df_classifier, 

                     param_grid=params_DT, 

                     cv=3,

                     verbose=1, 

                     scoring='accuracy')



gs_DT.fit(X, y);
gs_DT.best_params_
gs_DT.best_score_
results_DT = pd.DataFrame(gs_DT.cv_results_['params'])

results_DT['test_score'] = gs_DT.cv_results_['mean_test_score']

results_DT.columns
for i in ['gini', 'entropy']:

    temp = results_DT[results_DT['criterion'] == i]

    temp_average = temp.groupby('max_depth').agg({'test_score': 'mean'})

    plt.plot(temp_average, marker = '.', label = i)

    

    

plt.legend()

plt.xlabel('Max Depth')

plt.ylabel("Mean CV Score")

plt.title("DT Performance Comparison")

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt2 = DecisionTreeClassifier(criterion = 'gini', max_depth=4, min_samples_split =2)

dt2.fit(x_train,y_train)

acc_dt2 = dt2.score(x_test,y_test)*100

acc['Decision_tree_after_tuning'] = acc_dt2

print('Accuracy of Decision tree after tuning:{:.2f}%'.format(acc_dt2))
acc_df = pd.DataFrame(acc.items(), columns=['Algorithm', 'acc_score'])

acc_df.head()
plt.figure(figsize=(18,8))

ax = sns.barplot(x='Algorithm', y='acc_score', data = acc_df)
y_NB = gnb2.predict(x_test)

y_KNN = knn2.predict(x_test)

y_DT = dt2.predict(x_test)



from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_test,y_NB)

cm_knn = confusion_matrix(y_test,y_KNN)

cm_dt = confusion_matrix(y_test,y_DT)
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
sns.heatmap(cm_dt,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})