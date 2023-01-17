import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/spotifyclassification/data.csv")
df.head()
df.drop('Unnamed: 0', axis =1, inplace=True)
df.info()
df.isnull().sum()
df.duplicated().sum()
def stat(df, parameter):
  print('Mean: {:<.2f}'.format(df[parameter].mean()))
  print('Median: {:<.2f}'.format(df[parameter].median()))
  print('Std: {:<.2f}'.format(df[parameter].std()))
  print('Max: {:<.2f}'.format(df[parameter].max()))
  print('Min: {:<.2f}'.format(df[parameter].min()))
stat(df, 'acousticness')
stat(df, 'danceability')
stat(df, 'duration_ms')
stat(df, 'energy')
stat(df, 'liveness')
stat(df, 'loudness')
stat(df, 'speechiness')
stat(df, 'tempo')
stat(df, 'valence')
len(df['key'].unique())
len(df['time_signature'].unique())
def range_col(df, parameter, begin, end, step):
    rangecol = []
  
    for row in df.values:
        row[parameter] = float(row[parameter])
    
        if row[parameter] < begin:
            rangecol.append("<" + str(round(begin,2)))
        elif row[parameter] >= end:
            rangecol.append(">=" + str(round(end,2)))
        else:
            for r in np.arange(begin,end,step).round(2):
                if r <= row[parameter] < min((r+step, end)):
                    rangecol.append(str(round(r,2))+"-"+str(round(min(r+step, end),2)))
                    break
        
  
     
    return rangecol
df['acousticness'] = range_col(df, 0, 0, 1, 0.1)
df['danceability'] = range_col(df, 1, 0, 1, 0.1)
df['duration_ms'] = range_col(df, 2, 50000, 1000000, 50000)
df['energy'] = range_col(df, 3, 0, 1, 0.1)
df['instrumentalness'] = range_col(df, 4, 0, 1, 0.1)
df['liveness'] = range_col(df, 6, 0, 1, 0.1)
df['loudness'] = range_col(df, 7, -34, 0, 2)
df['speechiness'] = range_col(df, 9, 0, 1, 0.1)
df['tempo'] = range_col(df, 10, 40, 220, 10)
df['valence'] = range_col(df, 12, 0, 1, 0.1)
df.head()
onehot_table = df[['song_title','artist']]
def onehot(onehot_table, df, parameter):
  i=1
  for hotkey in df[parameter].unique():
    onehot_table.insert(loc = i-1, column = str(parameter)+'_'+ str(i), value=(df[parameter]==hotkey).astype(int), allow_duplicates=True)
    i+=1
onehot(onehot_table, df, 'acousticness')
onehot_table.head()
onehot(onehot_table, df, 'danceability')
onehot(onehot_table, df, 'duration_ms')
onehot(onehot_table, df, 'energy')
onehot(onehot_table, df, 'instrumentalness')
onehot(onehot_table, df, 'key')
onehot(onehot_table, df, 'liveness')
onehot(onehot_table, df, 'loudness')
onehot(onehot_table, df, 'mode')
onehot(onehot_table, df, 'speechiness')
onehot(onehot_table, df, 'tempo')
onehot(onehot_table, df, 'time_signature')
onehot(onehot_table, df, 'valence')
onehot(onehot_table, df, 'artist')

onehot_table = onehot_table.drop(['song_title','artist'], axis =1)
onehot_table.info()
!pip install catboost
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
X_train = onehot_table
Y_train = df['target']


x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size = 0.2, random_state = 0)
#1. Logistic Regression
log = LogisticRegression()
log.fit(x_train,y_train)
log_y_pred = log.predict(x_test)
log_result_train = log.score(x_train,y_train)

#2. Gaussian Naive Bayes
NB = GaussianNB()
NB.fit(x_train,y_train)
NB_y_pred = NB.predict(x_test)
NB_result_train = NB.score(x_train,y_train)

#3. Decision Tree
DT = DecisionTreeClassifier()
DT.fit(x_train,y_train)
DT_y_pred = DT.predict(x_test)
DT_result_train = DT.score(x_train,y_train)

#4. K-Nearest Neighbors (K-NN)
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train)
KNN_y_pred = KNN.predict(x_test)
KNN_result_train = KNN.score(x_train,y_train)

#5. Random Forest
forest = RandomForestClassifier(n_estimators=250)
forest.fit(x_train,y_train)
forest_y_pred = forest.predict(x_test)
forest_result_train = forest.score(x_train,y_train)

#6. Ada Boost
ada_boost = AdaBoostClassifier(base_estimator=DT)
ada_boost.fit(x_train,y_train)
ada_boost_y_pred = ada_boost.predict(x_test)
ada_boost_result_train = ada_boost.score(x_train,y_train)

#7. Gradient Boost
gb = GradientBoostingClassifier()
gb.fit(x_train,y_train)
gb_y_pred = gb.predict(x_test)
gb_result_train = gb.score(x_train,y_train)

#8. XGBoost
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
xgb_y_pred = xgb.predict(x_test)
xgb_result_train = xgb.score(x_train,y_train)

#9. Cat Boost
catb = CatBoostClassifier()
catb.fit(x_train,y_train)
catb_y_pred = catb.predict(x_test)
catb_result_train = catb.score(x_train,y_train)
print('Score train:')
print('1. Logistic Regression: {:.2%}'.format(log_result_train))
print('2. Gaussian Naive Bayes: {:.2%}'.format(NB_result_train))
print('3. Decision Tree: {:.2%}'.format(DT_result_train))
print('4. K-NN: {:.2%}'.format(KNN_result_train))
print('5. Random Forest: {:.2%}'.format(forest_result_train))
print('6. Ada Boost: {:.2%}'.format(ada_boost_result_train))
print('7. Gradient Boost: {:.2%}'.format(gb_result_train))
print('8. XGBoost: {:.2%}'.format(xgb_result_train))
print('9. Cat Boost: {:.2%}'.format(catb_result_train))
#1. Logistic Regression
log_result_test = log.score(x_test,y_test)

#2. Gaussian Naive Bayes
NB_result_test = NB.score(x_test,y_test)

#3. Decision Tree
DT_result_test = DT.score(x_test,y_test)

#4. K-Nearest Neighbors (K-NN)
KNN_result_test = KNN.score(x_test,y_test)

#5. Random Forest
forest_result_test = forest.score(x_test,y_test)

#6. Ada Boost
ada_boost_result_test = ada_boost.score(x_test,y_test)

#7. Gradient Boost
gb_result_test = gb.score(x_test,y_test)

#8. XGBoost
xgb_result_test = xgb.score(x_test,y_test)

#9. Cat Boost
catb_result_test = catb.score(x_test,y_test)

print('Score test:')
print('1. Logistic Regression: {:.2%}'.format(log_result_test))
print('2. Gaussian Naive Bayes: {:.2%}'.format(NB_result_test))
print('3. Decision Tree: {:.2%}'.format(DT_result_test))
print('4. K-NN: {:.2%}'.format(KNN_result_test))
print('5. Random Forest: {:.2%}'.format(forest_result_test))
print('6. Ada Boost: {:.2%}'.format(ada_boost_result_test))
print('7. Gradient Boost: {:.2%}'.format(gb_result_test))
print('8. XGBoost: {:.2%}'.format(xgb_result_test))
print('9. Cat Boost: {:.2%}'.format(catb_result_test))
log.get_params()
parameters= {'penalty' : ['l1', 'l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['liblinear'],
    'max_iter' : range(50,150,20)}

clf_model = GridSearchCV(estimator=log,param_grid=parameters)
clf_model.fit(x_train,y_train)

clf_y_pred = clf_model.predict(x_test)

clf_result_test = clf_model.score(x_test,y_test)
print('Logistic Regression with grid search score is {:.2%}'.format(clf_result_test))