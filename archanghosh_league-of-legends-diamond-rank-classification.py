import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
lol_df = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
lol_df.head()
#Checking the general Information



lol_df.info()
#copying the dataframe in another variable



lol = lol_df.copy()
#Lets Drop some columns that I think are unecessary

#We will also try to drop maximum red cloumns since the data asks us to predict for blue

columns = ['gameId', 'redFirstBlood', 'redKills', 'redEliteMonsters', 'redDragons','redTotalMinionsKilled',

       'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin', 'redHeralds',

       'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin', 'blueTotalMinionsKilled']



lol = lol.drop(columns, axis = 1)
lol.info()
#Compairing the relation between different features of Blue Team



p = sns.PairGrid(data = lol, vars = ['blueKills', 'blueAssists', 'blueWardsPlaced', 'blueTotalGold'],

                  hue = 'blueWins', size=5, palette='Set2')



p.map_diag(plt.hist)

p.map_offdiag(plt.scatter)

p.add_legend();
#plotting the Correlation Matrix



plt.figure(figsize=(18,18))

sns.heatmap(lol.drop('blueWins', axis =1 ).corr(),

cmap = 'autumn_r', annot = True, fmt = '0.2f', vmin = 0, alpha = 0.6);
#From the correlation matrix, we can clean the data more



cor_col = ['blueAvgLevel', 'redWardsPlaced', 'redWardsDestroyed', 'redDeaths', 'redAssists', 'redTowersDestroyed',

       'redTotalExperience', 'redTotalGold', 'redAvgLevel']



lol = lol.drop(cor_col, axis = 1)
#Dropping tables that have very less correlation with the winning of Blue



cor_list = lol[lol.columns[1:]].apply(lambda x: x.corr(lol['blueWins']))



cols = []



for col in cor_list.index:

    if(cor_list[col]>0.2 or cor_list[col]<-0.2):

        cols.append(col)

        

cols
lol = lol[cols]



lol.head()
lol.hist(alpha = 0.9, figsize=(15,15), bins=5);
#importing packages for model fitting and scalling



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



from sklearn import tree

from sklearn.model_selection import GridSearchCV



from sklearn.ensemble import RandomForestClassifier



from sklearn.linear_model import LogisticRegression



from sklearn.neighbors import KNeighborsClassifier



import tensorflow as tf

import tensorflow.keras as keras
X = lol

y = lol_df['blueWins']



scaler = MinMaxScaler()

scaler.fit(X)



X = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=32)
#Decision Tree





tree = tree.DecisionTreeClassifier()





grid = {'min_samples_split': [5, 10, 20, 50, 100]},



clf_tree = GridSearchCV(tree, grid, cv=5)

clf_tree.fit(X_train, y_train)



pred_tree = clf_tree.predict(X_test)





acc_tree = accuracy_score(pred_tree, y_test)

print(acc_tree)
#Naive Bayes





clf_nb = GaussianNB()

clf_nb.fit(X_train, y_train)



pred_nb = clf_nb.predict(X_test)





acc_nb = accuracy_score(pred_nb, y_test)

print(acc_nb)
#Logistic Regression



lm = LogisticRegression()

lm.fit(X_train, y_train)





pred_lm = lm.predict(X_test)

acc_lm = accuracy_score(pred_lm, y_test)

print(acc_lm)
#KNN Classification



knn = KNeighborsClassifier() 





grid = {"n_neighbors":np.arange(1,100)}

clf_knn = GridSearchCV(knn, grid, cv=5)

clf_knn.fit(X_train,y_train) 





pred_knn = clf_knn.predict(X_test) 

acc_knn = accuracy_score(pred_knn, y_test)

print(acc_knn)
#Random Forest



rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}



clf_rf = GridSearchCV(rf, grid, cv=5)

clf_rf.fit(X_train, y_train)



pred_rf = clf_rf.predict(X_test)

# get the accuracy score

acc_rf = accuracy_score(pred_rf, y_test)

print(acc_rf)
#Deep Learning



model = keras.Sequential([keras.layers.InputLayer(input_shape = X_train.shape[1:]),

                          keras.layers.Dense(100, activation = 'relu'),

                          keras.layers.Dense(50, activation = 'relu'),

                          keras.layers.Dense(25, activation = 'relu'),

                          keras.layers.Dense(1, activation = 'softmax')

                         ])



model.compile(optimizer='RMSProp', loss='mae', metrics=['accuracy'])



history = model.fit(X_train, y_train, validation_split=0.2, epochs = 100)



pred = model.evaluate(X_test, y_test)
prediction_table = {'Decision Tree': [acc_tree], 'Naive Bayes' : [acc_nb], 'Logistic Regression': [acc_lm], 'K_nearest Neighbors': [acc_knn], 'Random Forest': [acc_rf], 'Multi Layer NN': [pred[1]]}



p_table = pd.DataFrame.from_dict(prediction_table, orient = 'index', columns = ["Accuracy Score"])



print(p_table)