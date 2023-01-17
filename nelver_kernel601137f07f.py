import numpy as np

import pandas as pd

import seaborn as sns

import os



from matplotlib import pyplot as plt
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.head()
#Since feature `gameId` does not have any predictive power in our case, we remove it.

df = df.drop(['gameId'],axis=1)
#Get the features that are the most correlated with the label `blueWins`

#By "most correlated", we mean that the absolute value of the correlation coefficient exceeds 0.3



corrs = df.apply(lambda x: x.corr(df['blueWins']))

corr_feat = df[corrs[abs(corrs) > 0.3].index]
#Now check correlation between features.

plt.figure(figsize=(12, 12))

sns.heatmap(corr_feat.corr(),

            cmap='plasma',

            annot=True,

            fmt='0.2f',

            vmin=0)
#We can see that there are numerous pairs of distinct features that are perfectly correlated with each other.

#e.g (blueKills, redDeaths), (redGoldPerMin,redTotalGold) etc.

#Since having a pair of perfectly correlated features does not add a predictive power, for each pair, remove 1 feature.



cols_to_remove = ['redKills','redDeaths','blueGoldPerMin','redGoldPerMin','redGoldDiff','redExperienceDiff']

df1 = corr_feat.drop(cols_to_remove,axis=1)
X = df1[df1.columns[1:]].values

y = df1['blueWins']
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report





X = df1[df1.columns[1:]].values

y = df1['blueWins']

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=199)





clf = GaussianNB()

clf.fit(X_train,y_train)





print('---------- PERFORMANCE ON THE TEST DATA---------- \n')

print(classification_report(y_test,clf.predict(X_test)))



print('---------- PERFORMANCE ON THE TRAIN DATA---------- \n')

print(classification_report(y_train,clf.predict(X_train)))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report





X = df1[df1.columns[1:]].values

y = df1['blueWins']

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=199)





clf = LogisticRegression()

clf.fit(X_train,y_train)





print('---------- PERFORMANCE ON THE TEST DATA---------- \n')

print(classification_report(y_test,clf.predict(X_test)))



print('---------- PERFORMANCE ON THE TRAIN DATA---------- \n')

print(classification_report(y_train,clf.predict(X_train)))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report,f1_score

from sklearn.metrics import make_scorer



rec = make_scorer(f1_score,average='weighted')





X = df1[df1.columns[1:]].values

y = df1['blueWins']

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=199)





clf = KNeighborsClassifier()

grid_param = dict(n_neighbors=np.arange(1,11))

grid_s = GridSearchCV(clf,param_grid=grid_param, cv=5,refit=True,scoring=rec)

clf = grid_s.fit(X_train,y_train).best_estimator_





print('---------- PERFORMANCE ON THE TEST DATA---------- \n')

print(classification_report(y_test,clf.predict(X_test)))



print('---------- PERFORMANCE ON THE TRAIN DATA---------- \n')

print(classification_report(y_train,clf.predict(X_train)))
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report



X = df1[df1.columns[1:]].values

y = df1['blueWins']

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=199)





clf = SVC(C=10e+5) #I've also tried 10e+2,10e+3,10e+4,10e+5: they seem to be yielding same results.

clf.fit(X_train,y_train)





print('---------- PERFORMANCE ON THE TEST DATA---------- \n')

print(classification_report(y_test,clf.predict(X_test)))



print('---------- PERFORMANCE ON THE TRAIN DATA---------- \n')

print(classification_report(y_train,clf.predict(X_train)))
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras





X = df1[df1.columns[1:]].values

y = df1['blueWins']

#Note that we are using random seed, so the split of the data is identical to the previous splits.

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=199)



#Scale the data beforehand

scaler = MinMaxScaler()

X_train_transormed = scaler.fit_transform(X_train)

X_test_transformed = scaler.transform(X_test)





model = keras.Sequential()

model.add(keras.layers.Dense(3,input_shape=(10,),activation='relu'))

model.add(keras.layers.Dense(1,activation='sigmoid'))





adam = keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='binary_crossentropy')





history = model.fit(X_train_transormed,y_train,epochs=5,verbose=0,batch_size=20)





print(classification_report(y_test, model.predict_classes(X_test_transformed)))

print(classification_report(y_train, model.predict_classes(X_train_transormed)))




