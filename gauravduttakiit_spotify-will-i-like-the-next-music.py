import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
song=pd.read_csv('/kaggle/input/spotifyclassification/data.csv')

song.head()
song.drop('Unnamed: 0',axis=1,inplace=True)

song.head()
song.info()
song.describe()
song.target.value_counts()
songd=song.copy()

songd.drop_duplicates(subset=None,inplace=True)

songd.shape
song.shape
song=songd

del songd

song.shape
plt.figure(figsize=(20,10))

sns.heatmap(song.corr(), annot=True,cmap='RdBu')

plt.show()
song.columns
left = song[['target','song_title', 'artist']]

left.head()
song.drop(['target','song_title', 'artist'],axis=1,inplace=True)

song.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_show(X_vif):

    vif = pd.DataFrame()

    vif['Features'] = X_vif.columns

    vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    print(vif)

    print()

    if(vif.iloc[0,1] > 5.0 ):

        col = vif.iloc[0,0]

        X_vif.drop([vif.iloc[0,0]],axis =1, inplace = True)

        print("After removing \""+ col + "\" from datafame")

        vif_show(X_vif)



vif_show(song)
song.columns
song = pd.concat([song,left],axis = 1)

song.head()
song.shape
#Note: - We reduced 9 Columns from original 15 columns, which will reduce training time  for teh model
X=song.drop('target',axis=1)

y=song['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.80,stratify=y, random_state = 1)
X_train.info()
song.head()
from catboost import CatBoostClassifier

model=CatBoostClassifier()
obj=list(np.where(X_train.dtypes == np.object)[0])

obj
model.fit(X_train,y_train,cat_features=obj)
import sklearn.metrics

print(sklearn.metrics.classification_report(y_test,model.predict(X_test)))
# predict result

X_test['target']= model.predict(X_test)

X_test.head()
# Factors which decide that I will like the song or not 
imp_df = pd.DataFrame({

    "Varname": X_train.columns,

    "Imp": model.feature_importances_})

imp_df.sort_values(by="Imp", ascending=False)