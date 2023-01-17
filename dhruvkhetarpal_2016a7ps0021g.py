import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_tr = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv", sep=',')

data_te = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv", sep=',')
data_tr.head()

data_te.head()
data_tr= pd.get_dummies(data_tr, prefix='type', columns=['type'])

data_te= pd.get_dummies(data_te,prefix='type', columns=['type'])
#print(data_tr.replace(r'^\s*$', np.nan, regex=True))

#data_tr.replace('', np.nan, inplace=True)

data_tr=data_tr.fillna(data_tr.mean())
data_te=data_te.fillna(data_te.mean())
idd=data_te['id']
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data_tr.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
data_tr=data_tr.drop(['id'],1)

data_tr=data_tr.drop(['type_new'],1)

#data_tr=data_tr.drop(['feature4'],1)
data_te=data_te.drop(['id'],1)

data_te=data_te.drop(['type_new'],1)

#data_te=data_te.drop(['feature4'],1)
y=data_tr['rating']
data_tr=data_tr.drop(['rating'],1)

data_tes=data_te
column= list(data_tr.columns)

#Preprocess

#columns

column.remove('type_old')
x=data_tr
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42,stratify=y)
X_train.head()
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = ExtraTreesRegressor(n_estimators=i, random_state = 42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
param_grid = { 

    'n_estimators': [1000,2000,3000,4000],

    'max_depth' : [20,30,40,50],

}
#GRID_SEARCH

#from sklearn.model_selection import GridSearchCV

#CV = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, verbose=20,n_jobs=-1)

#CV.fit(x,y)
#CV.best_params_
rf=ExtraTreesRegressor(random_state=42, n_estimators= 2500, max_depth=27, max_features='auto')
rf.fit(x,y)
predy = rf.predict(data_tes)

predy
data_te.shape
dframe = pd.DataFrame(predy)
dframe=dframe.round(0)
dframe[0].value_counts()
dff=pd.DataFrame(idd)
final_data=dff.join(dframe,how='left')

final_data= final_data.rename(columns={0: "rating"})

final_data['rating']=final_data['rating'].apply(int)
final_data.head()
final_data['rating'].value_counts()
final_data.to_csv('ML_DhruvKhetarpal.csv', index = False)