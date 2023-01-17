import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df.head()
X=df[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","attribute"]].copy()

y=df["class"].copy()
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=40)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

reg33 = ExtraTreesClassifier()

parameters = {'random_state':[69,42], 'max_depth':[18,19,20],'n_estimators':[60,68]}

scorer = make_scorer(accuracy_score,greater_is_better=True) 

grid_obj = GridSearchCV(reg33,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X,y)

best_reg33 = grid_fit.best_estimator_  

df1=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

df1.info()

df1.head()
X_test=df1[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","attribute"]].copy()
y_pred=best_reg33.predict(X_test)

y_pred
df2=pd.DataFrame()

df2['id']=df1['id']

df2['class']=y_pred

df2.head()
df2.to_csv('sol1.csv',index=False)
features = ["chem_0","chem_1","chem_4","chem_5","chem_6","attribute"]

X=df[features].copy()

y=df["class"].copy()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

reg1 = RandomForestClassifier()

parameters = {'random_state':[1111,1234],'n_estimators':[20,200,2000]}

scorer = make_scorer(accuracy_score,greater_is_better=True) 

grid_obj = GridSearchCV(reg1,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X,y)

best_reg1 = grid_fit.best_estimator_  
X_test=df1[features].copy()

clf11=best_reg1.fit(X, y)

y_pred1=clf11.predict(X_test)



df3=pd.DataFrame()

df3['id']=df1['id']

df3['class']=y_pred1

df3.head()
df3.to_csv('sol2.csv',index=False)