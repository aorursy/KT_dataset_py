import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots

df=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head()
df.head()
from sklearn.model_selection import train_test_split
target=df['Survived']
df.drop('Survived',axis=1,inplace=True)
X=df
y=target
X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
reg_log=LogisticRegression()
reg_log.fit(X_train,y_train)
y_pred=reg_log.predict(X_test)
reg_log.score(X_train,y_train)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(max_depth=4)
dtc.fit(X_train,y_train)
y_pred_dtc=dtc.predict(X_test)
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
xgb.score(X_train,y_train)
from lightgbm import LGBMClassifier
lgb=LGBMClassifier()
lgb.fit(X_train,y_train)