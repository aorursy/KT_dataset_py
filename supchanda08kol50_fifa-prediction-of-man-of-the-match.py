# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from IPython.display import display_html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer,LabelEncoder,StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as ac
dataset = pd.read_csv('FIFA 2018 Statistics.csv').fillna(0)[0:100]
Xdataset= dataset.iloc[:,0:20].values
lbl = LabelEncoder()
Xdataset[:,16] = lbl.fit_transform(Xdataset[:,16])
#print(Xdataset)
Ydataset=dataset.iloc[:,20].values
X_train,X_test,y_train,y_test = train_test_split(Xdataset,Ydataset,test_size=0.2,random_state=0)
gb =GaussianNB()
dTree= DecisionTreeClassifier(criterion='entropy',random_state=0,max_leaf_nodes=100)
rCls= RandomForestClassifier(n_estimators=178,criterion='entropy',random_state=19,max_leaf_nodes=13,min_impurity_decrease=0.00048)
'''sc=StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''
gb.fit(X_train,y_train)
dTree.fit(X_train,y_train)
rCls.fit(X_train,y_train)
Y_predgb = gb.predict(X_test)
Y_preddTree = dTree.predict(X_test)
Y_predrCls = rCls.predict(X_test)
print(ac(y_test,Y_predgb))
print(ac(y_test,Y_preddTree))
print(ac(y_test,Y_predrCls))
dfpred= pd.DataFrame(Y_predrCls,columns=['Is_The_Player_Man_Of_The _Match Of RandomForestModel Predicted'])
dfactual= pd.DataFrame(y_test,columns=['Is_The_Player_Man_Of_The _Match Of RandomForestModel Actual'])
df1_styler = dfactual.style.set_table_attributes("style='display:inline'")
df2_styler = dfpred.style.set_table_attributes("style='display:inline'")
k=0

display_html(df1_styler._repr_html_() + df2_styler._repr_html_(),raw=True)
#display(dfactual)