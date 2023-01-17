# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.tail()
print('the dataset contains',df.shape[0],'rows and ',df.shape[1] ,'columns')
df.info()
df.isna().sum()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
X=df.drop(['Class'],axis=1)

y=df.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
def model_eval(model,X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predict=model.predict(X_test)

    proba=model.predict_proba(X_test)[:,-1]

    print('Accuracy',accuracy_score(y_test,predict))

    print('Confusion',confusion_matrix(y_test,predict))

    print('Roc sco',roc_auc_score(y_test,proba))
model_eval(model,X_train, X_test, y_train, y_test)
!pip install pydotplus
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(max_depth=3)

dtc.fit(X_train,y_train)
from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

import pydotplus



features = X.columns

# Create DOT data

dot_data = export_graphviz(dtc, out_file=None, feature_names=features)



# Draw graph

graph = pydotplus.graph_from_dot_data(dot_data)  



# Show graph

Image(graph.create_png())
model_eval(dtc,X_train, X_test, y_train, y_test)
rfc=RandomForestClassifier(n_estimators=10)
model_eval(rfc,X_train, X_test, y_train, y_test)