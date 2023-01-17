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
df=pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
df
df.dtypes
sns.jointplot('mean_radius','mean_area',data=df)
sns.jointplot('mean_texture','mean_perimeter',data=df)
sns.jointplot('mean_radius','diagnosis',data=df)
sns.jointplot('mean_texture','diagnosis',data=df)
df.corr()
sns.heatmap(df.corr(),annot=True)
X=df.drop(['diagnosis','mean_area','mean_perimeter'],axis=1)
X
y=df.diagnosis
y.value_counts()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
y_train.shape
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=50,n_jobs=-1,max_depth=5,verbose=5)
clf.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score
clf.score(X_test,y_test)
y_pred=clf.predict(X_test)
# y_pred
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
ypred=logreg.predict(X_test)
confusion_matrix(y_test,ypred)
