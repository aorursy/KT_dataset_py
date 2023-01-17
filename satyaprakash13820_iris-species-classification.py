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
df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df
df.describe()
df.info()
sns.distplot(df['SepalLengthCm'],kde=False,bins=60)
sns.distplot(df['SepalWidthCm'],kde=False,bins=60)
sns.jointplot('SepalLengthCm','SepalWidthCm',df,space=0.4)
sns.jointplot('PetalLengthCm','PetalWidthCm',df,space=0.4)
df.corr()
sns.heatmap(df.corr(),annot=True)
sns.pairplot(df,hue='Species')
X=df.drop('Species',axis=1)
y=df.Species
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.model_selection import cross_val_score
clf.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
clf1=RandomForestClassifier(n_estimators=79,verbose=5,random_state=42)
clf1.fit(X_train,y_train)
ypred=clf1.predict(X_test)
ypred
clf.score(X_test,ypred)
confusion_matrix(y_test,ypred)
