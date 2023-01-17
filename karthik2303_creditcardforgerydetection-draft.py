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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import mean_squared_error
# import scikitplot as skl
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.shape
pd.options.display.max_columns = None
pd.options.display.max_rows = None
display(df.head())
df.info()
df.describe()
df['Class'].value_counts()
#Need to do SMOTE for handling imbalance dataset
df.head()
#Need to do SMOTE for handling imbalanced dataset
# !pip install imblearn
X = df.drop(['Class'],axis=1)
Y = df.Class
xtrain,xtest,ytrain,ytest = train_test_split(X,Y, test_size= 0.15, random_state = 355)
smote = SMOTE()
xtrain_smote, ytrain_smote = smote.fit_sample(xtrain.astype('float'),ytrain)
ytrain_smote.value_counts()
ytrain.value_counts()
%%time
gbr = GradientBoostingClassifier()
gbr.fit(xtrain_smote,ytrain_smote)
pred = gbr.predict(xtest)
accuracy_score(ytest,pred)
con_mat =confusion_matrix(ytest,pred)
con_mat
# !pip install xgboost
%%time
xgb = XGBClassifier()
xgb.fit(xtrain_smote,ytrain_smote)
%%time
pred2 = xgb.predict(xtest)
print(accuracy_score(ytest,pred2))
con_mat =confusion_matrix(ytest,pred2)
con_mat
%%time
rfc = RandomForestClassifier()
rfc.fit(xtrain_smote,ytrain_smote)
%%time
pred3 = rfc.predict(xtest)
print(accuracy_score(ytest,pred3))
con_mat =confusion_matrix(ytest,pred3)
con_mat
%%time
knn = KNeighborsClassifier(algorithm='ball_tree')
knn.fit(xtrain_smote,ytrain_smote)
%%time
pred4 = knn.predict(xtest)
print(accuracy_score(ytest,pred4))
con_mat =confusion_matrix(ytest,pred4)
con_mat