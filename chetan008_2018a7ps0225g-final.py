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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 



%matplotlib inline
df = pd.read_csv("../input/minor-project-2020/train.csv")
test_df=pd.read_csv("../input/minor-project-2020/test.csv")
df.drop(['id'],axis=1,inplace=True)



df0=df[df['target']==0]

X0=df0.iloc[:, :-1]

y0=df0.iloc[:,88]



df1=df[df['target']==1]

X1=df1.iloc[:, :-1]

y1=df1.iloc[:,88]



X=df.iloc[:, :-1]

y=df.iloc[:,88]







from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

#X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2)

#X0_train,X0_test,y0_train,y0_test=train_test_split(X0,y0,test_size=0.2)









X1=pd.concat([X1]*45)

y1=pd.concat([y1]*45)







X=pd.concat([X,X1])

y=pd.concat([y,y1])

X.reset_index(drop=True, inplace=True)

y.reset_index(drop=True, inplace=True)

#df_oversampled = pd.concat([X, y], axis = 1)

#using PCA
#X = StandardScaler().fit_transform(X)


#from sklearn.decomposition import PCA

#pca = PCA(n_components=87)
#principalComponents = pca.fit_transform(X)
#cols=["pca_"+str(i) for i in range(1,88)]
#principalDf = pd.DataFrame(data = principalComponents

#             , columns = cols)
#principalDf.reset_index(drop=True, inplace=True)

#y.reset_index(drop=True, inplace=True)

#finalDf = pd.concat([principalDf, y], axis = 1)
#finalDf.describe()
#y
#y = finalDf['target'].values

#X = finalDf.drop(['target'], axis=1)



#X_train=pd.concat([X1_train,X0_train])

#y_train=pd.concat([y1_train,y0_train])

#X_test=pd.concat([X1_test,X0_test])

#y_test=pd.concat([y1_test,y0_test])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)



X_train_scaled = StandardScaler().fit_transform(X_train)



#X_test_scaled=StandardScaler().fit_transform(X_test)
lr=LogisticRegression(penalty='l2',max_iter=100000000)

#cv=RepeatedStratifiedKFold(n_splits=4,n_repeats=3)

#scores=cross_val_score(lr,X,y,scoring='accuracy',cv=cv.split(X,y))



lr.fit(X_train_scaled,y_train)
y_predict=lr.predict_proba(X_test)

roc_auc_score(y_test,y_predict[:,1])
y.shape
X_test
test_df.drop(['id'],axis=1,inplace=True)
test_df
y_res=lr.predict_proba(test_df)
y_res
ans=pd.read_csv("../input/minor-project-2020/test.csv")
ans=ans['id']

ans=pd.DataFrame(ans)
ans['target']=y_res[:,1]

ans
ans.to_csv('/kaggle/working/ans.csv',index=False)