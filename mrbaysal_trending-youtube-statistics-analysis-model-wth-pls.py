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
import pandas as pd

import numpy as np

from sklearn import model_selection

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

import matplotlib.pyplot as plt
data=pd.read_csv("../input/youtube-new/USvideos.csv")
df=data.copy()

df=df.dropna()
print(df.columns)
data=data[["views","likes","dislikes","comment_count"]]

print(data)
data.reset_index(drop=True,inplace=True)
data=data.dropna()
data.info()
data.describe().T
y=data[["comment_count"]]

X=data[["views","likes","dislikes"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
print("X_train",X_train.shape)
print("X_train",X_test.shape)

print("X_train",y_train.shape)

print("X_train",y_test.shape)
data.shape
from sklearn.cross_decomposition import PLSRegression,PLSSVD
pls_model=PLSRegression().fit(X_train,y_train)

pls_model.coef_
pls_model.predict(X_train)[0:10]
y_pred=pls_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))
r2_score(y_train,y_pred)
y_pred=pls_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
cv_10=model_selection.KFold(n_splits=10,shuffle=True,random_state=1)
RMSE=[]
for i in np.arange(1,X_train.shape[1]+1):

    pls=PLSRegression(n_components=i)

    score=np.sqrt(-1*cross_val_score(pls,X_train,y_train,cv=cv_10,scoring='neg_mean_squared_error').mean())

    RMSE.append(score)
plt.plot(np.arange(1,X_train.shape[1]+1),np.array(RMSE),'-v',c="r")

plt.xlabel('Components')

plt.ylabel('RMSE')

plt.title('Comment_Counts')
pls_model=PLSRegression(n_components=3).fit(X_train,y_train)

y_pred=pls_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))