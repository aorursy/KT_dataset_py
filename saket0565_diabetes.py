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
df=pd.read_csv('../input/diabetes.csv')
df.head()
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
col=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
     'DiabetesPedigreeFunction', 'Age']
sns.pairplot(df[col],height=2.5)
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
ss=StandardScaler()
X=df.iloc[:,0:8].values
y=df.iloc[:,8].values
print(X.shape, y.shape)
np.random.seed(0)
idx=np.random.permutation(y.shape[0])
y=y[idx]
X=X[idx]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=4)
X_train_std=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
cov_mat=np.cov(X_train_std.T)
cov_mat
eigen_vals, eigen_vecs=np.linalg.eig(cov_mat)
tot=sum(eigen_vals)
exp_ratio=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_exp_ratio=np.cumsum(exp_ratio)
plt.bar(range(1,9), exp_ratio, alpha=0.5, align='center',
        label='Principal Component')
plt.step(range(1,9), cum_exp_ratio,where='mid',
         label='Cumilative Explained Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Explained Ratio')
plt.legend(loc='best')
plt.show()
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
n_component=[1,2,3,4,5,6,7,8]
gama_range=[0.1,0.5,1,5]
kernel=['rbf','linear']
param_range=[0.0001,0.001,0.01,0.1,1,10]
pipe_lr=Pipeline([('kpca',KernelPCA()),
                 ('clf',LogisticRegression(penalty='l2',random_state=0))])
param_grid=[{'kpca__n_components':n_component,'kpca__gamma': gama_range,'kpca__kernel':kernel,
             'clf__C':param_range}]
gs=GridSearchCV(estimator=pipe_lr,
               param_grid=param_grid,
               scoring='accuracy',
               cv=10,
               n_jobs=-1)
gs=gs.fit(X_train_std,y_train)
gs.best_score_




