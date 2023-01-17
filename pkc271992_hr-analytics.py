# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')

plt.figure(figsize=(10,10))
dept_list = df['Department'].unique()
ax = sns.countplot(df['Department'])
plt.xticks(rotation=90)
for p in ax.patches:
    ax.annotate(p.get_height(),(p.get_x()+.2,p.get_height()+10))
ax.set_title('Department wise Head count')    
plt.show()

dep_grp = df.groupby('Department')
avg_hrs = []
yrs_spent = []
fig,ax = plt.subplots(4,1)
fig.set_figheight(20)
fig.set_figwidth(10)
for dept in dept_list:
    avg_hrs.append(dep_grp['average_montly_hours'].mean().loc[dept])
    yrs_spent.append(dep_grp['time_spend_company'].mean().loc[dept])
ax[0].bar(dept_list,avg_hrs)
ax[0].set_title('Average monthly hours for departments')
ax[0].set_xlabel('Department')
ax[0].set_ylabel('Hour count')
ax[0].set_xticklabels(labels = dept_list,rotation=60)
sns.countplot(df['Department'],hue = df['salary'],ax=ax[1])
ax[1].set_title('Distribution of salaries for departments')
ax[1].set_xticklabels(labels = dept_list,rotation=60)
for p in ax[1].patches:
    ax[1].annotate(p.get_height(),(p.get_x(),p.get_height()+5))
sns.countplot(df['Department'],hue = df['left'],ax=ax[2])
ax[2].set_title('Comparison of people left to present')
ax[2].set_xticklabels(labels = dept_list,rotation=60)
for p in ax[2].patches:
    ax[2].annotate(p.get_height(),(p.get_x(),p.get_height()+5))
ax[3].bar(dept_list,yrs_spent)
ax[3].set_title('Average number of years spent in company')
ax[3].set_xlabel('Department')
ax[3].set_ylabel('Year count')
ax[3].set_xticklabels(labels = dept_list,rotation=60)    
fig.tight_layout(pad=2.0)


#plt.show()
sal_dict = {'low':1,'medium':2,'high':3}
df['salary'] = df['salary'].map(sal_dict)
df = pd.get_dummies(df)
test_size = int(math.ceil(.2*len(df)))
df_future = df[-test_size:]
df = df[:-test_size]
y = np.array(df['left'])
X = np.array(df.drop('left',1))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .1)
pipe = Pipeline([('std', StandardScaler()),('LogReg',LogisticRegression())])
pipe.fit(X_train,y_train)
pipe.score(X_test,y_test)

pred_actual = df_future['left']
df_future.drop('left',1,inplace=True)
pred_model = pipe.predict(df_future)
print('Logistic Regression Accuracy')
metrics.accuracy_score(pred_actual,pred_model)

pipe2 = Pipeline([('std',StandardScaler()),
               ('clf',KNeighborsClassifier(n_neighbors=5))])
pipe2.fit(X_train,y_train)
pipe.score(X_test,y_test)
knn_fut = pipe2.predict(df_future)
print('KNN Accuracy')
metrics.accuracy_score(pred_actual,knn_fut)