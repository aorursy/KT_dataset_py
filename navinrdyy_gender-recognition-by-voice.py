import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics
df = pd.read_csv('../input/voice.csv')

df.head()
pd.isnull(df).sum()
df['label'].value_counts()
df['label_Male'] = (df['label']=='male').astype(int)
df.columns
for y in df.columns[0:20]:

    sns.boxplot(x='label', y =y,data=df)

    plt.show()
sns.lmplot(x='meanfun', y='label_Male', data=df, aspect=1.5, ci = None, fit_reg = True)

plt.show()
X = df[['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',

       'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',

       'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']]

y = df['label_Male']
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=123)
log = LogisticRegression()
log.fit(Xtrain,ytrain)
y_pred = log.predict(Xtest)
np.sqrt(metrics.mean_squared_error(ytest,y_pred))
metrics.accuracy_score(ytest,y_pred)
from sklearn.metrics import classification_report
print(metrics.classification_report(ytest,y_pred))