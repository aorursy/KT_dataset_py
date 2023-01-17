# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv('../input/train.csv')
df.describe()
import matplotlib.pyplot as plt

%matplotlib inline
df.Age.plot(kind='hist')
df_miss = df[df['Age'].isnull()]
df_miss['Pclass'].hist()
df['Pclass'].hist()
plt.hist2d(x=df['Pclass'],y=df['Survived'], normed=True)

plt.colorbar()
plt.hist2d(x=df_miss['Pclass'],y=df_miss['Survived'], normed=True)

plt.colorbar()
from sklearn.ensemble import GradientBoostingClassifier
df['Age'] = df['Age'].fillna(-1)
df['Age'].hist()
feats = ['Pclass','Age','SibSp','Parch','Fare']



X = df[feats].values

y = df['Survived'].values
y.size
gb = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,max_depth=10,min_samples_leaf=10,min_samples_split=20)
gb
gb.fit(X,y)
np.square(gb.predict(X)-y).sum()
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
tuned_parameters = dict(n_estimators=[10,50,80,100],

                        learning_rate=[0.01,0.1,0.2,0.5],

                        max_depth=[3,5,7,10,15],

                        min_samples_leaf=[5,10,20,40],

                        min_samples_split=[10,20,40,80]

                       )



clf = GridSearchCV(GradientBoostingClassifier(**gb.get_params()), tuned_parameters, cv=5)
clf.fit(X,y)
np.square(clf.predict(X)-y).sum()
scores = pd.DataFrame(clf.cv_results_)
scores.describe()
for col in  clf.cv_results_['params'][0].keys():

    scores[col] = np.array( list(map(lambda x: x[col], clf.cv_results_['params'])) )
plt.figure(figsize=(10,7))

plt.hexbin(x=scores['learning_rate'],y=scores['mean_test_score'])



plt.colorbar()
plt.figure(figsize=(10,7))

plt.hexbin(x=scores['learning_rate'],y=scores['std_test_score'])



plt.colorbar()

clf.best_params_
plt.figure(figsize=(10,7))

plt.hexbin(x=scores['max_depth'],y=scores['mean_test_score'])



plt.colorbar()
sel = scores[(scores['learning_rate'] == 0.1) & (scores['min_samples_leaf'] == 20 ) & (scores['n_estimators'] == 50)]

plt.scatter(x=sel['max_depth'],y=sel['mean_test_score'])



#plt.colorbar()
plt.scatter(x=sel['max_depth'],y=sel['std_test_score'])
survived = y>0

clf_pred = clf.predict_proba(X)



clf_pred[survived,1].sum()-y[survived].size
gb = gb.predict_proba(X)



gb[survived,1].sum()-y[survived].size