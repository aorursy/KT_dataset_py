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
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)
df =pd.read_csv('/kaggle/input/heart-disease-dataset/heart.csv')
df.head()
df['target'].unique()
#!pip install pandas-profiling 
#from pandas_profiling import ProfileReport

# ### To Create the Simple report quickly
#profile = ProfileReport(df, title='Heart Profiling Report', explorative=True)

#profile.to_widgets() 
# profile.to_file("output.html")
df.isnull().sum()
df.info()
df.dtypes
sns.pairplot(df)
df.columns
for feature in df.columns:
    plt.hist(x=df[feature])
    plt.title('{}'.format(feature))
    plt.show()
sns.countplot(df['target'] ,)
import numpy as np
corr = df.corr()
fig, ax= plt.subplots(figsize=(8,8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()
cat_features = [feature for feature in df.columns if len(df[feature].unique()) < 7]
print('Caegorical features count {}'.format(len(cat_features)))
cat_features
cat_features.remove('target')
cat_features
df.head()
df1 = pd.get_dummies(df, columns=cat_features)
df1.shape
df.shape
from sklearn.preprocessing import StandardScaler
standscal = StandardScaler()
scale_col = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df1[scale_col] =standscal.fit_transform(df1[scale_col])
df1[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].head()
X = df1.drop(columns='target')
y = df1['target']
X.shape
y.shape
df1.shape
df1.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print('X_train {} ,X_test {} ,y_train {} ,y_test{}'.format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
classifier = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train,y_train)
random_search.best_estimator_
classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
              learning_rate=0.25, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier, X, y, cv=10)
score
score.mean()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
cm
y_test
y_pred
