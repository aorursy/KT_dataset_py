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
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
df.head()
df.shape
df.columns
df.isnull().sum()
X = df.iloc[:,0:20]
y = df.iloc[:,-1]
print(len(X),len(y))

df.info()
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2)
print(len(X_train),len(y_train),len(X_valid),len(y_valid))
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16,16))
sns.heatmap(df.corr(),annot=True)
from sklearn.feature_selection import SelectKBest , chi2
best_features = SelectKBest(score_func=chi2,k=10)
best_features
fit = best_features.fit(X,y)
fit.scores_
df_scores = pd.DataFrame(fit.scores_)
df_col = pd.DataFrame(X.columns)
feature_Scores = pd.concat([df_col,df_scores],axis=1)
feature_Scores.columns = ['Features','Score']
feature_Scores = feature_Scores.sort_values(by='Score',ascending=False)
features = feature_Scores.head(10)['Features'].values
features
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
model.feature_importances_
feature_importance = pd.DataFrame(model.feature_importances_,index=X.columns,columns=['IMPORTANCE'])
feature_importance.sort_values(by='IMPORTANCE',ascending = False,inplace = True)
feature_importance
plt.figure(figsize=(20,8))
sns.barplot(x=feature_importance.index,y='IMPORTANCE',data = feature_importance)
test_src = '/kaggle/input/mobile-price-classification/test.csv'
test = pd.read_csv(test_src)
test.head()
pred = model.predict(X_valid)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_valid, pred))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,X_valid[features],y_valid,cv=10)
print("Accuracy is : {}".format(scores.mean()))

