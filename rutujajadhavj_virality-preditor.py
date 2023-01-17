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
import numpy as np # linear algebra

import pandas as pd 
df_sa = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/shared_articles.csv')

df_sa.head()
df_ui = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/users_interactions.csv')

df_ui
df = df_ui
df['COUNTER'] =1       #initially, set that counter to 1.

group_data = df.groupby(['contentId','eventType'])['COUNTER'].sum().reset_index() #sum function

print(group_data)
events_df = group_data.pivot_table('COUNTER', ['contentId'], 'eventType')
events_df = events_df.fillna(0)

events_df
def label(row):

   return (1* row['VIEW']) + (4*row['LIKE']) + (10*row['COMMENT CREATED']) +( 25*row['FOLLOW'] )+ (100*row['BOOKMARK'])



events_df['label'] = events_df.apply (lambda row: label(row), axis=1)

      
events_df
events_df.describe()
from sklearn.model_selection import train_test_split



train, test = train_test_split(events_df, test_size=0.2)
import seaborn as sns

import matplotlib.pyplot as plt 



sns.pairplot(train)
plt.figure(figsize=(16,12))

sns.heatmap(train.corr(),annot=True,fmt=".2f")

train_X, train_Y = train.drop('label',axis = 1), train['label']

test_X, test_Y = test.drop('label',axis = 1), test['label']
from sklearn.linear_model import LinearRegression 

from sklearn.metrics import r2_score

from sklearn.metrics import explained_variance_score





lr = LinearRegression()

lr.fit(train_X,train_Y)

lr.score(train_X,train_Y)

predict_test = lr.predict(test_X)





res = dict()

metrics = dict()



res['lr'] = lr.coef_

metrics['lr'] = r2_score(test_Y, predict_test),explained_variance_score(test_Y,predict_test)



from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)

clf.fit(train_X, train_Y)

predict_clf = clf.predict(test_X)

print(clf.coef_)

res['lasso'] = clf.coef_

metrics['lasso'] = r2_score(test_Y, predict_clf),explained_variance_score(test_Y,predict_clf)



from sklearn.linear_model import ElasticNetCV



regr.fit(train_X, train_Y)

ElasticNetCV(cv=3, random_state=0)

print(regr.coef_)

predict_regr = regr.predict(test_X)



res['cv'] = regr.coef_

metrics['cv'] = r2_score(test_Y, predict_regr),explained_variance_score(test_Y,predict_regr)

print ("Comparing coefficients of the features with Ground truth : array [100,10,25,4,1]")

for r in res.items():

    print(r)
print ("Comparing r2 and explained variance score of models with 1 as max value")

for m in metrics.items():

    print(m)


new_post = [2,3,78,4,23]



new_post_f = np.array(new_post).reshape(1,-1)



predict_new_post = clf.predict(new_post_f)

print(predict_new_post)