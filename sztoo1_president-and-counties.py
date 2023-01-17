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
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 



# let's a take a look at the data 

train_df = pd.read_csv('../input/train_potus_by_county.csv')

test_df = pd.read_csv('../input/train_potus_by_county.csv')



train_df.head()
test_df.head()
winner_df = train_df.groupby('Winner').size()

winner_df.plot.bar()



# clearly that the class rather imbalanced, but let's go ahead and see what happens.


train_df['Winner_1'] = np.where(test_df['Winner']=='Barack Obama', 1, 0)
# X = train_df.iloc[:, 0:14]

# y = train_df['Winner_1']



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# lm = LinearRegression() 

# model = lm.fit(X_train, y_train)

# model
test_df.head()
X_train = train_df.iloc[:,0:14] # dropped Winner, Winner_1 

y_train = train_df['Winner_1']

X_test = test_df.drop(['Winner'], axis=1).copy()



X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
log_y_pred = log_reg.predict(X_test)
log_reg_score = log_reg.score(X_train, y_train)

log_reg_score
from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_y_pred = rf.predict(X_test)

rf_score = rf.score(X_train, y_train)

rf_score
# Stochastic Gradient Descent 

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier() 

sgd.fit(X_train, y_train)

sgd_y_pred = sgd.predict(X_test)

sgd_score = sgd.score(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier() 

dt.fit(X_train, y_train)

dt_y_pred = dt.predict(X_test)

dt_score = dt.score(X_train, y_train)
results = pd.DataFrame({

    "Winner":test_df['Winner'],

    "Logistic Regression": log_y_pred,

    "Random Forest Classifier": rf_y_pred,

    "Stochastic Gradient Descent": sgd_y_pred,

    "Decision Tree": dt_y_pred

})

results.head()
from collections import Counter 

log_reg_count = Counter(log_y_pred)

rf_count = Counter(rf_y_pred)

sgd_count = Counter(sgd_y_pred)

dt_count = Counter(dt_y_pred)

print(log_reg_count)

print(rf_count)

print(sgd_count)

print(dt_count)
results = pd.DataFrame({

    "Model": ["Logistic Reg", "Random Forest", "Stochastic Gradient Descent","Decision Tree"],

    "Barack Obama": [log_reg_count[1], rf_count[1], sgd_count[1], dt_count[1]],

    "Mitt Romney": [log_reg_count[0], rf_count[0], sgd_count[0], dt_count[0]],

    "Score": [log_reg_score, rf_score, sgd_score, dt_score]

})



# size of the dataframes we're dealing with - for comparison purposes

shapes = pd.DataFrame({

    "Dataset": ["Barack Obama","Mitt Romney"],

    "Train": [train_df.groupby('Winner').size()[0], train_df.groupby('Winner').size()[1]],

    "Test": [test_df.groupby('Winner').size()[0], test_df.groupby('Winner').size()[1]]

})



print(shapes)

results.sort_values(by="Score", ascending=False)