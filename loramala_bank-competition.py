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
DF_URL = '/kaggle/input/devrepublik02/training_set.csv'

bank_df = pd.read_csv(DF_URL)

bank_df.head()
bank_df.columns
bank_df.isnull().sum()
# how many unique values in each column

bank_df.nunique()
bank_df.info()
bank_df.drop_duplicates() 


bank_df.drop('duration', axis=1, inplace=True)

bank_df.drop('pdays', axis=1, inplace=True)

bank_df.drop('contact',axis=1, inplace=True)

bank_df.head()
round(bank_df.isnull().sum() / bank_df.shape[0] * 100, 2)
bank_df['default']=bank_df['default'].apply(lambda x: 1 if x == 'yes' else 0)

bank_df['marital']=bank_df['marital'].apply(lambda x: 1 if x == 'married' else 0)

bank_df['housing']=bank_df['housing'].apply(lambda x: 1 if x == 'yes' else 0)

bank_df['loan']=bank_df['loan'].apply(lambda x: 1 if x == 'yes' else 0)

bank_df['poutcome']=bank_df['poutcome'].apply(lambda x: 1 if x == 'success' else 0)



bank_df
bank_df.describe()
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



import xgboost



from sklearn.model_selection import train_test_split #split

from sklearn.metrics import accuracy_score #metrics



#tools for hyperparameters search

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
bank_df[['poutcome','campaign', 'previous']].describe()
plt.scatter(bank_df['previous'], bank_df['poutcome'], color='green')

plt.xlabel('Campaign')

plt.ylabel('Poutcome')

plt.title('Correlation between Previous and Poutcome')

plt.legend('poutcome: outcome of the previous marketing campaign','previous: number of contacts performed before this campaign and for this client')

plt.show()
#Column,which we are going to predict -'deposit' column and compare its values to other columns.

value_counts = bank_df['deposit'].value_counts()



value_counts.plot.bar(title = 'Deposit value counts')
#job and deposit:

j_df = pd.DataFrame()



j_df['yes'] = bank_df[bank_df['deposit'] == 'yes']['job'].value_counts()

j_df['no'] = bank_df[bank_df['deposit'] == 'no']['job'].value_counts()



j_df.plot.bar(title = 'Job and deposit')
#number of contacts performed during this campaign ('campaign') and deposit

c_df = pd.DataFrame()

c_df['campaign_yes'] = (bank_df[bank_df['deposit'] == 'yes'][['deposit','campaign']].describe())['campaign']

c_df['campaign_no'] = (bank_df[bank_df['deposit'] == 'no'][['deposit','campaign']].describe())['campaign']



c_df
c_df.drop(['count', '25%', '50%', '75%']).plot.bar(title = 'Number of contacts performed during this campaign and deposit statistics')
#number of contacts performed during previous campaign ('previous') and deposit

p_df = pd.DataFrame()

p_df['previous_yes'] = (bank_df[bank_df['deposit'] == 'yes'][['deposit','previous']].describe())['previous']

p_df['previous_no'] = (bank_df[bank_df['deposit'] == 'no'][['deposit','previous']].describe())['previous']



p_df
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from sklearn import tree
bank_df
bank_df.dtypes.sample(10)
def get_dummy_from_bool(row, column_name):

    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''

    return 1 if row[column_name] == 'yes' else 0



def get_correct_values(row, column_name, threshold, df):

    ''' Returns mean value if value in column_name is above threshold'''

    if row[column_name] <= threshold:

        return row[column_name]

    else:

        mean = df[df[column_name] <= threshold][column_name].mean()

        return mean



def clean_data(df):

    cleaned_df = df.copy()

    

    

    

    #convert categorical columns to dummies

    cat_columns = ['job', 'marital', 'education', 'month', 'poutcome']

    

    for col in  cat_columns:

        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),

                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',

                                               drop_first=True, dummy_na=False)], axis=1)

    

    

    

    #impute incorrect values and drop original columns

    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df),axis=1)

    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df),axis=1)

    

    cleaned_df = cleaned_df.drop(columns = ['campaign', 'previous'])

    return cleaned_df
cleaned_df = clean_data(bank_df)

cleaned_df.head()
cleaned_df.dtypes.sample(10)
from sklearn.model_selection import GridSearchCV
params = {'max_depth':[3,7,10],

         'criterion': ['gini','entropy'],

         'max_leaf_nodes': [5,10,15]}

model = tree.DecisionTreeClassifier()
GS_model = GridSearchCV(

    estimator=model,

    param_grid=params,

    scoring='accuracy',

    verbose=1)

GS_model.fit(X_test,y_test)
GS_model.best_params_
GS_model.best_score_
GS_model.best_index_
GS_model_2 = GridSearchCV(

    estimator=model,

    param_grid=params,

    scoring='accuracy',

    verbose=1)

GS_model.fit(X_train,y_train)
GS_model.best_score_
model = RandomForestClassifier(n_estimators=500, oob_score=True)

model.fit(X_train, y_train)
model = RandomForestClassifier(

    n_estimators=500,

    criterion='entropy',

    oob_score=True

)

model.fit(X_train, y_train)

model.score(X_train, y_train)
model.oob_score_
from tqdm import tqdm
oob_scores = []

n_est = list(range(50,300,10))



for n in tqdm(n_est):

    temp_model = RandomForestClassifier(

        n_estimators=n,

    criterion='entropy',

    #max_samples=0.5,

    oob_score=True,

    random_state=42

    )

    temp_model.fit(X_train, y_train)

    oob_scores.append(temp_model.oob_score)

    
plt.plot(n_est, oob_scores)

plt.xlabel('n_estimators')

plt.ylabel('OOB score')

plt.show()
from sklearn.metrics import roc_curve
from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier

from sklearn.datasets import make_classification
bagging_model = BaggingClassifier(

    #base_estimator=decision_tree,

    max_samples=0.5,

    n_estimators=100,

    oob_score=True,

    random_state=42)

bagging_model.fit(X_train, y_train)
bagging_model.score(X_train, y_train)
bagging_model.score(X_test, y_test)
bagging_model.oob_score_
from sklearn.metrics import roc_curve
y_pred_prob = model.predict_proba(X_test)

y_pred_prob.shape
#fpr, tpr, thresholds = roc_curve(y_test[:50], y_pred_prob[:50,1])
oob_scores
from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_regression

# generate regression dataset

X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)

# fit final model

model = LinearRegression()

model.fit(X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_blobs

# generate 2d classification dataset

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# fit final model

model = LogisticRegression()

model.fit(X, y)

# new instances where we do not know the answer

Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)

# make a prediction

ynew = model.predict_proba(Xnew)

# show the inputs and predicted probabilities

for i in range(len(Xnew)):

	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
X = cleaned_df.drop(columns = 'deposit')

y = cleaned_df[['deposit']]
TEST_SIZE = 0.3

RAND_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RAND_STATE)
#train XGBoost model

xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)

xgb.fit(X_train,y_train.squeeze().values)



#calculate and print scores for the model for top 15 features

y_train_preds = xgb.predict(X_train)

y_test_preds = xgb.predict(X_test)



print('XGB accuracy score for train: %.3f: test: %.3f' % (

        accuracy_score(y_train, y_train_preds),

        accuracy_score(y_test, y_test_preds)))

#get feature importances from the model

headers = ["name", "score"]

values = sorted(zip(X_train.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)

xgb_feature_importances = pd.DataFrame(values, columns = headers)



#plot feature importances

x_pos = np.arange(0, len(xgb_feature_importances))

plt.bar(x_pos, xgb_feature_importances['score'])

plt.xticks(x_pos, xgb_feature_importances['name'])

plt.xticks(rotation=90)

plt.title('Feature importances (XGB)')



plt.show()
#Test

DF_URL = '/kaggle/input/devrepublik02/validation_set.csv'

test_df = pd.read_csv(DF_URL)

test_df.head()
test_df.columns
test_df.drop('duration', axis=1, inplace=True)

test_df.drop('pdays', axis=1, inplace=True)

test_df.drop('contact',axis=1, inplace=True)

#test_df.drop('duration',axis=1, inplace=True)

test_df.head()
test_df['default']=test_df['default'].apply(lambda x: 1 if x == 'yes' else 0)

test_df['marital']=test_df['marital'].apply(lambda x: 1 if x == 'married' else 0)

test_df['housing']=test_df['housing'].apply(lambda x: 1 if x == 'yes' else 0)

test_df['loan']=test_df['loan'].apply(lambda x: 1 if x == 'yes' else 0)

test_df['poutcome']=test_df['poutcome'].apply(lambda x: 1 if x == 'success' else 0)



test_df
df = test_df

df
test_df.describe()
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



import xgboost



from sklearn.model_selection import train_test_split #split

from sklearn.metrics import accuracy_score #metrics



#tools for hyperparameters search

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
test_df[['poutcome','campaign', 'previous']].describe()
plt.scatter(df['previous'], df['poutcome'], color='green')

plt.xlabel('Campaign')

plt.ylabel('Poutcome')

plt.title('Correlation between Previous and Poutcome')

plt.legend('poutcome: outcome of the previous marketing campaign','previous: number of contacts performed before this campaign and for this client')

plt.show()




from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from sklearn import tree
df.dtypes.sample(10)
def get_dummy_from_bool(row, column_name):

    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''

    return 1 if row[column_name] == 'yes' else 0



def get_correct_values(row, column_name, threshold, df):

    ''' Returns mean value if value in column_name is above threshold'''

    if row[column_name] <= threshold:

        return row[column_name]

    else:

        mean = df[df[column_name] <= threshold][column_name].mean()

        return mean



def clean_data(df):

    cleaned_df = df.copy()

    

    

    

    #convert categorical columns to dummies

    cat_columns = ['job', 'marital', 'education', 'month', 'poutcome']

    

    for col in  cat_columns:

        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),

                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',

                                               drop_first=True, dummy_na=False)], axis=1)

    

    

    

    #impute incorrect values and drop original columns

    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df),axis=1)

    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df),axis=1)

    

    cleaned_df = cleaned_df.drop(columns = ['campaign', 'previous'])

    return cleaned_df
cleaned_df = clean_data(df)

cleaned_df.head()
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
params = {'max_depth':[3,7,10],

         'criterion': ['gini','entropy'],

         'max_leaf_nodes': [5,10,15]}

model = tree.DecisionTreeClassifier()
GS_model = GridSearchCV(

    estimator=model,

    param_grid=params,

    scoring='accuracy',

    verbose=1)

GS_model.fit(X_test,y_test)
GS_model.best_params_
GS_model.best_score_
from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier

from sklearn.datasets import make_classification
bagging_model = BaggingClassifier(

    #base_estimator=decision_tree,

    max_samples=0.5,

    n_estimators=100,

    oob_score=True,

    random_state=42)

bagging_model.fit(X_train, y_train)
bagging_model.score(X_train, y_train)
bagging_model.score(X_test, y_test)
bagging_model.oob_score_
from sklearn.metrics import roc_curve
X_train.head()
cleaned_df.head()
y_pred_prob = bagging_model.predict_proba(cleaned_df)

y_pred_prob
y_pred_prob.shape
subm = pd.DataFrame()

subm['deposit'] = y_pred_prob[:,1]

subm.reset_index(drop=False, inplace=True)

subm

subm.to_csv('submussion.csv', index=False)