import warnings
warnings.filterwarnings('ignore')

import pandas  as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Load in dataset
data = pd.read_csv('../input/UCI_Credit_Card.csv')
data.head()
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.4, random_state = 42)
# Ensure the test data doesn't have the answer
test_solution = test['default.payment.next.month']
test = test.drop('default.payment.next.month', axis = 1)
train.describe()
plt.figure(figsize=(15,12))
cor = round(train.corr(),2)
sns.heatmap(cor, cmap = sns.color_palette('BuGn'), annot = True)
train[['PAY_0', 'default.payment.next.month']].groupby(['PAY_0'], as_index = False).mean()
# Function to get default payment means 
def get_pay_mean(PAY_NUM):
    temp = train[[PAY_NUM, 'default.payment.next.month']].groupby([PAY_NUM], as_index = True).mean()
    pay_mean = temp['default.payment.next.month']
    return pay_mean
pay_means = {}
for i in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
    pay_means[i] = get_pay_mean(i)
pay_means_results = pd.DataFrame(pay_means)
#pay_means_results.reset_index(level=0, inplace=True)
pay_means_results
pay_means_results.plot(kind = 'bar', title = 'PAY_# Mean Results', figsize=(15, 7), legend=True, rot = 0, colormap = 'Set2')
# Limit Balance and Default Rate Distribution
age_survival_dist = sns.FacetGrid(train, hue = 'default.payment.next.month', aspect = 2.5, size = 5, palette = 'BuGn')
age_survival_dist.map(sns.kdeplot, 'LIMIT_BAL', shade = True)
age_survival_dist.add_legend()
plt.suptitle('Limit Balance and Default Rate Distribution', fontsize = 20, y = 1.05)
# Age and Default Rate Distribution
age_survival_dist = sns.FacetGrid(train, hue = 'default.payment.next.month', aspect = 2.5, size = 5, palette = 'BuGn')
age_survival_dist.map(sns.kdeplot, 'AGE', shade = True)
age_survival_dist.add_legend()
plt.suptitle('Age and Default Rate Distribution', fontsize = 20, y = 1.05)
train[['SEX', 'default.payment.next.month']].groupby(['SEX'], as_index = False).mean()
train[['MARRIAGE', 'default.payment.next.month']].groupby(['MARRIAGE'], as_index = False).mean()
train[['EDUCATION', 'default.payment.next.month']].groupby(['EDUCATION'], as_index = False).mean()
credit_card = train.append(test, ignore_index = True)

credit_card['MARRIAGE'].replace(0, 3, inplace = True)
credit_card['EDUCATION'].replace([0, 5, 6], 4, inplace = True)
credit_card = credit_card.drop(['ID'], axis = 1)
credit_card.shape
train_cleaned = credit_card[0:18000]
test_cleaned = credit_card[18000:30000]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
# Train Test Split
features = train_cleaned.drop('default.payment.next.month', axis=1)
target = train_cleaned['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=100)
logr = LogisticRegression()
logr_parameters = {'penalty': ['l1', 'l2'], 
                   'C' : [10, 20, 30, 40, 50, 60]
                  }

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(logr, logr_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
logr = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
logr.fit(X_train, y_train)
logr.predict(X_test)
round(logr.score(X_train, y_train) * 100, 2)
knn = KNeighborsClassifier()
knn_parameters = {'n_neighbors': range(6,9),
                  'leaf_size': [3, 5, 7, 10]
                 }

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(knn, knn_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
knn = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
knn.fit(X_train, y_train)
knn.predict(X_test)
round(knn.score(X_train, y_train) * 100, 2)
nbc = GaussianNB()
nbc_parameters = {}

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(nbc, nbc_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
nbc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
nbc.fit(X_train, y_train)
nbc.predict(X_test)
round(nbc.score(X_train, y_train) * 100, 2)
ann = MLPClassifier()
ann_parameters = {'hidden_layer_sizes': [(100,1), (100,2), (100,3)],
                  'alpha': [.0001, .001, .01, .1, 1],
                 }

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(ann, ann_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
ann = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
ann.fit(X_train, y_train)
ann.predict(X_test)
round(ann.score(X_train, y_train) * 100, 2)
dt = DecisionTreeClassifier()
dt_parameters = {'max_depth': [2,6,10], 
                 'min_samples_split': range(2,5),
                 'min_samples_leaf': range(1,5),
                 'max_features': [5, 10, 15]   
                }

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(dt, dt_parameters, cv = 10, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
dt = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
dt.fit(X_train, y_train)
dt.predict(X_test)
round(dt.score(X_train, y_train) * 100, 2)
rfc = RandomForestClassifier()
rfc_parameters = {'max_depth': [2,6,10], 
                  'min_samples_split': range(2,5),
                  'min_samples_leaf': range(1,5),
                  'n_estimators': range(5,10)
                 }

acc_scorer = make_scorer(accuracy_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(rfc, rfc_parameters, cv = 10, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the algorithm to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)
rfc.predict(X_test)
round(rfc.score(X_train, y_train) * 100, 2)
guess = rfc.predict(test_cleaned.drop(['default.payment.next.month'], axis = 1)).astype(int)
sum(guess == test_solution)/guess.size
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')
clf = xgb.XGBClassifier(max_depth=7, learning_rate=0.2, n_estimators=200)
clf.fit(X_train, y_train)
clf.predict(X_test)
round(clf.score(X_train, y_train) * 100, 2)
# Performance on Real Data Set
guess = clf.predict(test_cleaned.drop(['default.payment.next.month'], axis = 1)).astype(int)
sum(guess == test_solution)/guess.size