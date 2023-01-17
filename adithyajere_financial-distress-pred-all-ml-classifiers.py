# Basics
import sys
import pandas as pd
import numpy as np
%matplotlib inline

# Imports for data loading
# import psycopg2
# import sqlalchemy
# import imp
# import os

# Sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import TimeSeriesSplit
# secrets_filepath = '/home/casey/secrets.py'
# secrets = imp.load_source('secrets', secrets_filepath)

# # Postgres connection info
# POSTGRES_ADDRESS = secrets.psql_ad
# POSTGRES_PORT = secrets.psql_port
# POSTGRES_USERNAME = secrets.psql_username
# POSTGRES_DBNAME = secrets.psql_db
# POSTGRES_PASSWORD = secrets.psql_pw

# # Form string
# postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
#                 .format(username=POSTGRES_USERNAME, 
#                         password=POSTGRES_PASSWORD, 
#                         ipaddress=POSTGRES_ADDRESS, 
#                         port=POSTGRES_PORT, 
#                         dbname=POSTGRES_DBNAME)) 

# # Make connection
# cnx = sqlalchemy.create_engine(postgres_str)
# companies = pd.read_sql_query('''SELECT * from casey;''', cnx)

### UNCOMMENT BELOW TO LOAD FROM FILE ###

companies = pd.read_csv('../input/Financial Distress.csv')
companies.rename(index=str, columns={"Company": "company", "Time": "time", "Financial Distress": "financial_distress"}, inplace=True)
# Take a look at our loaded data to ensure all is in order
companies.head()

# Print some summaries and checks

 # shape
print(companies.shape)

# dtypes
print(companies.iloc[:5,:5].dtypes)

# check for nulls
print(companies.iloc[:5,:5].isnull().any())

# Describe
print(companies.describe(percentiles=[0.25,0.5,0.75,0.99]))
total_n = len(companies.groupby('company')['company'].nunique())
print(total_n)

distress_companies = companies[companies['financial_distress'] < -0.5]
u_distress = distress_companies['company'].unique()
print(u_distress.shape)

feature_names = list(companies.columns.values)[3:] # ignore first 3: company, time, financial_distress
print(feature_names)
f80 = list(companies.groupby('company')['x80'].agg('mean'))
f80 = [int(c) for c in f80]

# print(f80)
# print(len(f80))
companies.hist(column=['time'], bins=14)
# We can see from this that most companies start at time period 1, 
# but there are some which start their life much later.

# print(companies.groupby(['company'])['time'].agg('min'))
# What about the histogram of the timestamps when the distress event occurs?
distress_companies.hist(column=['time'], bins=14)
# Generate new train/val/test sets.

# Populate the entire pandas array into a dict for easier processing

datadict = {}
distress_dict = {}

for i in range (1, total_n+1):
    datadict[i] = {}
    distress_dict[i] = {}

print("Populating dictionary...")
for idx, row in companies.iterrows():
    company = row['company']
    time = int(row['time'])
    
    datadict[company][time] = {}
    
    if row['financial_distress'] < -0.5:
        distress_dict[company][time] = 1
    else:
        distress_dict[company][time] = 0
        
    for feat_idx, column in enumerate(row[3:]):
        feat = feature_names[feat_idx]
        datadict[company][time][feat] = column
        
# print('Dict population complete. Sample below:')
# print("\nData for company 1, time 1:")
# print(datadict[1][1])

# print("\nDistress history for company 1:")
# print(distress_dict[1])

print('We can encode categorical feature 80 as a one-hot vector with this many dimensions:')
print(len(list(set(f80))))

label_binarizer = LabelBinarizer()
label_binarizer.fit(range(max(f80)))
f80_oh = label_binarizer.transform(f80)

# print(f80_oh[0:5])
# Make new features as np array. We'll even add x80 back!

def rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods):

    for company in range(1, total_n+1):
            
            all_periods_exist = True
            for j in range(0, lookback_periods):
                if not time-j in distress_dict[company]:
                    all_periods_exist = False
            if not all_periods_exist:
                continue
            
            distress_at_eop = distress_dict[company][time]
            new_row = [company]

            for feature in feature_names:
                if feature == 'x80':
                    continue
                feat_sum = 0.0
                variance_arr = []
                for j in range(0, lookback_periods):
                    feat_sum += datadict[company][time-j][feature]
                    variance_arr.append(datadict[company][time-j][feature])
                new_row.append(feat_sum)
                new_row.append(np.var(variance_arr))
                
            for j in range(0,len(f80_oh[0])):
                new_row.append(f80_oh[company-1][j])

            if len(new_row) == ((len(feature_names)-1)*2 + 1 + len(f80_oh[0])) : # we have a complete row
                new_row.append(distress_at_eop)
                new_row_np = np.asarray(new_row)
                train_array.append(new_row_np)
    

def custom_timeseries_cv(datadict, distress_dict, feature_names, total_n, val_time, test_time, 
                         lookback_periods, total_periods=14):

    # Train data
    train_array = []
    for _t in range(1, val_time+1):
        time = (val_time+1) -_t # Start from time period 10 and work backwards
        train_array_np = rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    train_array_np = np.asarray(train_array)
    print(train_array_np.shape)
    # print(train_array_np[0])
    
    # Val data
    if val_time != test_time:
        val_array = []
        for time in range(val_time+1, test_time+1):
            val_array_np = rolling_operation(time, val_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

        val_array_np = np.asarray(val_array)
        print(val_array_np.shape)
        # print(val_array_np[0])
    else:
        val_array_np = None

    # Test data
    test_array = []
    # start from time period 11 and work forwards
    for time in range(test_time+1,total_periods+1):
        test_array_np = rolling_operation(time, test_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    test_array_np = np.asarray(test_array)
    print(test_array_np.shape)
    # print(test_array_np[0])
    
    return train_array_np, val_array_np, test_array_np

# Generate our sets
train_array_np, val_array_np, test_array_np = custom_timeseries_cv(datadict, distress_dict, feature_names, total_n,
                                                     val_time=9, test_time=12, lookback_periods=3, total_periods=14)
X_train = train_array_np[:,0:train_array_np.shape[1]-1]
y_train = train_array_np[:,-1].astype(int)

X_val = val_array_np[:,0:val_array_np.shape[1]-1]
y_val = val_array_np[:,-1].astype(int)

X_test = test_array_np[:,0:test_array_np.shape[1]-1]
y_test = test_array_np[:,-1].astype(int)

np.set_printoptions(threshold=sys.maxsize)
print(X_train[0,:])
print(y_train)

print(X_val[0,:])
print(y_val)

print(X_test[0,:])
print(y_test)
# Try a couple of different basic classification models

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def model_trial(model_type, hyperparam):
    if model_type in ['logistic-regression']:
        # Logistic Regression. Try 11, l2 penalty, understand one-vs-rest vs multinomial (cross-entropy) 
        model = LogisticRegression(penalty=hyperparam, solver='saga', max_iter=4000)
    elif model_type in ['decision-tree']:
        model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)
    elif model_type in ['random-forest']:
        model = RandomForestClassifier(n_estimators=hyperparam)
    else:
        print("Warning: model {} not recognized.".format(model_type))
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    print("Mean acc: %f" % model.score(X_val, y_val))
    print("F1: %f" % f1)
    print("Recall: %f" % recall)
print("-"*20 + "Logistic regression, l1:" + "-"*20)
model_trial('logistic-regression', 'l1')

print("-"*20 + "Logistic regression, l2:" + "-"*20)
model_trial('logistic-regression', 'l2')

print("-"*20 + "Decision tree:" + "-"*20)
model_trial('decision-tree', None)

for i in [2, 4, 10, 50, 100, 1000]:
    print("-"*20 + "Random forest, {} estimators:".format(i) + "-"*20)
    model_trial('random-forest', i)

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
knn = KNeighborsClassifier(n_neighbors=15)
clf = knn.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_knb_model=roc_auc_score(y_test, y_pred)*100
acc_knb_model
lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
acc_log_reg=roc_auc_score(y_test, y_pred1)*100
acc_log_reg
clf2 = GaussianNB().fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
acc_nb=roc_auc_score(y_test, y_pred2)*100
acc_nb
clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
acc_dt=roc_auc_score(y_test, y_pred3)*100
acc_dt
clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
y_pred4 = clf4.predict(X_test)
acc_rmf_model=roc_auc_score(y_test, y_pred4)*100
acc_rmf_model
clf5 = SVC(gamma='auto').fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
acc_svm_model=roc_auc_score(y_test, y_pred5)*100
acc_svm_model
sgd_model=SGDClassifier()
sgd_model.fit(X_train,y_train)
sgd_pred=sgd_model.predict(X_test)
acc_sgd=round(sgd_model.score(X_train,y_train)*100,10)
acc_sgd
xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
xgb_pred=xgb_model.predict(X_test)
acc_xgb=round(xgb_model.score(X_train,y_train)*100,10)
acc_xgb
lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)
lgbm_pred=lgbm.predict(X_test)
acc_lgbm=round(lgbm.score(X_train,y_train)*100,10)
acc_lgbm
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
regr_pred=regr.predict(X_test)
acc_regr=round(regr.score(X_train,y_train)*100,10)
acc_regr
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Linear Regression','Naive Bayes','XGBoost','LightGBM','Decision Tree'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd,acc_regr,acc_nb,acc_xgb,acc_lgbm,acc_dt]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df