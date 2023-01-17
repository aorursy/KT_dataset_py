!pip freeze | grep pandas
import pandas as pd
import numpy as np
import time
import pandas_profiling
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import seaborn as sns
df = pd.read_csv("../input/flight-delays/flights.csv")                  # Reading the dataset
df.head()
df.describe()
df.info()
df.shape
# Selecting important features

df = df[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT", "ORIGIN_AIRPORT", 
         "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY", 
         "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY", "AIR_TIME", "DISTANCE"]]
df = df.sample(n=10000, random_state= 10, axis=0)
df.shape
report = pandas_profiling.ProfileReport(df)
report.to_file('flight_df.html')
from IPython.display import display, HTML, IFrame
display(HTML(open('flight_df.html').read()))
# Origin and Destination airport has few values which are numeric

# Making a function to replace all numerical values in origin and destination airport feature with np.nan
def Replace(i):
    try:
      if str(i).isalpha():
        return str(i)
    except:
      i == np.nan
      return i

# Applying function to replace
df['DESTINATION_AIRPORT'] = df['DESTINATION_AIRPORT'].apply(func=Replace)
df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].apply(func=Replace)
df.isna().sum()
# Dropping all NAN missing values
df.dropna(inplace=True)
df.shape
df.head()
df_delay = df[df.DEPARTURE_DELAY >= 1]
dep_delayed_flights = df_delay.groupby(['AIRLINE'], as_index=False).agg({'DEPARTURE_DELAY': 'mean'})

f,ax = plt.subplots(figsize=(10, 8))
sns.barplot('AIRLINE','DEPARTURE_DELAY', data=dep_delayed_flights ,ax=ax)
ax.set_title('Airline Departure Delay Distribution', fontsize=16)
ax.set_ylabel("Departure Delay", fontsize=16)
ax.set_xlabel("Airlines", fontsize=16)
plt.close(2)
plt.show()
df_delay1 = df[df.ARRIVAL_DELAY >= 1]
dep_delayed_flights = df_delay.groupby(['AIRLINE'], as_index=False).agg({'ARRIVAL_DELAY': 'mean'})

f,ax = plt.subplots(figsize=(10, 8))
sns.barplot('AIRLINE','ARRIVAL_DELAY', data=dep_delayed_flights ,ax=ax)
ax.set_title('Airline Arrival Delay Distribution', fontsize=16)
ax.set_ylabel("Arrival Delay", fontsize=16)
ax.set_xlabel("Airlines", fontsize=16)
plt.close(2)
plt.show()
# To find the max 10th departure delay
df.nlargest(10, 'DEPARTURE_DELAY')[9:]
# We see that the 10th larges value for Departure Delay is 429 minutes

dep_delay_airports = df[df['DEPARTURE_DELAY']>427][['ORIGIN_AIRPORT', 'DEPARTURE_DELAY']]

dep_delay_airports['ORIGIN_AIRPORT'] = dep_delay_airports['ORIGIN_AIRPORT'].astype('category')

f, ax= plt.subplots(figsize=(10, 6))
sns.barplot('ORIGIN_AIRPORT', 'DEPARTURE_DELAY', data=dep_delay_airports, ax=ax)
ax.set_title('Departure Delay Distribution of Origin Airports', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.close(2)
plt.show()

# To find the max 10th arrival delay
df.nlargest(10, 'ARRIVAL_DELAY')[9:]
# We see that the 10th larges value for Arrival Delay is 434 minutes

arr_delay_airports = df[df['ARRIVAL_DELAY']>427][['DESTINATION_AIRPORT', 'ARRIVAL_DELAY']]
arr_delay_airports['DESTINATION_AIRPORT'] = arr_delay_airports['DESTINATION_AIRPORT'].astype('category')


f, ax= plt.subplots(figsize=(10, 6))
sns.barplot('DESTINATION_AIRPORT', 'ARRIVAL_DELAY', data=arr_delay_airports, ax=ax, saturation=.8)
ax.set_title('Arrival Delay Distribution of Destination Airports', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.close(2)
plt.show()

f, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot('MONTH', "DEPARTURE_DELAY", data=df, size='DEPARTURE_DELAY', hue='AIRLINE', sizes=(50, 200))
plt.legend(bbox_to_anchor=(1.5,1) , loc='upper right')
f, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot('MONTH', "ARRIVAL_DELAY", data=df, size='ARRIVAL_DELAY', hue='AIRLINE', sizes=(50, 200))
plt.legend(bbox_to_anchor=(1.5,1) , loc='upper right')
arr_delay_flightnum = df[df['ARRIVAL_DELAY']>430][['FLIGHT_NUMBER', 'ARRIVAL_DELAY', 'AIRLINE']]
arr_delay_log = np.log(df['ARRIVAL_DELAY'])
f, ax = plt.subplots(figsize=(14, 8))
sns.barplot('FLIGHT_NUMBER', 'ARRIVAL_DELAY', data=arr_delay_flightnum, hue='AIRLINE')

ax.legend(bbox_to_anchor=(1, 1), loc='upper right')

# using labelencoding and give conditions to Arrival delay colum
df['ARRIVAL_DELAY'].value_counts()
df["ARRIVAL_DELAY"] = (df["ARRIVAL_DELAY"]>10)*1    # Checking if delay is greater than 10 mins
df['ARRIVAL_DELAY'].value_counts()
# So we see that 2033 fights in our sample data has arrival delay more than 10 minutes
df.head()
df.info()
# We have features like AIRLINE, DESTINATION_AIRPORT, ORIGIN_AIRPORT which are categorical data
# Hence convert them to category
# Categorical columns

cols = ["AIRLINE","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
for item in cols:
    df[item] = df[item].astype("category")

# Lets check data type again
df.info()
# Now lets LabelEncode the categorical features for Model building
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col = ['AIRLINE', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT']
le.fit(df[col].values.flatten())

df[col] = df[col].apply(le.fit_transform)
df.head()
X = df.drop('ARRIVAL_DELAY', 1)
y = df['ARRIVAL_DELAY']
X.head()
# Normalizing data X

from sklearn.preprocessing import StandardScaler

#Lets Use Sandardscaler to normalise the data
scaler = StandardScaler()
scaler.fit(X)

# Scale and center the data
X_norm = scaler.transform(X)

# Create a pandas DataFrame
X = pd.DataFrame(data=X_norm, index=X.index, columns=X.columns)

# Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.3)
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
# Function for model evaluation

def auc(m, X_train, X_test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(X_train)[:,1]),
            metrics.roc_auc_score(y_test,m.predict_proba(X_test)[:,1]))
# XGBoost Model
%time
model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,\
                          n_jobs=-1 , verbose=1, learning_rate=0.2)
model.fit(X_train, y_train)

auc(model, X_train, X_test)
y_pred = model.predict(X_test)
import matplotlib.pyplot as plt                               # Visualization package

%matplotlib inline
import seaborn as sns

print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))
print('')
print('********************************************')
print('Confusion matrix')
lr_cfm=metrics.confusion_matrix(y_test, y_pred)


lbl1=["Predicted 1", "Predicted 2"]
lbl2=["Actual 1", "Actual 2"]

sns.heatmap(lr_cfm, annot=True, cmap="Blues", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
plt.show()

print('**********************************************')
print(metrics.classification_report(y_test,y_pred))
import lightgbm as lgb  # ligther version of GBM 
# Function to evaluate LightGBM model

def auc2(m, X_train, X_test):
    y_train_pred = m.predict(X_train)
    y_test_pred = m.predict(X_test)

    return (print('ROC AUC Train Score: ', metrics.roc_auc_score(y_train, y_train_pred)),
    print('ROC AUC Test Score: ', metrics.roc_auc_score(y_test, y_test_pred)),
    print('Avg. Precision Score: ', metrics.average_precision_score(y_test, y_test_pred)),
    print('Confusion Metrics: \n', metrics.confusion_matrix(y_test, y_test_pred)))
def gini(y_test, y_test_pred):
    fpr, tpr, thr = metrics.roc_curve(y_test, y_pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y_test, y_test_pred,) / gini(y_test, y)
    return 'gini', score, True

%time
model2 = lgb.LGBMClassifier(n_estimators=90, 
                     silent=False, 
                     random_state =94, 
                     max_depth=5, 
                     num_leaves=30, 
                     objective='binary', 
                     metrics ='auc')

model2.fit(X_train, y_train, eval_metric=gini_lgb)
auc2(model2, X_train, X_test)
import matplotlib.pyplot as plt                               # Visualization package
y_test_pred = model2.predict(X_test)
%matplotlib inline
import seaborn as sns
print(metrics.accuracy_score(y_test,y_test_pred))
print('********************************************')
print('Confusion matrix')
lr_cfm=metrics.confusion_matrix(y_test, y_test_pred)


lbl1=["Predicted 1", "Predicted 2"]
lbl2=["Actual 1", "Actual 2"]

sns.heatmap(lr_cfm, annot=True, cmap="Blues", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
plt.show()

print('**********************************************')
print(metrics.classification_report(y_test,y_test_pred))
!pip install catboost
import catboost as cb
cat_features_index = [0,1,2,3,4,5,6]  # externally defines the category index 
clf = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
clf.fit(X_train,y_train)

auc2(clf, X_train, X_test)
import matplotlib.pyplot as plt                               # Visualization package
y_test_p = clf.predict(X_test)
%matplotlib inline
import seaborn as sns
print(metrics.accuracy_score(y_test,y_test_p))
print('********************************************')
print('Confusion matrix')
lr_cfm=metrics.confusion_matrix(y_test, y_test_p)


lbl1=["Predicted 1", "Predicted 2"]
lbl2=["Actual 1", "Actual 2"]

sns.heatmap(lr_cfm, annot=True, cmap="Blues", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
plt.show()

print('**********************************************')
print(metrics.classification_report(y_test, y_test_p))