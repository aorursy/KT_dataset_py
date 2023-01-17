import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
data.head()
data.info(), len(data.columns)
np.abs(data.corr()['is_canceled']).sort_values(ascending=False)
data.isna().sum()
data = data.drop(labels = ['country', 'agent', 'company'], axis =1)
data.isna().sum()
# split the data to train-val-test
train_val, test = train_test_split(data, train_size=0.9, test_size=0.1, random_state=42)
train,val = train_test_split(train_val, train_size = 0.89, test_size=0.11, random_state=42)
len(data), len(train), len(val), len(test), len(train)/len(data)
# Check most correlated features
most_corr = np.abs(train.corr()['is_canceled']).sort_values(ascending=False)[1:7]
most_corr
#sns.pairplot(train[most_corr.index])
train['is_canceled'].value_counts(normalize = True)
train['lead_time'].plot(kind='kde')
train['lead_time'].quantile(np.arange(0.1,1,0.1))
#np.arange(0.1,1,0.1)
qlt = pd.qcut(train['lead_time'],10, labels=False)
qlt_df=pd.concat([qlt, train['is_canceled']], axis=1)
qlt_df
sns.barplot(x= 'lead_time', y='is_canceled', data=qlt_df, )
qlt_df['is_canceled'].groupby(qlt_df['lead_time']).value_counts(normalize=True)
qlt_df['lead_time'].groupby(qlt_df['is_canceled']).value_counts(normalize=True)
qlt_df['lead_time'].groupby(qlt_df['is_canceled']).value_counts(normalize=True)[1].plot(kind='bar')
df_prediction = pd.DataFrame(columns = ['Score'])
qlt_df['Decision'] = 0
#cond1  = (qlt_df['lead_time']==(11.0, 26.0]) | (qlt_df['lead_time']==(2.0, 11.0]) | ( qlt_df['lead_time']==(-0.001, 2.0])
qlt_df.loc[qlt_df['lead_time']>=7, 'Decision'] = 1
qlt_df
qlt_df['Decision'].value_counts(normalize=True)
def print_score(prediction,validation,name):
    acc = np.round(accuracy_score(validation,prediction) * 100, 2)
    print('Accuracy Score : ',acc)
    df_prediction.loc[name,'Score'] = acc
    cm_norm = confusion_matrix(prediction,validation)/(confusion_matrix(prediction,validation).sum())
    return sns.heatmap(cm_norm, cmap='hot', annot=True)
print_score(qlt_df['Decision'],qlt_df['is_canceled'],'Top 1 Numeric Feat')
train_5 = train[most_corr.index.values]
train_5.head()
train_5.info()
val_5 = val[most_corr.index.values]
val_5.info()
target_5 = train['is_canceled']
target_val_5 = val['is_canceled']
rf_5 = RandomForestClassifier()
rf_5.fit(train_5, target_5)
rf5_pred = rf_5.predict(val_5)
print_score(rf5_pred,target_val_5,'Top 5 Numeric Feat')
train['total_of_special_requests'].value_counts(normalize=True).plot(kind='bar')
train['is_canceled'].groupby(train['total_of_special_requests']).value_counts(normalize=True)
sns.barplot(x='total_of_special_requests', y='is_canceled', data=train)
qlt_df['total_of_special_requests']=train['total_of_special_requests']
qlt_df.pivot_table(index='lead_time', columns='total_of_special_requests', values='is_canceled')
train['required_car_parking_spaces'].value_counts(normalize=True)
train['required_car_parking_spaces'].value_counts()
# it is obvious that more than two cars are rare and stated in a different category, 
#therefore we will change the classification of this feature, not enough data for 3 cars or more
train.loc[train['required_car_parking_spaces']>1,'required_car_parking_spaces']=2
val.loc[val['required_car_parking_spaces']>1,'required_car_parking_spaces']=2
test.loc[test['required_car_parking_spaces']>1,'required_car_parking_spaces']=2
train['required_car_parking_spaces'].value_counts()
train['is_canceled'].groupby(train['required_car_parking_spaces']).value_counts(normalize=True)
sns.barplot(x = 'required_car_parking_spaces',y ='is_canceled', data=train)
np.abs(train.corr()['required_car_parking_spaces']).sort_values(ascending=False)[1:7]
train['hotel'].groupby(train['required_car_parking_spaces']).value_counts()
train['required_car_parking_spaces'].groupby(train['hotel']).value_counts(normalize=True)
sns.barplot(x='required_car_parking_spaces', y='hotel', data=train)
# Checking if cancelation is correlated to the type of hotel, regardless to parking spaces
train['is_canceled'].groupby(train['hotel']).value_counts(normalize=True)
train['booking_changes'].value_counts(normalize=True).plot(kind='bar')
train['booking_changes'].value_counts(),train['booking_changes'].value_counts(normalize=True)
train.loc[train['booking_changes']>5,'booking_changes']=6
val.loc[val['booking_changes']>5,'booking_changes']=6
test.loc[test['booking_changes']>5,'booking_changes']=6
train['booking_changes'].value_counts(normalize=True)
train['is_canceled'].groupby(train['booking_changes']).value_counts(normalize=True)

sns.barplot(x='booking_changes',y='is_canceled',data=train)
train['is_change']=0
train.loc[train['booking_changes']>0,'is_change']=1
train[['is_change','booking_changes']][10:30]
train.corr()[['is_change','booking_changes']]
np.abs(train.corr()['is_change']-train.corr()['booking_changes']).sort_values(ascending=False)[2:]
# Replace booking_changes by is_change and apply changes to validation and test sets
val['is_change']=0
test['is_change']=0
val.loc[val['booking_changes']>0,'is_change']=1
test.loc[test['booking_changes']>0,'is_change']=1

train = train.drop('booking_changes',axis=1)
val = val.drop('booking_changes',axis=1)
test = test.drop('booking_changes',axis=1)

train.head()
train.corr()['is_change'].sort_values(ascending=False)[1:7]
train.corr()['is_change'].sort_values(ascending=True)[:7]
train.pivot_table(index='is_change', columns='babies',values='is_canceled')
train['previous_cancellations'].value_counts().sort_index()
train['previous_cancellations'].value_counts().sort_index()[2:].plot()
train['previous_cancellations'].value_counts().sort_index()[3:].plot(kind='bar')
prc = train['previous_cancellations']
sns.barplot(x='previous_cancellations',y='is_canceled',data=train)
np.abs(train.corr()['previous_cancellations']).sort_values(ascending=False)[1:]
train['previous_bookings_not_canceled'].value_counts()
# we will create a new feature of the cancelation percentage
total_canc = train['previous_bookings_not_canceled'] + train['previous_cancellations']
train['previous_cancellation_per'] =  train['previous_cancellations'].div(total_canc)
train['previous_cancellation_per'] = train['previous_cancellation_per'].fillna(0)
train[['previous_cancellations','previous_bookings_not_canceled','previous_cancellation_per']]
train['previous_cancellation_per'].value_counts()
bins = [-1,0,0.5,1]
train['previous_cancellation_per'] =  pd.cut(train['previous_cancellation_per'], bins, labels = [0,1,2])
sns.barplot(x='previous_cancellation_per',y='is_canceled',data=train)
train['previous_cancellation_per'] = train['previous_cancellation_per'].astype('int64')
train.info()
np.abs(train.corr()[['previous_cancellation_per','previous_cancellations']]).sort_values(ascending=False,by='previous_cancellation_per')
train[['previous_cancellation_per','previous_cancellations','previous_bookings_not_canceled','is_canceled']].corr()
def add_previous_cancelation_per(df):
    total_canc = df['previous_bookings_not_canceled'] + df['previous_cancellations']
    df['previous_cancellation_per'] =  df['previous_cancellations'].div(total_canc)
    df['previous_cancellation_per'] = df['previous_cancellation_per'].fillna(0)
    bins = [-1,0,0.5,1]
    df['previous_cancellation_per'] =  pd.cut(df['previous_cancellation_per'], bins, labels = [0,1,2])
    df['previous_cancellation_per'] = df['previous_cancellation_per'].astype('int64')
    df = df.drop('previous_cancellations',axis=1)
    return df

val = add_previous_cancelation_per(val)
test = add_previous_cancelation_per(test)
train = train.drop('previous_cancellations',axis=1)
val['previous_cancellation_per'].value_counts()
train['is_repeated_guest'].value_counts(normalize=True)
train['is_canceled'].groupby(train['is_repeated_guest']).value_counts(normalize=True)
sns.barplot(x='is_repeated_guest',y='is_canceled',data=train)
num_feat = list(train.columns[(train.dtypes.values=='int64')|(train.dtypes.values=='float64')])
non_num_feat = list(train.columns[(train.dtypes.values!='int64')&(train.dtypes.values!='float64')])
num_feat, non_num_feat
if 'is_canceled' in num_feat:
    num_feat.remove('is_canceled')
if 'is_change' in num_feat:
    num_feat.remove('is_change')
if 'previous_cancellation_per' in num_feat:
    num_feat.remove('previous_cancellation_per')
if 'previous_bookings_not_canceled' in num_feat:
    num_feat.remove('previous_bookings_not_canceled')
    
for item in list(most_corr.index):
    if item in num_feat:
        num_feat.remove(item)
num_feat
train[num_feat]
train[num_feat].isna().sum()
np.abs(train.corr()['children']).sort_values(ascending=False)[1:6]
train['adr'].groupby(train['children']).median()
sns.barplot(x='children', y='adr', data=train)
train.loc[train['children'].isnull(),'children']=0
train.isna().sum()
cat_flag = False
train[non_num_feat].head()
# Create a dataframe of all the unique categories
cat_df = pd.DataFrame(index = non_num_feat, columns = ['Unique Values', 'Number of Categories'])
for feature in non_num_feat:
    cat_df.loc[feature,'Unique Values'] = train[feature].unique()
    cat_df.loc[feature,'Number of Categories'] = len(train[feature].unique())
cat_df.drop('reservation_status_date', axis=0, inplace=True)
cat_df
train['deposit_type'].value_counts()
sns.barplot(x='deposit_type',y='is_canceled', data = train)
train['reservation_status'].value_counts()
sns.barplot(x='reservation_status', y='is_canceled', data = train)
def dummies(df,var,prefix=None):
    dummies = pd.get_dummies(df[var], prefix = prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(var, axis=1)
    return df

def set_cat_feat(df):

    #Ordinal/binary parameters   
    month = {
        'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,\
        'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
    hotel = { 'Resort Hotel' : 0, 'City Hotel' : 1}
    df['arrival_date_month'] = df['arrival_date_month'].map(month)
    df['hotel'] = df['hotel'].map(hotel)
    
    df['assigned/reserved'] = 0
    df.loc[df['reserved_room_type']==df['assigned_room_type'],'assigned/reserved']=1
    df = df.drop('reserved_room_type', axis=1)
    df = df.drop('assigned_room_type', axis =1)
    
    df = df.drop('reservation_status', axis=1)
    
    dummy_feat = ['meal','market_segment','distribution_channel','customer_type','deposit_type']
    dummy_prefix=['meal','MS','DC','CT','DT']
    
    # get_dummies for parameters with a few categories, as we face classification problem no need to drop any category
    for i in range(len(dummy_feat)):
        df = dummies(df,dummy_feat[i],dummy_prefix[i])

    return df
temp = train.loc[:,non_num_feat+ ['is_canceled']]
temp
# Apply changes in all the datasets
if cat_flag == False:
    print('Perform encoding...')
    train = set_cat_feat(train)
    val = set_cat_feat(val)
    test = set_cat_feat(test)
    cat_flag=True

train.head(10)
train.info()
object_list = train.dtypes.index[train.dtypes=='object'].values
object_list
object_feat = object_list[0] # reservation_status_date
minibatch = train.loc[:,[object_feat]]
minibatch
minibatch[['res_year','res_month','res_day']] = minibatch[object_feat].str.split('-', expand=True).astype(int)
minibatch
minibatch['res_year'].value_counts(normalize=True)
train['arrival_date_year' ].value_counts(normalize=True)
def adjust_dates(df):
    object_feat = 'reservation_status_date'
    df[['res_year','res_month','res_day']] = df[object_feat].str.split('-', expand=True).astype(int)
    df = df.drop(object_feat, axis = 1)
    years_dict = {2014:1,2015:2,2016:3,2017:4}
    df['res_year'] = df['res_year'].map(years_dict)
    df['arrival_date_year'] = df['arrival_date_year'].map(years_dict)
    
    return df
train = adjust_dates(train)
val = adjust_dates(val)
test = adjust_dates(test) 
train.info()
missing_test = train.columns.values.tolist()
for i in train.columns.values:
    for j in test.columns.values:
        if i==j:
            missing_test.remove(i)
missing_val = train.columns.values.tolist()
for i in train.columns.values:
    for j in val.columns.values:
        if i==j:
            missing_val.remove(i)
missing_test, missing_val
for col in missing_val:
    val[col]=0
for col in missing_test:
    test[col]=0

test.head()
# Resort the columns
train = train.reindex(sorted(train.columns), axis=1)
val = val.reindex(sorted(train.columns), axis=1)
test = test.reindex(sorted(train.columns), axis=1)
train.head()
val.head()
np.abs(train.corr()['is_canceled']).sort_values(ascending = False)
df_prediction
score_deposit = np.round((np.sum(train['DT_Non Refund']==train['is_canceled'])/len(train))*100,2)
df_prediction.loc['Top-1 Cat. Feat','Score']=score_deposit
score_deposit
#feat_to_remove = train.columns[["RS" in x for x in train.columns]].values
train.columns.shape, val.columns.shape, test.columns.shape
def get_Xy(df,target):
    X = df.drop(target, axis=1) 
    y = df[target]
    return X,y
X_train, y_train = get_Xy(train,'is_canceled')
print('X dim = {}, y dim = {}'.format(X_train.shape, y_train.shape))
print(y_train[:5])
X_test, y_test = get_Xy(test, 'is_canceled')
X_val, y_val = get_Xy(val, 'is_canceled')
# Normalize input matrix
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)
X_test.shape, X_val.shape
# we will start by running Random Forest with default hypreparameters
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
print_score(rf_pred, y_val, 'Random Forest')
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_val)
print_score(gb_pred, y_val, 'Gradient Boosting')
log = LogisticRegression()
log.fit(X_train, y_train)
log_pred = log.predict(X_val)
print_score(log_pred, y_val, 'Logistic Regression')
from sklearn.decomposition import PCA

pca = PCA(n_components = None)
Xp_train = pca.fit(X_train)

target_var = 0.99
explained_variance = pca.explained_variance_ratio_
ev_curve = np.cumsum(explained_variance)
plt.plot(ev_curve)
plt.plot(np.arange(len(explained_variance)),np.ones(len(explained_variance))*target_var, color='red')
n_components = np.min(np.where(ev_curve>target_var))
n_components
pca = PCA(n_components = n_components)
Xp_train = pca.fit_transform(X_train)
Xp_test = pca.transform(X_test)
Xp_val = pca.transform(X_val)
rfp = RandomForestClassifier()
rfp.fit(Xp_train, y_train)
rfp_pred = rfp.predict(Xp_val)
print_score(rfp_pred, y_val, 'Random Forest (PCA)')
log_p = LogisticRegression()
log_p.fit(Xp_train, y_train)
log_p_pred = log_p.predict(Xp_val)
print_score(log_p_pred, y_val, 'Logistic Regression (PCA)')
rf_model = RandomForestClassifier()
#Run a gridsearch
rf_params = {"max_depth": [10,20,30,40],
            "max_features": [10,20,35],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}
            
rf_val = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 5, 
                           n_jobs = -1, 
                           verbose = 2) 

rf_val.fit(X_val, y_val)
rf_val.best_params_
rf_tuned = RandomForestClassifier(max_depth = rf_val.best_params_.get('max_depth'), 
                                  max_features = rf_val.best_params_.get('max_features'), 
                                  min_samples_split = rf_val.best_params_.get('min_samples_split'),
                                  n_estimators = rf_val.best_params_.get('n_estimators'))

rf_tuned.fit(X_train, y_train)
#Evaluation on Test set
rft_pred = rf_tuned.predict(X_test)
print_score(rft_pred,y_test,'Random Forest (tuned)')
C = np.logspace(2, 8, 4)
penalty = ['l1', 'l2']
max_iter = [100, 200, 500]
#log_params = dict(C=C, penalty=penalty, max_iter=max_iter) 
log_params = dict(C=C, penalty=['l2'], solver = ['lbfgs'], max_iter = max_iter) 
log_params
log_model = LogisticRegression()
#Run a gridsearch  
#log_val = GridSearchCV(log_model, log_params, cv=5, verbose=0)
log_val = GridSearchCV(log_model, log_params, cv=5, verbose=0)

log_val.fit(X_val, y_val)
log_val.best_params_
log_val.best_params_.get('C')
log_tuned = LogisticRegression(C=log_val.best_params_.get('C'), max_iter=log_val.best_params_.get('max_iter'), solver= 'lbfgs')

log_tuned.fit(X_train, y_train)
#Evaluation on Test set
log_t_pred = log_tuned.predict(X_test)
print_score(log_t_pred,y_test,'Logistic Regression (tuned)')
df_prediction
df_prediction.plot(kind='bar')
Importance = pd.DataFrame( {"Importance": rf_tuned.feature_importances_*100},
                         index = train.drop('is_canceled',axis=1).columns)
Importance.sort_values(by = "Importance", axis = 0, ascending = False)[:10].plot(kind ="barh", color = "r")

plt.xlabel("Variable Importance Level")
wrong = []
for i in np.arange(len(y_test)):
    if y_test.iloc[i]!=log_t_pred[i]:
        wrong.append(i)
feat_ranked = Importance.sort_values(by = "Importance", axis = 0, ascending = True).index.values
#pd.concat([test[feat_ranked],log_t_pred],axis=1)
df_wrong = test[feat_ranked].iloc[wrong]
df_wrong['Prediction'] = log_t_pred[wrong]
df_wrong['Target'] = test['is_canceled'].iloc[wrong]
df_wrong
