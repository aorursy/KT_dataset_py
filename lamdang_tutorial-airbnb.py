import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import FunctionTransformer, Imputer, OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
def load_data(load_age_gender=False, load_sessions=False):
    if load_age_gender:
        age_gender = pd.read_csv('../input/age_gender_bkts.csv')
    else:
        age_gender = None
    if load_sessions:
        sessions = pd.read_csv('../input/sessions.csv')
    else:
        sessions = None
    data = pd.read_csv('../input/train_users_2.csv')    
    data.drop(['id', 'date_first_booking'], axis=1, inplace=True)
    return data, age_gender, sessions
data, _, _ = load_data()
data.shape
COLUMNS_FORMAT=dict(
    category_columns = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
                        'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
                        'first_device_type', 'first_browser','country_destination'],
    numeric_columns = ['age'],
    date_columns = ['date_account_created','timestamp_first_active']
)
data['date_account_created'] = pd.to_datetime(data.date_account_created,
                                                format='%Y-%m-%d')
data['timestamp_first_active'] = pd.to_datetime(data.timestamp_first_active, 
                                                  format='%Y%m%d%H%M%S')
for c in COLUMNS_FORMAT['category_columns']:
    data[c] = data[c].astype('str')
data.shape[0] - data.count()
### Write your code here
data.shape[0] - data.count()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(data.gender)
encoder.classes_
encoder.transform(data.gender)
def create_label_encoder(df, columns_list):
    label_encoders={} # this dictionary will store the label encoder object
    for c in columns_list:
        # initiate an LabelEncoder object and fit it to column c
        
        
        # store the fitted object in the dictionary
        # label_encoders[c] = 
        pass
    return label_encoders
def apply_label_encoder(df, label_encoders):
    for c in label_encoders.keys():
        df[c] = label_encoders[c].transform(df[c])
    return df
label_encoders = create_label_encoder(data, COLUMNS_FORMAT['category_columns'])
data = apply_label_encoder(data, label_encoders)
data.head()
(data.timestamp_first_active - data.date_account_created).head()
(data.timestamp_first_active - data.date_account_created).dt.days.head()
### Write your code here
data.head()


from sklearn.model_selection import train_test_split
split = StratifiedShuffleSplit(data.country_destination, n_iter=1, test_size=0.2)
train_ind, test_ind = train_test_split(data.index, stratify=data.country_destination, 
                                       test_size=0.2, random_state=1234)
len(train_ind), len(test_ind)
X_train = data.iloc[train_ind].drop('country_destination', axis=1)
y_train = data.iloc[train_ind]['country_destination']
X_test = data.iloc[test_ind].drop('country_destination', axis=1)
y_test = data.iloc[test_ind]['country_destination']
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def test_model(model, X_test, y_test):
    p_test = model.predict_proba(X_test)
    return accuracy_score(y_test, p_test.argmax(axis=1))
model = LogisticRegression(penalty='l2', C=1.0, n_jobs=4)
model.fit(X_train, y_train)
test_model(model, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, n_jobs=4, min_samples_leaf=10)
rf.fit(X_train, y_train)
test_model(rf, X_test, y_test)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
params_grid = {
    'C': [10, 1, 1e-1, 1e-2, 1e-3]
}
search_ = GridSearchCV(model, params_grid, n_jobs=4, verbose=1, cv=5)
search_.fit(X_train, y_train)
search_.best_score_
search_.best_params_
search_.best_estimator_.score(X_test, y_test)
rf_grid = {
    'n_estimators':[300], 
    'min_samples_leaf':[1, 5, 10, 20],
    'max_depth':[3,5,9, None],
}
search_rf = RandomizedSearchCV(rf, rf_grid, n_iter=5, n_jobs=4, cv=5)
search_rf.fit(X_train, y_train)
search_rf.grid_scores_
search_rf.best_estimator_.score(X_test, y_test)
test_model(search_rf.best_estimator_, X_test, y_test)
def format_columns(df, column_format):
    res = df.copy()
    res['date_account_created'] = pd.to_datetime(res.date_account_created,
                                                format='%Y-%m-%d')
    res['timestamp_first_active'] = pd.to_datetime(res.timestamp_first_active, 
                                                  format='%Y%m%d%H%M%S')
    for c in column_format['category_columns']:
        res.loc[:, c] = res.loc[:, c].astype('str')
    return res
def fill_missing_values(df):
    res = df.copy()
    res['age_missing'] = res.age.isnull().astype('int')
    res['age'] = res.age.fillna(df.age.median())
    return res
def transform_dates(df):
    res = df.copy()
    ### Create an "activation_delay" as difference in days between first active time and account creation
    res['activation_delay'] = (res.timestamp_first_active - res.date_account_created).dt.days
    ### Create an "date_creation_float" as difference in days between account creation and an arbitrary date
    res['date_creation_float'] = (res.date_account_created - pd.to_datetime('2010-01-01')).dt.days
    res.drop(['timestamp_first_active', 'date_account_created'], axis=1, inplace=True)
    return res
def pipeline(raw_data, label_encoders, model):
    data = format_columns(raw_data, COLUMNS_FORMAT)
    data = fill_missing_values(data)
    data = apply_label_encoder(data, label_encoders)
    data = transform_dates(data)
    X = data.drop('country_destination', axis=1)
    
    country_names = label_encoders['country_destination'].classes_
    prediction = pd.DataFrame(model.predict_proba(X), columns=country_names)
    return prediction
raw_data, _, _ = load_data()
raw_data = raw_data.iloc[test_ind]
p_test = pipeline(raw_data, label_encoders, rf)
p_test.head()
accuracy_score(y_test, np.argmax(p_test.values, axis=1))
def permutation_importance(predict_function, X, y, loss_function):
    baseline = loss_function(y, predict_function(X))
    feature_list = X.columns
    importance={}
    for i, feature_name in enumerate(feature_list):
        X_permute = X.copy()
        X_permute[feature_name] = np.random.permutation(X_permute[feature_name])
        importance[feature_name] = loss_function(y, predict_function(X_permute)) - baseline
    return pd.Series(importance, name='permutation_importance')
def classif_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, np.argmax(y_pred.values, axis=1))
def predict_function(X):
    return pipeline(X, label_encoders=label_encoders, model=rf)
feature_importance = permutation_importance(predict_function, raw_data, y_test, classif_error)
feature_importance.sort_values().plot(kind='barh')
def partial_dependence(X, column, values, predict_function):
    result = {}
    for v in values:
        X_copy = X.copy()
        X_copy[column] = v
        result[v] = predict_function(X_copy).mean(axis=0)
    return pd.DataFrame(result).T
age_steps = [raw_data.age.dropna().quantile(q) for q in np.arange(0.1, 1, 0.05)]
age_steps
pd_age = partial_dependence(raw_data, 'age', age_steps, predict_function)
for c in pd_age.columns:
    pd_age[c].plot(title=c)
    plt.show()
signup_method_val = raw_data.signup_method.unique()
signup_method_val
pd_signup_method = partial_dependence(raw_data, 'signup_method', signup_method_val, predict_function)
for c in pd_signup_method.columns:
    pd_signup_method[c].plot(kind='bar',title=c)
    plt.show()