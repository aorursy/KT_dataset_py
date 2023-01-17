import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# https://www.kaggle.com/satomacoto/airbnb-recruiting-new-user-bookings/script-0-1

#Loading data
df_train_raw = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train_raw['country_destination'].values
df_train = df_train_raw.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]
#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

#####Feature engineering#######
#date_account_created
df_all.date_account_created = pd.to_datetime(df_all.date_account_created)
df_all['dac_year'] = df_all.date_account_created.apply(lambda x: x.year)
df_all['dac_month'] = df_all.date_account_created.apply(lambda x: x.month)
df_all['dac_day'] = df_all.date_account_created.apply(lambda x: x.day)
df_all['dac_weekday'] = df_all.date_account_created.apply(lambda x: x.weekday())
df_all['dac_week'] = df_all.date_account_created.apply(lambda x: x.week)
df_all['dac_log_elapsed'] = np.log((datetime.date(2016, 1, 1) - df_all.date_account_created).astype('timedelta64[D]'))
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
df_all.timestamp_first_active = pd.to_datetime(df_all.timestamp_first_active, format='%Y%m%d%H%M%S')
df_all['tfa_year'] = df_all.timestamp_first_active.apply(lambda x: x.year)
df_all['tfa_month'] = df_all.timestamp_first_active.apply(lambda x: x.month)
df_all['tfa_day'] = df_all.timestamp_first_active.apply(lambda x: x.day)
df_all['tfa_weekday'] = df_all.timestamp_first_active.apply(lambda x: x.weekday())
df_all['tfa_week'] = df_all.timestamp_first_active.apply(lambda x: x.week)
df_all['tfa_log_elapsed'] = np.log((datetime.date(2016, 1, 1) - df_all.timestamp_first_active).astype('timedelta64[D]'))
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>90), -1, av)
df_all['age_year'] = np.where(av > 1900, -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
             'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type',
             'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X_train = vals[:piv_train]
le = LabelEncoder()
y_train = le.fit_transform(labels)
X_test = vals[piv_train:]

#Sampling
np.random.seed(42)
samples = np.random.choice(piv_train, 10000)
X_train = vals[samples]
y_train = le.fit_transform(labels)[samples]
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
model = DummyClassifier('prior')
cross_val_score(model, X_train, y_train)
model = DummyClassifier('stratified')
cross_val_score(model, X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
cross_val_score(model, X_train, y_train)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
cross_val_score(model, X_train, y_train)
from sklearn.ensemble import ExtraTreesClassifier
model = RandomForestClassifier()
cross_val_score(model, X_train, y_train)
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
cross_val_score(model, X_train, y_train)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
cross_val_score(model, X_train, y_train)
from sklearn.svm import SVC
model = SVC(kernel="linear", C=0.025)
cross_val_score(model, X_train, y_train)
model = SVC(gamma=2, C=1)
cross_val_score(model, X_train, y_train)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
cross_val_score(model, X_train, y_train)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(10, 10))
cross_val_score(model, X_train, y_train)
#from xgboost.sklearn import XGBClassifier
#model = XGBClassifier()
#cross_val_score(model, X_train, y_train)
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GradientBoostingClassifier(random_state=1)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Ensemble']):
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))