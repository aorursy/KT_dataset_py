# Importing relevant libraries 

import matplotlib

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (10, 6)

import pandas as pd

import numpy as np

import scipy.stats

import pylab as plt

from matplotlib.colors import LogNorm



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from sklearn.model_selection import GridSearchCV



import xgboost as xgb
data = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv',dtype={'PatientId': object})
data.head(10)
print('Number of items in data set',data.shape[0])

print('Number of unique patients', len(np.unique(data['PatientId'])))

print('Number of unique appointments', len(np.unique(data['AppointmentID'])),'same as number of datasets')

print('Gender Values:', np.unique(data['Gender']))

print('Gender distribution: Male', len(data[data['Gender']=='M']), 'Female', len(data[data['Gender']=='F']))

print('Age Values: Min',data['Age'].min(), 'Max', data['Age'].max())

print('Neighborhood Values:', len(np.unique(data['Neighbourhood'])))
data['AD_year'] = pd.to_datetime(data['DataAgendamento']).dt.year

data['AD_month'] = pd.to_datetime(data['DataAgendamento']).dt.month

data['AD_week'] = pd.to_datetime(data['DataAgendamento']).dt.week

data['AD_dow'] = pd.to_datetime(data['DataAgendamento']).dt.dayofweek

data['AD_hour'] = pd.to_datetime(data['DataAgendamento']).dt.hour

data['AD_min'] = pd.to_datetime(data['DataAgendamento']).dt.minute



data['SD_year'] = pd.to_datetime(data['DataMarcacaoConsulta']).dt.year

data['SD_month'] = pd.to_datetime(data['DataMarcacaoConsulta']).dt.month

data['SD_week'] = pd.to_datetime(data['DataMarcacaoConsulta']).dt.week

data['SD_dow'] = pd.to_datetime(data['DataMarcacaoConsulta']).dt.dayofweek

data['SD_hour'] = pd.to_datetime(data['DataMarcacaoConsulta']).dt.hour

data['SD_min'] = pd.to_datetime(data['DataMarcacaoConsulta']).dt.minute



data['AD_SD'] = (pd.to_datetime(data['DataAgendamento'])-pd.to_datetime(data['DataMarcacaoConsulta']))/np.timedelta64(1, 'D')
data['Gender'] = pd.Categorical(data['Gender'])

data['GenderB'] = data['Gender'].cat.codes

dGENDER = dict(enumerate(data['Gender'].cat.categories))
data['Neighbourhood'] = pd.Categorical(data['Neighbourhood'])

data['NeighborhoodC'] = data['Neighbourhood'].cat.codes

dNEIGHBORHOOD = dict(enumerate(data['Neighbourhood'].cat.categories))
data['No-show'] = pd.Categorical(data['No-show'])

data['StatusB'] = data['No-show'].cat.codes

dSTATUS = dict(enumerate(data['No-show'].cat.categories))
data.head()
fig, ax = plt.subplots(ncols=3)

ax[0].hist(data['AD_SD'],bins=100)

ax[0].set_xlabel('Time booking to appointment [D]')

ax[1].hist(data['AD_SD'],bins=100, range=(0,1))

ax[1].set_xlabel('Time booking to appointment [D]')

ax[2].hist(data['AD_SD'],bins=100, range=(-2,0))

ax[2].set_xlabel('Time booking to appointment [D]')

plt.show()
data.columns.values
features = [ 'Age', 'Scholarship',

       'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received',

        'AD_year', 'AD_month', 'AD_week', 'AD_dow', 'AD_hour',

       'AD_min', 'SD_year', 'SD_month', 'SD_week', 'SD_dow', 'SD_hour',

       'SD_min', 'AD_SD', 'GenderB', 'NeighborhoodC']
N_SAMPLE = 10000

X = data[features][:N_SAMPLE]

y = data['StatusB'][:N_SAMPLE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)

y_pred_rf = rfc.predict(X_test)

y_score_rf = rfc.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)

print('Random Forest ROC AUC', auc(fpr_rf, tpr_rf))

print(classification_report(y_test, y_pred_rf, target_names=['NoShow','ShowUp']))
gbc = GradientBoostingClassifier(n_estimators=1000)

gbc.fit(X_train,y_train)

y_pred_gb = gbc.predict(X_test)

y_score_gb = gbc.predict_proba(X_test)[:,1]

fpr_gb, tpr_gb, _ = roc_curve(y_test, y_score_gb)

print('Gradient Boosting ROC AUC', auc(fpr_gb, tpr_gb))

print(classification_report(y_test, y_pred_gb, target_names=['NoShow','ShowUp']))
xgbc = xgb.XGBClassifier()

xgbc.fit(X_train, y_train)

y_pred_xgb = xgbc.predict(X_test)

y_score_xgb = xgbc.predict_proba(X_test)[:,1]

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_score_xgb)

print('XGBoost ROC AUC', auc(fpr_xgb, tpr_xgb))

print(classification_report(y_test, y_pred_xgb, target_names=['NoShow','ShowUp']))
fig, ax = plt.subplots()

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_rf, tpr_rf, label='RF')

plt.plot(fpr_gb, tpr_gb, label='GB')

plt.plot(fpr_xgb, tpr_xgb, label='XGB')

ax.set_aspect('equal')

plt.legend()

plt.show()
model = xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.1)

model.fit(X_train, y_train)

y_pred_xgb = model.predict(X_test)

y_score_xgb = model.predict_proba(X_test)[:,1]

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_score_xgb)

print('XGBoost ROC AUC', auc(fpr_xgb, tpr_xgb))

print(classification_report(y_test, y_pred_xgb, target_names=['NoShow','ShowUp']))
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

ind_params = {'learning_rate': 0.1, 'n_estimators': 100, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 

             'objective': 'binary:logistic'}

optimized_model = GridSearchCV(xgb.XGBClassifier(**ind_params), 

                            cv_params, 

                             scoring = 'accuracy', cv = 5, n_jobs = -1) 

optimized_model.fit(X_train, y_train)

best_optimized_model = optimized_model.best_estimator_
y_pred_xgb = optimized_model.predict(X_test)

y_score_xgb = optimized_model.predict_proba(X_test)[:,1]

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_score_xgb)

print('XGBoost ROC AUC', auc(fpr_xgb, tpr_xgb))

print(classification_report(y_test, y_pred_xgb, target_names=['NoShow','ShowUp']))
i_index = 2

def getAUC_XGB(i_index):

    slFIELDS = features[:i_index] + features[i_index+1:]

    X = data[slFIELDS]

    y = data['StatusB']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    xgbc_best = best_optimized_model

    xgbc_best.fit(X_train, y_train)

    y_pred_xgb = xgbc_best.predict(X_test)

    y_score_xgb = xgbc_best.predict_proba(X_test)[:,1]

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_score_xgb)

    return auc(fpr_xgb, tpr_xgb)
lAUC = []

lIND = []

for ii in range(len(features)):

    lIND.append(features[ii])

    lAUC.append(getAUC_XGB(ii))
plt.scatter(range(len(lIND)),lAUC)

plt.xticks([ii for ii in range(len(features))], features, rotation='vertical')

plt.show()
data_scores = pd.DataFrame({'Feature':lIND, 'Scores':lAUC })

data_scores.sort_values('Scores').head()