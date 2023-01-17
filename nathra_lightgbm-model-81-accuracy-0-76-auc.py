import numpy as np
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
for col in df.columns:
    uniq = df[col].unique()
    print('\n{}: {} ({})'.format(col, uniq, len(uniq)))
df.rename(columns = {'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
df['target'] = df['No-show'].apply(lambda x: 1 if x=='Yes' else 0)
df.pop('No-show')
df['Age'] = df['Age'].apply(lambda x: 0 if x < 0 else x)
df[['ScheduledDay', 'AppointmentDay']] = df[['ScheduledDay', 'AppointmentDay']].astype('datetime64')

cat_columns = [
    'Gender',
    'Neighbourhood',
    'Scholarship',
    'Hypertension',
    'Diabetes',
    'Alcoholism',
    'Handicap',
    'SMS_received',
]

for col in cat_columns:
    df[col] = pd.Categorical(df[col]).codes
df[cat_columns] = df[cat_columns].astype('category')

df.info()
df['Month'] = df['AppointmentDay'].apply(lambda x: x.month).astype('category')
df['DayOfMonth'] = df['AppointmentDay'].apply(lambda x: x.day)
df['DayOfWeek'] = df['ScheduledDay'].apply(lambda x: x.dayofweek).astype('category')
df['HourCalled'] = df['ScheduledDay'].apply(lambda x: x.hour)
df['DaysInAdvance'] = (df['AppointmentDay'] - df['ScheduledDay']).apply(lambda x: x.days + x.seconds / (3600*24))

df.drop(['AppointmentDay', 'ScheduledDay'], axis=1, inplace=True)

with pd.option_context('display.max_columns', 100):
    print(df.sample(5))
def load_data(df):
    data = df.drop(['PatientId', 'AppointmentID'], axis=1)
    
    X_test = data.sample(frac=.1)
    y_test = X_test.pop('target')
    data = data.drop(X_test.index, axis=0)

    X_train = data
    y_train = X_train.pop('target')

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(df)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_error', 'auc'],
    'learning_rate': 0.05,
    'verbose': 0,
    'num_boost_round': 1000,
    'num_leaves': 512,
    'max_depth': 256,
    'seed': 1
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
}

gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)
y_pred = gbm.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
imp = pd.DataFrame()
imp['feature'] = X_train.columns
imp['importance'] = gbm.feature_importance()
imp = imp.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", 
            data=imp)

# lgb.plot_importance(gbm, max_num_features=50)