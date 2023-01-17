import pandas as pd
raw = pd.read_csv('../input/train.csv', nrows=10000000)
raw.head()
raw = raw.drop(['click_time', 'attributed_time'], axis=1)
raw.head()
raw['is_attributed'].value_counts()
raw.isnull().sum()
raw.columns.values
features = raw.columns.values.tolist()[0:5]
features
X = raw[features]
y = raw['is_attributed']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
X_train.shape
y_train.value_counts()
X_test.shape
y_test.value_counts()
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf = RandomForestClassifier(max_depth=20, random_state=2018)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
roc_auc
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='Random Forest (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
unseen_data = pd.read_csv('../input/test.csv')
unseen_data.shape
unseen_data.head()
unseen_data.drop(['click_time'], axis=1, inplace=True)
unseen_data.head()
unseen_data[features].isnull().sum()
unseen_data_pred = rf.predict(unseen_data[features])
unseen_data_pred
submission = pd.concat([unseen_data['click_id'], pd.Series(unseen_data_pred, name='is_attributed')], axis=1)
submission.head()
submission.shape
submission.to_csv('submission.csv', index=False)