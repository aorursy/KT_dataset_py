import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
train_data = pd.read_csv('../input/train.csv')
train_data.head()
train_data.groupby('target')['id'].count()
train_set, test_set = train_test_split(train_data)



print('Train size:', train_set.shape[0])

print('Test size:', test_set.shape[0])
feature_columns = train_data.columns.tolist()

feature_columns.remove('target')



X_train = train_set[feature_columns]

X_test = test_set[feature_columns]



y_train = train_set['target']

y_test = test_set['target']



print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)

print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
# Unbalanced model

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

roc_auc_score(y_test, y_pred)
# Balanced model

model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

roc_auc_score(y_test, y_pred)
test_data = pd.read_csv('../input/test.csv')
p_pred = model.predict_proba(test_data[feature_columns])
p_pred[:10]
prediction = pd.concat([test_data.id, pd.Series(p_pred[:,1])], axis=1)

prediction.columns = ['id', 'target']
prediction = prediction.round(4)
prediction.head()
#prediction.to_csv('../submissions/submission1.csv', header=True, index=False)