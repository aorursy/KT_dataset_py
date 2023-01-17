import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
sample = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
train.head()
print("Training data size is :",train.shape[0])
print("Testing data size is :",test.shape[0])
train.isna().sum()
train_cat = OrdinalEncoder().fit_transform(train[['Sex', 'SmokingStatus']]) 
train_cat = pd.DataFrame({'Sex': train_cat[:, 0], 'SmokingStatus': train_cat[:, 1]})
train_num = StandardScaler().fit_transform(train[['Weeks', 'Percent','Age']])  # standard scaling 
train_num = pd.DataFrame({'Weeks': train_num[:, 0], 'Percent': train_num[:, 1],'Age':train_num[:,2]})
df = pd.concat([train_cat, train_num, train['FVC']], axis = 1)
X = df.drop('FVC',axis =1)
y = df['FVC']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
params = {'n_estimators': 300,
          'max_depth': 7,
          'learning_rate': 0.01}

reg = GradientBoostingRegressor(**params)

%time reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("The mean squared error (MSE) on test set: {}".format(mse))
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
from xgboost import XGBRegressor

params = {'n_estimators': 300,
          'max_depth': 7,
          'learning_rate': 0.25}

model = XGBRegressor(**params)

%time model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("The mean squared error (MSE) on test set: {}".format(mse))

print(model.feature_importances_)
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
test_cat = OrdinalEncoder().fit_transform(test[['Sex', 'SmokingStatus']]) # categorical Encoding 
test_cat = pd.DataFrame({'Sex': test_cat[:, 0], 'SmokingStatus': test_cat[:, 1]})
test_num = StandardScaler().fit_transform(test[['Weeks', 'Percent','Age']])  # standard scaling 
test_num = pd.DataFrame({'Weeks': test_num[:, 0], 'Percent': test_num[:, 1],'Age':test_num[:,2]})
Xtest = pd.concat([test_cat, test_num], axis = 1)
y_pred_score = reg.predict(Xtest)
y_pred_score
pred = pd.DataFrame(y_pred_score,columns = ['FVC'])
pred['Confidence'] = pred['FVC'].std()
sub = pd.DataFrame({'Patient_Week': sample.Patient_Week, 'FVC': pred['FVC']})
sub = sub[['Patient_Week', 'FVC',]]
filename = 'submission.csv'
sub['Confidence'] = pred['Confidence']
sub.to_csv(filename, index=False) 
sub.head()
