import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df_data_1 = pd.read_csv('../input/elearning-student-reactions/online_classroom_data.csv', index_col=0)
df_data_1.replace(',', '.', regex=True, inplace=True)
df_data_1.head()
from sklearn import preprocessing

y = df_data_1['Approved'].values

X = df_data_1[['total_posts', 'helpful_post', 'nice_code_post', 'collaborative_post', 'confused_post','creative_post','bad_post','amazing_post','timeonline']].values
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
import xgboost as xgb

xgb_model = xgb.XGBRegressor(random_state=42)

xgb_model.fit(X_train, y_train)
xgboost_yhat = xgb_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,xgboost_yhat.round()))
print(classification_report(y_test,xgboost_yhat.round()))
print(accuracy_score(y_test, xgboost_yhat.round()))