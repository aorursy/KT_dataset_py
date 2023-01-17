import numpy as np
import pandas as pd
df_test = pd.read_csv('../input/test.csv')
df_train = pd.read_csv('../input/train.csv')
# numeric_features = df_train.select_dtypes(include=[np.number])
# corr = numeric_features.corr()
# print(corr['SalePrice'].sort_values(ascending=False)[:5])
# print(corr['SalePrice'].sort_values(ascending=False)[-5:])

# 构造特征，转化为one-hot向量
# df_train['enc_street'] = pd.get_dummies(df_train.Street, drop_first=True)
# df_test['enc_street'] = pd.get_dummies(df_train.Street, drop_first=True)

# 异常数据处理
df_train = df_train[df_train['GarageArea']<1200]
data = df_train.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(df_train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 测试集MSE
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predictions))
feats = df_test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = model.predict(feats)
final_predictions = np.exp(predictions)

submission = pd.DataFrame()
submission['Id'] = df_test.Id
submission['SalePrice'] = final_predictions
submission.to_csv('submission.csv', index=False)