from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import pandas as pd

train_data = pd.read_csv('train_data.csv')



test_data = pd.read_csv('test_data.csv')

train_y = pd.read_csv('train_y.csv')

print(train_data.head(3))

print(train_y.head(3))
train_data.drop(columns = ['ID'], inplace = True)

train_y.drop(columns = ['ID'], inplace = True)
reg = LogisticRegression()

reg.fit(train_data, train_y)



reg.score(train_data, train_y) # sklearn linear model 獨有的score參數可以直接評估訓練的好壞
# 在訓練及上評估mapping是否正確 (決策樹不限制的話，在簡單訓練資料上很容易有1.0的正確率)

y_pred = reg.predict(train_data)



accuracy_score(train_y['class'], y_pred)
# 對測試資料進行預測

test_df = test_data.copy()

ID = test_df.pop('ID')



y_pred = reg.predict(test_df)

y_pred
# 製造submission dataframe

ans = pd.DataFrame(ID, columns = ['ID'])

ans['class'] = y_pred

ans
ans.to_csv('logisticReg_pred.csv', index = False)