!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
#캐글 주소
!kaggle competitions download -c unemployment-rate
!unzip unemployment-rate.zip
import numpy as np
import pandas as pd
train_data=pd.read_csv('train_unemployment_rate.csv',header=None)
test_data=pd.read_csv('test_unemployment_rate.csv',header=None)
train_data
#년,월값 수정
train_data[[1]]=train_data[[1]]%10000/100
x_train_data=train_data.loc[1:,1:3]
y_train_data=train_data.loc[1:,4]

x_data=np.array(x_train_data)
y_data=np.array(y_train_data)
x_data
test_data[[1]]=test_data[[1]]%10000/100
x_test_data=test_data.loc[1:,1:3]
test_data=np.array(x_test_data)
import xgboost as xgb
xg_model = xgb.XGBRegressor(max_depth=7, learning_rate=0.01, n_estimators=2000,objective='reg:squarederror',seed=501)
xg_model.fit(x_data, y_data)

y_pred = xg_model.predict(test_data)
y_pred
predictions = [round(value) for value in y_pred]
predictions
submit = pd.read_csv("submission.csv")
for x in range(len(predictions)):
    submit['Expected'][x] = predictions[x]
submit
submit.to_csv('submission.csv',index=False)
!kaggle competitions submit -c unemployment-rate -f submission.csv -m "Message"