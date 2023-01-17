# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install keras
# Import libraries
import matplotlib.pyplot as plt
import matplotlib as  pylab


from keras.models import Sequential
from keras.layers import  Dense
from keras.layers import  LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tools.eval_measures import rmse
df_defects = pd.read_csv("../input/monthly-defect-data-simulated/software_defects_simulated.csv")
df_defects['created_date'] = pd.to_datetime(df_defects['created_date'], format="%Y-%m-%d")
df_defects.set_index('created_date', inplace=True)
df_defects = df_defects.groupby(pd.Grouper(freq='M')).sum()

#split the dataset in train and test
train = df_defects.iloc[:40]
test = df_defects.iloc[40:]
start = len(train)
end = len(train) + len(test)-1
n_input = 2
n_features =1

# The LSTM expects data input to have the shape [samples, timesteps, features],
# whereas the generator described so far is providing lag observations as features or the shape [samples, features].
x_train = train['count'].values.reshape((len(train['count']),n_features))
train_generator = TimeseriesGenerator(x_train,x_train,length=n_input,batch_size=1)

for i in range(len(train_generator)):
    x, y = train_generator[i]
    print('%s => %s' % (x, y))
model = Sequential()
model.add(LSTM(150,activation='relu',input_shape=(n_input,n_features)))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.summary()

model.fit(train_generator,epochs=200)
model_loss = model.history.history['loss']
plt.plot(range(len(model_loss)),model_loss)
# Forecast single predictions
first_eval_batch = x_train[-n_input:]
first_eval_batch = first_eval_batch.reshape(1,n_input,n_features)
test_predictions_first_eval_batch = model.predict(first_eval_batch)
test_predictions_first_eval_batch[0][0]
test_predictions = []
current_batch = first_eval_batch.reshape(1,n_input,n_features)

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    print(current_pred)
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

df_pred = pd.DataFrame(test_predictions)
df_pred.rename( columns={0:'count'}, inplace=True )
rmse(test['count'],df_pred['count'])
error = rmse(test['count'],df_pred['count'])
print("Test Mean",test.mean())
print("Predictions Mean",df_pred['count'].mean())
print("Predictions Error",error)

test['count'].plot(figsize=(12,8),legend=True,title="Actual Test Data Plot")
df_pred.plot(legend=True ,title="Prediction Data Plot")
