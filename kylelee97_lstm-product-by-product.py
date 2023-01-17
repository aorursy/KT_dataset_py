import numpy as np 
import pandas as pd 
import keras
sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, input_shape=(1,28))))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        
d_days = [i for i in sales.columns if i[:2] == 'd_']
X = sales[d_days]

for r in range(7,len(X)):
    def create_dataset(X, time_steps):
        Xs, ys = [], []
        for i in range(len(X.columns) - time_steps):
            x = X.iloc[r, i:(i + time_steps)].values
            y = X.iloc[r, i + time_steps]
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)
    xx, yy = create_dataset(X, 28)
    xx = xx.reshape(-1,1,28)
   
    model.fit(xx, yy, epochs=100)
    
    pred = model.predict(np.array(X.iloc[r, len(X.columns)-28:]).reshape(1,1,-1))
    new_array = np.insert(X.iloc[r, len(X.columns)-28 + 1:].values.astype(float), 27, round(pred[0][0]))
    for i in range(2,29):
        pred = model.predict(new_array.reshape(1,1,-1))
        if pred < 0:
            pred[0][0] = 0
        new_array = np.insert(new_array[1:], 27, round(pred[0][0]))
    print(r,'th row right now')
    print(new_array)
    
    sub.iloc[r,1:] = np.array(new_array).reshape(-1)
