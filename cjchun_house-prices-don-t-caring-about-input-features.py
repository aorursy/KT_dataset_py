import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import TensorBoard

from sklearn.model_selection import KFold
input_var = 'Id'
target_var = 'SalePrice'
df_train_org = pd.read_csv('../input/train.csv')
df_train_rev = df_train_org.copy()

print(df_train_org.shape, df_train_rev.shape)
df_train_org.head(10)
def Norm(x, mu, sigma):
    tmp = ((x - mu) / sigma)
    
    return (1.0 / (1.0 + np.exp(-tmp)))
df_train_rev = df_train_org.drop(columns=[input_var, target_var])
for col in df_train_rev.columns:
    if df_train_rev[col].dtype == object:  
        df_train_rev[col] = df_train_rev[col].fillna(df_train_org[col].describe().top)        
        df_train_rev[col] = df_train_rev[col].astype('category').cat.codes        
    elif (df_train_rev[col].dtype == int) or (df_train_rev[col].dtype == float):        
        df_train_rev[col] = Norm(df_train_rev[col].astype(np.float32), df_train_org[col].mean(), df_train_org[col].std())
        df_train_rev[col] = df_train_rev[col].fillna(0.5)
        
del_list = []
for col in df_train_rev.columns:
    if abs(df_train_rev[col].corr(df_train_org[target_var])) < 0.2:
        del_list.append(col)
        
df_train_rev = df_train_rev.drop(columns=del_list)
print(df_train_rev.shape)
df_train_rev.head(10)
print(np.abs(df_train_rev.corrwith(df_train_org[target_var])))
x_train_val = df_train_rev.values
y_train_val = df_train_org[target_var].values.reshape(-1, 1)

print(x_train_val.shape, y_train_val.shape)
model_num = 10

kf = KFold(n_splits=model_num, shuffle=True)
print(kf)
# neural network architecture
hidden_unit_num_ = 128
activation_ = 'relu'

def mlp_model():
    model = Sequential()
    
    model.add(Dense(units=hidden_unit_num_, input_dim=np.size(x_train_val, 1)))
    model.add(BatchNormalization())
    model.add(Activation(activation_))
    
    model.add(Dense(units=hidden_unit_num_))
    model.add(BatchNormalization())
    model.add(Activation(activation_))
    
    model.add(Dense(units=hidden_unit_num_))
    model.add(BatchNormalization())
    model.add(Activation(activation_))
    
    model.add(Dense(units=hidden_unit_num_))
    model.add(BatchNormalization())
    model.add(Activation(activation_))
    
    model.add(Dense(units=hidden_unit_num_))
 
    model.add(Dense(units=1))

    return model
# loss function & optimization method
model = []
for i in range(model_num):
    tmp = mlp_model()
    tmp.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.append(tmp)
# training
tensorBoard = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

idx = 0
for train_index, test_index in kf.split(x_train_val):
    x_train, x_val = x_train_val[train_index], x_train_val[test_index]
    y_train, y_val = y_train_val[train_index], y_train_val[test_index]
    
    hist = model[idx].fit(x_train, y_train, epochs=300, batch_size=32, verbose=False, validation_data=(x_val, y_val), shuffle=True, callbacks=[tensorBoard])            
    print(model[idx].evaluate(x=x_val, y=y_val, batch_size=512))
    
    idx += 1
df_test_org = pd.read_csv('../input/test.csv')
df_test_rev = df_test_org.copy()

print(df_test_org.shape, df_test_rev.shape)
df_test_rev = df_test_org.drop(columns=[input_var])
for col in df_test_rev.columns:
    if df_test_rev[col].dtype == object:  
        df_test_rev[col] = df_test_rev[col].fillna(df_train_org[col].describe().top)        
        df_test_rev[col] = df_test_rev[col].astype('category').cat.codes        
    elif (df_test_rev[col].dtype == int) or (df_test_rev[col].dtype == float):        
        df_test_rev[col] = Norm(df_test_rev[col].astype(np.float32), df_train_org[col].mean(), df_train_org[col].std())
        df_test_rev[col] = df_test_rev[col].fillna(0.5)        
        
x_test = df_test_rev.drop(columns=del_list)
# evaluation
idx = 0
y_pred = []
for i in range(model_num):
    tmp = model[idx].predict(x_test, batch_size=512)
    y_pred.append(tmp)
    
    idx += 1
    
y_pred = np.mean(y_pred, axis=0)
print(y_pred.shape)
submission = pd.DataFrame({
    input_var : df_test_org[input_var].astype(int),
    target_var : y_pred.reshape(-1, ).astype(float)
})
submission.head(10)
submission.to_csv('result.csv', index=False)