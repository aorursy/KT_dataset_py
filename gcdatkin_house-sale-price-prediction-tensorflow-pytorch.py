import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf

import torch

import torch.nn as nn

import torch.nn.functional as F
data = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
data
data.info()
data = data.drop('id', axis=1)
data['year'] = data['date'].apply(lambda x: x[0:4])

data['month'] = data['date'].apply(lambda x: x[4:6])



data = data.drop('date', axis=1)
len(data['zipcode'].unique())
def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
data = onehot_encode(data, 'zipcode', 'zip')
data.query("yr_renovated != 0")
data = data.drop('yr_renovated', axis=1)
data
y = data['price'].copy()

X = data.drop('price', axis=1).copy()
scaler = StandardScaler()



X = scaler.fit_transform(X)
tf_X_train, tf_X_test, tf_y_train, tf_y_test = train_test_split(X, y, train_size=0.7, random_state=1)
tf_X_train.shape
inputs = tf.keras.Input(shape=(88,))

hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)

hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)

outputs = tf.keras.layers.Dense(1, activation='linear')(hidden)



tf_model = tf.keras.Model(inputs, outputs)





tf_model.compile(

    optimizer='adam',

    loss='mse'

)





history = tf_model.fit(

    tf_X_train,

    tf_y_train,

    validation_split=0.12,

    batch_size=32,

    epochs=10

)
tf_rmse = np.sqrt(tf_model.evaluate(tf_X_test, tf_y_test))
class Net(nn.Module):

    

    def __init__(self):

        super(Net, self).__init__()

        self.layer1 = nn.Linear(88, 64)

        self.layer2 = nn.Linear(64, 64)

        self.out = nn.Linear(64, 1)

    

    def forward(self, x):

        x = F.relu(self.layer1(x))

        x = F.relu(self.layer2(x))

        x = self.out(x)

        return x



net = Net()
for i in range(len(list(net.parameters()))):

    print(list(net.parameters())[i].shape)
torch_X_train = torch.tensor(tf_X_train).type(torch.float32)

torch_y_train = torch.tensor(np.array(tf_y_train)).type(torch.float32)



torch_X_test = torch.tensor(tf_X_test).type(torch.float32)

torch_y_test = torch.tensor(np.array(tf_y_test)).type(torch.float32)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

criterion = nn.MSELoss()
for x, target in zip(torch_X_train, torch_y_train):

    optimizer.zero_grad()

    output = net(x)

    loss = criterion(output, target)

    loss.backward()

    optimizer.step()
total_loss = 0



for x, target in zip(torch_X_test, torch_y_test):

    output = net(x)

    loss = criterion(output, target)

    total_loss += loss

    

avg_loss = total_loss / len(torch_X_test)
torch_rmse = torch.sqrt(avg_loss).detach().numpy()
print("TensorFlow RMSE:", tf_rmse)

print("   PyTorch RMSE:", torch_rmse)