import numpy as np
import pandas as pd
Data = pd.read_csv('../input/reliance-stock-20002020/RELIANCE.csv')
Data.head()
Data = Data.sort_values('Date')
Data.shape
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(Data[['Close']])
plt.xticks(range(0,Data.shape[0],500),Data['Date'].loc[::500],rotation=45)
plt.title("RELIANCE Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.show()
Data[['Close']].info()
from sklearn.preprocessing import MinMaxScaler
price=Data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
price['Close'].tail(3)
prediction_days=40
Price=price.to_numpy() # convert to numpy array
data = []
    
    
for index in range(len(Price) -prediction_days): 
    data.append(Price[index: index + prediction_days])
    
data = np.array(data);
test_set_size = int(np.round(0.2*data.shape[0]));
train_set_size = data.shape[0] - (test_set_size);

x_train = data[:train_set_size,:-1]
y_train = data[:train_set_size,-1]

x_test = data[train_set_size:,:-1]
y_test = data[train_set_size:,-1]

print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)
x_train
y_train
import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,drop_p):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,dropout=drop_p, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim,int( hidden_dim*0.3))
        self.fc2 = nn.Linear(int(0.3*hidden_dim), output_dim)
        self.dropout=nn.Dropout(drop_p)

    def forward(self, x):
        #initialize the hidden state to zero while allowing to grad
        h0 = torch.zeros(self.num_layers, x.size(0),self.hidden_dim).requires_grad_()
        #initialize the cell state to zero 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        #detach will pervent backprop from goingall the way to the start 
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out=self.fc1(out)
        out = self.fc2(out[:, -1, :]) 
        return out
model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2,drop_p=0.2)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
import time
num_epochs=100
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []


for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))
optimiser.step()
#return the scaled data to the original shape
predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
import seaborn as sns
sns.set_style("darkgrid")    

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
ax.set_title('Stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)


plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
import math, time
from sklearn.metrics import mean_squared_error
# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)
# Shift train predictions for plotting
# np.empty_like: returns a new array with the same shape and type as a given array
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[prediction_days:len(y_train_pred)+prediction_days,:] = y_train_pred

# Shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(y_train_pred)+prediction_days-1:len(price)-1,:] =y_test_pred



sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(scaler.inverse_transform(price),label="original price",color='r')

plt.plot(trainPredictPlot,label='train prediction',color='yellow')

plt.plot(testPredictPlot,label='test prediction',color='cyan')
plt.legend()
plt.xticks(range(0,Data.shape[0],500),Data['Date'].loc[::500],rotation=45)
plt.title("RELIANCE Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.show()