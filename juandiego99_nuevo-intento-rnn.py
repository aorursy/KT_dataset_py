import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
df = pd.read_csv('../input/dataset-medellin-air-quality-sensors/Ventana360_SinFaltantes.csv')
df.info()
df=df.rename(columns={'Unnamed: 0':'Sensores'})
df = df.astype({"Sensores": str})
df
#df_s = df.iloc[0,1:]
#df_s = df_s.to_frame().T
#df_s

df_s = df.iloc[0:,1:]
df_s.head(10)

X = df_s.values
X.astype("float64")
#Xn = np.expand_dims(np.concatenate((X[2,:],X[4,:],X[6,:],X[7,:]),axis=0),1)
#Xn = Xn[:360,:]
Xn = np.expand_dims(X[1,:],1)

print(Xn.shape)

fig = plt.figure()

ax11 = fig.add_subplot(211)
ax11.plot(Xn)
#Creando las secuencias
def create_sequences(data, seq_lenght):
    xs = []
    ys = []
    
    for i in range(len(data)-seq_lenght-1):
        x = data[i:(i+seq_lenght)]
        y = data[i+seq_lenght]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
Xtrain, Xtest = train_test_split(Xn,train_size=0.9, test_size=0.1, shuffle=False)
Xtr, Xval = train_test_split(Xtrain,train_size=0.9, test_size=0.1, shuffle=False)

print(Xtr.shape,Xval.shape,Xtest.shape)
#scal_lab = MinMaxScaler()
#scal_lab_fit = scal_lab.fit(Xtr[:,0].reshape(-1, 1))
#scal_lab_fit = scal_lab.fit(Xtr.reshape(-1, 1))
scal = MinMaxScaler()
Xtr = scal.fit_transform(Xtr)
Xval = scal.transform(Xval)
Xtest = scal.transform(Xtest)

print("Shapes")
print("Xtr:",Xtr.shape)
print("Xval:",Xval.shape)
print("Xtest:",Xtest.shape)
tw = 5 # Tw debe ser menor a 33
# "RuntimeError: Expected hidden[0] size (2, 20, 256), got (2, 64, 256)" Cambio a 20
batch_size = 24 #batch_size = 64
Xtr, ytr = create_sequences(Xtr,tw)
Xval, yval = create_sequences(Xval,tw)
Xtest, ytest = create_sequences(Xtest,tw)

#Apenas corro esta línea, solo me quedan Xtr y ytr, el resto es vacío
print(Xtr.shape,Xval.shape,Xtest.shape)
train_data = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

val_data = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
#val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=False)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
plt.plot(ytr)
n_features = 1 # this is number of parallel inputs
n_timesteps = tw # this is number of timesteps
hidden_dim = 256
output_dim = 1
n_layers = 2
# create NN
#model = MV_LSTM(n_features,n_timesteps)

model = LSTMNet(n_features,hidden_dim,output_dim,n_layers)
model.to(device)

criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_episodes = 60
model.train()
for t in range(train_episodes):
    htr = model.init_hidden(batch_size)
    for x,y in train_loader:
        htr = tuple([e.data for e in htr])
        model.zero_grad()
        output, htr = model(x.to(device).float(),htr) 
        loss = criterion(output, y.to(device).float())  
        #Calcular gradiente
        loss.backward() 
        #Mover los datos
        optimizer.step()        
        
    hval = model.init_hidden(batch_size)    
    for xval,yval in val_loader:
        hval = tuple([e.data for e in hval])
        output_val, hval = model(xval.to(device).float(),htr) 
        loss_val = criterion(output_val, yval.to(device).float()) 
    
    
        
        
        
    print('step : ' , t , ' loss_train: ' , loss.item(), ' loss_val: ', loss_val.item())
for xval,yval in val_loader:
        print(yval)
print(x)
#Evaluate the model
model.eval()
Xtest = torch.from_numpy(Xtest)
htest = model.init_hidden(Xtest.shape[0])
out, htest = model(Xtest.to(device).float(), htest)
print("Antes de:",out)

out = out.cpu().detach().numpy()
out = scal.inverse_transform(out)

print("Despues de:",out)
#ytest = scal_lab_fit.inverse_transform(ytest)
ytest = scal.inverse_transform(ytest)
print("\nytest:",ytest)
print(Xtest[0:10,:])
fig = plt.figure()

ax11 = fig.add_subplot(211)
ax11.plot(out)
ax11.plot(ytest,'r')