import pandas as pd
import numpy as np
import pickle
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as Accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
#X = pickle.load( open( "../input/urbansoundsdata/Data16k.pickle", "rb" ) )
#Y = pickle.load( open( "../input/urbansoundsdata/Labels16k.pickle", "rb" ))
df = pd.read_csv("../input/urbansoundsdata/UrbanSound8K.csv")
Labs = df["class"]
List_of_labels = list(set(Labs))
List_of_classes = list(set(df["classID"]))

print(X.shape,Y.shape)
class ConvNN(torch.nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv_layer1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride = 2)
        self.conv_layer2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride = 2)
        self.conv_layer3 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride = 2)
        self.conv_layer4 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride = 2)
        self.fc_layer1 = torch.nn.Linear(in_features=8*128, out_features=128, bias=True)
        self.fc_layer2 = torch.nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc_layer3 = torch.nn.Linear(in_features=64, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x,kernel_size = 8,stride = 8)
        
        x = self.conv_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x,kernel_size = 8,stride = 8)
        
        x = self.conv_layer3(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv_layer4(x)
        x = torch.nn.functional.relu(x)
        
        x = x.view(x.size(0),-1)
        
        x = self.fc_layer1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5, training=self.training)
        
        x = self.fc_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5, training=self.training)
        
        x = self.fc_layer3(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        
        return x
batch_size = 100 #Número de muestras del minibacth
train_episodes = 100 #Número de épocas para gradiente descendente
order = False


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,train_size=0.9, test_size=0.1, random_state = 42)
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain,Ytrain,train_size=0.9, test_size=0.1, random_state = 42)

train_data = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain), torch.from_numpy(Ytrain))
val_data = torch.utils.data.TensorDataset(torch.from_numpy(Xval), torch.from_numpy(Yval))

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)
val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=False)

model.parameters
model = ConvNN()
model.to(device)

criterion = torch.nn.NLLLoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

order = False
acc_bm = 0
model_path_ConvNN = '/kaggle/working/best_model_ConvNN'
if order == False:
        model_path_ConvNN = '/kaggle/working/best_model_ConvNN_random'
        try: 
            ind_rand = pickle.load( open( "/kaggle/working/ind_rand.pickle", "rb" ) )
        except:
            ind_rand = random.sample(range(16000), 16000)
            with open('/kaggle/working/ind_rand.pickle', 'wb') as f:
                pickle.dump(ind_rand, f)
            
torch.cuda.empty_cache()
for t in range(train_episodes):
    model.train(True)
    loss_train = 0
    loss_val = 0
    acc_val = 0
    for i, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        if order == False:
            x = x[:,ind_rand]
        x = x.unsqueeze(1)
        prob = model(x.to(device).float())
        loss = criterion(prob, y.to(device).long())  
        loss_train += loss.item()
        loss.backward()
        optimizer.step() 
    
    model.train(False)
    for i, (x,y) in enumerate(val_loader):
        if order == False:
            x = x[:,ind_rand]
        x = x.unsqueeze(1)
        prob = model(x.to(device).float())
        loss_v = criterion(prob, y.to(device).long())
        loss_val += loss_v.item()
        pred = torch.argmax(prob, dim=1)
        
        acc_v = torch.sum(pred==y.to(device).data)
        acc_val += acc_v
        
    acc_epoch = acc_val.double()/len(Yval)    
    if(acc_epoch>acc_bm):
        acc_bm= acc_epoch
        torch.save(model.state_dict(), model_path_ConvNN)

        
    print(f'epoch : [{t+1}/{train_episodes}], loss_train: {loss_train/len(train_loader):.4f}, loss_val: {loss_val/len(val_loader):.4f}, acc_val: {acc_epoch:.4f} ')

order = True
model_path_ConvNN = '/kaggle/working//best_model_ConvNN'
if order == False:
        model_path_ConvNN = '/kaggle/working//best_model_ConvNN_random'
        
model = ConvNN()
model.load_state_dict(torch.load(model_path_ConvNN,map_location=torch.device('cpu')))
model.to(device)

#Evaluate the model

test_data = torch.utils.data.TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))
batch_size = len(Xtest)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False)
model.eval()
torch.cuda.empty_cache()

for i, (x,y) in enumerate(test_loader):
    yn = y.cpu()
    if order == False:
        x = x[:,ind_rand]
    x = x.unsqueeze(1)
    with torch.no_grad():
        prob = model(x.to(device).float()) 
    pred = torch.argmax(prob, dim=1)
    pred = pred.cpu()
    
%matplotlib inline
cm = confusion_matrix(y_true = yn.numpy(), y_pred = pred.numpy(), labels=List_of_classes)
Acc = Accuracy(yn.numpy(),pred.numpy())
df_cm = pd.DataFrame(cm, columns=List_of_labels, index = List_of_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (7,7))
plt.title('Matriz de confusión')
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
print(f'Tasa de acierto: {Acc}')
%matplotlib inline
x = X[1,:]
ind_rand = random.sample(range(16000), 16000)
xr = x[ind_rand]
t = np.linspace(0,1,16000)

fig_sigs = plt.figure()
ax1 = fig_sigs.add_subplot(211)
ax1.plot(t,x, 'k')

ax2 = fig_sigs.add_subplot(212)
ax2.plot(t,xr, 'k')


order = False
model_path_ConvNN = '/kaggle/working//best_model_ConvNN'
if order == False:
        model_path_ConvNN = '/kaggle/working/best_model_ConvNN_random'
        ind_rand = pickle.load( open( "/kaggle/working/ind_rand.pickle", "rb" ) )
model = ConvNN()
model.load_state_dict(torch.load(model_path_ConvNN,map_location=torch.device('cpu')))
model.to(device)

#Evaluate the model

test_data = torch.utils.data.TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))
batch_size = len(Xtest)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False)
model.eval()
torch.cuda.empty_cache()

for i, (x,y) in enumerate(test_loader):
    yn = y.cpu()
    if order == False:
        x = x[:,ind_rand]
    x = x.unsqueeze(1)
    with torch.no_grad():
        prob = model(x.to(device).float()) 
    pred = torch.argmax(prob, dim=1)
    pred = pred.cpu()
%matplotlib inline
cm = confusion_matrix(y_true = yn.numpy(), y_pred = pred.numpy(), labels=List_of_classes)
Acc = Accuracy(yn.numpy(),pred.numpy())
df_cm = pd.DataFrame(cm, columns=List_of_labels, index = List_of_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (7,7))
plt.title('Matriz de confusión')
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
print(f'Tasa de acierto: {Acc}')
class FCNN(torch.nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc_layer1 = torch.nn.Linear(in_features=16000, out_features=7969, bias=True)
        self.fc_layer2 = torch.nn.Linear(in_features=7969, out_features=996, bias=True)
        self.fc_layer3 = torch.nn.Linear(in_features=996, out_features=483, bias=True)
        self.fc_layer4 = torch.nn.Linear(in_features=483, out_features=128, bias=True)
        self.fc_layer5 = torch.nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc_layer6 = torch.nn.Linear(in_features=64, out_features=10, bias=True)
        
        
    def forward(self, x):
        
        x = self.fc_layer1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5, training=self.training)
        
        x = self.fc_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5, training=self.training)
        
        x = self.fc_layer3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5, training=self.training)
        
        x = self.fc_layer4(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5, training=self.training)
        
        x = self.fc_layer5(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5, training=self.training)
        
        x = self.fc_layer6(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        
    
        
        return x
model = FCNN()
model.to(device)

criterion = torch.nn.NLLLoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

order = False
acc_bm = 0
model_path_FCNN = '/kaggle/working/best_model_FCNN'
if order == False:
        model_path_FCNN = '/kaggle/working/best_model_FCNN_random'
        try: 
            ind_rand = pickle.load( open( "/kaggle/working/ind_rand.pickle", "rb" ) )
        except:
            ind_rand = random.sample(range(16000), 16000)
            with open('/kaggle/working/ind_rand.pickle', 'wb') as f:
                pickle.dump(ind_rand, f)

torch.cuda.empty_cache()
for t in range(train_episodes):
    model.train(True)
    loss_train = 0
    loss_val = 0
    acc_val = 0
    for i, (x,y) in enumerate(train_loader):
        if order == False:
            x = x[:,ind_rand]
        optimizer.zero_grad()
        prob = model(x.to(device).float())
        loss = criterion(prob, y.to(device).long())  
        loss_train += loss.item()
        loss.backward()
        optimizer.step() 
    
    model.train(False)
    for i, (x,y) in enumerate(val_loader):
        if order == False:
            x = x[:,ind_rand]
        prob = model(x.to(device).float())
        loss_v = criterion(prob, y.to(device).long())
        loss_val += loss_v.item()
        pred = torch.argmax(prob, dim=1)
        
        acc_v = torch.sum(pred==y.to(device).data)
        acc_val += acc_v
        
    acc_epoch = acc_val.double()/len(Yval)    
    if(acc_epoch>acc_bm):
        acc_bm= acc_epoch
        torch.save(model.state_dict(), model_path_FCNN)

        
    print(f'epoch : [{t+1}/{train_episodes}], loss_train: {loss_train/len(train_loader):.4f}, loss_val: {loss_val/len(val_loader):.4f}, acc_val: {acc_epoch:.4f} ')

# Test FCNN
order = True
model_path_FCNN = '/kaggle/working//best_model_FCNN'
if order == False:
        model_path_FCNN = '/kaggle/working//best_model_FCNN_random'

model = FCNN()
model.load_state_dict(torch.load(model_path_FCNN))
model.to(device)

#Evaluate the model
test_data = torch.utils.data.TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))
batch_size = len(Xtest)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False)
model.eval()
torch.cuda.empty_cache()
predi = []

acc = []
yn = []
for i, (x,y) in enumerate(test_loader):
    if order == False:
        x = x[:,ind_rand]
    yn = y.cpu()
    with torch.no_grad():
        prob = model(x.to(device).float()) 
    pred = torch.argmax(prob, dim=1)
    pred = pred.cpu()

%matplotlib inline
cm = confusion_matrix(y_true = yn.numpy(), y_pred = pred.numpy(), labels=List_of_classes)
Acc = Accuracy(yn.numpy(),pred.numpy())
df_cm = pd.DataFrame(cm, columns=List_of_labels, index = List_of_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (7,7))
plt.title('Matriz de confusión')
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
print(f'Tasa de acierto: {Acc}')
# Test FCNN
order = False
model_path_FCNN = '/kaggle/working//best_model_FCNN'
if order == False:
        model_path_FCNN = '/kaggle/working//best_model_FCNN_random'

model = FCNN()
model.load_state_dict(torch.load(model_path_FCNN))
model.to(device)

#Evaluate the model
test_data = torch.utils.data.TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))
batch_size = len(Xtest)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False)
model.eval()
torch.cuda.empty_cache()
predi = []

acc = []
yn = []
for i, (x,y) in enumerate(test_loader):
    if order == False:
        x = x[:,ind_rand]
    yn = y.cpu()
    with torch.no_grad():
        prob = model(x.to(device).float()) 
    pred = torch.argmax(prob, dim=1)
    pred = pred.cpu()

%matplotlib inline
cm = confusion_matrix(y_true = yn.numpy(), y_pred = pred.numpy(), labels=List_of_classes)
Acc = Accuracy(yn.numpy(),pred.numpy())
df_cm = pd.DataFrame(cm, columns=List_of_labels, index = List_of_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (7,7))
plt.title('Matriz de confusión')
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
print(f'Tasa de acierto: {Acc}')