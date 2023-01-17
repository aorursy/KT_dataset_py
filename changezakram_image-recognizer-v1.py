import keras
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")

# Convert ino numpy arrays
train_y_np = train["label"].to_numpy()
train_x_np = train.drop(columns = ["label"], axis = 1).to_numpy()
train_x_np = train_x_np.astype("float32")

train_y_np.dtype, train_x_np.dtype, train_x_np.shape
train_x, val_x, train_y, val_y = train_test_split(train_x_np,train_y_np,test_size=0.2,random_state=42)
train_x = train_x.reshape(-1,28,28,1)
val_x = val_x.reshape(-1,28,28,1)

train_x.shape, val_x.shape
img = train_x[7].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.title(train_y[7])
plt.axis("off")
plt.show()
train_x_tensor = torch.from_numpy(train_x)
train_y_tensor = torch.from_numpy(train_y)

val_x_tensor = torch.from_numpy(val_x)
val_y_tensor = torch.from_numpy(val_y)

train_x_tensor = train_x_tensor.permute(0,3,1,2)
val_x_tensor = val_x_tensor.permute(0,3,1,2)

#Normalize
train_x_tensor = train_x_tensor / 255.0
val_x_tensor = val_x_tensor / 255.0

train = torch.utils.data.TensorDataset(train_x_tensor,train_y_tensor)
train_loader = torch.utils.data.DataLoader(train, batch_size = 256, shuffle = True)

val = torch.utils.data.TensorDataset(val_x_tensor,val_y_tensor)
val_loader = torch.utils.data.DataLoader(val, batch_size = 256, shuffle = False)
# Create CNN Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16 * 7 * 7, 10) 
    
    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        
        #Flatten
        out = out.view(out.size(0), -1)

        #Dense
        out = self.fc1(out)
        
        return out
import datetime

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        val_loss = 0.0
        correct = 0
        
        for imgs, labels in train_loader:
            #imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item() 
            
        with torch.no_grad():
            model.eval()
            for imgs, labels in val_loader:                
                val_output = model(imgs)
                val_loss += loss_fn(val_output, labels).item() # sum up batch loss
                pred = val_output.argmax(dim=1, keepdim= True) # get index of max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item() #
                        
        if epoch == 1 or epoch % 10 == 0:
            print('Epoch: {}, Training loss: {}, Validation loss: {}, Accuracy: {}'.format(
                epoch,
                loss_train / len(train_loader.dataset),
                val_loss / len(val_loader.dataset),
                correct / len(val_loader.dataset)))  
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss() 

train(
    n_epochs = 200,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    val_loader = val_loader
)
test_x = test.to_numpy()
test_x = test_x.astype("float32")

test_x = test_x.reshape(-1,28,28,1)

test_x.dtype, test_x.shape
test_x_tensor = torch.from_numpy(test_x)/255.0
test_x_tensor = test_x_tensor.permute(0,3,1,2)
test_x_tensor.shape
test = torch.utils.data.TensorDataset(test_x_tensor)
test_loader = torch.utils.data.DataLoader(test, batch_size = 16, shuffle = False)
def make_predictions(data_loader):
    model.eval()
    test_preds = torch.LongTensor()
    
    for i, data in enumerate(test_loader):
        data = data.unsqueeze(1)
        
        if torch.cuda.is_available():
            data = data.cuda()
            
        output = conv_model(data)
        
        preds = output.cpu().data.max(1, keepdim=True)[1]
        test_preds = torch.cat((test_preds, preds), dim=0)
        
    return test_preds


model.eval()   
with torch.no_grad():   
    ps = model(test_x_tensor)
    prediction = torch.argmax(ps, 1)
    print('Prediction',prediction)
df_submission = pd.DataFrame(prediction.cpu().tolist(), columns = ['Label'])
df_submission['ImageId'] = df_submission.index +1
df_submission = df_submission[['ImageId', 'Label']]
df_submission.head()
df_submission.to_csv('submission.csv', index=False)