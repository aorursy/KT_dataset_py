import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Necessary Dependancies

import torch

from torch import nn,optim

import torch.nn.functional as F

from torchvision import transforms

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split



import pandas as pd

import numpy as np



from tqdm import tqdm
# Read data from csv

df = pd.read_csv('../input/digit-recognizer/train.csv')

print(df.shape)
y = df['label']

x = df[df.columns[1:]]

print(x.shape)

print(y.shape)
xtrain,xval,ytrain,yval = train_test_split(x,y,train_size=0.8)

print(f'Training: X.shape = {xtrain.shape}  Y.shape = {ytrain.shape}')

print(f'Validation: X.shape = {xval.shape}  Y.shape = {yval.shape}')
# Normalizing the input data

# Normalizing helps gradients converge faster 

xtrain = xtrain / 255

xval = xval / 255
# Converting the numpy arrays to tensors

xtrain_tensor = torch.tensor(np.array(xtrain),dtype = torch.float)

ytrain_tensor = torch.tensor(np.array(ytrain))

ytrain_tensor = ytrain_tensor.type(torch.LongTensor)    # Categorical attributes should be in LongTensor



xval_tensor = torch.tensor(np.array(xval),dtype = torch.float)

yval_tensor = torch.tensor(np.array(yval))

yval_tensor = yval_tensor.type(torch.LongTensor)
# Obtaining TensorDatasets and dataloaders



train_batch_size = 64

val_batch_size = 64



# Create the train dataset

train_dataset = TensorDataset(xtrain_tensor,ytrain_tensor)

# Create the train dataloader

train_dataloader = DataLoader(train_dataset,batch_size = train_batch_size)



# Create the val dataset

val_dataset = TensorDataset(xval_tensor,yval_tensor)

# Create the validation dataloader

val_dataloader = DataLoader(val_dataset,batch_size = val_batch_size)
# Model Architecture

class Net(nn.Module):

    

    def __init__(self):

        

        hidden1 = 512

        hidden2 = 256

        

        super().__init__()

        # Input layer

        self.fc1 = nn.Linear(784,hidden1)

        # layer 2

        self.fc2 = nn.Linear(hidden1,hidden2)

        #layer 3 / output layer

        self.output = nn.Linear(hidden2,10) # 10 as we have ten classes

        # dropout to prevent overfitting

        self.dropout = nn.Dropout(p=0.2) # while training drops 20% of neurons for a particular layer

        

    def forward(self,x):

        

        # We apply Dropout to the first two layers

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        

        # The output layer has no dropout and a log_softmax activation is applied to it 

        x = F.log_softmax(self.output(x),dim=1)

        

        # We then return the value obtained from forward prop

        return x



# Instanstiating the model

model = Net()
# Loss Function

criterion = nn.NLLLoss()



# Optimizer 

optimizer = optim.Adam(model.parameters(),lr = 0.003)
# Check if GPU is available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
model.to(device)
epochs = 100



# History dictionary

history = {'loss':[],'acc':[],'val_loss':[],'val_acc':[]}



max_val_acc = 0

# Training over all epochs

for e in range(epochs):

    

    running_loss, total, correct,val_running_loss, val_total, val_correct = 0,0,0,0,0,0

    

    # TRAINING SET

    

    # Set model to train mode

    model.train()

    

    # Loading the data in batches

    for images, labels in train_dataloader:

        

        # sending the images and labels tensor to the GPU

        images, labels = images.to(device), labels.to(device)

        # FORWARD PROPOGATION

        

        # Pass the images as input to the model

        outputs = model(images)

        # Calculate loss between prediction and actual output

        loss = criterion(outputs, labels)

        

        # BACKWARD PROPOGATION

        

        # zero_grad() is to avoid accumalation of gradients

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        # Metrics

        # summing the losses over entire training set

        running_loss += loss.item()

        total += labels.size(0)

        predicted = torch.argmax(torch.exp(outputs).data,1)

        correct += (predicted == labels).sum().item()

        

    # VALIDATION SET

    

    with torch.no_grad():  # This disables gradient calculations on validation set

        

        # Lets set model to eval mode. This will turn-off the dropouts

        model.eval()

        

        # Loading the validation data in batches

        for val_images, val_labels in val_dataloader:

            

            # sending the images and labels tensor to the GPU

            val_images, val_labels = val_images.to(device), val_labels.to(device)

    

            # FORWARD PROPOGATION



            # Pass the images as input to the model

            val_outputs = model(val_images)

            # Calculate loss between prediction and actual output

            val_loss = criterion(val_outputs, val_labels)



            # Metrics

            # summing the losses over entire training set

            val_running_loss += val_loss.item()

            val_total += val_labels.size(0)

            val_predicted = torch.argmax(torch.exp(val_outputs).data,1)

            val_correct += (val_predicted == val_labels).sum().item()

            

    # Finally saving all the metrics to the history dictionary

    acc = correct/total * 100

    val_acc = val_correct/val_total * 100

    history['loss'].append(running_loss)

    history['acc'].append(acc)

    history['val_loss'].append(val_running_loss)

    history['val_acc'].append(val_acc)

    

    # Logs 

    print(f'Epoch {e}: (Training: Loss:{running_loss:.3f} Acc:{acc:.2f})  (Validation: Loss:{val_running_loss:.3f} Acc:{val_acc:.2f})')

    

    # Save the model with the best accuracy on Validation data

    

    if val_acc > max_val_acc:

        torch.save(model.state_dict(),str(e) + "_model.pth")

        max_val_acc = val_acc
import plotly.graph_objs as go



# Plot Loss

fig = go.Figure()

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = history['loss'],

    mode = 'lines',

    name = 'Train loss'   

))

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = history['val_loss'],

    mode = 'lines',

    name = 'Validation loss'   

))



fig.update_layout(

    title = 'Train and Validation Loss'

)

fig.show()
# Plot Accuracy

fig = go.Figure()

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = history['acc'],

    mode = 'lines',

    name = 'Train Accuracy'   

))

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = history['val_acc'],

    mode = 'lines',

    name = 'Validation Accuracy'   

))



fig.update_layout(

    title = 'Train and Validation Accuracy'

)

fig.show()
# Load the test file

test = pd.read_csv('../input/digit-recognizer/test.csv')



# Normalize the data

xtest = test / 255



# Convert to numpy from dataframe 

xtest = np.array(xtest)



xtest.shape
submission = {'ImageId':list(range(1,28001)),'Label':[]}

for row in tqdm(xtest):

    image_tensor = torch.tensor(np.array(row),dtype=torch.float)

    image_tensor = image_tensor.unsqueeze(0)

    

    image_tensor = image_tensor.to(device)

    

    # Set model to evaluation mode

    model.eval()

    

    outputs = model(image_tensor)

    predicted = torch.argmax(torch.exp(outputs).data,1).item()

    submission['Label'].append(predicted)
sub = pd.DataFrame(submission)

sub.to_csv('digits_sub_pytorch.csv',index=False)