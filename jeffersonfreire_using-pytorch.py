# Importing the libraries
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torchvision
data_train = pd.read_csv('../input/digit-recognizer/train.csv')
data_test = pd.read_csv('../input/digit-recognizer/test.csv')
sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv' )
# visualize the dimensions of the training data
data_train.shape
# visualize the dimensions of the test data
data_test.shape
# Visualize the first five training data
data_train.head()
# Visualize the first five test data
data_test.head()
# Getting all rows from column 1 to column 785 using the pandas class .iloc [], returning an array
train = data_train.iloc[:, 1: 785 ].values
label = data_train.iloc[:, 0].values
# visualizing the element in position 3 of our training data set 
train[3]
# Mapeamos nosso data set de treinamento
normalizer = MinMaxScaler()
train = normalizer.fit_transform( train, ( 0, 1 ))
# We divided the training data set into 20% for training and 80% for testing, but it is up to the reader to choose.
X_train, X_test, y_train, y_test =  train_test_split( train, label, test_size= 0.20, random_state= 0)
X_train.shape
# convert for tensors and resize them to four dimensions 
X_train = torch.tensor( X_train ).reshape( -1, 1, 28, 28)
X_teste = torch.tensor( X_test).reshape( -1, 1, 28, 28)
# Plotting our first image in position six
image = next(iter( X_train[6] )).view(28, 28) # (1, 1, 28, 28)
plt.imshow(image, cmap='gray')
# Plotting our second image in position two.
image = next(iter( X_train[2] )).view(28, 28) # (1, 1, 28, 28)
plt.imshow(image, cmap='gray')
print(X_train.shape)
print(X_teste.shape)
# Transforming data for tensors
Y_train = torch.tensor( y_train )
Y_test = torch.tensor( y_test )
print(Y_train.shape)
print(Y_test.shape)
# A data set of tensor
dataset_train = torch.utils.data.TensorDataset( X_train, Y_train)
dataset_test = torch.utils.data.TensorDataset( X_teste, Y_test )

# Data Loader
train_loader = torch.utils.data.DataLoader( dataset_train, batch_size= 32, shuffle= True)
test_loader = torch.utils.data.DataLoader( dataset_test, batch_size= 32, shuffle= True)
# Mapping the test data
test = normalizer.fit_transform( data_test, (0,1))
test.shape
# Resizing the data
test = torch.tensor( test , dtype= torch.float ).reshape(-1, 1, 28, 28)
test.shape
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
    
        self.conv_input = nn.Conv2d( in_channels= 1, out_channels= 32, kernel_size= ( 3, 3 ))
        self.conv_hidden = nn.Conv2d( in_channels= 32, out_channels=32, kernel_size= ( 3, 3 ))
        self.function_activation = nn.ReLU()
    
        self.pool = nn.MaxPool2d(kernel_size=( 2, 2))
        self.flaten = nn.Flatten()
        
        self.layer_input = nn.Linear( in_features= 32*5*5, out_features= 128 )
        self.layer_hidden = nn.Linear( 128, 128 )
        self.layer_output = nn.Linear( 128, 10 )
        self.dropout = nn.Dropout( 0.2 )

    def forward( self,  X ):
        X = self.pool( self.function_activation( self.conv_input( X )))
        X = self.pool( self.function_activation( self.conv_hidden( X )))
        X = self.flaten( X )

        X = self.dropout( self.function_activation( self.layer_input( X )))
        X = self.dropout( self.function_activation( self.layer_hidden( X )))
        X = self.layer_output( X )

        return X
# Constructor of our Model class
model = Model()
# Our criteria for calculating our losses
critetion = nn.CrossEntropyLoss()
# Our optimizer. Passing our CNN model as a parameter, 
# but it is up to the reader to make their choices of optimizer.
optmizer = optim.Adam( model.parameters())
model
def Train_loop( loader, epocha):
    running_loss = 0.
    running_accuracy = 0.
    
    for i, data in enumerate( loader ):

        inputs, labels = data
        optmizer.zero_grad()
        
        outputs = model( inputs.float() )
        loss = critetion( outputs,  labels)
        loss.backward()

        optmizer.step()

        running_loss += loss.item()

        ps = F.softmax( outputs )
        top_p, top_class = ps.topk( k=1, dim= 1)
        equals = top_class == labels.view( *top_class.shape )

        accuracy = torch.mean( equals.type( torch.float ))

        running_accuracy += accuracy

        # Printing the data for this loop
        print('\rEphoc {:3d} - Loop {:3d} in {:3d}: loss {:03.2f} - accuracy {:03.2f}'.format(epocha + 1, i + 1, len(loader), 
                                                                                              loss, accuracy), end = '\r')

    print('\rEphoc {:3d} Finish: loss {:.5f} - accuracy {:.5f}'.format(epocha+1, running_loss/len(loader), 
                     running_accuracy/len( loader )))
for epocha in range( 15 ):
    print('Training....')
    Train_loop( train_loader, epocha )
    print('Testing....')
    #   Moving the model to the evaluation mode  
    model.eval()
    Train_loop( test_loader, epocha )
    #   Moving the model to the training mode
    model.train()
# Save our model
torch.save( model.state_dict(), 'checkpoint.pth')
model_forecast = model.eval()
forecast = model_forecast.forward( test )
forecast
forecast = F.softmax( forecast )
forecast = forecast.cpu().detach().numpy()
forecast
results = [] 
for i in range(len( forecast )):
    results.append(np.argmax( forecast[i] )) 
results_f = np.array( results )
# Salving submission
results =pd.DataFrame()
results['ImageId'] = sample_submission['ImageId']
results['Label'] = results_f.astype(int)
results.head()
results.to_csv('submission_finish2.csv',index=False)