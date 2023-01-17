import pandas as pd
import torch
dataset = pd.read_csv('../input/prima-diabetes-dataset/diabetes.csv')
print(dataset.shape)
dataset.head()
dataset.isnull().sum()
X = dataset.drop('Outcome', axis = 1).values ## independent features
y = dataset['Outcome'].values ## dependent features
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
### Libraries from Pytorch
import torch
import torch.nn as nn   ## Helps you to create and train the neural networks
import torch.nn.functional as F  ## this functional F contains functions like sigmoid, relu etc...
### Creating Tensors

## Remember that your independent features need to be converted into Float Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

## No need to convert to float tensors in case of dependent features
y_train = torch.LongTensor(y_train)  ## LongTensor won't convert values into float tensors
y_test = torch.LongTensor(y_test)
### Creating model with pytorch

class ANN_Model(nn.Module):
    def __init__(self, input_features = 8, hidden1 = 20, hidden2 = 20, out_features = 2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)
    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x
## Instantiate ANN_MODEL

torch.manual_seed(20) ##it will set the seed of the random number generator to a fixed value, so that when you call for example torch.rand(2), the results will be reproducible
model = ANN_Model()
model.parameters
model.parameters() ## it's an generator, so we can iterate and retirve all the parameter one by one
## Backward Propagation | Define the Loss Function | Define the Optimizer

loss_function = nn.CrossEntropyLoss() ## for multiclassification problem use CrossEntropyLoss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
epochs = 500
final_loss = []
for i in range(epochs):
    i = i + 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    final_loss.append(loss)
    
    ## after eveery 10 epochs print this
    if i % 10 == 1:
        print('Epoch number: {} and the loss: {}'.format(i, loss.item()))
        
    optimizer.zero_grad() ## Clears the gradients of all optimized class
    loss.backward() ## for backward propagation and to find the derivative
    optimizer.step() ## performs a single optimization step.
## Plot the loss function
import matplotlib.pyplot as plt
%matplotlib inline
## To see the if loss is decreasing or not
plt.plot(range(epochs), final_loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
## Let's find out the prediction

## Prediction in X_test data
prediction = []  

## To remove this gradient --> (grad_fn=<AddBackward0>). it is required when we are taining the model. here it has no use.
with torch.no_grad():
    for i, data in enumerate(X_test):
#         print(model(data)) ## Here model is out ANN_Model
        y_pred = model(data)
        prediction.append(y_pred.argmax().item())
#         print(y_pred.argmax().item())  ## 1 = Diabetic Person     0 = No Diabetic Person
## list of prediction. ## 1 = Diabetic Person     0 = No Diabetic Person
prediction
## Check the accuracy by using confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
cm

# result: 92+34 = 126 are true predicted values and 15+13 = 28 are wrong predicted values
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, prediction)
acc_score ## Accuracy is 81%
## Save the model
torch.save(model, 'diabetes.pt') ## pt is extension for pytorch files
## To load the model
model = torch.load('diabetes.pt')
model.eval()
## how to do prediction of the new data points

## Let's take the only first column of dataset and change it's values for prediction
dataset.iloc[0, :-1].head()
list(dataset.iloc[0, :-1])
## New data
lst = [6.0, 130.0, 72.0, 40.0, 0.0, 33.6, 0.627, 45.0]
new_data = torch.tensor(lst) ## Converting into torch tensors.
new_data
new_data
## Predict new_data using PyTorch

## since this is just a single list, we don't have to use enumeraet
with torch.no_grad():
    print(model(new_data))
    print(model(new_data).argmax())
    print(model(new_data).argmax().item())  
    
## result = 1. which means for this data point(single list that we provided) person is diabetic.
