import torch

import torchvision

import torch.nn as nn

import pandas as pd

import matplotlib.pyplot as plt

import torch.nn.functional as F

from torchvision.datasets.utils import download_url

from torch.utils.data import DataLoader, TensorDataset, random_split

import jovian
DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"

DATA_FILENAME = "insurance.csv"

download_url(DATASET_URL, '.')
your_name = 'Jhonathan' # at least 5 characters
def customize_dataset(dataframe_raw, rand_str):

    dataframe = dataframe_raw.copy(deep=True)

    # drop some rows

    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))

    # scale input

    dataframe.bmi = dataframe.bmi * ord(rand_str[1])/100.

    # scale target

    dataframe.charges = dataframe.charges * ord(rand_str[2])/100.

    # drop column

    if ord(rand_str[3]) % 2 == 1:

        dataframe = dataframe.drop(['region'], axis=1)

    return dataframe
dataframe_raw = pd.read_csv(DATA_FILENAME)

dataframe_raw.head()
dataframe = customize_dataset(dataframe_raw, your_name)

dataframe.head()
dataframe.info()
num_rows = dataframe.shape[0]

print(num_rows)
num_cols = dataframe.shape[1]

print(num_cols)
input_cols = dataframe.columns.to_list()

input_cols.remove('charges')

input_cols
categorical_cols = [x for x in dataframe.columns.to_list() if dataframe[x].dtype == 'object']

categorical_cols
output_cols = ['charges']


#Searching the max value into BMI column

max_value = dataframe['charges'].max()

print('Max value for charges: ', max_value)



#Searching the min value into BMI column

min_value = dataframe['charges'].min()

print('Min value for charges: ', min_value)



## The max value for BMI is 55.2552, while thw min value is 16.5984
import seaborn as sns



sns.distplot(dataframe['charges'])

plt.title('CHARGES DATA DISTRIBUTION')

plt.show()

sns.heatmap(dataframe.corr(), annot=True) 

plt.show()
project_name='02-insurance-linear-regression' # will be used by jovian.commit

jovian.commit(project=project_name, envoirement=None)
def dataframe_to_array(dataframe):

    

    dataframe1 = dataframe.copy()

    #Convert categorical data to numbers

    for cols in categorical_cols:

        dataframe1[cols] = dataframe[cols].astype('category').cat.codes



    #From data frame to numpy array

    inputs = dataframe1.drop('charges', axis=1).values

    target = dataframe1[['charges']].values



    return inputs, target

inputs_array, targets_array = dataframe_to_array(dataframe)



print(inputs_array.shape, targets_array.shape)
inputs = torch.tensor(inputs_array, dtype=torch.float32)



targets = torch.tensor(targets_array, dtype=torch.float32)
inputs.dtype, targets.dtype
dataset = TensorDataset(inputs,targets)
val_percent = 0.2 # between 0.1 and 0.2

val_size = int(num_rows * val_percent)

train_size = (num_rows - val_size)





train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 10
train_loader = DataLoader(train_ds, batch_size, shuffle=True)

val_loader = DataLoader(val_ds, batch_size)
for xb, yb in train_loader:

    print("inputs:", xb)

    print("targets:", yb)

    break
input_size = len(input_cols)

output_size = len(output_cols)



input_size, output_size
class InsuranceModel(nn.Module):

    def __init__(self):

        super().__init__()

        #Define linear pytorch method returns the input weighted sum like y = wx + b

        self.linear = nn.Linear(input_size, output_size)

        

    def forward(self, x):

        output = self.linear(x)

        return output

    

    def training_step(self, batch):

        inputs, targets = batch

        #Make prediction

        prediction = self(inputs)

        #Compute loss

        loss = F.l1_loss(prediction, targets)

        return loss

    

    def validation_step(self, batch):

        inputs, targets = batch

        # Generate predictions

        out = self(inputs)

        # Calculate loss

        loss = F.l1_loss(out, targets)                           # fill this    

        return {'val_loss': loss.detach()}



    

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        return {'val_loss': epoch_loss.item()}

    

    def epoch_end(self, epoch, result, num_epochs):

        # Print result every 20th epoch

        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:

            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))

            

        
model = InsuranceModel()
list(model.parameters())
def evaluate(model, val_loader):

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)





#Train loop

def fit(epochs, lr, model, train_batch, val_batch, opt_func=torch.optim.SGD):

    history = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        for batch in train_batch:

            loss = model.train_step(batch)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        #val_step

        result = evaluate(model, val_batch)

        model.epoch_end(epoch, result, epochs)

        history.append(result)

        return history
result = evaluate(model, val_dataset) # Use the the evaluate function

print(result)
model = InsuranceModel()

epochs = 1000

lr = 1e-1

history1= fit(epochs, lr, model, train_loader, val_loader)

epochs = 1500

lr = 1e-3

history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000

lr = 1e-3

history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1000

lr = 1e-4

history4 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 250

lr = 1e-6

history5 = fit(epochs, lr, model, train_loader, val_loader)
val_loss = [result] + history1 + history2 + history3 + history4 + history5

#print(val_loss)

val_loss_list = [vl['val_loss'] for vl in val_loss]



plt.plot(val_loss_list, '-x')



plt.xlabel('epochs')

plt.ylabel('losses')

plt.show()
x = torch.rand((2, 3, 3))

print(x.size())

def predict_single(inputs, target, model):

    inp = inputs.unsqueeze(0)

    predictions = model(inputs)

    prediction = predictions.detach()

    print('Input: ', inputs)

    print('Target: ', target)

    print('Prediction: ',  prediction)
inputs, target = val_ds[0]

predict_single(inputs, target, model)
jovian.commit(project=project_name, environment=None)
jovian.commit(project=project_name, environment=None)
!git init
!git remote  https://github.com/JhonathanOrtiz/FreeCodeCampCourse.git