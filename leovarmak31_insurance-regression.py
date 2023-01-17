# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install jovian --upgrade
import jovian

import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url

from torch.utils.data import DataLoader, TensorDataset, random_split
project_name='02-insurance-linear-regression' 
DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')
dataframe_raw = pd.read_csv(DATA_FILENAME)
dataframe_raw.head()
len(dataframe_raw)

dataframe_raw.dropna()
dataframe_raw.describe()
your_name = 'Preeti'
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
dataframe = customize_dataset(dataframe_raw, your_name)
dataframe.head()
num_rows = dataframe.shape[0]
num_rows
num_cols = dataframe.shape[1]
num_cols
input_cols = ["age","sex","bmi","children","smoker","charges"]
input_cols
categorical_cols = ['sex', 'smoker']
dataframe.head()[categorical_cols]
output_cols = ['charges']
print(dataframe[output_cols])
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
t = dataframe[output_cols]   #avg
avg = t/len(dataframe)
sns.distplot(avg)

e = t.max()
sns.barplot(e )
f = t.min()
sns.barplot(f)
jovian.commit(project=project_name, environment=None)
dataframe[output_cols].mean()
dataframe[output_cols].min()

dataframe[output_cols].max()
plt.title('Distribution')
sns.distplot(dataframe[output_cols], kde=False)
plt.title('Distribution')
sns.scatterplot(dataframe['charges'],dataframe_raw['bmi'])
sns.barplot(dataframe['age'],dataframe_raw['bmi'])
avg_smokers_charges = dataframe[dataframe.smoker == 'yes'][output_cols].mean()
avg_non_smokers_charges = dataframe[dataframe.smoker == 'no'][output_cols].mean()
print('Average smokers', avg_smokers_charges)
print('Average non smokers', avg_non_smokers_charges)
plt.show()
avg_smokers_charges/avg_non_smokers_charges
sns.catplot(x="sex", y="charges", hue="smoker",
            kind="violin", data=dataframe, palette = 'magma')
sns.boxplot(y="smoker", x="charges",data = dataframe[(dataframe.age == 18)],palette='pink')
plt.figure(figsize=(12,5))
plt.title("Distribution of bmi")
ax = sns.distplot(dataframe["bmi"], color = 'm')
def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_arrays(dataframe)
inputs_array.shape, inputs_array, targets_array
inputs = torch.from_numpy(inputs_array).type(torch.float32)
targets = torch.from_numpy(targets_array).type(torch.float32)
inputs.dtype, targets.dtype
dataset = TensorDataset(inputs, targets)
from sklearn.model_selection import train_test_split

val_percent = 0.1 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_ds, val_ds = random_split(dataset, [train_size, val_size])
#refrence from https://jovian.ml/kir-prz/02-insurance-linear-regression
batch_size = 32
train_loader = DataLoader(train_ds,  
                          batch_size,
                         shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

for xb, yb in train_loader:
    print("inputs:", xb)
    print("targets:", yb)
    break
jovian.commit(project=project_name, environment=None)
input_size = len(input_cols)
output_size = len(output_cols)
class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)               # fill this (hint: use input_size & output_size defined above)
        
    def forward(self, xb):
        out = self.linear(xb)                        # fill this
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets)                       # fill this
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out,targets)               # fill this    
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

jovian.commit(project=project_name, environment=None)
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history

result = evaluate(model, val_loader)
print(result)
epochs = 5000
lr = 1.5e-1
history2 = fit(epochs, lr, model, train_loader, val_loader)
loss = []
for values in history2:
    loss.append(values['val_loss'])
plt.plot(loss)
jovian.commit(project=project_name, environment=None)



