import matplotlib

import matplotlib.pyplot as plt

import torch

import pandas as pd

import numpy as np

print(f'matplotlib: {matplotlib.__version__}')

print(f'pytorch   : {torch.__version__}')

print(f'pandas    : {pd.__version__}')

print(f'numpy     : {np.__version__}')
# Load the data

df = pd.read_csv("../input/titanic/train.csv")

df.describe()
df.head(10)
from sklearn import preprocessing



def format_feats(in_feats):

    x = in_feats.values #returns a numpy array

    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)

    return pd.DataFrame(x_scaled, columns=in_feats.columns)



# Apply some data formatting

def format_data(data):

    # One-hot encode 'Embarked' column

    data = pd.get_dummies(data, columns=['Sex','Embarked'])

    # Drop columns that require additional processing

    data = data.drop(['Name','Ticket','Cabin'], axis=1)

    # Fill null values with the mean of the column

    data.fillna(data.mean(), inplace=True)

    if 'Survived' in data.columns:

        data_y = data['Survived']

        data_x = data.drop(['Survived'], axis=1)

        data_x = format_feats(data_x)

        return data_x, data_y

    else:

        return format_feats(data)



# This should split the data into our features and our labels

feats, labels = format_data(df)

feats.describe()
# Split the data into training and testing samples

# The training sample should consist of ~80% of our data

mask  = np.random.rand(len(feats)) < 0.8

train_X = feats[mask]

train_y = labels[mask]

test_X  = feats[~mask]

test_y  = labels[~mask]



# Look at the training sample

train_X.describe()

print(train_X.describe(), test_y.describe())
# Format the data into PyTorch tensors

trn_X = torch.Tensor(train_X.to_numpy())

trn_y = torch.Tensor(train_y.to_numpy()).type(torch.LongTensor)

tst_X = torch.Tensor(test_X.to_numpy())

tst_y = torch.Tensor(test_y.to_numpy()).type(torch.LongTensor)



# Get the number of inputs

drpout = 0.2
# Generate the model

from torch import nn



# Set Dropout rate

drpout = 0.1

# Define number of inputs

inputs = len(trn_X[0])



# Method for initializing weights and biases

def set_weight_bias(layer):

    layer.bias.data.fill_(0)

    layer.weight.data.normal_(std=0.01)



# Create a function for model construction

# This will help 

def model_construct(inputs, n=[16], outputs=2,

                    activ=nn.ReLU):

    # Add the outputs to the list of nodes

    n.append(outputs)

    

    # Input layer

    layers = []

    layers.append(nn.Linear(inputs, n[0]))

    set_weight_bias(layers[-1])

    layers.append( nn.Dropout(p=drpout) )

    layers.append(activ())

    

    # Loop over the hidden layers

    for i in range(len(n)-1):

        layers.append(nn.Linear(n[i], n[i+1]))

        set_weight_bias(layers[-1])

        layers.append( nn.Dropout(p=drpout) )

        layers.append(activ())

        

    # Remove the last dropout layer

    layers.pop()

    layers.pop()

    # Change final activation function

    #layers[-1] = nn.Softmax(dim=1)

    

    # Put it all together

    return nn.Sequential(*layers)
# Write another function for training and testing the model

from torch import optim

from sklearn.utils import shuffle

from torch.autograd import Variable



def train_model(model, train_data, test_data, epochs=5, verbose=False):

    

    # Setup

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    

    # Loop over the epochs

    train_losses, test_losses = [0]*epochs, [0]*epochs

    accuracy = [0]*epochs

    for e in range(epochs):

        

        # Iterate the model, note we are passing in the

        # entire training set as a single batch

        optimizer.zero_grad()

        ps = model(train_data[0])

        loss = criterion(ps, train_data[1])

        loss.backward()

        optimizer.step()

        train_losses[e] = loss.item()



        # Compute the test stats

        with torch.no_grad():

            # Turn on all the nodes

            model.eval()

            

            # Comput test loss

            ps = model(test_data[0])

            loss = criterion(ps, test_data[1])

            test_losses[e] = loss.item()

            

            # Compute accuracy

            top_p, top_class = ps.topk(1, dim=1)

            equals      = (top_class == test_data[1].view(*top_class.shape))

            accuracy[e] = torch.mean(equals.type(torch.FloatTensor))

            

        model.train()

        

    # Print the final information

    print(f'   Accuracy  : {100*accuracy[-1].item():0.2f}%')

    print(f'   Train loss: {train_losses[-1]}')

    print(f'   Test loss : {test_losses[-1]}')

        

    # Plot the results

    plt.subplot(211)

    plt.ylabel('Accuracy')

    plt.plot(accuracy)

    plt.subplot(212)

    plt.ylabel('Loss')

    plt.plot(train_losses, label='train')

    plt.plot(test_losses, label='test')

    plt.legend();

    return
# Give it a try

print("Test 1:")

model = model_construct(inputs, n=[256])

print(model)

train_model(model, epochs=100,

            train_data=(trn_X,trn_y), test_data=(tst_X,tst_y))
print("Test 2:")

model = model_construct(inputs, n=[256, 64])

print(model)

train_model(model, epochs=200,

            train_data=(trn_X,trn_y), test_data=(tst_X,tst_y))
print("Test 3:")

model = model_construct(inputs, n=[16])

print(model)

train_model(model, epochs=1000,

            train_data=(trn_X,trn_y), test_data=(tst_X,tst_y))
# Assign the training data to the full training set

trn_X = torch.Tensor(feats.to_numpy())

trn_y = torch.Tensor(labels.to_numpy()).type(torch.LongTensor)



# Construct and fit the model

model = model_construct(inputs, n=[16])

train_model(model, epochs=100,

            train_data=(trn_X,trn_y), test_data=(tst_X,tst_y))
def gen_model(scale):

    """

    Generate and fit a model



    Parameters

    ----------

    scale : int

        Number of models that are being trained

    

    Returns

    -------

    Model generated and trained.

    """

    # Generate a model

    mod = model_construct(inputs, n=[32,16])

    

    # Update the data subset

    mask  = np.random.rand(len(feats)) < 1.0/(scale/2.0)

    train_dat = (torch.Tensor(feats[mask].to_numpy()),

                 torch.Tensor(labels[mask].to_numpy()).type(torch.LongTensor))

    test_dat = (torch.Tensor(feats[~mask].to_numpy()),

                torch.Tensor(labels[~mask].to_numpy()).type(torch.LongTensor))

    

    # Train the model

    train_model(mod, epochs=100,

                train_data=train_dat, test_data=test_dat)

    

    # Return the trained model

    return mod
# Function to combine the probabilities from all models

def combined_pred(models, data):

    # Loop through models and get the probabilities

    prob = np.array([[0.0]*2]*len(data))

    with torch.no_grad():

        for mod in models:

            mod.eval()

            prob += torch.exp(mod(data)).numpy()

            mod.train()

    return prob/len(models)
# Generate a fit a group of models

models = []

mod_size = 4

for i in range(mod_size):

    models.append(gen_model(mod_size))

preds = combined_pred(models, tst_X)

top_class = np.argmax(preds, axis=1)

equals = (top_class == tst_y.numpy())

print(100*np.mean(equals))
# Load and process the testing data

test_df    = pd.read_csv("../input/titanic/test.csv")

test_feats = format_data(test_df)

test_feats = torch.Tensor(test_feats.to_numpy())



# Compute the results

#results          = model(test_feats)

#top_p, top_class = results.topk(1, dim=1)

results = combined_pred(models, test_feats)

top_class = np.argmax(results, axis=1)



# Load it all into a dataframe

submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 

                              'Survived'   : top_class})

submission_df.describe()
submission_df.to_csv('submission.csv', index=False)