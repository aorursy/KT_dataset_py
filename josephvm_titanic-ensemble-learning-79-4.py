import pandas as pd

import numpy as np

import matplotlib.pyplot as plt # Not used as of now
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")
train_df.head()
test_df.head()
train_df.info()
test_df.info()
corr = train_df.corr()

corr
# Impute, remove, encode, scale

# (Only scale for ANN (for now))
# Age, cabin, & embarked are incomplete (in the training set)

# Age, cabin, & fare are incomplete (in the test set)



# age and fare can be imputed using the mean

# embarked - most common (only missing 2)

# cabin can probably be safely dropped due to how many are missing
train_labels = train_df['Survived'] # get separate labels array

X_train = train_df.drop(['Survived'], axis=1) # drop labels out of df
X_train # Missing 'Survived' now, good
# Import

from sklearn.impute import SimpleImputer
# Imputer for age and fare

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Imputer for embarked

common_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # decent guess as any
mean_imputer.fit(X_train[['Age', 'Fare']])

common_imputer.fit(X_train[['Embarked']])
X_train[['Age', 'Fare']] = mean_imputer.transform(X_train[['Age', 'Fare']]);

test_df[['Age', 'Fare']] = mean_imputer.transform(test_df[['Age', 'Fare']]);



X_train[['Embarked']] = common_imputer.transform(X_train[['Embarked']]);

test_df[['Embarked']] = common_imputer.transform(test_df[['Embarked']]);
X_train.info()
col_to_drop = ['Name', 'Ticket', 'Cabin']
X_train = X_train.drop(col_to_drop, axis=1)

test_df = test_df.drop(col_to_drop, axis=1)
X_train.info() #; test_df.info() # easier to comment out
X_train.head()
# Need to encode sex & embarked

# Drop Id
X_train['AgeRange'] = ['child' if x < 13 else 'teenager' if x < 22 

                       else 'adult' if x < 66 else 'senior' for x in X_train['Age']]

test_df['AgeRange'] = ['child' if x < 13 else 'teenager' if x < 22 

                       else 'adult' if x < 66 else 'senior' for x in test_df['Age']]
X_train.head()
# Make a new column that contains the # of family members aboard

X_train['FamNum'] = X_train['SibSp'] + X_train['Parch']

test_df['FamNum'] = test_df['SibSp'] + test_df['Parch']



# Remove SibSp and Parch as they were just used

# Remove Age as we have AgeRange now

# Remove PassengerId as it doesn't really tell us anything

X_train = X_train.drop(['SibSp', 'Parch', 'PassengerId', 'Age'], axis=1)

test_df = test_df.drop(['SibSp', 'Parch', 'PassengerId', 'Age'], axis=1)
# A quick idea of what is strongly correlated with Survived

pd.concat([X_train, train_labels], axis=1).corr()['Survived']
# Nothing super crazy, but Pclass is nice to see
need_encoding = ['Sex', 'Embarked', 'AgeRange'] # Marked for later removal
# Manual one-hot encoding on train set using list comprehensions

# Gender

X_train['Male'] = [1 if x == 'male' else 0 for x in X_train['Sex']]

X_train['Female'] = [1 if x == 'female' else 0 for x in X_train['Sex']]



# Embarked

X_train['S'] = [1 if x == 'S' else 0 for x in X_train['Embarked']]

X_train['C'] = [1 if x == 'C' else 0 for x in X_train['Embarked']]

X_train['Q'] = [1 if x == 'Q' else 0 for x in X_train['Embarked']]



# AgeRange

X_train['child'] = [1 if x == 'child' else 0 for x in X_train['AgeRange']]

X_train['teenager'] = [1 if x == 'teenager' else 0 for x in X_train['AgeRange']]

X_train['adult'] = [1 if x == 'adult' else 0 for x in X_train['AgeRange']]

X_train['senior'] = [1 if x == 'senior' else 0 for x in X_train['AgeRange']]
# Manual one-hot encoding on test set using list comprehensions

# Gender

test_df['Male'] = [1 if x == 'male' else 0 for x in test_df['Sex']]

test_df['Female'] = [1 if x == 'female' else 0 for x in test_df['Sex']]



# Embarked

test_df['S'] = [1 if x == 'S' else 0 for x in test_df['Embarked']]

test_df['C'] = [1 if x == 'C' else 0 for x in test_df['Embarked']]

test_df['Q'] = [1 if x == 'Q' else 0 for x in test_df['Embarked']]



# AgeRange

test_df['child'] = [1 if x == 'child' else 0 for x in test_df['AgeRange']]

test_df['teenager'] = [1 if x == 'teenager' else 0 for x in test_df['AgeRange']]

test_df['adult'] = [1 if x == 'adult' else 0 for x in test_df['AgeRange']]

test_df['senior'] = [1 if x == 'senior' else 0 for x in test_df['AgeRange']]
# Can safely drop the original columns now out of both

X_train = X_train.drop(need_encoding, axis=1)

test_df = test_df.drop(need_encoding, axis=1)
# Let's see what it looks like now

X_train.head()
from sklearn.ensemble import RandomForestClassifier



rf_classifier = RandomForestClassifier()
# Grid Search

from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [100], 

               'criterion': ['gini'],

               'min_samples_split': [2, 3],

               'min_samples_leaf': [1, 2],

               'max_features': ['auto', None],

               'max_depth': [None, 8, 10, 12]

              }]

# Get it ready

grid_search = GridSearchCV(rf_classifier, param_grid, cv=7, verbose=1)
grid_search.fit(X_train, train_labels) # Train it (shouldn't take too long)
grid_search.best_params_ 

# Usually {gini, 8, None, 2, 2, 100}
grid_search.best_score_ 

# ~ 83.28%
# Save the best random forest

rf_classifier = grid_search.best_estimator_
from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier()
# Grid Search

from sklearn.model_selection import GridSearchCV

param_grid = [{'n_neighbors': [2, 3, 4], 

               'algorithm': ['ball_tree', 'kd_tree'],

               'leaf_size': [10, 20, 30]

              }]

# Get it ready

grid_search = GridSearchCV(knn_classifier, param_grid, cv=7, verbose=1)
grid_search.fit(X_train, train_labels) # Fit it
grid_search.best_params_ 

# ball_tree, 10, 3 last time
grid_search.best_score_ 

# ~ 80.36%
# Save the best K-NN

knn_classifier = grid_search.best_estimator_
# This one is a bit messier
# Quick function that gets how many out of 'preds' match 'labels'

def get_num_correct(preds, labels):

    return preds.argmax(dim=1).eq(labels).sum().item()
# Imports

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
# Scale the data for the ANN

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
# Scale the data

X_train_scaled = scaler.fit_transform(X_train)

test_ann = scaler.transform(test_df) # To be used later
# Want a train and val set now

from sklearn.model_selection import train_test_split



# Split the train set into train & val 

# Could try again w/o splitting the train set w/ a set NN architecture

X_train_part, X_val, y_train, y_val = train_test_split(X_train_scaled, train_labels, test_size=0.10)
X_train_part.shape
# Our Artificial Neural Network class

# Could play around with this a lot more

class ANN(nn.Module):

    def __init__(self):

        super(ANN, self).__init__()

      

        self.fc1 = nn.Linear(in_features=12, out_features=32) # linear 1

        self.fc1_bn = nn.BatchNorm1d(num_features=32)

        self.drop1 = nn.Dropout(.1)

        

        self.fc2 = nn.Linear(in_features=32, out_features=8) # linear 2

        self.fc2_bn = nn.BatchNorm1d(num_features=8)

        

        self.out = nn.Linear(in_features=8, out_features=2) # output

    

    def forward(self, t):

        t = F.relu(self.fc1_bn(self.fc1(t)))

        t = self.drop1(t)

        t = F.relu(self.fc2_bn(self.fc2(t)))

        

        t = self.out(t)

        return t
# Gets the GPU if there is one, otherwise the cpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
batch_size = 64 # got a weird error if there was a different batch size



# Some weird TensorDataset stuff

# It wants tensors of type float, but they were numpy arrays of not all float

train_set = TensorDataset(torch.from_numpy(X_train_part.astype(float)), 

                          torch.from_numpy(y_train.as_matrix().astype(float)))

val_set = TensorDataset(torch.from_numpy(X_val.astype(float)), 

                        torch.from_numpy(y_val.as_matrix().astype(float)))



# Load up the data and shuffle away

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True)
lr = 0.001 # initial learning rate

epochs = 200 # number of epochs to run



network = ANN().float().to(device) # put the model on device (hopefully a GPU!)

optimizer = optim.Adam(network.parameters(), lr=lr) # Could try a different optimizer



# It wanted the X to be float and the y to be long, so I complied

for epoch in range(epochs):

    network.train() # training mode

    

    # decrease the learning rate a bit

    if epoch == 40:

        optimizer = optim.Adam(network.parameters(), lr=0.0001)

    

    # decrease the learning rate a bit more

    if epoch == 80:

        optimizer = optim.Adam(network.parameters(), lr=0.00000000001)

        

    for features, labels in train_dl:

        X, y = features.to(device), labels.to(device) # put X & y on device

        y_ = network(X.float()) # get predictions

        

        optimizer.zero_grad() # zeros out the gradients

        loss = F.cross_entropy(y_, y.long()) # computes the loss

        loss.backward() # computes the gradients

        optimizer.step() # updates weights

          

    # Evaluation with the validation set

    network.eval() # eval mode

    val_loss = 0

    val_correct = 0

    with torch.no_grad():

        for features, labels in val_dl:

            X, y = features.to(device), labels.to(device) # to device

            

            preds = network(X.float()) # get predictions

            loss = F.cross_entropy(preds, y.long()) # calculate the loss

            

            val_correct += get_num_correct(preds, y.long())

            val_loss = loss.item() * batch_size

            

    # Print the loss and accuracy for the validation set

    if (epoch % 10) == 9: # prints every 10th epoch

        print("Epoch: ", epoch+1)

        print(" Val Loss: ", val_loss)

        print(" Val Acc: ", (val_correct/len(X_val))*100)



# Output can get a bit long
# Helps a bit to reduce dimensions for linear regression (or so I remember)

X_lin = X_train.drop(['S', 'C', 'Q', 'teenager', 'adult'], axis=1)

test_lin = test_df.drop(['S', 'C', 'Q', 'teenager', 'adult'], axis=1) # need to use later
from sklearn.linear_model import LinearRegression



linearreg = LinearRegression(normalize=True, copy_X=True)
linearreg.fit(X_lin, train_labels)
# rf_classifier, network, knn_classifier, linearreg

# test_df = test set
# Get the test set ready for the ANN because it's special

test = TensorDataset(torch.from_numpy(test_ann.astype(float)) )

test_tensor = DataLoader(test, batch_size=batch_size, shuffle=False)
# Get predictions



# Predictions for the Random Forest ---------------

rf_preds = rf_classifier.predict(test_df)



# Predictions for the ANN -------------------------

ann_preds = torch.LongTensor().to(device) # Tensor for all predictions

network.eval() # safety

for batch in test_tensor:

    batch = batch[0].to(device) # just batch is a [tensor] for some reason

    predictions = network(batch.float()) # again with the float thing

    ann_preds = torch.cat((ann_preds, predictions.argmax(dim=1)), dim=0) 

# bring it back to the cpu and convert to an array

ann_preds = ann_preds.cpu().numpy()



# Predictions for the K-Nearest Neighbors ---------

knn_preds = knn_classifier.predict(test_df)



# Predictions for the Linear Regression -----------

lin_preds = linearreg.predict(test_lin) # special test set with less columns
lin_preds[:5] # Not quite how we want them
lin_preds = np.around(lin_preds, decimals=0).astype(int) # Rounds them
# Interesting to see

print(np.sum(rf_preds==ann_preds), "/", rf_preds.shape[0], " same predictions between Random Forest and ANN")

print(np.sum(rf_preds==knn_preds), "/", rf_preds.shape[0], " same predictions between Random Forest and K-NN")

print(np.sum(rf_preds==lin_preds), "/", rf_preds.shape[0], " same predictions between Random Forest and Linear Reg")

print(np.sum(ann_preds==knn_preds), "/", rf_preds.shape[0], " same predictions between ANN and K-NN")

print(np.sum(ann_preds==lin_preds), "/", rf_preds.shape[0], " same predictions between ANN and Linear Reg")

print(np.sum(knn_preds==lin_preds), "/", rf_preds.shape[0], " same predictions between K-NN and Linear Reg")
# Add them all up

agg_preds = rf_preds + ann_preds + lin_preds + knn_preds

agg_preds # values 0-4 now
values, counts = np.unique(agg_preds, return_counts=True) # sum number of 0s..., 4s

for i in range(5):

    print(values[i], " classifiers predicted 'Survive'", counts[i], " times", )
# Time to get the final predictions

final_preds = np.empty(len(agg_preds), dtype=int) # empty predictions array



# Survived if agg_preds has 4 or 3

# Didn't survive if agg_preds has 0 or 1

# Up to the Random Forest if agg_preds is split at 2

for i in range(len(agg_preds)): # go through agg_preds

    if agg_preds[i] < 2:

        final_preds[i] = 0

    elif agg_preds[i] > 2:

        final_preds[i] = 1

    else: # final call goes to random forest

        final_preds[i] = rf_preds[i]
final_preds # Beautiful!
# Read in sample csv

sample_df = pd.read_csv("../input/titanic/gender_submission.csv")
# Edit it

sample_df['Survived'] = final_preds
# Write to a new csv

sample_df.to_csv("predictions.csv", index=False) # Be sure to not include the index
# 79.425 on test set (Kaggle) (2164th place when submitted)

# Top 20%