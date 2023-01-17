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
# load in the 6 class csv
df = pd.read_csv('/kaggle/input/star-dataset/6 class csv.csv')
# inspect the first few rows of the dataset
df.head()
# provide a column-wise statistical description of the dataframe
df.describe()
# obtain the shape of the dataframe (no. rows x no. columns)
print(df.shape)
# import visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt

# set visualisation parameters
sns.set(rc={'figure.figsize': (45.0, 20.0)})
sns.set(font_scale=8.0)
sns.set_context("notebook", font_scale=5.5, rc={"lines.linewidth": 0.5})
sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'white', 
            'figure.figsize': (45.0, 20.0)}, font_scale=8.0)
ax = sns.set_context("notebook", font_scale=5.5, rc={"lines.linewidth": 0.5})

ax = sns.scatterplot(data=df[df['Star type'] == 0], x="Temperature (K)", y="Absolute magnitude(Mv)",
                s=300, color='red')
ax = sns.scatterplot(data=df[df['Star type'] == 1], x="Temperature (K)", y="Absolute magnitude(Mv)",
                s=300, color='sienna')
ax = sns.scatterplot(data=df[df['Star type'] == 2], x="Temperature (K)", y="Absolute magnitude(Mv)",
                s=300, color='white')
ax = sns.scatterplot(data=df[df['Star type'] == 3], x="Temperature (K)", y="Absolute magnitude(Mv)",
                s=300, color='blue')
ax = sns.scatterplot(data=df[df['Star type'] == 4], x="Temperature (K)", y="Absolute magnitude(Mv)",
                s=300, color='purple')
ax = sns.scatterplot(data=df[df['Star type'] == 5], x="Temperature (K)", y="Absolute magnitude(Mv)",
                s=300, color='yellow')

plt.legend(['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiants', 'Hypergiants'],
          facecolor='w', markerscale=2)
plt.show()
# set visualisation parameters
sns.set(rc={'figure.figsize': (45.0, 20.0)})
sns.set(font_scale=8.0)
sns.set_context("notebook", font_scale=5.5, rc={"lines.linewidth": 0.5})

# plot the distribution of the temperature variable
plt.hist(df['Temperature (K)'], bins='auto', color='red', edgecolor='black', linewidth=2.0)
plt.gca().set(title='Temperature frequency histogram', ylabel='Frequency', xlabel='Temperature (K)')
plt.show()
# plot the distribution of the luminosity variable
plt.hist(df['Luminosity(L/Lo)'], bins='auto', color='green', edgecolor='black', linewidth=2.0)
plt.gca().set(title='Luminosity frequency histogram', ylabel='Frequency', xlabel='Luminosity(L/Lo)')
plt.show()
# plot the distribution of the radius variable
plt.hist(df['Radius(R/Ro)'], bins='auto', color='blue', edgecolor='black', linewidth=2.0)
plt.gca().set(title='Radius frequency histogram', ylabel='Frequency', xlabel='Radius(R/Ro)')
plt.show()
# plot the distribution of the magnitude variable
plt.hist(df['Absolute magnitude(Mv)'], bins='auto', color='orange', edgecolor='black', linewidth=2.0)
plt.gca().set(title='Magnitude frequency histogram', ylabel='Frequency', xlabel='Absolute magnitude(Mv)')
plt.show()
# obtain the data types of each column within the data frame
print(df.dtypes)
# convert columns to 'float64' from 'int64'
df['Temperature (K)'] = df['Temperature (K)'].astype(float)
df['Star type'] = df['Star type'].astype(float)
# obtain One Hot Encoding of the 'Star color' and 'Spectral Class' columns
star_col_dummies = pd.get_dummies(df['Star color'], dtype=float)
spec_cla_dummies = pd.get_dummies(df['Spectral Class'], dtype=float)

# remove original columns from the main dataframe
df = df.drop(['Star color', 'Spectral Class'], axis=1)

# join the One Hot Encoded dataframes onto the main dataframe
df = pd.concat([df, star_col_dummies], axis=1)
df = pd.concat([df, spec_cla_dummies], axis=1)
from sklearn.preprocessing import MinMaxScaler

# variables with the exponential distribution scaled logarithmically
df['Temperature (K)'] = np.log(df['Temperature (K)'])
df['Luminosity(L/Lo)'] = np.log(df['Luminosity(L/Lo)'])
df['Radius(R/Ro)'] = np.log(df['Radius(R/Ro)'])

# define the scaler
scaler = MinMaxScaler()

# variables now scaled with the minmax scaler
df['Temperature (K)'] = scaler.fit_transform(np.expand_dims(df['Temperature (K)'], axis=1))
df['Luminosity(L/Lo)'] = scaler.fit_transform(np.expand_dims(df['Luminosity(L/Lo)'], axis=1))
df['Radius(R/Ro)'] = scaler.fit_transform(np.expand_dims(df['Radius(R/Ro)'], axis=1))
df['Absolute magnitude(Mv)'] = scaler.fit_transform(np.expand_dims(df['Absolute magnitude(Mv)'], axis=1))
# import train_test_split function
from sklearn.model_selection import train_test_split

# split our x and y values into new dataframes
X_values = df.drop(['Star type'], axis=1)
y_values = df['Star type']

# now split our x and y values into train/test sets with a 75/25 percentage split
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.25)
print("X_train shape is", X_train.shape)
print("X_test shape is", X_test.shape)
print("y_train shape is", y_train.shape)
print("y_test shape is", y_test.shape)
# import evaluation metrics
from sklearn import metrics
# import random forest classifier model
from sklearn.ensemble import RandomForestClassifier

# define our random forest classifier
rfc = RandomForestClassifier(n_estimators=100)

# train the model using the x and y training sets
rfc.fit(X_train,y_train)
# apply the model on unseen testing data
rfc_preds = rfc.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, rfc_preds))
# import the support vector classifier model
from sklearn.svm import SVC

# define our support vector classifier model
svc = SVC(kernel='poly')  # polynomial kernel performed best with this experiment

# train the model using the x and y training sets
svc.fit(X_train,y_train)
# apply the model on unseen testing data
svc_preds = svc.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, svc_preds))
# import the K nearest neighbours classifier
from sklearn.neighbors import KNeighborsClassifier

# define the K nearest neighbours model
knc = KNeighborsClassifier()

# train our K nearest neighbours model with the x and y training sets
knc.fit(X_train,y_train)
# apply the model on unseen testing data
knc_preds = knc.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, knc_preds))
# import the gaussian naive bayes model
from sklearn.naive_bayes import GaussianNB

# define our gaussian naive bayes model
gnc = GaussianNB()

# fit our model with training x and y data
gnc.fit(X_train,y_train)
# apply the model on unseen testing data
gnc_preds = gnc.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, gnc_preds))
import torch
import torch.nn as nn

# tensorize our x/y train/test data to form pytorch tensors
X_train_tensor = torch.from_numpy(X_train.to_numpy()).float()
X_test_tensor = torch.from_numpy(X_test.to_numpy()).float()
y_train_tensor = torch.from_numpy(y_train.to_numpy()).long()
y_test_tensor  = torch.from_numpy(y_test.to_numpy()).long()
print("X_train_tensor shape is", X_train_tensor.shape)
print("X_test_tensor shape is", X_test_tensor.shape)
print("y_train_tensor shape is", y_train_tensor.shape)
print("y_test_tensor shape is", y_test_tensor.shape)
# construct the deep learning MLP classifier
class MLP_Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, layer_sizes, dropout):
        super(MLP_Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[4], output_dim),
        )        
        
    # the forward pass through the network
    def forward(self, input_tensor):
        
        output_tensor = self.mlp(input_tensor)  # pass the input tensor through the mlp
        
        return output_tensor
    
# now lets define the model
mlp_classifier = MLP_Classifier(X_train_tensor.shape[1],
                               len(torch.unique(y_train_tensor)),
                               [100, 500, 600, 400, 100],
                               0.1)
print(mlp_classifier)
loss_function = nn.CrossEntropyLoss()  # cross entropy loss function
optimizer = torch.optim.Adam(mlp_classifier.parameters(), lr=0.00005)  # adam's optimiser
epochs = 10000  # number of epochs
loss_vals_train = []  # hold the training loss values
loss_vals_valid = []  # hold the validation loss values

for i in range(epochs):
    y_pred_tensor = mlp_classifier(X_train_tensor)  # obtain y predictions
    single_loss = loss_function(y_pred_tensor[:-20], y_train_tensor[:-20])  # calculate training loss
    loss_vals_train.append(single_loss.item())
    
    # now calculate the validation loss
    with torch.no_grad():  # disable the autograd engine
        val_loss = loss_function(y_pred_tensor[-20:], y_train_tensor[-20:])  # calculate validation loss
        loss_vals_valid.append(val_loss.item())
    
    optimizer.zero_grad()  # zero the gradients
    single_loss.backward()  # backpropagate through the model
    optimizer.step()  # update parameters
    
    if i%250 == 0:
        print(f'epoch: {i:5} training loss: {single_loss.item():10.8f} validation loss: {val_loss.item():10.8f}')
sns.set(rc={'figure.figsize': (45.0, 20.0)})
sns.set(font_scale=8.0)
sns.set_context("notebook", font_scale=5.5, rc={"lines.linewidth": 0.5})
x_vals = np.arange(0, epochs, 1)
ax = sns.lineplot(x=x_vals, y=loss_vals_train)
ax = sns.lineplot(x=x_vals, y=loss_vals_valid)
ax.set_ylabel('Loss', labelpad=20, fontsize=75)
ax.set_xlabel('Epochs', labelpad=20, fontsize=75)
plt.legend(labels=['Training loss', 'Validation loss'], facecolor='white', framealpha=1)
plt.show()
# prepare the model for evaluation
mlp_classifier.eval()

# obtain predictions from unseen testing data, and apply argmax
mlp_preds = mlp_classifier(X_test_tensor)
mlp_preds = np.argmax(mlp_preds.detach().numpy(), axis=1)

print("Accuracy:", metrics.accuracy_score(y_test, mlp_preds))
# convert accuracy scores to percentages and store them as variables
rfc_acc = metrics.accuracy_score(y_test, rfc_preds) * 100
svc_acc = metrics.accuracy_score(y_test, svc_preds) * 100
knc_acc = metrics.accuracy_score(y_test, knc_preds) * 100
gnc_acc = metrics.accuracy_score(y_test, gnc_preds) * 100
mlp_acc = metrics.accuracy_score(y_test, mlp_preds) * 100

# place accuracy scores in lists, and then create a dataframe
models = ['Random Forest \nClassifier', 'Support Vector \nClassifier', 'K Nearest Neighbour\n Classifier',
         'Gaussion Naive \nBayes Classifier', 'Deep Learning \nMLP Classifier']
accuracy = [rfc_acc, svc_acc, knc_acc, gnc_acc, mlp_acc]
model_comp_df = pd.DataFrame({'Model': models, 'Accuracy': accuracy})

# plot a barchart with the accuracy of each model
sns.set_context("notebook", font_scale=4.5, rc={"lines.linewidth": 0.5})
ax = sns.barplot(x="Model", y="Accuracy", data=model_comp_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
ax.set_ylabel('Accuracy (%)', labelpad=50, fontsize=85)
ax.set_xlabel('Model', labelpad=50, fontsize=85)

plt.title("A Barplot comparing the training data accuracy \nof all five classifiers", fontsize=100)
plt.show()
