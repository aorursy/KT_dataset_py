# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

%config InlineBackend.figure_format = 'retina'
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv") # Replace the path the path to your file

df.head()
plt.style.use("ggplot")
sns.countplot(x="odor", data=df)
sns.countplot(x="cap-surface", data=df)
df.shape
df.columns
categorical_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',

       'ring-type', 'spore-print-color', 'population', 'habitat']

numerical_columns = []

outputs = ['class']
for category in categorical_columns:

    df[category] = df[category].astype('category')
df['stalk-color-below-ring'].cat.categories
vals = [df[cat].cat.codes.values for cat in categorical_columns]

categorical_data = np.stack(vals, 1)

categorical_data[:10]
import torch

from torch import nn, optim
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

categorical_data[:10]
df[outputs].values
df['class'] = df['class'].astype('category')

outputs = torch.tensor(df['class'].cat.codes.values, dtype=torch.long).flatten()

outputs[:5]
numerical_data = torch.tensor([[] for i in range(8124)])
print(categorical_data.shape)

print(numerical_data.shape)

print(outputs.shape)
categorical_column_sizes = [len(df[column].cat.categories) for column in categorical_columns]

categorical_embedding_sizes = [(col_size, min((col_size + 1) // 2, 50)) for col_size in categorical_column_sizes]

categorical_embedding_sizes
total_records = 8124

test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records-test_records]

categorical_test_data = categorical_data[total_records-test_records:total_records]

numerical_train_data = numerical_data[:total_records-test_records]

numerical_test_data = numerical_data[total_records-test_records:total_records]

train_outputs = outputs[:total_records-test_records]

test_outputs = outputs[total_records-test_records:total_records]
print(len(categorical_train_data))

print(len(numerical_train_data))

print(len(train_outputs))



print(len(categorical_test_data))

print(len(numerical_test_data))

print(len(test_outputs))
class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):

        super(Model, self).__init__()

        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])

        self.embedding_drouput = nn.Dropout(p)

        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        

        all_layers = []

        num_categorical_cols = sum((nf for ni, nf in embedding_size))

        input_size = num_categorical_cols + num_numerical_cols

        

        for i in layers:

            all_layers.append(nn.Linear(input_size, i))

            all_layers.append(nn.ReLU(inplace=True))

            all_layers.append(nn.BatchNorm1d(i))

            all_layers.append(nn.Dropout(p))

            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):

        embeddings = []

        for i, e in enumerate(self.all_embeddings):

            embeddings.append(e(x_categorical[:, i]))

        x = torch.cat(embeddings, 1)

        x = self.embedding_drouput(x)

        x_numerical = self.batch_norm_num(x_numerical)

        x = torch.cat([x, x_numerical], 1)

        x = self.layers(x)

        return x
model = Model(categorical_embedding_sizes, numerical_data.shape[1], 2, [300,200, 100,50], p=0.3)
print(model)
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 300

aggregated_losses = []
for i in range(1, epochs + 1):

    y_pred = model(categorical_train_data, numerical_train_data)

    loss = loss_function(y_pred, train_outputs)

    aggregated_losses.append(loss)



    if i%25 == 1:

        print(f'epoch: {i:3} loss: {loss.item():10.8f}')

        

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

print(f'epoch: {i:3} loss: {loss.item():10.10f}')
plt.plot(range(epochs), aggregated_losses)

plt.ylabel('Loss')

plt.xlabel('epoch');
with torch.no_grad():

    y_val = model(categorical_test_data, numerical_test_data)

    loss = loss_function(y_val, test_outputs)

print(f'Loss: {loss:.8f}')
y_val = np.argmax(y_val, axis=1)
print(y_val[:5])
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(test_outputs,y_val))

print(classification_report(test_outputs,y_val))

print(accuracy_score(test_outputs, y_val))