import numpy as np

import pandas as pd 

from scipy import stats



from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.gridspec import GridSpec



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader



from tqdm import tqdm



np.random.seed(42)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device: {}".format(device))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
print('training dataset size: {}'.format(train_data.shape))

print('test dataset size: {}'.format(test_data.shape))
fig, ax = plt.subplots(ncols=2, figsize = (20,6))



sns.countplot(train_data['Activity'], ax=ax[0])

sns.countplot(test_data['subject'], ax=ax[1])

ax[0].tick_params(axis='x', rotation=45)

ax[0].set_ylabel('Number of observations')

ax[0].set_title('Observations by activity')

ax[1].set_ylabel('Number of observations')

ax[1].set_title('Observations by subject')



plt.show()
le = LabelEncoder()



X_train = train_data.iloc[:,0:(train_data.shape[1]-2)].values

y_train = le.fit_transform(train_data.iloc[:,train_data.shape[1]-1].values)



X_test = test_data.iloc[:,0:(test_data.shape[1]-2)].values

y_test = le.transform(test_data.iloc[:,test_data.shape[1]-1].values)
pca = PCA(n_components=X_train.shape[1])



pca.fit(X_train)

cumulative_variance = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(X_train.shape[1])]



fig, ax = plt.subplots(figsize=(12,6))

ax.plot(cumulative_variance)

ax.set_title('Cumulative proportion of variance explained by principal components')

ax.set_xlabel('Number of components')

ax.set_ylabel('Proportion of variance explained')



plt.show()
tsne = TSNE()



X_reduced = tsne.fit_transform(X_train)
tsne_data = pd.DataFrame(

    {'X':X_reduced[:,0], 'Y':X_reduced[:,1], 

     'activity':train_data['Activity']})



activities = list(tsne_data['activity'].unique())

colormap = ['b', 'g', 'r', 'c', 'm', 'y']



fig, ax = plt.subplots(figsize=(12,12))

for i in range(len(activities)):

    plot_data = tsne_data.loc[tsne_data['activity'] == activities[i]]

    ax.scatter('X', 'Y', data=plot_data, color=colormap[i], label=activities[i])

ax.set_title('2-dimensional $t$-SNE embedding of 561-dimensional accelerometer data')

ax.legend()



plt.show()
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)



test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
class QuickModel(nn.Module):

    def __init__(self, layer_sizes):

        super().__init__()

        

        self.layer_sizes = layer_sizes

        

        structure = []

        for i in range(len(layer_sizes)):

            if i == len(layer_sizes) - 2:

                structure.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

                break

            else:

                structure.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

                structure.append(nn.BatchNorm1d(layer_sizes[i+1]))

                structure.append(nn.ReLU())

                structure.append(nn.Dropout(0.5))

                

        self.layers = nn.ModuleList(structure)

        

    def forward(self, x):

        for layer in self.layers:

            x = layer(x)

        return x

    

def generateRandomArchitecture(num_layers=None, max_layers=10):

    np.random.seed(42)

    if num_layers is None:

        if max_layers <= 3:

            max_layers = 4

        num_layers = np.random.randint(low=3, high=max_layers)

    layer_sizes = []

    for i in range(num_layers):

        if i == 0:

            layer_sizes.append(561)

        elif i == num_layers - 1:

            layer_sizes.append(6)

        else:

            layer_max = min(int(layer_sizes[-1] * 1.5),2000)

            layer_min = int(layer_sizes[-1] / 3)

            if layer_min < 6:

                layer_size = 6

            else:

                layer_size = np.random.randint(low=layer_min, high=layer_max)

            layer_sizes.append(layer_size)

    return layer_sizes, num_layers
nets = []



nets.append(QuickModel([561, 250, 6]).double())

nets.append(QuickModel([561, 124, 32, 16, 6]).double())

nets.append(QuickModel([561, 256, 256, 124, 64, 32, 16, 6]).double())

nets.append(QuickModel(generateRandomArchitecture(num_layers=12)[0]).double())

nets.append(QuickModel(generateRandomArchitecture(num_layers=15)[0]).double())

nets.append(QuickModel(generateRandomArchitecture(num_layers=20)[0]).double())



n_nets = len(nets)

nets_details = [(len(net.layer_sizes), net.layer_sizes) for net in nets]



for ii, net in enumerate(nets_details):

    print("Net {}: {} layers - {}".format(ii+1, net[0], net[1]))
for net in nets:

    net.to(device)



criterion = nn.CrossEntropyLoss()

optimizers = [optim.SGD(net.parameters(), lr=0.003, momentum=0.9) for net in nets]



epochs = 800

train_losses = [[] for net in nets]

test_losses = [[] for net in nets]

test_accuracies = [[] for net in nets]

ensemble_accuracies, ensemble_aucs = [], []



best = [{} for net in nets]



for e in tqdm(range(epochs)):

    for net in nets:

        net.train()

    

    for ii, (inputs, labels) in enumerate(train_dataloader):

        inputs, labels = inputs.to(device), labels.to(device)

        

        for optimizer in optimizers:

            optimizer.zero_grad()



        logits = [net(inputs) for net in nets]

        losses = [criterion(logit, labels) for logit in logits]

        

        for loss in losses:

            loss.backward()

        

        for optimizer in optimizers:

            optimizer.step()

        

        for i in range(n_nets):

            train_losses[i].append(losses[i].item() / inputs.shape[0])

    

    test_loss = [0 for net in nets]

    accuracy = [0 for net in nets]

    

    for net in nets:

        net.eval()    

    for ii, (inputs, labels) in enumerate(test_dataloader):

        inputs, labels = inputs.to(device), labels.to(device)

        

        with torch.no_grad():

            logits = [net(inputs) for net in nets]

            losses = [criterion(logit, labels) for logit in logits]

            

            for i in range(n_nets):

                test_loss[i] += losses[i].item() / inputs.shape[0]

            

            preds = [logit.argmax(dim=1) for logit in logits]

            for i in range(n_nets):

                correct = (preds[i] == labels).type(torch.FloatTensor)

                accuracy[i] += torch.mean(correct).item()

    

    for i in range(n_nets):

        if e == 0:

            best[i]['epoch'] = e

            best[i]['state_dict'] = nets[i].state_dict()

            best[i]['accuracy'] = accuracy[i] / len(test_dataloader)

        elif accuracy[i] / len(test_dataloader) > max(test_accuracies[i]):

            best[i]['epoch'] = e

            best[i]['state_dict'] = nets[i].state_dict()

            best[i]['accuracy'] = accuracy[i] / len(test_dataloader)

        

        test_losses[i].append(test_loss[i] / len(test_dataloader))

        test_accuracies[i].append(accuracy[i] / len(test_dataloader))
train_losses_averaged = [[] for net in nets]

interval = 250

for n in range(len(train_losses)):

    for i in range(0,len(train_losses[n]),interval):

        try:

            train_losses_averaged[n].append(np.mean(train_losses[n][i:i+interval]))

        except IndexError:

            train_losses_averaged[n].append(np.mean(train_losses[n][i:]))
fig = plt.figure(figsize=(20,12))

gs = GridSpec(2, 2, figure=fig)

ax1 = plt.subplot(gs[0, :])

ax2 = plt.subplot(gs[1, 0])

ax3 = plt.subplot(gs[1, 1])



for i in range(n_nets):

    ax1.plot(train_losses_averaged[i], 

             label="Net {}: {} layers".format(i+1, nets_details[i][0]))

    ax2.plot(test_losses[i], alpha=0.6)

    ax3.plot(test_accuracies[i], alpha=0.6)



ax1.set_title('Training loss (mean per 250 batches)')

ax1.set_xlabel('Batch')

ax1.set_ylabel('Loss')

ax1.legend(loc='upper right')



ax2.set_title('Test loss (per epoch)')

ax2.set_xlabel('Epoch')

ax2.set_ylabel('Loss')



ax3.set_title('Test accuracy (per epoch)')

ax3.set_xlabel('Epoch')

ax3.set_ylabel('Accuracy')



plt.show()
for i in range(n_nets):

    print("Net {}: {} layers, maximum test accuracy of {:.4f}".format(

        i+1, nets_details[i][0], best[i]['accuracy']))
for i in range(n_nets):

    nets[i].load_state_dict(best[i]['state_dict'])
tsne = TSNE()



X_test_reduced = tsne.fit_transform(X_test)
tsne_data = pd.DataFrame(

    {'X':X_test_reduced[:,0], 'Y':X_test_reduced[:,1], 

     'activity':test_data['Activity'], 'label':y_test})



activities = list(tsne_data['activity'].unique())

colormap = ['b', 'g', 'r', 'c', 'm', 'y']



fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(30,30))

ax = ax.flatten()



for i in range(n_nets):

    predictions = nets[i](

        torch.from_numpy(X_test).to(device)

    ).argmax(dim=1).cpu().numpy()

    correct_predictions = predictions == y_test



    for j in range(len(activities)):

        data = tsne_data.loc[tsne_data['activity'] == activities[j]]

        ax[i].scatter('X', 'Y', 

                      data=data, color=colormap[j], label=activities[j])

    misclassified_points = tsne_data[np.logical_not(correct_predictions)]

    ax[i].scatter('X', 'Y', 

                  data=misclassified_points, color='black', label='WRONG')

    ax[i].set_title("Model {}: {} layers ({:.2f}% accuracy)".format(

        i+1, nets_details[i][0], best[i]['accuracy']*100))

    ax[i].legend()



plt.show()