import numpy as np

from os import path

import matplotlib.pyplot as plt



import sklearn

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



import torch

import torch.nn as nn

import torch.optim as optim
!pip install captum
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
boston = load_boston()
feature_names = boston.feature_names

X = boston.data

y = boston.target
torch.manual_seed(12134)
np.random.seed(1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(30, 20))

for i, (ax, col) in enumerate(zip(axs.flat, feature_names)):

    x = X[:, i]

    pf = np.polyfit(x, y, 1)

    p = np.poly1d(pf)

    ax.plot(x, y, 'o')

    ax.plot(x, p(x), 'r--')

    ax.set_title(col + ' vs Prices')

    ax.set_xlabel(col)

    ax.set_ylabel('Prices')
X_train = torch.tensor(X_train).float()

y_train = torch.tensor(y_train).view(-1, 1).float()

X_test = torch.tensor(X_test).float()

y_test = torch.tensor(y_test).view(-1, 1).float()

datasets = torch.utils.data.TensorDataset(X_train, y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size = 10, shuffle=True)
batch_size= 50

num_epochs = 200

learning_rate = 0.0001

size_hidden1 = 100

size_hidden2 = 50

size_hidden3 = 10

size_hidden4 = 1
class BostonModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lin1 = nn.Linear(13, size_hidden1)

        self.relu1 = nn.ReLU()

        self.lin2 = nn.Linear(size_hidden1, size_hidden2)

        self.relu2 = nn.ReLU()

        self.lin3 = nn.Linear(size_hidden2, size_hidden3)

        self.relu3 = nn.ReLU()

        self.lin4 = nn.Linear(size_hidden3, size_hidden4)

    def forward(self, input):

        return self.lin4(self.relu3(self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(input)))))))
model = BostonModel()

model.train()
!pip install torchviz
dummy_input = torch.zeros([1, 13])
dummy_input
dummy_out = model(dummy_input)
from torchviz import make_dot

make_dot(dummy_out)
criterion = nn.MSELoss(reduction='sum')

def train(model_inp, num_epochs = num_epochs):

    optimizer = optim.RMSprop(model_inp.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        running_loss = 0.

        for inputs , labels in train_iter:

            outputs = model_inp(inputs)

            loss= criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            running_loss += loss.item()

            optimizer.step()

        if epoch % 20 == 0:

            print(f'Epoch {epoch+1}/{num_epochs} running accumulative loss {running_loss:.3f}')
def train_load_save_model(model_obj, model_path):

    if path.isfile(model_path):

        print("Loading pretrain model  from {}".format(model_path))

        model_obj.load_state_dict(torch.load(model_path))

    else:

        train(model_obj)

        print('Finished training the model. Sasving teh model to the path: {}'.format(model_path))

        torch.save(model_obj.state_dict(), model_path)
SAVED_MODEL_PATH = 'boston_model.pt'

train_load_save_model(model, SAVED_MODEL_PATH)
model.eval()

outputs = model(X_test)

err = np.sqrt(mean_squared_error(outputs.detach().numpy(), y_test.detach().numpy()))

print('model err: ', err)
%%time

ig = IntegratedGradients(model)

ig_attr_test = ig.attribute(X_test, n_steps = 50)
%%time

ig_nt = NoiseTunnel(ig)

ig_nt_attr_test = ig_nt.attribute(X_test)
%%time

dl = DeepLift(model)

dl_attr_test = dl.attribute(X_test)
%%time

gs = GradientShap(model)

gs_attr_test = gs.attribute(X_test, X_train)
%%time

fa = FeatureAblation(model)

fa_attr_test = fa.attribute(X_test)
x_axis_data = np.arange(X_test.shape[1])

x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))



ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)

ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)



ig_nt_attr_test_sum = ig_nt_attr_test.detach().numpy().sum(0)

ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)



dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)

dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)



gs_attr_test_sum = gs_attr_test.detach().numpy().sum(0)

gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)



fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)

fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)



lin_weight = model.lin1.weight.detach().numpy().sum(0)

y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)
model.lin1.weight.shape
width = 0.14

legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']



plt.figure(figsize=(20, 10))



ax = plt.subplot()

ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')

ax.set_ylabel('Attributions')



FONT_SIZE = 16

plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes

plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title

plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels

plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend



ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')

ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')

ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')

ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')

ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')

ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')

ax.autoscale_view()

plt.tight_layout()



ax.set_xticks(x_axis_data + 0.5)

ax.set_xticklabels(x_axis_data_labels)



plt.legend(legends, loc=3)

plt.show()
# Compute the attributions of the output with respect to teh inpus of teh fourth linear layer.

lc = LayerConductance(model, model.lin4)

lc_attr_test = lc.attribute(X_test, n_steps= 100, attribute_to_layer_input = True)
# Shape

lc_attr_test
lc_attr_test = lc_attr_test[0]
lc_attr_test
# weights fromo forth linear layer

lin4_weight = model.lin4.weight
lin4_weight
plt.figure(figsize=(15, 8))

x_axis_data = np.arange(lc_attr_test.shape[1])



y_axis_lc_attr_test = lc_attr_test.mean(0).detach().numpy()

y_axis_lc_attr_test = y_axis_lc_attr_test/np.linalg.norm(y_axis_lc_attr_test, ord=1)

y_axis_lin4_weight = lin4_weight[0].detach().numpy()

y_axis_lin4_weight = y_axis_lin4_weight/np.linalg.norm(y_axis_lin4_weight, ord=1)





x_axis_labels = [f'Neuron {i}' for i in range(len(y_axis_lin4_weight))]



ax = plt.subplot()

ax.set_title('Aggreggated neuron importances and learned weights in the last linear layer of the model')

ax.bar(x_axis_data + width, y_axis_lc_attr_test, width, align='center', alpha = 0.5, color='red')

ax.bar(x_axis_data + 2 * width, y_axis_lin4_weight, width, align='center', alpha=0.5, color='green')

plt.legend(legends, loc=2, prop={'size': 20})

ax.autoscale_view()

plt.tight_layout()

ax.set_xticks(x_axis_data + 0.25)

ax.set_xticklabels(x_axis_labels)
