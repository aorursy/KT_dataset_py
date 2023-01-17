import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torchvision
train_all = pd.read_csv('../input/train.csv')

train_all.head()
class Mnist(torch.utils.data.dataset.Dataset):

    """

    Converts Kaggle's weird pixel-dataframe into pytorch-DataSet.

    Stores transformed images in memory for quick access.

    """

    

    def __init__(self, df: pd.DataFrame, transformation=None, labeled=True):

        

        self.labeled = labeled

        

        X = df.drop('label', axis=1, errors='ignore').values

        self.images = [torch.FloatTensor(X[i, :]).view(28, 28).unsqueeze(0) for i in range(len(df))]

        

        if labeled:

            self.labels = df.label.values

        

        if transformation:

            self.images = [transformation(x) for x in self.images]

        

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self, i):

        if self.labeled:

            return self.images[i], self.labels[i]

        else:

            return self.images[i]

    

    

def normalize(image: torch.Tensor):

    """ Normalize values into range [-0.5, 0.5] """

    return (image / 255) - 0.5



def un_normalize(image: torch.Tensor):

    return (image + 0.5) * 255
val_set_mask = np.random.rand(len(train_all)) < 0.2



train_ds = Mnist(train_all.loc[~val_set_mask], transformation=normalize)

val_ds   = Mnist(train_all.loc[val_set_mask], transformation=normalize)



batchsize = 256

train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True)

val_loader   = DataLoader(val_ds, batch_size=batchsize, shuffle=False)
for i, (X, y) in enumerate(train_loader):

    pass



plt.rcParams['figure.figsize'] = [15, 15]

grid_image = un_normalize(torchvision.utils.make_grid(X, nrow=20))

plt.imshow(grid_image.permute(1, 2, 0))

plt.title(f'one batch ({batchsize} images)');
def output_dim(in_dim, k=3, s=1, p=0, p_left=None, p_right=None):

    

    if p is not None:

        p_left = p_right = p

    assert p_left is not None and p_right is not None

        

    tmp = (in_dim - k + p_left + p_right) / s

    out_dim = int(np.floor(tmp) + 1)

    

    if tmp % 1 != 0:

        print('no exact output-dim; using Gauss-brackets.')

    return out_dim
l1_dim = output_dim(in_dim=28, k=3, s=1, p=1)

print(f'after conv-layer 1: {l1_dim}')

l2_dim = output_dim(l1_dim, k=2, s=2)

print(f'after max-pooling 1: {l2_dim}')

l3_dim = output_dim(l2_dim, k=3, s=1, p=1)

print(f'\nafter conv-layer 2: {l3_dim}')

l4_dim = output_dim(l3_dim, k=2, s=2)

print(f'after max-pooling 2: {l4_dim}')

l5_dim = output_dim(l4_dim, k=3, s=1, p=1)

print(f'\nafter conv-layer 3: {l5_dim}')

l6_dim = output_dim(l5_dim, k=3, s=2)

print(f'after max-pooling 3: {l6_dim}')

class SimpleConvNet(nn.Module):

    

    def __init__(self):

        super().__init__()

        

        self.features = nn.Sequential(

            

            # same convolution

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),

            nn.ReLU(inplace=True),

            

            # down-sample to 14**2

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReLU(inplace=True),

            

            # same convolution

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(inplace=True),

            

            # down-sample to 7**2

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReLU(inplace=True),

            

            # same convolution

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(inplace=True),

            

            # downsample to 3x3

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.ReLU(inplace=True),

        )

        

        self.classifier = nn.Linear(32 * 3**2, 10)

        # note: no softmax because it is included in cost-function

        

    def forward(self, X):

        

        features = self.features(X)

        batchsize, channels, dim, _ = features.shape

        output = self.classifier(features.view(batchsize, channels * dim**2))

        

        return output
net = SimpleConvNet()

net.cuda()



loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)





error_df = pd.DataFrame(columns=['train_bce', 'train_acc', 'val_bce', 'val_acc'])  # track errors

print('\t\t|--------training---------|\t\t|-------validation--------|')

print('epoch\t\tCE\t\tAccuracy\t\tBCE\t\tAccuracy')



for epoch_i in range(50):

    

    train_bce = train_acc = 0

    

    for X, y in train_loader:

            

        X, y = X.cuda(), y.cuda()

        

        optimizer.zero_grad()

        

        output = net(X)        

        loss = loss_function(output, y)

        train_bce += loss.item()

        

        # no softmax required because largest the largest value is also the largest value after softmax

        prediction = torch.max(output.data, 1)[1] 

        train_acc += (prediction == y).sum().item()

        

        # update parameters:

        loss.backward()   

        optimizer.step()

        

    val_bce = val_acc = 0

        

    for X, y in val_loader:

            

        X, y = X.cuda(), y.cuda()



        output = net(X)

        loss = loss_function(output, y)

        val_bce += loss.item()

        

        prediction = torch.max(output.data, 1)[1]

        val_acc += (prediction == y).sum().item()

    

    # convert to batch loss:

    train_bce /= len(train_loader)

    train_acc /= (len(train_loader) * batchsize)

    val_bce   /= len(val_loader)

    val_acc   /= (len(val_loader) * batchsize)

    

    error_stats = (train_bce, train_acc, val_bce, val_acc)

    error_df = error_df.append(pd.Series(error_stats, index=error_df.columns), ignore_index=True)

    print('{}\t\t{:.4f}\t\t{:.4f}\t\t\t{:.4f}\t\t{:.4f}'.format(epoch_i, *error_stats))

def cout_net_parameters(net, count_shared=False):

    n_params = sum(x.numel() for x in net.parameters() if x.requires_grad)

    print(f'number of parameters: {n_params}')



    

def plot_error_curves(training_error, validation_error, error_name='error', ylim=None, save_fig=False):

    """

    @ plot training and validation-error vs. training epoch

    """

    assert len(training_error) == len(validation_error) and len(training_error) > 1

    

    fig, ax = plt.subplots()



    ax.plot(range(len(training_error)), training_error)

    ax.plot(range(len(validation_error)), validation_error)

    ax.set_xticks(range(len(training_error) + 1, len(training_error) // 2))

    

    if ylim:

        ax.set_ylim(*ylim)

    

    ax.set_xlabel('epoch')

    ax.set_ylabel('CE')

    ax.legend(('training', 'validation'))

    ax.set_title(f'{error_name} over time')

    

    if save_fig:

        fig.savefig(f'{error_name.replace(" ", "_")}_learning_curve', bbox_inches='tight', transparent=True)

    

    plt.show();

    

    

def predict_and_display(net, testset, n=10, verbose=False):

    """

    Predict @n random examples from the test-set and show images + predictions

    """

    for i in np.random.choice(range(len(testset)), size=n):

        

        X = testset[i]

        out = net(X.unsqueeze(0).cuda()).squeeze()

        pred = out.argmax().item()



        plt.figure( figsize=(2, 2) )

        plt.imshow(X.squeeze())

        plt.title(f'prediction: {pred}')

        plt.show();
cout_net_parameters(net)



plt.rcParams['figure.figsize'] = [10, 5]

plot_error_curves(error_df.train_bce, error_df.val_bce, error_name='Binary Cross Entropy', ylim=(0, 1))

plot_error_curves(error_df.train_acc, error_df.val_acc, error_name='Accuracy')
test_df = pd.read_csv('../input/test.csv')

test_ds = Mnist(test_df, transformation=normalize, labeled=False)
X = test_ds[np.random.choice(range(len(test_ds)))]

        

out = net(X.unsqueeze(0).cuda()).squeeze()

probabilities = torch.softmax(out, dim=0)

pred = probabilities.argmax().item()



print(f'\nlinear layer output:\n{out.detach().cpu().numpy().tolist()}')

print(f'\nprobabilities:\n{probabilities.detach().cpu().numpy().tolist()}')

print(f'sum: {probabilities.sum().item()}')

print(f'\nmax probability: {probabilities.max().item()}')

print(f'\nprediction: {pred}')



plt.imshow(X.squeeze());
predict_and_display(net, test_ds, n=10)
test_loader = DataLoader(test_ds, batch_size=batchsize)

predictions = []

net.eval()



for X in test_loader:

    out = net(X.cuda())

    batch_predictions = torch.max(out, 1)[1]

    predictions.extend(batch_predictions.detach().cpu().numpy().tolist())
submission_df = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})

assert len(submission_df) == len(test_df)

display(submission_df.head())



submission_df.to_csv('submission.csv', index=False)