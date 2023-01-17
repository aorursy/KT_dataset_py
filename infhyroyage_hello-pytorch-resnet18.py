!pip install skorch
%matplotlib inline
from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from skorch import NeuralNetClassifier

import torch

from torch.optim import Adam

from torchvision.models import resnet18
seaborn.set(style="darkgrid", context="notebook", palette="muted")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 7

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape)

train.head()
print(test.shape)

test.head()
x_train = train.drop(["label"], axis=1) 

x_test = deepcopy(test)

x_train.shape
seaborn.distplot(x_train[4:5], kde=False, rug=True)
x_train = x_train / 255.0

x_test = x_test / 255.0
seaborn.distplot(x_train[4:5], kde=False, rug=True)
print("train.shape=%s, test.shape=%s" % (x_train.shape, x_test.shape))
x_train = x_train.values.reshape(-1, 1, 28, 28)

x_test = x_test.values.reshape(-1, 1, 28, 28)
print("train.shape=%s, test.shape=%s" % (x_train.shape, x_test.shape))
x_train = torch.from_numpy(x_train).type('torch.FloatTensor')

x_test = torch.from_numpy(x_test).type('torch.FloatTensor')
y_train = train["label"]
y_train = torch.Tensor(y_train).type('torch.LongTensor')
y_train[3]
plt.imshow(x_train[3][0,:,:])
network = resnet18()

network
network.conv1 = torch.nn.Conv2d(1, 64,

                                kernel_size=(7, 7),

                                stride=(2, 2),

                                padding=(3, 3),

                                bias=False)

network.fc = torch.nn.Linear(in_features=512,

                             out_features=10,

                             bias=True)

network.add_module("softmax",

                   torch.nn.Softmax(dim=-1))

network
network.zero_grad()

classifier = NeuralNetClassifier(

    network,

    max_epochs=20,

    lr=0.01,

    batch_size=256,

    optimizer=torch.optim.Adam,

    device=device,

    criterion=torch.nn.CrossEntropyLoss,

    train_split=None

)

classifier.fit(x_train, y_train)
pred_train = classifier.predict(x_train)

pred_train.shape
pred_train[4]
plt.imshow(x_train[4][0,:,:])
cm = confusion_matrix(y_train.numpy(), pred_train) 

cm_df = pd.DataFrame(cm, columns=np.unique(y_train.numpy()), index = np.unique(y_train.numpy()))

cm_df.index.name = "True Label"

cm_df.columns.name = "Predicted Label"

cm_df
seaborn.heatmap(cm_df,

                annot=True,

                cmap="Blues",

                fmt="d")
errors = (pred_train - y_train.numpy() != 0)

pred_train_errors = pred_train[errors]

x_train_errors = x_train.numpy()[errors]

y_train_errors = y_train.numpy()[errors]
pred_train_errors = pred_train_errors[:6]

x_train_errors = x_train_errors[:6]

y_train_errors = y_train_errors[:6]
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

for row in range(2):

    for col in range(3):

        idx = 3 * row + col

        ax[row][col].imshow(x_train_errors[idx][0])

        args = (pred_train_errors[idx], y_train_errors[idx])

        title = "Predict:%s,True:%s" % args

        ax[row][col].set_title(title)
pred_test = classifier.predict(x_test)

pred_test.shape
pred_test[4]
plt.imshow(x_test[4][0,:,:])
result = pd.DataFrame({"ImageId" : range(1,28001),

                       "Label" : pred_test})

result.to_csv("result.csv",index=False)