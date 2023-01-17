# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import numpy as np

import pandas as pd

import os

import matplotlib.image as mpimg



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms

from sklearn.decomposition import PCA

import os

import cv2

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time



from scipy import ndimage



from sklearn import manifold, datasets

import glob



from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering



# Any results you write to the current directory are saved as output.
train_dir = "../input/train/train/"

test_dir = "../input/test/test/"
train_path=train_dir

test_path=test_dir

train_set = pd.read_csv('../input/train.csv').sort_values('id')

train_set.sort_values('id')

train_labels = train_set['has_cactus']

train_labels.head()
sns.countplot(train_labels)
files = sorted(glob.glob(train_path + '*.jpg'))



train = [cv2.imread(image) for image in files]



train = np.array(train, dtype='int32')



train_images_set = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])
def plot_clustering(X_red, labels, title=None):

    

    # calculating the minimum and maximum values, so that we can use it to normalize X_red within min/max range for plotting

    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)

    X_red = (X_red - x_min) / (x_max - x_min)

    # setting the figure size or plot size

    plt.figure(figsize=(6, 4))

    for i in range(X_red.shape[0]):

        # Plotting the text i.e. numbers

        plt.text(X_red[i, 0], X_red[i, 1], str(labels[i]),

                 color=plt.cm.seismic(labels[i]),

                 fontdict={'weight': 'bold', 'size': 9})

        

    plt.xticks([])

    plt.yticks([])

    if title is not None:

        plt.title(title, size=17)

    plt.axis('off')

    plt.tight_layout()
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train_images_set)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principalcomponent1',

                                                                  'principalcomponent2'])
print("Computing embedding")

# Converting the data into 2D embedding

X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(principalDf)

print("Done.")
from sklearn.cluster import AgglomerativeClustering



# Calling the agglorimative clustering function from sklearn library.

clustering = AgglomerativeClustering(linkage='ward', n_clusters=10)

# startitng the timier

t0 = time()

# Fitting the data in agglorimative function on order to train it

clustering.fit(X_red)

# printing the time taken

print("%s : %.2fs" % ("linkage", time() - t0))

# Plotting the cluster distribution

plot_clustering(X_red, train_labels, "Agglomerative Clustering- distribution of clusters" )



plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc



X_train, X_test, y_train, y_test = train_test_split(principalDf, train_set['has_cactus'], test_size=0.33, random_state=42)

  

clf = QDA(store_covariance = True, tol = 0.000000001)

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
LDA_CLF = LDA(solver = 'lsqr', tol=0.000000001)

LDA_CLF.fit(X_train,y_train)

y_lda_pred = LDA_CLF.predict(X_test)

accuracy_score(y_test, y_lda_pred)
labels = pd.read_csv("../input/train.csv")





class ImageData(Dataset):

    def __init__(self, df, data_dir, transform):

        super().__init__()

        self.df = df

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):       

        img_name = self.df.id[index]

        label = self.df.has_cactus[index]

        

        img_path = os.path.join(self.data_dir, img_name)

        image = mpimg.imread(img_path)

        image = self.transform(image)

        return image, label

labels.head()
epochs = 25

batch_size = 20

device = torch.device('cpu')



data_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)

train_loader = DataLoader(dataset = train_data, batch_size = batch_size)



#train_num = train_loader.numpy()



num_classes = 2
class ConvNet(nn.Module):

    def __init__(self, num_classes=10):

        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(10),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=0.2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(10, 32, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),nn.Dropout2d(p=0.5))

        self.layer3 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(1024, num_classes)

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = out.view(out.shape[0],-1)

        out = self.fc(out)

        return out
net = ConvNet().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_func = nn.CrossEntropyLoss()
train_loader
for epoch in range(epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device)

        # Forward

        outputs = net(images)

        loss = loss_func(outputs, labels)

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (i+1) % 500 == 0:

            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
submit = pd.read_csv('../input/sample_submission.csv')

test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)

test_loader = DataLoader(dataset = test_data, shuffle=False)



predict = []

for batch, (data, target) in enumerate(test_loader):

    data, target = data.to(device), target.to(device)

    output = net(data)

    

    num, pred = torch.max(output.data, 1)

    predict.append(int(pred))



submit['has_cactus'] = predict

submit.to_csv('submission.csv', index=False)