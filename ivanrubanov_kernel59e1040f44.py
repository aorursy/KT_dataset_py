%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *



import torchvision.datasets as dsets

from torchvision import transforms
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
import os

os.listdir("../input/fashionmnist/")
labels =['Tshirt/top',"Trouser","Pullover","Dress","Coat","Sandal","Shirt",'Sneaker',"Bag","Ankle boot"]
tfms = get_transforms(do_flip = False)
train_data = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

test_data = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
train_data.head()
test_data.head()
train_data['label']
train_images = train_data.drop('label', axis = 1)
img = train_images.iloc[2,:].values.reshape(28, 28)

plt.imshow(img, cmap = 'gray')
class FashionMNIST(Dataset):

    def __init__(self, path, transform=None):

        self.transform = transform

        fashion_df = pd.read_csv(path)

        self.labels = fashion_df.label.values

        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        label = self.labels[idx]

        img = Image.fromarray(self.images[idx])

        

        if self.transform:

            img = self.transform(img)



        return img, label
train_ds = FashionMNIST('../input/fashionmnist/fashion-mnist_train.csv')

test_ds = FashionMNIST('../input/fashionmnist/fashion-mnist_test.csv')
from PIL import Image



test_ds[0][0]
train_data_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

test_data_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

#data = DataBunch.create(train_ds = train_ds, valid_ds = test_ds)

data = DataBunch.create(train_data_loader, test_data_loader)
learn = cnn_learner(data, models.resnet34, metrics=[accuracy])