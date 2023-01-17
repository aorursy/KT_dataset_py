%matplotlib inline

from fastai.basics import *

from fastai.vision import *
path = Config().data_path()/'mnist'
path.mkdir(parents=True)
path.ls()
!wget http://deeplearning.net/data/mnist/mnist.pkl.gz -P {path}
with gzip.open(path/'mnist.pkl.gz', 'rb') as f:

    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
y_train[0], x_train[0].shape
x_train[0].reshape((28,28))[10:20, 10:20]
plt.imshow(x_train[0].reshape((28,28))[10:20, 10:20], cmap="gray")
plt.imshow(x_train[0].reshape((28,28)), cmap="gray")

x_train.shape
x_train,y_train,x_valid,y_valid = map(torch.tensor, (x_train,y_train,x_valid,y_valid))

n,c = x_train.shape

x_train.shape, y_train.min(), y_train.max()
bs=64

train_ds = TensorDataset(x_train, y_train)

valid_ds = TensorDataset(x_valid, y_valid)

data = DataBunch.create(train_ds, valid_ds, bs=bs)
[x for x in data.train_dl][: 2]
loss_func = nn.CrossEntropyLoss()
class Mnist_NN(nn.Module):

    def __init__(self):

        super().__init__()

        self.lin1 = nn.Linear(784, 50, bias=True)

        self.lin2 = nn.Linear(50, 10, bias=True)



    def forward(self, xb):

        x = self.lin1(xb)

        x = F.relu(x)

        return self.lin2(x)
model = Mnist_NN().cuda()
[p.shape for p in model.parameters()]
print(model)
def update(x,y,lr):

    opt = optim.Adam(model.parameters(), lr)

    y_hat = model(x)

    loss = loss_func(y_hat, y)

    loss.backward()

    opt.step()

    opt.zero_grad()

    return loss.item()
losses = [update(x,y,1e-3) for x,y in data.train_dl]
plt.plot(losses);
learn = Learner(data, Mnist_NN(), loss_func=loss_func, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 1e-2)
learn.recorder.plot_lr(show_moms=True)
learn.recorder.plot_losses()
x_train,y_train,x_valid,y_valid = map(torch.tensor, (x_train.reshape(-1, 1, 28, 28),y_train,x_valid.reshape(-1, 1, 28, 28),y_valid))

x_train.shape, y_train.min(), y_train.max()
bs=64

train_ds = TensorDataset(x_train, y_train)

valid_ds = TensorDataset(x_valid, y_valid)

data = DataBunch.create(train_ds, valid_ds, bs=bs)
class SimpleCNN(torch.nn.Module):

    

    #Our batch shape for input x is (1, 32, 32)

    

    def __init__(self):

        super(SimpleCNN, self).__init__()

        

        #Input channels = 1, output channels = 18

        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        

        #4608 input features, 64 output features (see sizing flow below)

        self.fc1 = torch.nn.Linear(18 * 14 * 14, 64)

        

        #64 input features, 10 output features for our 10 defined classes

        self.fc2 = torch.nn.Linear(64, 10)

        

    def forward(self, x):

        

        #Computes the activation of the first convolution

        #Size changes from (1, 28, 28) to (18, 28, 28)

        x = F.relu(self.conv1(x))

        

        #Size changes from (18, 28, 28) to (18, 14, 14)

        x = self.pool(x)

        

        #Reshape data to input to the input layer of the neural net

        #Size changes from (18, 14, 14) to (1, 3528)

        #Recall that the -1 infers this dimension from the other given dimension

        x = x.view(-1, 18 * 14 *14)

        

        #Computes the activation of the first fully connected layer

        #Size changes from (1, 3528) to (1, 64)

        x = F.relu(self.fc1(x))

        

        #Computes the second fully connected layer (activation applied later)

        #Size changes from (1, 64) to (1, 10)

        x = self.fc2(x)

        return(x)
learn = Learner(data, SimpleCNN(), loss_func=loss_func, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 1e-2)
learn.model
class Mnist(Dataset):

    def __init__(self, X, y=None):

        super().__init__()

        self.classes = np.unique(y)

        self.c = len(np.unique(y))

        self.X = X

        if y is not None: self.y = y



    def __getitem__(self, i):

        return (self.X[i], self.y[i])

    

    def __len__(self): return len(self.X)
train_ds = Mnist(x_train.expand(-1, 3, -1, -1), y_train)

valid_ds = Mnist(x_valid.expand(-1, 3, -1, -1), y_valid)
bs=64

data = ImageDataBunch.create(train_ds, valid_ds, bs=bs)
learn = cnn_learner(data, models.vgg16_bn, loss_func=loss_func, metrics=accuracy)
learn.lr_find()


learn.recorder.plot()
learn.fit_one_cycle(4, 1e-2)
learn.recorder.plot_losses()
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5, 2e-4))
learn.model