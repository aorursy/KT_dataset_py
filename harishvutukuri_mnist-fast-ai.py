import warnings

warnings.filterwarnings("ignore")



from fastai import *

from fastai.vision import *

from pathlib import Path
np.random.seed(42)



# setting path and reading data

path = Path('../input/digit-recognizer/')

train_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')
# splitting data by 20% 

valid_df = train_df.sample(frac=0.2, random_state=42)

train_df.drop(valid_df.index, inplace=True)



print(train_df.shape, valid_df.shape)
# dependent and independent variable splitting

trainY = train_df["label"].values

validY = valid_df["label"].values

trainX = train_df.drop("label", axis=1).values

validX = valid_df.drop("label", axis=1).values
# reshaping to 28 x 28 1 channel image

trainX = trainX.reshape(len(trainX), 1, 28, 28)

validX = validX.reshape(len(validX), 1, 28, 28)

test_df = test_df.values.reshape(len(test_df), 1, 28, 28)
# converting to torch tensors

trainX, trainY, validX, validY, test_df = map(torch.Tensor, (trainX, trainY, validX, validY, test_df))
# batch size and classes

sz = 28

bs = 64

classes = len(np.unique(trainY))
# creating TensorDataset & DataBunch

train_ds = TensorDataset(trainX, trainY)

valid_ds = TensorDataset(validX, validY)

test_ds = TensorDataset(test_df, torch.zeros(len(test_df)))



data = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, bs=bs)
class MnistModel(nn.Module):

    def __init__(self, classes):

        super(MnistModel, self).__init__()

        

        self.classes = classes

        

        # initialize the layers in the first (CONV => RELU) * 2 => POOL + DROP

        self.conv1A = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)  # (N,1,28,28) -> (N,16,24,24)

        self.act1A = nn.ReLU()

        self.conv1B = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # (N,16,24,24) -> (N,32,20,20)

        self.act1B = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2) # (N,32,20,20) -> (N,32,10,10)

        self.do1 = nn.Dropout(0.25)

        

        # initialize the layers in the second (CONV => RELU) * 2 => POOL + DROP

        self.conv2A = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0) # (N,32,10,10) -> (N,64,8,8)

        self.act2A = nn.ReLU()

        self.conv2B = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) # (N,64,8,8) -> (N,128,6,6)

        self.act2B = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2) # (N,128,6,6) -> (N,128,3,3)

        self.do2 = nn.Dropout(0.25)

        

        # initialize the layers in our fully-connected layer set

        self.dense3 = nn.Linear(128*3*3, 32) # (N,128,3,3) -> (N,32)

        self.act3 = nn.ReLU()

        self.do3 = nn.Dropout(0.25)

        

        # initialize the layers in the softmax classifier layer set

        self.dense4 = nn.Linear(32, self.classes) # (N, classes)

    

    def forward(self, x):

        

        # build the first (CONV => RELU) * 2 => POOL layer set

        x = self.conv1A(x)

        x = self.act1A(x)

        x = self.conv1B(x)

        x = self.act1B(x)

        x = self.pool1(x)

        x = self.do1(x)

        

        # build the second (CONV => RELU) * 2 => POOL layer set

        x = self.conv2A(x)

        x = self.act2A(x)

        x = self.conv2B(x)

        x = self.act2B(x)

        x = self.pool2(x)

        x = self.do2(x)

        

        # build our FC layer set

        x = x.view(x.size(0), -1)

        x = self.dense3(x)

        x = self.act3(x)

        x = self.do3(x)

        

        # build the softmax classifier

        x = nn.functional.log_softmax(self.dense4(x), dim=1)

        

        return x
# Loss Function

def loss_func(output, target):

    return nn.CrossEntropyLoss()(output, target.long())
# pytorch approach

model = MnistModel(classes=classes)



lr = 1e-2



def update(x,y,lr):

    wd = 1e-5

    y_hat = model(x)

    w2 = 0.

    for p in model.parameters(): 

        w2 += (p**2).sum()

    loss = loss_func(y_hat, y) + w2*wd

    loss.backward()

    with torch.no_grad():

        for p in model.parameters():

            p.sub_(lr * p.grad)

            p.grad.zero_()

    return loss.item()

            



losses = [update(x,y,lr) for x,y in data.train_dl]



plt.plot(losses);
# fast.Ai approach

# create a learner 

learn = Learner(data, MnistModel(classes), loss_func=loss_func, metrics=accuracy)
# finiding the best learning rate

learn.lr_find()

learn.recorder.plot(suggestion=True)
# fiting the learner for the learning rate

learn.fit_one_cycle(1, 1e-2)
# saving the learner and unfreezing, finiding new lr

learn.save('stage-1')

learn.unfreeze()

learn.lr_find(stop_div=False, num_it=200)

learn.recorder.plot(skip_start=0, skip_end=5, suggestion=True)
# fitting learner and saving the learner

learn.fit_one_cycle(10, slice(1e-5))

learn.save('stage-2')
# plotting losses

learn.recorder.plot_losses()