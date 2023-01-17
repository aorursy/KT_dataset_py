import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pathname = os.path.join(dirname, filename)

        print(f"{pathname}\t{os.path.getsize(pathname)//1024/1024:.3f} MB")
import pandas as pd



TRAIN_FILE = '/kaggle/input/fashionmnist/fashion-mnist_train.csv'

TEST_FILE = '/kaggle/input/fashionmnist/fashion-mnist_test.csv'



train = pd.read_csv(TRAIN_FILE)

test = pd.read_csv(TEST_FILE)
train.head()
from matplotlib import pyplot as plt



TEXT_LABELS = [

    't-shirt',

    'trouser',

    'pullover',

    'dress',

    'coat',

    'sandal',

    'shirt',

    'sneaker',

    'bag',

    'ankle boot'

]

IMG_WIDTH=28

IMG_HEIGHT=28



def plot_images(X, y, y_predicted=None, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, ncols=4, figsize=None):

    """Plot a set of images along with their labels.

    

    Keyword arguments:

    X -- A set of images to be plotted

    y -- The labels of the images

    y_predicted -- If specified, displays another label for the prediction.

                   This will be used toward the end of this notebook when

                   we start using the trained network to make predictions.

    img_width -- The width of each image. Default to 28, the width of images in the dataset.

    img_height -- The height of each image. Default to 28, the height of images in the dataset.

    ncols -- The number of images to display in each row of the grid. Default to 4.

    figsize -- The overall size of the grid. Leave None for auto calculation.

    

    """

    # Find the number of images to be plotted.

    image_count = X.shape[0]

    nrows = (image_count + ncols - 1) // ncols



    MAX_IMAGE_COUNT = 30

    if image_count > MAX_IMAGE_COUNT:

        raise ValueError(f"Trying to plot too many images. The maximum is {MAX_IMAGE_COUNT}")

    

    MAX_NCOLS = 5

    if ncols > MAX_NCOLS:

        raise ValueError(f"Too many column. The maximum is {MAX_NCOLS}")    



    if figsize is None:

        figsize = (ncols*4, nrows*5) # for each row, we leave space for the text



    if y_predicted is None:

        y_predicted = [None] * len(y)



    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for ax, image, label, predicted_label in zip(axes.reshape(-1), X.reshape(-1, img_height, img_width), y, y_predicted):

        ax.imshow(image, cmap='gray')

        if predicted_label is not None:

            ax.set_title("Real: %s\nPrediction: %s" % (

                TEXT_LABELS[label], TEXT_LABELS[predicted_label]))

        else:

            ax.set_title(TEXT_LABELS[label])

        ax.title.set_fontsize(20)

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    # Turn off unused subplots at the end.

    for ax in axes.reshape(-1)[image_count:]:

        ax.axis('off')

    plt.show()



sample_count = 12

X_sample = train.iloc[0:sample_count,1:].to_numpy()

y_sample = train.iloc[0:sample_count,0].to_numpy()

plot_images(X_sample, y_sample)

train_mean = (train.iloc[:, 1:]/255).mean().mean()

train_std = (train.iloc[:, 1:]/255).std().std()

(train_mean, train_std)
from mxnet import nd as nd



X_train = nd.array(train.iloc[:, 1:].to_numpy().reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)) # gray scale image, so just one channel

y_train = nd.array(train.iloc[:,0].to_numpy())

type(X_train), X_train.shape, type(y_train), y_train.shape
X_test = nd.array(test.iloc[:, 1:].to_numpy().reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)) # gray scale image, so just one channel

y_test = nd.array(test.iloc[:, 0].to_numpy())

type(X_test), X_test.shape, type(y_test), y_test.shape
from mxnet.gluon.data.vision import transforms



def create_transformer():

    # Create a transformer that tensorize and normalize the images.

    tensorize = transforms.ToTensor()

    normalize = transforms.Normalize(train_mean, train_std)

    return transforms.Compose([tensorize, normalize])



# Let's try the transformer out.

trans = create_transformer()

trans(X_train[0])
from mxnet.gluon.data import ArrayDataset, DataLoader

import mxnet.gluon

from mxnet import gpu, cpu



ctx = gpu() # This context will be used later. It tells MXNet to use

            # the GPU instead of the CPU to achieve faster training

            # and prediction time.



BATCH_SIZE = 100

# Use the transformer above to transform the images to the format required by LeNet.

# Notice that our dataset is a zipping of the images and their labels, since later on

# during training, we need to give the neural network a set of images to train on

# along with their labels.

# Also notice the transform is only applied to the image; notice the use of

# transform_first() method instead of transform().

train_loader = DataLoader(ArrayDataset(list(zip(X_train, y_train))).transform_first(trans),

                          BATCH_SIZE, shuffle=True)

test_loader = DataLoader(ArrayDataset(list(zip(X_test, y_test))).transform_first(trans),

                         BATCH_SIZE, shuffle=True)
for batch in train_loader:

    print(batch)

    break
import mxnet as mx

from mxnet.gluon import nn, HybridBlock



class LeNet(HybridBlock):

    def __init__(self):

        super(LeNet, self).__init__()

        with self.name_scope():

            self.conv1 = nn.Conv2D(channels=6, kernel_size=5, activation='relu')

            self.pool1 = nn.MaxPool2D(pool_size=2, strides=2)

            self.conv2 = nn.Conv2D(channels=16, kernel_size=3, activation='relu')

            self.pool2 =  nn.MaxPool2D(pool_size=2, strides=2)

            self.hidden1 = nn.Dense(120, activation='relu')

            self.hidden2 = nn.Dense(84, activation='relu')

            self.out = nn.Dense(10)

    

    def hybrid_forward(self, F, x):

        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.pool2(x)

        x = x.reshape((0, -1))

        x = self.hidden1(x)

        x = self.hidden2(x)

        x = self.out(x)

        return x

        



net = LeNet()

net.initialize(mx.init.Xavier())

net.hybridize()

net
from datetime import datetime

from mxnet import gluon, autograd



def accuracy(output, label):

    # output: (batch, num_output) float32 ndarray

    # label: (batch, ) int32 ndarray

    acc = (output.argmax(axis=1) == label.astype('float32'))

    return acc.mean().asscalar()



def train_lenet(iters = 7, learning_rate = 0.1, ctx=ctx):

    net = LeNet()

    net.initialize(mx.init.Xavier(), ctx=ctx)

    net.hybridize()



    losses, train_accuracies, test_accuracies = [], [], []



    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

    trainer = gluon.Trainer(net.collect_params(), 'sgd',

                            {'learning_rate': learning_rate})



    X_train_trans = trans(X_train).as_in_context(ctx)

    X_test_trans = trans(X_test).as_in_context(ctx)

    

    print("N | Elapsed Time   | Loss  | Training | Test")

    print("----------------------------------------------")

    start_time = datetime.now()

    for i in range(iters):

        for _, (X, y) in enumerate(train_loader):

            X, y = X.as_in_context(ctx), y.as_in_context(ctx)

            with autograd.record():

                y_hat = net(X)

                loss = loss_fn(y_hat, y)

            loss.backward()

            trainer.step(X.shape[0])



        # TODO: do this in batches to avoid loading a huge amount

        # of data into the GPU and potentially running out of

        # GPU memory

        y_hat_train = net(X_train_trans)

        y_hat_test = net(X_test_trans)

        # TODO: Avoid repeatedly calling as_in_context().

        loss = nd.mean(loss_fn(y_hat_train, y_train.as_in_context(ctx))).asscalar()

        train_accuracy = accuracy(y_hat_train, y_train.as_in_context(ctx))

        test_accuracy = accuracy(y_hat_test, y_test.as_in_context(ctx))

        print(" | ".join([f"{i+1}",

                          str(datetime.now() - start_time),

                          f"{loss:.3f}",

                          f"{train_accuracy*100:.2f}%  ",

                          f"{test_accuracy*100:.2f}%"]))

        losses.append(loss)

        train_accuracies.append(train_accuracy)

        test_accuracies.append(test_accuracy)

    return net, losses, train_accuracies, test_accuracies

net, losses, train_accuracies, test_accuracies = train_lenet(ctx=gpu())
def plot_stuff(losses, train_accuracies, test_accuracies):

    plt.figure(num=None,figsize=(8, 6))

    plt.plot(losses)

    plt.xlabel('Epoch',fontsize=14)

    plt.ylabel('Mean loss',fontsize=14)



    plt.figure(num=None,figsize=(8, 6))

    plt.plot(range(len(train_accuracies)), train_accuracies, range(len(test_accuracies)), test_accuracies)

    plt.xlabel('Epoch',fontsize=14)

    plt.ylabel('Accuracy',fontsize=14)

    plt.legend(['train accuracy', 'test accuracy'])

    plt.ylim([0, 1])

    

plot_stuff(losses, train_accuracies, test_accuracies)
# The following values were randomly generated and then hard coded 

# to ensure that the same samples are used regardless of the

# machine configuration.

random_samples = [1787,

    6944,

    3792,

    5969,

    1244,

    7577,

    596,

    786,

    3548,

    2195,

    4346,

    6370,

    3817,

    3603,

    1239,

    1106,

    6058,

    328,

    5325,

    8843]

sample_size = len(random_samples)



X_test_sample = X_test[random_samples]

y_test_sample = y_test.astype('int32')[random_samples]

y_test_sample_prediction = net(trans(X_test_sample).as_in_context(ctx)).asnumpy().argmax(axis=1)



plot_images(X_test_sample.asnumpy(),

            y_test_sample.asnumpy(),

            y_test_sample_prediction)

net, losses, train_accuracies, test_accuracies = train_lenet(iters=20, ctx=gpu())
plot_stuff(losses, train_accuracies, test_accuracies)