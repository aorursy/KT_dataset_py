import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import matplotlib

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



is_python = "inline" in matplotlib.get_backend()

if is_python:

    from IPython import display
def load_data(path):

    with np.load(path) as mnist:

        X_train, y_train = mnist['x_train'], mnist['y_train']

        X_test, y_test = mnist['x_test'], mnist['y_test']

    return X_train,y_train, X_test, y_test
path = "../input/mnist-numpy/mnist.npz"
x_train, y_train, x_test, y_test = load_data(path)
print("X_train: {}, y_train: {}".format(x_train.shape,y_train.shape))

print("X_test: {}, y_test: {}".format(x_test.shape,y_test.shape))
mnist_image = np.vstack((x_train,x_test))

mnist_image = mnist_image.reshape(-1,784)

print(mnist_image.shape)

mnist_label = np.vstack((y_train.reshape(-1,1),y_test.reshape(-1,1)))

print(mnist_label.shape)
mnist_train = '../input/digit-recognizer/train.csv'

mnist_test = '../input/digit-recognizer/test.csv'

sample = '../input/digit-recognizer/sample_submission.csv'
train_data = pd.read_csv(mnist_train)

test_data = pd.read_csv(mnist_test)
print(train_data.shape)

print(test_data.shape)
train_images = train_data.copy()

train_images = train_images.values

X_train = train_images[:,1:]

y_train = train_images[:,0]

X_test = test_data.values



print(X_train.shape)

print(y_train.shape)

print(test_data.shape)
X_train = X_train.reshape(-1,28,28)

y_train = y_train.reshape(-1,1)



print(X_train.shape)

print(y_train.shape)
plt.figure(figsize=(100,100))

sns.set_style('whitegrid')

for idx in range(400):

    plt.subplot(20,20,idx+1)

    plt.imshow(X_train[idx],interpolation='nearest',cmap='gray')

    plt.title(y_train[idx],fontsize=20)

    plt.xticks([])

    plt.yticks([])

    plt.colorbar()
predictions = np.zeros((mnist_label.shape))
x1=0

x2=0

print("Classifying Kaggle's 'test.csv' using KNN where K=1 and MNIST 70k images..")

for i in range(0,28000):

    for j in range(0,70000):

        if np.absolute(X_test[i,:]-mnist_image[j,:]).sum()==0:

            predictions[i]=mnist_label[j]

            if i%1000==0:

                print("  %d images classified perfectly"%(i),end="")

            if j<60000:

                x1+=1

            else:

                x2+=1

            break



if x1+x2==28000:

    print(" 28000 images classified perfectly.")

    print("All 28000 images are contained in MNIST.npz Dataset.")

    print("%d images are in MNIST.npz train and %d images are in MNIST.npz test"%(x1,x2))
final_pred = predictions[0:28000]
final_pred[-10:]
sample_data = pd.read_csv(sample)
sample_data.head()
my_submission = pd.DataFrame({'ImageId':np.arange(28000),'Label':final_pred.squeeze().astype(np.int)})
my_submission.head()
my_submission.to_csv('submission.csv', index=False)