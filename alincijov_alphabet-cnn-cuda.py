import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from numba import cuda, jit, float32

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import string
df = pd.read_csv('../input/handwritten-az/handwritten_data_785.csv')
features = df.values[:,1:]

labels = df.values[:,0]



features = features.reshape(len(features), 28, 28)



nr_to_letter = {k:v.upper() for k,v in enumerate(list(string.ascii_lowercase))}
plt.title('Letter ' + nr_to_letter[labels[0]])

plt.imshow(features[0])
# normalize

features = features / 255.



# on eye encoding

labels = np.eye(len(np.unique(labels)))[labels]



# select only 100000 of features and labels

features, labels = features[:100000], labels[:100000]



# split the dataset

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
@cuda.jit

def convolution(result, mask, image):

    i, j = cuda.grid(2)



    image_rows, image_cols = image.shape

    if (i >= image_rows) or (j >= image_cols):

        return



    delta_rows = mask.shape[0] // 2

    delta_cols = mask.shape[1] // 2



    s = 0

    for k in range(mask.shape[0]):

        for l in range(mask.shape[1]):

            i_k = i - k + delta_rows

            j_l = j - l + delta_cols



            if (i_k >= 0) and (i_k < image_rows) and (j_l >= 0) and (j_l < image_cols):

                s += mask[k, l] * image[i_k, j_l]

    result[i, j] = s
def softmax(x):

    ex = np.exp(x)

    return ex / ex.sum()
def cost(probs,y):

    return -np.log(np.sum(y * probs))
def get_max(result):

    max_index = []

    max_arr = []



    for i in range(0, len(result[0]), 7):

        for j in range(0, len(result), 7):

            block = result[i:i+7, j:j+7]

            a, b = np.unravel_index(block.argmax(), block.shape)

            max_index.append((a + i, b + j))

            max_arr.append(result[a + i, b + j])



    return max_index, max_arr
def forward(image, theta, blockparams):

    w, mask = theta

    result = np.empty_like(image)

    convolution[blockparams](result, mask, image)



    max_index, max_arr = get_max(result)



    m = max_arr @ w

    probs = softmax(m)



    return max_index, max_arr, probs
def get_dpool(max_index, dm):

    dpool = np.zeros((28, 28))

    for i,k in enumerate(max_index):

        dpool[k[0],k[1]] = dm[i]



    return dpool
def backward(image, label, delta, theta, blockparams):

    w, mask = theta

    max_index, max_arr, probs = delta

    dout = probs - label



    dw = np.array(max_arr).reshape(16,1).dot(np.array(dout).reshape(1,26))

    dm = w @ dout.reshape(26,1) * np.array(max_arr).reshape(16,1)

    dm = dm.flatten()



    dpool = get_dpool(max_index, dm)

    

    # rotate 190 the derivatives of max pool

    drotated = np.rot90(np.rot90(dpool))

    

    dmask = np.zeros((7, 7))

    convolution[blockparams](dmask, np.ascontiguousarray(drotated, dtype=np.float32), image)



    return dw, dmask
np.random.seed(12342423)

w = np.random.randn(16, 26) * 0.01

mask = np.random.randn(7, 7) * 0.01



image = X_train[0]



blockdim = (28, 28)

griddim = (image.shape[0] // blockdim[0] + 1, image.shape[1] // blockdim[1] + 1)



theta = w, mask

blockparams = griddim, blockdim
batches = np.array_split(np.arange(len(X_train)), len(X_train)/128)



costs = []
def train(X, y, theta, batches):

    w, mask = theta



    for j in range(10):

        for batch in tqdm(batches):

            for i in batch:

                image = np.copy(X[i]).reshape(28,28)

                delta = forward(X[i], theta, blockparams)

                dw, dmask = backward(image, y[i], delta, theta, blockparams)



                w -= dw * 0.1 / len(batch)

                mask -= dmask * 0.1 / len(batch)

                theta = w, mask



        if j % 1 == 0:

            c = cost(delta[-1], y[i])

            costs.append(c)

    return theta
theta = train(X_train, y_train, theta, batches)
plt.title('Costs over Epochs')

plt.plot(costs)
def get_accuracy(X, y, batches, theta, blockparams):

    acc = 0

    for batch in batches:

        for i in batch:

            a = np.argmax(forward(X[i], theta, blockparams)[-1])

            if a == np.argmax(y[i]):

                acc += 1



    return acc / len(batches) / len(batch)
test_batches = np.array_split(np.arange(len(X_test)), len(X_test)/128)

accuracy = get_accuracy(X_test, y_test, test_batches, theta, blockparams)

print('Accuracy for all batches:', accuracy)