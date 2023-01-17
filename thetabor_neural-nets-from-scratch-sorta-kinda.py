import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
		e = nonlin(x)
		return e*(1-e)
	return 1/(1+np.exp(-x))
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
		[1],
		[1],
		[0]])
np.random.seed(1)

# randomly initialize our network weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):
	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
syn0
syn1
type(syn0), type(syn1)
# Feed forward through layers 0, 1, and 2
l0 = X
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))
# how much did we miss the target value?
l2_error = y - l2
print(l2_error)
l2
np.random.random()
np.random.random((10, 10))
m = np.random.random((2,2))
n = np.random.random((2,2))
k = np.dot(m,n)
print("m:", m)
print("n:", n)
print("k:", k)
j = nonlin(k)
print("j:", j)
print(nonlin(5))
l2_delta.dot(syn1.T)
syn1.T
syn1
m
m.T
m = np.random.random((3,3))
m
m.T
def initialize_net(inputs, hiddens, outputs):
    # randomly initialize our weights with mean 0
    syn0 = 2*np.random.random((inputs,hiddens)) - 1
    syn1 = 2*np.random.random((hiddens,outputs)) - 1
    
    return syn0, syn1
def predict_proba(x):
    # Feed forward through layers 0, 1, and 2
    l0 = x
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    
    return l2, l1

# don't use this with a single output network!
def predict(x):
    return predict_proba(x)[0].argmax()
def fit(X, y, syn0, syn1, cycles = 6000, alpha = 0.01):
    for j in range(cycles):
        predicts, hiddens = predict_proba(X)

        # how much did we miss the target value?
        error = y - predicts

        if (j% 100) == 0:
            print("Error:" + str(np.mean(np.abs(error))))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        predicts_delta = alpha*error*nonlin(predicts,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        hiddens_error = predicts_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        hiddens_delta = hiddens_error * nonlin(hiddens,deriv=True)

        syn1 += hiddens.T.dot(predicts_delta)
        syn0 += X.T.dot(hiddens_delta)
# training variables
alpha = 0.005
cycles = 6000

syn0, syn1 = initialize_net(3, 4, 1)
    
fit(X, y, syn0, syn1, cycles, 1)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
data = pd.read_csv("../input/train.csv")[:50]
data.head()
data.describe()
image_width,image_height = 28,28

images = data.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))
# display image
def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

# output image     
IMAGE_TO_DISPLAY = 42
display(images[IMAGE_TO_DISPLAY])
labels = data['label'].values
labels[42]
# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels, 10)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))
X = images
y = labels
np.random.seed(42)

# training variables
alpha = 0.005
cycles = 1000

syn0, syn1 = initialize_net(784, 15, 10)
    
fit(X, y, syn0, syn1, cycles, alpha)
val_data = pd.read_csv("../input/train.csv")[50:100]
val = val_data.iloc[:,1:].values
val = val.astype(np.float)

# convert from [0:255] => [0.0:1.0]
val = np.multiply(val, 1.0 / 255.0)

val_labels = val_data['label'].values
val_labels_oh = dense_to_one_hot(val_labels, 10)
val_labels_oh = val_labels_oh.astype(np.uint8)

print('val({0[0]},{0[1]})'.format(val.shape))
def display_predict(i):
    IMAGE_TO_DISPLAY = i
    print ('val_labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,val_labels[IMAGE_TO_DISPLAY]))
    print("prediction: ", predict(val[i]))
    display(val[IMAGE_TO_DISPLAY])
display_predict(np.random.randint(50))
def score(imgs, labels, labels_oh):
    l0 = imgs
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l2_error = labels_oh - l2
    predicts = [probs.argmax() for probs in l2]
    num_correct = sum([p == l for (p, l) in zip(predicts, labels)])
    print("Validation Error: {0}, {1} of {2} correctly labeled".format(str(np.mean(np.abs(l2_error))), num_correct, len(predicts)))
score(val, val_labels, val_labels_oh)
test_data = pd.read_csv("../input/train.csv")[50:]
test = test_data.iloc[:,1:].values
test = test.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test = np.multiply(test, 1.0 / 255.0)

print('test({0[0]},{0[1]})'.format(test.shape))

test_labels = test_data['label'].values
test_labels_oh = dense_to_one_hot(test_labels, 10)
test_labels_oh = test_labels_oh.astype(np.uint8)
score(test, test_labels, test_labels_oh)
