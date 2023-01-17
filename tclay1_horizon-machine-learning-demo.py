"""

We import various python libraries for data science and AI. Other popular libraries include: 

1. ggplot for pretty graphs, 

2. pylearn2 for more models,

3. statsmodel for statistics, but a statistical language is better. 

    - I use R for small work and Stata for a lot of data. Stata requires a student license or payment tho.

4. scipy for computing, mostly linear algebra

"""



import numpy as np #  for multidimensional arrays

import pandas as pd  # to display the data, but we don't use it here

import matplotlib.pyplot as plt # to get pretty graphs

from sklearn import neural_network # implementation of the model



"""

Here, we format the data, so we import some libraries. However, these libraries aren't 

particularly needed, because you could have done this yourself with for loops. To do so, 

read in the rows and cols of a picture and put the values into a numpy array. Do this for 

all pictures.



Formally, we turn the picture object into a stream of bits. This is called as serialization. Also, 

remember that images are commonly just N by M matrices whose individual pixel values are some

value in range 0 - 255, but when we pickled the data, it normalized the pixel value between 0 and 1 

for the sake of simplicity.



"""



import gzip, pickle, sys 

f = gzip.open('../input/mnist.pkl.gz', 'rb')

(input_train, output_train), (input_test, output_test), _ = pickle.load(f, encoding='bytes') # we don't need the last return from the load function, so we didn't bind it to a variable - hence, the underscore.



""" 

To make sure that we loaded the data correctly, let's look at it.

"""

# lets look at the size of  our data

img_len = int(np.sqrt(input_train.shape[1])) # we have N by N images

print(input_train.shape, output_train.shape)

print('image dimension = ' + str(img_len) + 'x' + str(img_len))



# take a look at the data set to see what we are given

print(input_train.shape)

print(input_train.min(), input_train.max())



print(output_train.shape)

print(output_train[0])



# plot out an example to see what we are looking at

plot_num(input_train[0,:]);
"""

Let's explore the data further, so we create functions that make a graphical

representation of the image.

"""

# visualize one number

def plot_num(X): 

    X = X.reshape(1,-1) 

    assert(X.shape[1] == img_len*img_len)

    image = np.reshape(X, (img_len, img_len) )

    _ = plt.figure()

    _ = plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    

# visualize 100 numbers, store numbers in matrix format where each row represents a number

def plot_num_100(X):

    assert(X.shape[1] == img_len*img_len)

    n = len(X) if (len(X) < 100) else 100

    X = X[np.random.choice(np.arange(len(X)), size=n, replace=False),:]

    

    fig = plt.figure()

    for i in range(len(X)):

        image = np.reshape(X[i,:], (img_len, img_len) )

        _ = plt.subplot(10, 10, i + 1)

        _ = plt.axis('off')

        _ = plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
"""

Let's use the function that we just made to visualize 100 numbers in our training set

"""

plot_num_100(input_train[:100,:])
"""

Now that we've explored our dataset and formatted the data in a way that the sklearn library 

can understand, we're ready to train our model. 



We decided to use a NN, because NNs are good models for recognizing complex patterns.  



The inputs and options to a model must be made carefully. The only option for this model is 

verbose, a bool to turn the logs off or on.The interesting decisions made are about the inputs. 

Max iterations is the amount of times that the training will iterate. More iterations could 

tend towards a 0 percent error on the training set, but to do so would overfit the model. 

Determine the max iterations carefully. Tol is just the tolerance for the model's improvement. 

Hidden layers is the number of layer nodes to use. This post has a good discussion on how to decide 

the number of hidden layers: 



https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw



Other inputs of interest include alpha and learning rate. The others should be tweaked carefully.



**For the first round of training a model, I always turn on verbose and see where the inflection 

point is on testing error. But professional data scientist have better heuristics - such as, the 

formula in the post.** 



**The cool thing is that finding the best inputs for a model for a particular dataset is also an 

interesting AI problem. If you give it a try, I suggest that you use gradient ascent, but for the 

sake of performance, only give a discrete set of options for tol.**



"""



hidden_layer_sizes = [25] #Intuitively, this means we are breaking up the image into 25 different regions, and training the nodes to identify patterns within that quadrant.

model1 = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,

                                      verbose=True,

                                      tol=0.00200, # made the tol higher to get rid of the warning, but this still  needs to be optimized better

                                      max_iter = 100)

model1 = model1.fit(input_train[:1000], output_train[:1000]) # giving the model more data will make it slow to train. If you do give it more data, change tol accordingly :(
"""

Now, we have trained models! Let's see if they are any good. We have 88 accurancy on the testing 

set with 99 percent accuracy in the training set. I would say that this model isn't overfitting, 

and I'm not suspicious of anything else that could have went wrong. It's not the best model, however.



But, if you know some statistics, sometimes residual plots, t-tests, pcas, and other methods of 

determining the impact of each variable in a model are useful for figuring out if you're overfiting. 

"""



output_pred = model1.predict(input_test)

output_pred_train = model1.predict(input_train)

print(1 - np.sum(output_pred_train[:1000] != output_train[:1000]) / len(output_train[:1000]))

print(1 - np.sum(output_pred != output_test) / len(output_test))
"""

Now, let's literally see if they are any good. That's about it :)

"""



for i in np.random.choice(len(input_test), 5):

    plot_num(input_test[i, :])

    _=plt.title('Prediction: {}'.format(output_pred[i]))