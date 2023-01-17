import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.display import display,Image

from string import Template

import pandas as pd

import numpy as np

import IPython.display

import warnings 
display(Image(filename='/kaggle/input/cnn-images/cnn/conv.png'))
display(Image(filename='/kaggle/input/cnn-images/cnn/conv2.png'))
class Conv:

    

    def __init__(self, num_filters):

        self.num_filters = num_filters

        

        #why divide by 9...Xavier initialization

        self.filters = np.random.randn(num_filters, 3, 3)/9

    

    def iterate_regions(self, image):

        #generates all possible 3*3 image regions using valid padding

        

        h,w = image.shape

        

        for i in range(h-2):

            for j in range(w-2):

                im_region = image[i:(i+3), j:(j+3)]

                yield im_region, i, j

                

    def forward(self, input):

        self.last_input = input

        

        h,w = input.shape

        

        output = np.zeros((h-2, w-2, self.num_filters))

        

        for im_regions, i, j in self.iterate_regions(input):

            output[i, j] = np.sum(im_regions * self.filters, axis=(1,2))

        return output

    

    def backprop(self, d_l_d_out, learn_rate):

        '''

        Performs a backward pass of the conv layer.

        - d_L_d_out is the loss gradient for this layer's outputs.

        - learn_rate is a float.

        '''

        d_l_d_filters = np.zeros(self.filters.shape)



        for im_region, i, j in self.iterate_regions(self.last_input):

            for f in range(self.num_filters):

                d_l_d_filters[f] += d_l_d_out[i,j,f] * im_region



        #update filters

        self.filters -= learn_rate * d_l_d_filters



        return None
display(Image(filename='/kaggle/input/cnn-images/cnn/maxpool.png'))
display(Image(filename='/kaggle/input/cnn-images/cnn/maxpool2.png'))
class MaxPool:

    def iterate_regions(self, image):

        h, w, _ = image.shape

        

        new_h = h // 2

        new_w = w // 2

        

        for i in range(new_h):

            for j in range(new_w):

                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]

                yield im_region, i, j

                

    def forward(self, input):

        

        self.last_input = input

        

        h, w, num_filters = input.shape

        output = np.zeros((h//2, w//2, num_filters))

        

        for im_region, i, j in self.iterate_regions(input):

            output[i,j] = np.amax(im_region,axis=(0,1))

            

        return output

    

    def backprop(self, d_l_d_out):

        '''

        Performs a backward pass of the maxpool layer.

        Returns the loss gradient for this layer's inputs.

        - d_L_d_out is the loss gradient for this layer's outputs.

        '''

        d_l_d_input = np.zeros(self.last_input.shape)



        for im_region, i, j in self.iterate_regions(self.last_input):

            h, w, f = im_region.shape

            amax = np.amax(im_region, axis=(0,1))



            for i2 in range(h):

                for j2 in range(w):

                    for f2 in range(f):

                        #if the pixel was the max value, copy the gradient to it

                        if(im_region[i2,j2,f2] == amax[f2]):

                            d_l_d_input[i*2+i2, j*2+j2 ,f2] = d_l_d_out[i, j, f2]

                            break;

        return d_l_d_input
display(Image(filename='/kaggle/input/cnn-images/cnn/softmax.png'))
display(Image(filename='/kaggle/input/cnn-images/cnn/sogtmax2.png'))
class Softmax:

    def __init__(self, input_len, nodes):

        # We divide by input_len to reduce the variance of our initial values

        self.weights = np.random.randn(input_len, nodes)/input_len

        self.biases = np.zeros(nodes)

    

    def forward(self, input):

        

        self.last_input_shape = input.shape

        

        input = input.flatten()

        self.last_input = input

        

        input_len, nodes = self.weights.shape

        

        totals = np.dot(input, self.weights) + self.biases

        self.last_totals = totals

        

        exp = np.exp(totals)

        return(exp/np.sum(exp, axis=0)) 

    

    def backprop(self, d_l_d_out, learn_rate):

        """  

        Performs a backward pass of the softmax layer.

        Returns the loss gradient for this layers inputs.

        - d_L_d_out is the loss gradient for this layers outputs.

        """

        

        #We know only 1 element of d_l_d_out will be nonzero

        for i, gradient in enumerate(d_l_d_out):

            if(gradient == 0):

                continue

            

            #e^totals

            t_exp = np.exp(self.last_totals)

            

            #Sum of all e^totals

            S = np.sum(t_exp)

            

            #gradients of out[i] against totals

            d_out_d_t = -t_exp[i] * t_exp/ (S**2)

            d_out_d_t[i] = t_exp[i] * (S-t_exp[i]) /(S**2)

            

            # Gradients of totals against weights/biases/input

            d_t_d_w = self.last_input

            d_t_d_b = 1

            d_t_d_inputs = self.weights

            

            #Gradients of loss against totals

            d_l_d_t = gradient * d_out_d_t

            

            #Gradients of loss against weights/biases/input

            d_l_d_w = d_t_d_w[np.newaxis].T @ d_l_d_t[np.newaxis]

            d_l_d_b = d_l_d_t * d_t_d_b  

            d_l_d_inputs = d_t_d_inputs @ d_l_d_t

            

            #update weights/biases

            self.weights -= learn_rate * d_l_d_w

            self.biases -= learn_rate * d_l_d_b

            return d_l_d_inputs.reshape(self.last_input_shape)
import mnist



train_images = mnist.train_images()[:1000]

train_labels = mnist.train_labels()[:1000]

test_images = mnist.test_images()[:1000]

test_labels = mnist.test_labels()[:1000]
display(Image(filename='/kaggle/input/cnn-images/cnn/train.png'))
conv = Conv(8)

pool = MaxPool()

softmax = Softmax(13 * 13 * 8, 10)



def forward(image, label):

    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier

    # to work with. This is standard practice.

    

    out = conv.forward((image/255) - 0.5)

    out = pool.forward(out)

    out = softmax.forward(out)

    

    #calculate cross-entropy loss and accuracy

    loss = -np.log(out[label])

    acc = 1 if(np.argmax(out) == label) else 0

    

    return out, loss, acc





def train(im, label, lr=0.005):

    #forward

    out,loss,acc = forward(im, label)

    

    #calculate initial gradient

    gradient = np.zeros(10)

    gradient[label] = -1/out[label]

    

    

    #Backprop

    gradient = softmax.backprop(gradient, lr)

    gradient = pool.backprop(gradient)

    gradient = conv.backprop(gradient, lr)

    

    return loss, acc

    

    

print('MNIST CNN initialized')



for epoch in range(3):

    print('----EPOCH %d ---'%(epoch+1))

    

    #shuffle the training data

    permutation = np.random.permutation(len(train_images))

    train_images = train_images[permutation]

    train_labels = train_labels[permutation]





    loss = 0

    num_correct = 0



    for i, (im, label) in enumerate(zip(train_images, train_labels)):



        #print stats every 100 steps

        if(i>0 and i %100 == 99):

            print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))



            loss = 0

            num_correct = 0

        l, acc = train(im, label)

        loss += l

        num_correct += acc
display(Image(filename='/kaggle/input/backprop-cnn/backprop_CNN/1.png'))
display(Image(filename='/kaggle/input/backprop-cnn/backprop_CNN/2.png'))
display(Image(filename='/kaggle/input/backprop-cnn/backprop_CNN/3.png'))
display(Image(filename='/kaggle/input/backprop-cnn/backprop_CNN/4.png'))
display(Image(filename='/kaggle/input/backprop-cnn/backprop_CNN/5.png'))