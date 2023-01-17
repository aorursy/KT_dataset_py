import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
x = tf.constant([4,2],dtype=tf.int32)  # can not take Tensor Type as input

m = tf.convert_to_tensor([2,3]) # can take Tensor Type as input

c = tf.constant(5) # A list with different dtypes or numpy array can be passed too.
y = (m*x)+c # try with tf.matmul if you want to perform matric multiplication

y.numpy()
tf.concat([m,x],axis=0) # default axis is 0 but you can concat on other axes too
t = [[1, 2, 3],[4, 5, 6]] # shape [2, 3]

print("axis=-1:\n",tf.expand_dims(t, -1),'\n') # use last index dimension so axis=-1 works as axis=2

print("axis=0:\n",tf.expand_dims(t, 0),'\n') # see the difference 

print("axis=1:\n",tf.expand_dims(t, 1),'\n') #  between 0 and 1

print("axis=2:\n",tf.expand_dims(t, 2),'\n') # same as -1 or last index dimension
cons = tf.constant([1.3,5.8,9]) #not good practice. Here data will be converted to same type internally 

# cons[0] = 5.0 # try it. I could be decieving you
var = tf.Variable([1.3,5.8,9])

var[0].assign(3.1)

var
ans = [2,2,2]*((var*cons)/(var+cons))

# try doing multiplying by 2 instead of [2,2,2]



print(type(ans))  # Look closely and see what is happening in the output

ans
weight = tf.Variable(3.2)



lr = 0.01



def calculate_loss(w):

    return w**1.3





manual_losses = []

for i in range(100):

    weight = weight.assign(weight - lr*1.3*weight) 

    # this 1.3*weight makes the GD and comes due to the mighty Calculus

    manual_losses.append(calculate_loss(weight)) # calculate the loss ar epoch i





plt.plot(manual_losses)

plt.show()
def calculate_gradient(w):

    '''

    Method will take the weight vector w and calculate the gradients using chaining differential calculus

    '''

    with tf.GradientTape() as tape:

        loss = calculate_loss(w) # calculate loss WITHIN tf.GradientTape() for a single or list of weights

        

    grad = tape.gradient(loss,w) # gradient of loss wrt. w the second argument w can be a list too

    return grad







# train and apply the things here

weight = tf.Variable(3.2)



opt = tf.keras.optimizers.SGD(lr=lr)



tf_losses = []



for i in range(100):

    grad = calculate_gradient(weight)

    opt.apply_gradients(zip([grad],[weight]))  # zip makes a list of tuple of (grad_i,weight_i)

    # map all the gradients to all the layers. There can be many weights and respective gradients

    

    tf_losses.append(calculate_loss(weight))

    



plt.plot(tf_losses)

plt.show()
class RegressionModel(tf.keras.Model): # every new model has to use Model

    '''

    A Model that performs Linear Regression 

    '''

    def __init__(self,in_units,out_units):

        '''

        args:

            in_units: number of input neurons

            out_units: number of output units

        '''

        super(RegressionModel,self).__init__() # constructor of Parent class 

        

        self.in_units = in_units

        self.out_units = out_units

        

        self.w = tf.Variable(tf.initializers.GlorotNormal()((self.in_units,self.out_units))) 

        # make weight which has initial weights according to glorot_normal distribution

        

        self.b = tf.Variable(tf.zeros(self.out_units)) # bias is mostly zeros

        

        self.params = [self.w,self.b] # we can use the model.params directly inside the GradientTape()

            

    

    def call(self,input_tensor):

        '''

        execurte forward pass

        args:

            input_tensor: input tensor which will be fed to the network

        '''

        return tf.matmul(input_tensor, self.w) + self.b
def generate_random_data(shape=(100,1)):

    '''

    Generate correlated X and y points which are correlated according to straight line  y=mx+c

    args:

        feat_shape: {tuple} shape of the X data

    '''

    X = np.random.random(shape)* np.random.randn() - np.random.randn() # complete randomness

    m = np.random.random((shape[1],1))

    c = np.random.randn()

    y = X.dot(m) +  c + np.random.randn(shape[0],1)*0.13 # add some noise too

    return X,y





def compute_loss(model,x_features,y_true):

    '''

    Calculate the loss. You can use RMSE or MAE 

    args:

        model:  a model that'll give  predicted values

        x_features: Array of data points

        y_true: respective target values

    '''

    y_pred = model(x_features)

    error = y_true  - y_pred

    return tf.reduce_mean(tf.square(error)) # MSE: Mean Squred Error







def compute_grad(model,x_features,y_true):

    '''

    Compute the Gradient here

    '''

    with tf.GradientTape() as tape:

        loss = compute_loss(model,x_features,y_true)

        

    return tape.gradient(loss,model.params) # you see model.params. It'll include all the params 

X,y = generate_random_data()  # try generating the data different times to see performance of model





X = X.astype(np.float32) # default is double or float 64 in numpy

y = y.astype(np.float32) # tf would have converted it to float32 automatically





plt.scatter(X,y)

plt.show()
losses = []

optimizer = tf.keras.optimizers.SGD(lr=0.01)

model = RegressionModel(1,1) # 1 feature columns and 1 output for regression easy for plotting

print(f"Initial:\n{model.w}\n{model.b}\n\n") # model's initial weights and biases



for epoch in range(500):

    gradients = compute_grad(model,X,y)

    optimizer.apply_gradients(zip(gradients,model.params)) # apply back prop to all the "trainable" params

    

    losses.append(compute_loss(model,X,y)) # make a list of loss per epoch



print(f"Final:\n{model.w}\n{model.b}")



plt.plot(losses)

plt.show()

    

    
reg_x_points = np.linspace(X.min(),X.max(),2) # get 100 equally spaced points between x_min and max

reg_y_points = model.predict(reg_x_points.reshape(-1,1)).flatten() # get y predictions of those points



plt.plot(reg_x_points,reg_y_points,color='m',label='Regression Line') 

# it'll draw a straight line as all the points are on the same line



plt.scatter(X,y) # original data points

plt.show()
@tf.function

def operation(x, y):

    return x*y



operation(tf.constant([9, 2]), tf.Variable([3,5]))
v = tf.Variable(4.07)



with tf.GradientTape() as tape:

    result = operation(v, 2.45)

    

tape.gradient(result, v)
@tf.function

def common_function(inp):

    print("Tracing with", inp)

    return inp + inp



print(common_function(tf.constant(1))) # int as input

print('*'*50,'\n')



print(common_function(tf.constant(1.1))) # float as input

print('-'*50,'\n')



print(common_function(tf.constant("inp is string")))