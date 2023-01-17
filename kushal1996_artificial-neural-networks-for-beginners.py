'''example of perceptron using scikit learn'''

import numpy as np 

from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron 



iris = load_iris()

X = iris.data[: ,  (2 , 3)] # petal lenght and petal width

y = (iris.target == 0 ).astype(np.int) # binary classification 



classifier = Perceptron(max_iter = 1000 ,tol = 1e-3 ,random_state = 42)

classifier.fit(X , y)

y_pred = classifier.predict([[1.5 , 0.5] , [3 , 3.2]])



print(y_pred)
import tensorflow as tf

import matplotlib

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits 



digits = load_digits()

m , n = digits.data.shape



n_inputs = n 

n_hidden_neurons_1 = 300

n_hidden_neurons_2 = 100

n_outputs = 10 # classes 0 to 9
X = tf.placeholder(tf.float32 ,shape = (None , n_inputs) , name = 'X')

y = tf.placeholder(tf.int64 , shape = (None) , name = 'y')
def fetch_batch(epoch, batch_index, batch_size):

    np.random.seed(epoch * n_batches + batch_index)      

    indices = np.random.randint(digits.data[:1700].shape[0] , size = batch_size)  

    X_batch = digits.data[:1700][indices] 

    y_batch = digits.target[:1700][indices] 

    return X_batch, y_batch
def neuron_layer(X , n_neurons , name , activation = None):

    with tf.name_scope(name):

        n_inputs = int(X.get_shape()[1])

        stddev = 2 /np.sqrt(n_inputs)

        '''initializing weights randomly , using truncated normal distribution

        with standard deviation of 2/sqrt(n_inputs)'''

        init = tf.truncated_normal((n_inputs , n_neurons) , stddev = stddev)

        W = tf.Variable(init , name = 'weights')

        '''biases'''

        b = tf.Variable(tf.zeros([n_neurons]) , name = 'bias')

        '''weighted sum'''

        Z = tf.matmul(X , W) + b

        '''activation function'''

        if activation is not None:

            return activation(Z)

        else:

            return Z
with tf.name_scope('dnn'):

    hidden_layer_1 = neuron_layer(X = X , n_neurons = n_hidden_neurons_1 ,

                                  name = 'hiddenLayer1' , activation = tf.nn.relu)

    hidden_layer_2 = neuron_layer(X = hidden_layer_1 , n_neurons = n_hidden_neurons_2 ,

                                 name = 'hiddenLayer2' , activation = tf.nn.relu)

    logits = neuron_layer(X = hidden_layer_2 , n_neurons = n_outputs , name = 'outputs')
with tf.name_scope('loss'):

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y , 

                                                              logits = logits)

    loss = tf.reduce_mean(xentropy , name = 'loss')
learning_rate = 0.01 

with tf.name_scope('train'):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    training_op = optimizer.minimize(loss)
with tf.name_scope('eval'):

    correct = tf.nn.in_top_k(logits , y , 1)

    accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))
init = tf.global_variables_initializer()

saver = tf.train.Saver()
n_epochs = 1000

batch_size = 50 

n_batches = int(np.ceil(digits.data.shape[0] / batch_size))



with tf.Session() as session:

    init.run()

    for epoch in range(n_epochs):

        

        for batch_index in range(n_batches):

            X_batch , y_batch = fetch_batch(epoch , batch_index  , batch_size)

            session.run(training_op , feed_dict = {X : X_batch , y : y_batch})

            

        acc_train = accuracy.eval(feed_dict = {X : X_batch , y : y_batch})

        acc_test = accuracy.eval(feed_dict = {X : digits.data[1700:] , y : digits.target[1700:]})

        if epoch % 100 == 0 :

            print('epoch : {0}, Train Acc : {1}  , Test Acc : {2}'.format(epoch ,acc_train , acc_test))

    save_path = saver.save(session , './model_final.ckpt') 
with tf.Session() as session:

    saver.restore(session , './model_final.ckpt')

    X_test = digits.data[1780:]

    Z = logits.eval(feed_dict = {X : X_test})

    y_pred = np.argmax(Z , axis = 1)



plt.figure(1 , figsize = (15 , 10))

for n in np.arange(1 , 17):

    plt.subplot(4 , 4 , n )

    plt.subplots_adjust(hspace = 0.3 , wspace=0.3)

    plt.imshow(digits.data[1780:][n].reshape(8 , 8)  ,cmap = matplotlib.cm.binary , interpolation="sinc")

    x_l = "True : {0} , Predicted : {1}".format(digits.target[1780:][n] , y_pred[n])

    plt.xlabel(x_l)

    

plt.show()
'''creating an input function which returns X , y'''

'''reason to create this input function is because tf.estimator.DNNClassifier() class

methods train() , eval() and predict() needs a function in the parameter which returns 

X and y'''

def input(df):

    return df.data[:1789] , df.target[:1789].astype(np.int32)



'''tf.feature_column are used to convert some sort of input data feature into continuous 

variables that can be used by a regression or neural network model.'''

'''In the case of a numeric_column, there is no such conversion needed,

so the class basically just acts like a tf.placeholder.'''

feature_col = [tf.feature_column.numeric_column('x' , shape = [8 , 8])] #shape = (8,8) cause digits data images are of 64 pixel





'''creating architecture of DNN with two hidden layer with 300 and 100 neurons 

respectively and a softmax output layer with 10 neurons'''

dnn_clf = tf.estimator.DNNClassifier(hidden_units = [200 , 100] ,

                                     n_classes = 10  ,

                                     feature_columns = feature_col 

                                    ) 





'''Defining the training inputs'''

input_fn_train = tf.estimator.inputs.numpy_input_fn(

    x={"x": input(digits)[0]}, # X

    y=input(digits)[1], # y

    num_epochs=None,

    batch_size=50,

    shuffle=True # randomness

)



'''training the classifier'''

dnn_clf.train(input_fn = input_fn_train , steps = 1000 )



'''Defining the Eval inputs'''

input_fn_eval = tf.estimator.inputs.numpy_input_fn(

    x={"x": input(digits)[0]},

    y=input(digits)[1],

    num_epochs=1,

    shuffle=False

)



'''evaluating'''

metrics = dnn_clf.evaluate(input_fn=input_fn_eval , steps=10)



'''Defining the predict inputs'''

input_fn_predict = tf.estimator.inputs.numpy_input_fn(x = {"x" : digits.data[1788:]} , 

                                                          num_epochs = 1 , 

                                                          shuffle = False

                                                         )

'''predicting'''

predictions = dnn_clf.predict(input_fn= input_fn_predict)



'''Note you should always split the data into train , eval and test data which i

have not done , because this is just an example.'''
'''getting the predicted classes'''

cls = [p['classes'] for p in predictions]

'''converting into int array'''

y_pred = np.array(cls , dtype = 'int').squeeze()



'''ploting true value and predicted value'''

plt.figure(1 , figsize = (15 , 10))

for n in np.arange(1 , 9):

    plt.subplot(4 , 4 , n )

    plt.subplots_adjust(hspace = 0.3 , wspace=0.3)

    plt.imshow(digits.data[1788:][n].reshape(8 , 8)  ,cmap = matplotlib.cm.binary , interpolation="sinc")

    x_l = "True : {0} , Predicted : {1}".format(digits.target[1788:][n] , y_pred[n])

    plt.xlabel(x_l)

    

plt.show()