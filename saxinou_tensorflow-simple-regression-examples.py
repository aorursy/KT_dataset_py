# Prepare the environment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_boston 
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
boston = load_boston()
# The True passed to load_boston() lets it know that we want features and prices in separate numpy arrays.
print("Shape of design (feature) matrix : \n ", boston.data.shape)
print("List of features : \n ", boston.feature_names)

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
print("Simples statistics : \n ", bos.describe())

"""
Correlation
"""
corr= bos.corr(method='pearson')

# 1. HeatMap with Seaborn
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)

# 2. If you want to figures out
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
    
"""
Pair plot
"""
sns.pairplot(bos)
# Get the data 
total_features, total_prices = load_boston(True) 

# Keep 300 samples for training 
train_features = scale(total_features[:300]) 
train_prices = total_prices[:300] 
 
# Keep 100 samples for validation 
valid_features = scale(total_features[300:400]) 
valid_prices = total_prices[300:400] 
 
# Keep remaining samples as test set 
test_features = scale(total_features[400:]) 
test_prices = total_prices[400:] 
nb_obs = total_features.shape[0]
print("There is {} observations in our dataset ".format(nb_obs))

nb_feature = total_features.shape[1]
print("There is {} features in our dataset ".format(nb_feature))
# Set model weights - with random initialization
W = tf.Variable(tf.truncated_normal([nb_feature, 1], 
                                    mean=0.0, 
                                    stddev=1.0, 
                                    dtype=tf.float64), 
                name="weight") 
# Set model biais - initialized to 0
b = tf.Variable(tf.zeros(1, dtype = tf.float64), name="bias") 
# tf Graph Input 
X = tf.placeholder("float") 
Y = tf.placeholder("float") 
def linear_reg(x,y):
    # Define your equation Ypred = X * W + b
    Ypred = tf.add(b,tf.matmul(x,W))

    # Define your loss function
    error = tf.reduce_mean(tf.square(y - Ypred))
    
    # Return values 
    return([Ypred,error])

y, cost = linear_reg(train_features, train_prices)
# Define your parameter : 
learning_rate = 0.01
epochs = 200
cost_history = [[], []]

# Use gradient descent to minimize loss
optim = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in list(range(epochs)):
        sess.run(optim) # Execute the gradient descent, according our learning_rate and our cost function
        
        # For each 10 epochs, save costs values - we can plot it later  
        if i % 10 == 0.: 
            cost_history[0].append(i+1) 
            cost_history[1].append(sess.run(cost)) 
        if i % 100 == 0: 
            print("Cost = ", sess.run(cost)) 
        
    # Plot costs values
    plt.plot(cost_history[0], cost_history[1], 'r--')
    plt.ylabel('Costs')
    plt.xlabel('Epochs')
    plt.show() 
    
    train_cost = linear_reg(train_features, train_prices)[1]
    print('Train error =', sess.run(train_cost))
    valid_cost = linear_reg(valid_features, valid_prices)[1]
    print('Validation error =', sess.run(valid_cost))
sess.close()
"""
1.5.1. Create variables - we just add "W2"
"""
# tf Graph Input 
X = tf.placeholder("float") 
Y = tf.placeholder("float") 
 
# Set model weights - with random initialization
W1 = tf.Variable(tf.truncated_normal([12, 1], mean=0.0, stddev=1.0, dtype=tf.float64), name="weight1") 
W2 = tf.Variable(tf.truncated_normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float64), name="weight2") 

# Set model biais - initialized to 0
b = tf.Variable(tf.zeros(1, dtype = tf.float64), name="bias") 

"""
1.5.2. Create your model - With an additional term : X^2 * W2  
"""
def linear_reg_quad(x1, x2,y):
    # Define your equation Ypred = X * W1 + b + X^2 * W2
    # add another tf.add because this function accept only two arguments
    Ypred = tf.add(tf.add(b,tf.matmul(x1,W1)),
                   tf.matmul(tf.square(x2),W2) 
                  )      
    # Define your loss function
    error = tf.reduce_mean(tf.square(y - Ypred))
    # Return values 
    return([Ypred,error])

y, cost = linear_reg_quad(train_features[:,1:13],train_features[:,0:1], train_prices)

"""
1.5.3. Gradient descent (doesn't change)
"""
# Define your parameter : 
learning_rate = 0.01
epochs = 200
cost_history = [[], []]

# Use gradient descent to minimize loss
optim = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

"""
1.5.4. Initialize the variables (i.e. assign their default value) (doesn't change)
"""
init = tf.global_variables_initializer()

"""
1.5.5. Run your model (doesn't change, except for cost calculation at the end)
"""
with tf.Session() as sess:
    sess.run(init)
    for i in list(range(epochs)):
        sess.run(optim) # Execute the gradient descent, according our learning_rate and our cost function
        
        # For each 10 epochs, save costs values - we can plot it later  
        if i % 10 == 0.: 
            cost_history[0].append(i+1) 
            cost_history[1].append(sess.run(cost)) 
        if i % 100 == 0: 
            print("Cost = ", sess.run(cost)) 
        
    # Plot costs values
    plt.plot(cost_history[0], cost_history[1], 'r--')
    plt.ylabel('Costs')
    plt.xlabel('Epochs')
    plt.show() 
    
    train_cost = linear_reg_quad(train_features[:,1:13],
                                 train_features[:,0:1], 
                                 train_prices)[1]
    
    print('Train error =', sess.run(train_cost))
    valid_cost = linear_reg_quad(valid_features[:,1:13],
                                 valid_features[:,0:1], 
                                 valid_prices)[1]
    print('Validation error =', sess.run(valid_cost))

"""
2.1.1.1. Load datas
"""
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
iris = pd.read_csv('../input/Iris.csv')
print("Shape of iris database : ", iris.shape)

# Keep first 100 rows of data in order to do a binary classification
iris = iris[:100]
print("New shape of iris database, for binary classification : ", iris.shape)


# Plotting some features informations
plt.scatter(iris[:50].SepalLengthCm, 
            iris[:50].SepalWidthCm, 
            label='Iris-setosa')
plt.scatter(iris[51:].SepalLengthCm, 
            iris[51:].SepalWidthCm, 
            label='Iris-versicolo')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend(loc='best')

"""
2.1.1.2. Preprocessing and creation of train and valid set
"""
iris.Species = iris.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor'], value=[0, 1])

# Create your design matrix and y vector
X = iris.drop(labels=['Id', 'Species'], axis=1).values
Y = iris.Species.values

# Save dimension of our database
nb_obs = X.shape[0]
nb_feature = X.shape[1]

# Normalize your features 
Xscaled = scale(X) 

"""
2.1.1.3. Create your train and valid set
"""
X_train, X_test, y_train, y_test = train_test_split(Xscaled, 
                                                    Y, 
                                                    test_size=0.33, 
                                                    random_state=42)
# Recall : fix "random_state" for reproducible results

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

print("Recall : there is {} features ".format(nb_feature))

# X is placeholdre for iris features. We will feed data later on
X = tf.placeholder(tf.float32, [None, nb_feature])
# y is placeholder for iris labels. We will feed data later on
y = tf.placeholder(tf.float32, [None, 1])
    
# W is our weights. This will update during training time
W = tf.Variable(tf.zeros([nb_feature, 1]))
# b is our bias. This will also update during training time
b = tf.Variable(tf.zeros([1]))
  
# our prediction function
Ypred = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
    
# calculating cost
cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, 
                                               logits=Ypred)

# optimizer
# we use gradient descent for our optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# Define the learning rate， batch_size etc.
learning_rate = 0.01
epochs = 300
cost_history = [[], []]
# Initialize the variables (i.e. assign their default value) 
init = tf.global_variables_initializer() 
cost_history = [[], []]

with tf.Session() as sess:
    sess.run(init) # # Run the initializer
    
    for i in list(range(epochs)):
        cost_in_each_epoch = 0
        # let's start training
        _, c = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})
        cost_in_each_epoch += c
            
    print("Optimization Finished!")
    
    # Test the model
    correct_prediction = tf.equal(tf.argmax(Ypred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy on train set:", accuracy.eval({X: X_train, y: y_train}))
    print("Accuracy on test set:", accuracy.eval({X: X_test, y: y_test}))
