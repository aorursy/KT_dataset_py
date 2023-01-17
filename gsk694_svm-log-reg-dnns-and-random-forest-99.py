import warnings,math

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

import pandas as pd

import tensorflow as tf

import operator,time,sys; import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from sklearn import linear_model,metrics

from sklearn.mixture import GMM

from sklearn.svm import SVC 

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV,train_test_split,KFold, cross_val_score

import sklearn,matplotlib

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report

#from scipy.optimize import fmin_bfgs as bfgs

from sklearn.metrics import log_loss



def load(str):



	df = pd.read_csv(str, header = 0)

	x = df.ix[:,df.columns!='diagnosis']

	y = df.ix[:,df.columns=='diagnosis']

	#print(y)

	y = y['diagnosis'].map({'M':1,'B':0})

	

	x = x.drop(['id','Unnamed: 32','area_mean','perimeter_mean','concavity_mean','concave points_mean','area_worst','perimeter_worst',

		'concave points_worst','concavity_worst','area_se','perimeter_se'],axis = 1)

	features = []

	for i in x:

		features.append(i)

	frames = [x,y]

	total = pd.concat(frames,axis = 1)

	#print(total)

	#total.info()

	return total, features



def report(grid_scores , n_top = 3):

    top_scores = sorted(grid_scores, key = operator.itemgetter(1), reverse= True)[:n_top]



def gmm(x,features):

	del features[-1]

	train, test = train_test_split(x, test_size = 0.2)

	train_x,test_x = train[features],test[features]

	train_y, test_y = train.diagnosis, test.diagnosis

	start = time.time()

	x_all = np.r_[train_x,test_x]

	lowest_bic = np.infty

	bic = []

	n_components_range = range(1,7)



	cv_types = ['spherical', 'tied', 'diag', 'full']

	for i in cv_types:

		for j in n_components_range:

			gmm = GMM(n_components= j, covariance_type = i,min_covar = 0.0001)



			gmm.fit(x_all)

			bic.append(gmm.aic(x_all))

			if bic[-1] < lowest_bic:

				lowest_bic = bic[-1]

				best_gmm = gmm

	g = best_gmm

	g.fit(x_all)



	x=g.predict_proba(train_x)



	clf = RandomForestClassifier(n_estimators=1500, criterion='entropy', max_depth=15, min_samples_split = 2,min_samples_leaf=3, max_features='auto',

	 n_jobs=-1,random_state=343)

	param_grid = dict()



	grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', cv = 5).fit(x,train_y)

	#print(grid_search.best_estimator_)

	report(grid_search.cv_results_)

	svc = grid_search.best_estimator_.fit(x,train_y)

	###################

	grid_search.best_estimator_.score(x, train_y)

	scores = cross_val_score(svc, x, train_y, cv=5, scoring='accuracy')

	#print(scores.mean(), scores.min())

	#print(scores)

	x_t = g.predict_proba(test_x)



	y_pred = grid_search.best_estimator_.predict(x_t)

	end = time.time()

	print("Time taken for RandomForestClassifier with GMM = %.2f seconds"%(end-start))

	#print("Accuracy using RandomForestClassifier with GMM = %.3f"%(metrics.accuracy_score(y_pred,test_y)*100))

	return(metrics.accuracy_score(y_pred,test_y)*100)



def RandomForests(x,features):

	#RandomForests

	del features[-1]

	train, test = train_test_split(x, test_size = 0.2)

	train_x,test_x = train[features],test[features]

	train_y, test_y = train.diagnosis, test.diagnosis

	start = time.time()

	clf = RandomForestClassifier(n_estimators = 1000, max_features = 'auto', criterion = 'entropy',random_state = 30, min_samples_leaf = 5, n_jobs = 2)

	clf.fit(train_x,train_y)

	end = time.time()

	y_pred = clf.predict(test_x)

	acc_2 = metrics.accuracy_score(y_pred,test_y)

	print("\nTime taken for Random Forest Classifier = %.4f seconds"%(end-start))





	return acc_2*100



def KNN():

	#K-Nearest Neighbors

	pass



def LogReg(x, features):

	#Logistic Regression

	del features[-1]

	train, test = train_test_split(x,test_size = 0.2) # 5 fold CV

	train_x, test_x = train[features], test[features]

	train_y, test_y = train.diagnosis,test.diagnosis

	start = time.time()

	c_p = []

	acc_dict = {}

	t= 0.01

	training_accuracy = []

	testing_accuracy = []

	while t<=pow(10,4):

		c_p.append(t)

		t*=5

	for c in c_p:

		#print("Running Log Reg with %.6f"%(c))

		clf = LogisticRegression(C = c, penalty = 'l2')

		# training model using given data

		clf.fit(train_x,train_y)



		#training set accuracy

		y_pred_train = clf.predict(train_x)

		acc_train = metrics.accuracy_score(y_pred_train,train_y)

		training_accuracy.append(acc_train)



		# test set accuracy

		y_pred_test = clf.predict(test_x)

		acc_test = metrics.accuracy_score(y_pred_test,test_y)

		testing_accuracy.append(acc_test)

		acc_dict[c] = acc_test

		#print("Accuracy = ",acc_test)

	end = time.time()

	print("\nMaximum accuracy using Logistic Regression = %.4f with C = %.6f"%( max(acc_dict.items(),key = operator.itemgetter(1))[1]*100,

		max(acc_dict.items(), key = operator.itemgetter(1))[0]))

	print("Time taken for LogReg = %.2f seconds"%(end-start))

	#print(c_p) 

	plt.plot(c_p,training_accuracy, label = 'Training Accuracy')

	plt.plot(c_p,testing_accuracy, label = 'Testing Accuracy')

	plt.legend()

	plt.show()



def ADB():

	#adaBoost

	pass



def SVM(x,features,kernel = 'rbf'):

	del features[-1]

	#for pos,val in enumerate(features):	print(pos,val)

	train, test = train_test_split(x, test_size = 0.2)

	train_x,test_x = train[features],test[features]

	train_y, test_y = train.diagnosis, test.diagnosis

	#for pos,val in enumerate(train_x):	print(pos,val)

	#print(train_x.shape)

	#print(test_x.shape)

	kernel = str(kernel)

	start = time.time()

	c_param_range = []

	t=0.001

	while t<=pow(10,1):

		c_param_range.append(t)

		t*=2





	results_table_svm = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Accuracy'])

	results_table_svm['C_parameter'] = c_param_range

	j = 0

	

	acc_dict_svm={}

	for c_param in c_param_range:

		#print("Running SVM with  C = %f "%(c_param))

			# using bagging classifier with n-cores to speed up performance

		#clf = BaggingClassifier(SVC(C = c_param, kernel = kernel),n_jobs=-1)

# time taken with bagging classifier = 88 seconds

		clf = SVC(C = c_param, kernel = kernel, probability = True)

		clf.fit(train_x,train_y)



		y_pred = clf.predict(test_x)



		acc = metrics.accuracy_score(y_pred,test_y)





		acc_dict_svm[c_param]=acc

		results_table_svm.ix[j,'Accuracy'] = acc

		j += 1





	best_c_svm = results_table_svm.loc[results_table_svm['Accuracy'].idxmax()]['C_parameter']

	#print("Best Accuracy: %f with C= %.4f using %s kernel"%(max(acc_dict_svm.items(),key = operator.itemgetter(1))[1]*100,

	#	max(acc_dict_svm.items(),key = operator.itemgetter(1))[0],kernel))

	end = time.time()

	#print("Average time taken for SVM = %.4f seconds"%((end-start)/len(c_param_range)))

	return acc_dict_svm[best_c_svm]



def sigmoid(Z):

    """

    Implements the sigmoid activation in numpy

    

    Arguments:

    Z -- numpy array of any shape

    

    Returns:

    A -- output of sigmoid(z), same shape as Z

    cache -- returns Z as well, useful during backpropagation

    """

    

    A = 1/(1+np.exp(-Z))

    cache = Z

    

    return A, cache



def relu(Z):

    """

    Implement the RELU function.



    Arguments:

    Z -- Output of the linear layer, of any shape



    Returns:

    A -- Post-activation parameter, of the same shape as Z

    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently

    """

    

    A = np.maximum(0,Z)

    

    #assert(A.shape == Z.shape)

    

    cache = Z 

    return A, cache

def initialize_parameters(n_x, n_h, n_y):

    """

    Argument:

    n_x -- size of the input layer

    n_h -- size of the hidden layer

    n_y -- size of the output layer

    

    Returns:

    parameters -- python dictionary containing your parameters:

                    W1 -- weight matrix of shape (n_h, n_x)

                    b1 -- bias vector of shape (n_h, 1)

                    W2 -- weight matrix of shape (n_y, n_h)

                    b2 -- bias vector of shape (n_y, 1)

    """

    

    np.random.seed(13456)

    

    ### START CODE HERE ### (≈ 4 lines of code)

    W1 = np.random.randn(n_h,n_x) * 0.01 /np.sqrt(445)

    b1 = np.random.randn(n_h,1)*0.01

    W2 = np.random.randn(n_y,n_h) * 0.01 /np.sqrt(445)

    b2 = np.random.randn(n_y,1)*0.01 /np.sqrt(n_y)

    ### END CODE HERE ###np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01

    

    #assert(W1.shape == (n_h, n_x))

    #assert(b1.shape == (n_h, 1))

    #assert(W2.shape == (n_y, n_h))

    #assert(b2.shape == (n_y, 1))

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters



def initialize_parameters_deep(layer_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the dimensions of each layer in our network

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])

                    bl -- bias vector of shape (layer_dims[l], 1)

    """

    

    np.random.seed(3)

    parameters = {}

    L = len(layer_dims)            # number of layers in the network



    for l in range(1, L):

        ### START CODE HERE ### (≈ 2 lines of code)

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        ### END CODE HERE ###

        

        #assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))

        #assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))



        

    return parameters



def linear_forward(A, W, b):

    """

    Implement the linear part of a layer's forward propagation.



    Arguments:

    A -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)



    Returns:

    Z -- the input of the activation function, also called pre-activation parameter 

    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """

    Z = np.dot(W,A) + b

    cache = (A, W, b)

    

    return Z, cache



def linear_activation_forward(A_prev, W, b, activation):

    """

    Implement the forward propagation for the LINEAR->ACTIVATION layer



    Arguments:

    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)

    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"



    Returns:

    A -- the output of the activation function, also called the post-activation value 

    cache -- a python dictionary containing "linear_cache" and "activation_cache";

             stored for computing the backward pass efficiently

    """

    

    if activation == "sigmoid":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = sigmoid(Z)

    

    elif activation == "relu":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)



    return A, cache



def L_model_forward(X, parameters):

    """

    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    

    Arguments:

    X -- data, numpy array of shape (input size, number of examples)

    parameters -- output of initialize_parameters_deep()

    

    Returns:

    AL -- last post-activation value

    caches -- list of caches containing:

                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)

                the cache of linear_sigmoid_forward() (there is one, indexed L-1)

    """



    caches = []

    A = X

    L = len(parameters) // 2                  # number of layers in the neural network

    

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.

    for l in range(1, L):

        A_prev = A 

        ### START CODE HERE ### (≈ 2 lines of code)

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)],parameters['b' + str(l)],activation='relu')

        

        caches.append(cache)

        ### END CODE HERE ###

    

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    ### START CODE HERE ### (≈ 2 lines of code)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],parameters['b' + str(L)],activation='sigmoid')

    caches.append(cache)

    ### END CODE HERE ###

    

    #assert(AL.shape == (1,X.shape[1]))

            

    return AL, caches



def compute_cost(AL, Y):

    """

    Implement the cost function defined by equation (7).



    Arguments:

    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)

    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)



    Returns:

    cost -- cross-entropy cost

    """

    

    m = Y.shape[1]

    #print(AL)

    # Compute loss from aL and y.

    ### START CODE HERE ### (≈ 1 lines of code)

    cost = (-1 / m) * np.sum(np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))

    ### END CODE HERE ###

    

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    #assert(cost.shape == ())

    

    return cost



def linear_backward(dZ, cache):

    """

    Implement the linear portion of backward propagation for a single layer (layer l)



    Arguments:

    dZ -- Gradient of the cost with respect to the linear output (of current layer l)

    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer



    Returns:

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

    dW -- Gradient of the cost with respect to W (current layer l), same shape as W

    db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    A_prev, W, b = cache

    m = A_prev.shape[1]

    #print(b.shape)

    ### START CODE HERE ### (≈ 3 lines of code)

    dW = np.dot(dZ, cache[0].T) / m

    db = (np.sum(dZ,axis = 1, keepdims = True)) / m

    #print(db.shape)

    dA_prev = np.dot(cache[1].T,dZ)

    ### END CODE HERE ###

    

    #assert (dA_prev.shape == A_prev.shape)

    #assert (dW.shape == W.shape)

    #assert (db.shape == b.shape)

    

    return dA_prev, dW, db



def relu_backward(dA, cache):

    """

    Implement the backward propagation for a single RELU unit.



    Arguments:

    dA -- post-activation gradient, of any shape

    cache -- 'Z' where we store for computing backward propagation efficiently



    Returns:

    dZ -- Gradient of the cost with respect to Z

    """

    

    Z = cache

    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    

    # When z <= 0, you should set dz to 0 as well. 

    dZ[Z <= 0] = 0

    

    #assert (dZ.shape == Z.shape)

    

    return dZ



def sigmoid_backward(dA, cache):

    """

    Implement the backward propagation for a single SIGMOID unit.



    Arguments:

    dA -- post-activation gradient, of any shape

    cache -- 'Z' where we store for computing backward propagation efficiently



    Returns:

    dZ -- Gradient of the cost with respect to Z

    """

    

    Z = cache

    

    s = 1/(1+np.exp(-Z))

    dZ = dA * s * (1-s)

    

    #assert (dZ.shape == Z.shape)

    

    return dZ



def linear_activation_backward(dA, cache, activation):

    """

    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    

    Arguments:

    dA -- post-activation gradient for current layer l 

    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently

    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    

    Returns:

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

    dW -- Gradient of the cost with respect to W (current layer l), same shape as W

    db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    linear_cache, activation_cache = cache

    

    if activation == "relu":

        ### START CODE HERE ### (≈ 2 lines of code)

        dZ = relu_backward(dA, activation_cache)

        #dA_prev, dW, db = linear_backward(dZ,cache)

        ### END CODE HERE ###

        

    elif activation == "sigmoid":

        ### START CODE HERE ### (≈ 2 lines of code)

        dZ = sigmoid_backward(dA, activation_cache)

        #dA_prev, dW, db = linear_backward(dZ,cache)

        ### END CODE HERE ###

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db



def L_model_backward(AL, Y, caches):

    """

    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    

    Arguments:

    AL -- probability vector, output of the forward propagation (L_model_forward())

    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)

    caches -- list of caches containing:

                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)

                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    

    Returns:

    grads -- A dictionary with the gradients

             grads["dA" + str(l)] = ... 

             grads["dW" + str(l)] = ...

             grads["db" + str(l)] = ... 

    """

    grads = {}

    L = len(caches) # the number of layers

    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    

    # Initializing the backpropagation

    ### START CODE HERE ### (1 line of code)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    ### END CODE HERE ###

    

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]

    ### START CODE HERE ### (approx. 2 lines)

    current_cache = caches[-1]

    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,

                                                                                                  current_cache,

                                                                                                  "sigmoid")

    ### END CODE HERE ###

    

    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients.

        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        ### START CODE HERE ### (approx. 5 lines)

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA{}".format(l + 2)],

                                                                    current_cache,

                                                                    "relu")

        grads["dA" + str(l + 1)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

        ### END CODE HERE ###



    return grads



def update_parameters(parameters, grads, learning_rate):

    """

    Update parameters using gradient descent

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    grads -- python dictionary containing your gradients, output of L_model_backward

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

                  parameters["W" + str(l)] = ... 

                  parameters["b" + str(l)] = ...

    """

    

    L = len(parameters) // 2 # number of layers in the neural network



    # Update rule for each parameter. Use a for loop.

    ### START CODE HERE ### (≈ 3 lines of code)

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W{}".format(l + 1)] - learning_rate * grads["dW{}".format(l + 1)]

        parameters["b" + str(l+1)] = parameters["b{}".format(l + 1)] - learning_rate * grads["db{}".format(l + 1)]

    ### END CODE HERE ###

    return parameters



def predict(X, y, parameters):

    """

    This function is used to predict the results of a  L-layer neural network.

    

    Arguments:

    X -- data set of examples you would like to label

    parameters -- parameters of the trained model

    

    Returns:

    p -- predictions for the given dataset X

    """

    

    m = X.shape[1]

    n = len(parameters) // 2 # number of layers in the neural network

    p = np.zeros((1,m))

    

    # Forward propagation

    probas, caches = L_model_forward(X, parameters)



    

    # convert probas to 0/1 predictions

    for i in range(0, probas.shape[1]):

        if probas[0,i] > 0.5:

            p[0,i] = 1

        else:

            p[0,i] = 0

    

    #print results

    #print ("predictions: " + str(p))

    #print ("true labels: " + str(y))

    print("Test Accuracy: "  + str(np.sum((p == y)*100/m)))

        

    return np.sum((p == y)*100/m)



def two_layer_model(data, features,n_h, learning_rate = 0.015, num_iterations = 30000, print_cost=False):

    """

    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    

    Arguments:

    X -- input data, of shape (n_x, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- dimensions of the layers (n_x, n_h, n_y)

    num_iterations -- number of iterations of the optimization loop

    learning_rate -- learning rate of the gradient descent update rule

    print_cost -- If set to True, this will print the cost every 100 iterations 

    

    Returns:

    parameters -- a dictionary containing W1, W2, b1, and b2

    """

    start = time.time()

    del features[-1]

    train, test = train_test_split(data, test_size = 0.2)

    train_x,test_x = train[features],test[features]

    train_y, test_y = train.diagnosis.values.reshape((train.diagnosis.shape[0],1)),test.diagnosis.values.reshape((test.diagnosis.shape[0],1))

    train_x,test_x = train_x.T,test_x.T

    train_y = train_y.T

    test_y = test_y.T

    np.random.seed(1)

    grads = {}

    costs = []                              # to keep track of the cost

    m = train_x.shape[1]                           # number of examples

    n_x = train_x.shape[0]

    n_h = n_h

    n_y = train_y.shape[0]

    

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented

    ### START CODE HERE ### (≈ 1 line of code)

    parameters = initialize_parameters(n_x, n_h, n_y)

    ### END CODE HERE ###

    

    # Get W1, b1, W2 and b2 from the dictionary parameters.

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    

    # Loop (gradient descent)



    for i in range(0, num_iterations):



        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".

        ### START CODE HERE ### (≈ 2 lines of code)

        A1, cache1 = linear_activation_forward(train_x, W1, b1, 'relu')

        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        ### END CODE HERE ###

        

        # Compute cost

        ### START CODE HERE ### (≈ 1 line of code)

        cost = compute_cost(A2, train_y)

        ### END CODE HERE ###

        if cost <= 0.1:

        	break

        # Initializing backward propagation

        dA2 = - (np.divide(train_y, A2) - np.divide(1 - train_y, 1 - A2))

        

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".

        ### START CODE HERE ### (≈ 2 lines of code)

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')

        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        ### END CODE HERE ###

        

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2

        grads['dW1'] = dW1

        grads['db1'] = db1

        grads['dW2'] = dW2

        grads['db2'] = db2

        

        # Update parameters.

        ### START CODE HERE ### (approx. 1 line of code)

        parameters = update_parameters(parameters, grads, learning_rate)

        ### END CODE HERE ###



        # Retrieve W1, b1, W2, b2 from parameters

        W1 = parameters["W1"]

        b1 = parameters["b1"]

        W2 = parameters["W2"]

        b2 = parameters["b2"]

        

        # Print the cost every 100 training example

        if print_cost and i % 1000 == 0:

            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        if print_cost and i % 1000 == 0:

            costs.append(cost)

       



    

    p = np.zeros((1,m))

    

    # Forward propagation

    probas, caches = L_model_forward(train_x, parameters)



    

    # convert probas to 0/1 predictions

    for i in range(0, probas.shape[1]):

        if probas[0,i] > 0.5:

            p[0,i] = 1

        else:

            p[0,i] = 0

    #print(p.shape)

    #print('\n',test_y.shape)

    #print results

    #print ("predictions: " + str(p))

    #print ("true labels: " + str(y))

    print("Training Accuracy: "  + str(np.sum((p == train_y)*100)/455))

    a = predict(test_x,test_y,parameters)

    end = time.time()

    print("Time taken: %.4f"%(end-start))

    return a

    ###### plot the cost   #######

"""

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

"""

	



def L_layer_model(data, features, layers_dims, learning_rate = 0.015, num_iterations = 5000, print_cost=False):#lr was 0.009

    """

    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    

    Arguments:

    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).

    learning_rate -- learning rate of the gradient descent update rule

    num_iterations -- number of iterations of the optimization loop

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """

    start = time.time()

    del features[-1]

    train, test = train_test_split(data, test_size = 0.2)

    train_x,test_x = train[features],test[features]

    train_y, test_y = train.diagnosis.values.reshape((train.diagnosis.shape[0],1)),test.diagnosis.values.reshape((test.diagnosis.shape[0],1))

    train_x,test_x = train_x.T,test_x.T

    train_y = train_y.T

    test_y = test_y.T

    np.random.seed(11)

    costs = []                         # keep track of cost

    m = train_x.shape[1]

    # Parameters initialization.

    parameters = initialize_parameters_deep(layers_dims)

    

    # Loop (gradient descent)

    for i in range(0, num_iterations):



        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

        AL, caches = L_model_forward(train_x, parameters)



        # Compute cost.



        cost = compute_cost(AL, train_y)

        if cost < 0.1:

        	pass

    

        # Backward propagation.



        grads = L_model_backward(AL, train_y, caches)



 

        # Update parameters.



        parameters = update_parameters(parameters, grads, learning_rate)

              



        if print_cost and i % 1000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 1000 == 0:

            costs.append(cost)

        p = np.zeros((1,m))

    

    # Forward propagation

        probas, caches = L_model_forward(train_x, parameters)



    

    # convert probas to 0/1 predictions

    for i in range(0, probas.shape[1]):

        if probas[0,i] > 0.5:

        	p[0,i] = 1

        else:

            p[0,i] = 0



    print("Training Accuracy: "  + str(np.sum((p == train_y) * 100 )/455))

    predict(test_x,test_y,parameters)

    end = time.time()

    #print("Time taken: %.4f"%(end-start))

    ###### plot the cost   #######





    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters

def NeuralNetwork(x,features):

	# Neural Network with one hidden layer

	# Will try and work with more hidden layers

	# Optimization of hyperparameters for one hidden layer NN supported

	# WORK IN PROGRESS

	del features[-1]

	train, test = train_test_split(x, test_size = 0.2)

	train_x,test_x = train[features],test[features]

	train_y, test_y = train.diagnosis.values.reshape((train.diagnosis.shape[0],1)),test.diagnosis.values.reshape((test.diagnosis.shape[0],1))

	train_y = train_y.T

	test_y = test_y.T

	"""

	# Checking dimensions

	print(test_y, '\n',test.diagnosis)

	print("shape of train_x = ",train_x.shape)

	print("shape of test_x = ",test_x.shape)

	print("shape of train_y = ",train_y.shape)

	print("shape of test_y = ",test_y.shape)

	"""

	

	# DEFINING PARAMETERS 

	n_h = 5 # number of hidden units

	

	# layer 1 (input layer)



	w1 = np.random.randn(n_h,train_x.shape[1])*0.01/np.sqrt(train_x.shape[0])

	b1 = np.random.randn(n_h,1)*0.01

	w2 = np.random.randn(train_y.shape[0],n_h)*0.01/np.sqrt(train_y.shape[0])

	b2 = np.random.randn(train_y.shape[0],1)*0.01



	num_iters = 10

	learning_rate = 0.1

	m = (1/train_y.shape[1])

	for i in range(num_iters):

		#print(w1.shape,w2.shape,b1.shape,b2.shape) # checkind dimensions



		###### FORWARD PROPAGATION #######



		z1 = np.dot(w1,train_x.T) + b1 # linear transform

		A1 = np.maximum(z1,0.001*z1) # ReLu Activation or maybe we can use leaky ReLu 

		z2 = np.dot(w2,A1) + b2

		A2 = (1/(1+np.exp(-z2))) # sigmoid activation to get y hat 



		

		#print(np.log(abs(1-A2)))

		#cost = -(1/m)*np.sum(np.dot(train_y,np.log(A2).T)+np.dot(1-train_y,np.log(1-A2).T))

		cost = log_loss(train_y,A2) # this is throwing some error, 

									 # no patiance to workaround, will write own loss function

		#print(z2)

		#print("Iteration %i, cost %.4f"%(i,cost))



		



		###### 	BACKWARD PROPAGATION #######



		#dA2 = train_y/A2 + (1-train_y)/(1-A2)



		dA2 = - (np.divide(train_y, A2) - np.divide(1 - train_y, 1 - A2))

		#print(dA2)

		dz2 = (dA2*(z2*(1-z2))) # deriative of sigmoid 



		dw2 = (np.dot(dz2,A1.T))/m



		# print(w2.shape,dw2.shape) # checking fwd bwd param dims



		db2 = np.sum(dz2, axis = 1, keepdims = True) / m



		dA1 = np.dot(w2.T,dz2)



		

		dz1 = (dA1 ) #relu derivative (modified, ReLu is not differentiable at 0 )



		dw1 = np.dot(dz1,train_x) / m



		db1 = np.sum(dz1, axis = 1, keepdims = True) / m

		#print(w1.shape,dw1.shape, b1.shape, db1.shape) # checking fwd bwd param dims

		

		##### UPDATING PARAMETERS ######



		w1 = w1 - learning_rate * dw1

		b1 = b1 - learning_rate * db1

		w2 = w2 - learning_rate * dw2

		b2 = b2 - learning_rate * db2

		cost2 = -(1/m)*np.sum(np.dot(train_y,np.log(A2).T))

		#print("Iteration %i, cost2 %.4f"%(i,cost2))



def call_function(x):

	data, features =load('../input/data.csv')

	""" FUNCTION CALLS """

	function = str(x)

	

	if function == "LR":

		LogReg(data,features)

	elif function == "NN":

		#hidden_unit_size = [5 , 10 , 15 , 20 , 25 , 50 , 100 , 200 , 500, 1000]

		hidden_unit_size = [5 , 10 , 15 , 20 , 25 , 26, 27, 28, 29, 30]

		arr_res=[]

		parameters = two_layer_model(data, features,100, learning_rate = 0.0050, num_iterations = 30000, print_cost = False)

		"""start = time.time()

		for i in range(1,50):

			data, features =load('../input/data.csv')

			print('\n',i)

			parameters = two_layer_model(data, features,i, learning_rate = 0.0050, num_iterations = 30000, print_cost = False)

			arr_res.append(parameters)

		end = time.time()



		print("Time taken = %.4f"%(end-start))

		plt.plot([i+1 for i in range(1,50)],arr_res)

		plt.ylabel("Test Accuracy")

		plt.xlabel("Hidden Unit Size")

		plt.show()

		"""

	elif function == "GMM":	

		print("\nAccuracy with Random Forest Classifier using GMM = %.4f"%(gmm(data,features)))

	elif function == "RFC":	

		print("\nAccuracy with Random Forest Classifier = ",RandomForests(data , features))

	elif function == "SVM":	

		kernels = ['rbf','linear','poly','sigmoid']

		for i in kernels:	

			start = time.time()

			print("\nAccuracy using SVM with ",i,":",SVM(data,features,i)*100)

			end = time.time()

			print("Time taken for ",i," = %.4f"%(end-start))

	elif function == "DNN":

		layers_dims = [19, 48, 48, 48, 48, 24, 12, 1]

		L_layer_model(data, features, layers_dims, learning_rate = 0.05, num_iterations = 15000, print_cost = True)		







if __name__ == '__main__':

# LR = LOGISTIC REGRESSION

# NN = NEURAL NETWORK WITH ONE HIDDEN LAYER

# SVM = SVM WITH FOUR KERNELS: RBF POLY SIGMOID AND LINEAR

# GMM = RANDOM FOREST WITH GAUSSIAN MISTURE

# RFC = RANDOM FOREST CLASSIFICATION

# DNN = DEEP NEURAL NETWORK WITH EXPLICIT (L -LAYER) HIDDEN LAYERS

	#call_function("LR")

	call_function("NN")
if __name__ == '__main__':

# LR = LOGISTIC REGRESSION

# NN = NEURAL NETWORK WITH ONE HIDDEN LAYER

# SVM = SVM WITH FOUR KERNELS: RBF POLY SIGMOID AND LINEAR

# GMM = RANDOM FOREST WITH GAUSSIAN MISTURE

# RFC = RANDOM FOREST CLASSIFICATION

# DNN = DEEP NEURAL NETWORK WITH EXPLICIT (L -LAYER) HIDDEN LAYERS

	#call_function("LR")

	#call_function("NN")

	#call_function("SVM")

	#call_function("GMM")

	call_function("RFC")
if __name__ == '__main__':

# LR = LOGISTIC REGRESSION

# NN = NEURAL NETWORK WITH ONE HIDDEN LAYER

# SVM = SVM WITH FOUR KERNELS: RBF POLY SIGMOID AND LINEAR

# GMM = RANDOM FOREST WITH GAUSSIAN MISTURE

# RFC = RANDOM FOREST CLASSIFICATION

# DNN = DEEP NEURAL NETWORK WITH EXPLICIT (L -LAYER) HIDDEN LAYERS

	#call_function("LR")

	#call_function("NN")

	#call_function("SVM")

	#call_function("GMM")

	#call_function("RFC")

	TIME = time.time()

	call_function("DNN")

	END = time.time()

	print("Time taken for DNN = %.4f seconds "%(END-TIME))
if __name__ == '__main__':

# LR = LOGISTIC REGRESSION

# NN = NEURAL NETWORK WITH ONE HIDDEN LAYER

# SVM = SVM WITH FOUR KERNELS: RBF POLY SIGMOID AND LINEAR

# GMM = RANDOM FOREST WITH GAUSSIAN MISTURE

# RFC = RANDOM FOREST CLASSIFICATION

# DNN = DEEP NEURAL NETWORK WITH EXPLICIT (L -LAYER) HIDDEN LAYERS

	#call_function("LR")

	#call_function("NN")

	#call_function("SVM")

	#call_function("GMM")

	#call_function("RFC")

	#call_function("DNN")

	TIME = time.time()

	call_function("SVM")

	END = time.time()

	print("Time taken for SVM = %.4f seconds "%(END-TIME))
if __name__ == '__main__':

# LR = LOGISTIC REGRESSION

# NN = NEURAL NETWORK WITH ONE HIDDEN LAYER

# SVM = SVM WITH FOUR KERNELS: RBF POLY SIGMOID AND LINEAR

# GMM = RANDOM FOREST WITH GAUSSIAN MISTURE

# RFC = RANDOM FOREST CLASSIFICATION

# DNN = DEEP NEURAL NETWORK WITH EXPLICIT (L -LAYER) HIDDEN LAYERS

	#call_function("LR")

	#call_function("NN")

	#call_function("SVM")

	#call_function("GMM")

	#call_function("RFC")

	#call_function("DNN")

	TIME = time.time()

	call_function("SVM")

	END = time.time()

	print("Time taken for SVM = %.4f seconds "%(END-TIME))