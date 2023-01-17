import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/train.csv', header=0)

train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



print(train.head())

test = pd.read_csv('../input/test.csv', header=0)

test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#test.head()

print(train.describe())

print(train.info())



dead=train[train['Survived']==0]

alive=train[train['Survived']==1]

surv_col = "green"

nosurv_col = "red"



sns.heatmap(train.drop('PassengerId',axis=1).corr(),annot=True)

plt.plot()

cols = ['Survived','Pclass','Age','SibSp','Parch','Fare','Gender']

sns.pairplot(train.dropna(),hue='Survived',vars=cols,palette=[nosurv_col,surv_col],size=2)



plt.figure(figsize=[12,10])

plt.subplot(331)

sns.distplot(alive['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)

sns.distplot(dead['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,

            axlabel='Age')

plt.subplot(332)

sns.barplot('Sex', 'Survived', data=train)

plt.subplot(333)

sns.barplot('Pclass', 'Survived', data=train)

plt.subplot(334)

sns.barplot('Embarked', 'Survived', data=train)

plt.subplot(335)

sns.barplot('SibSp', 'Survived', data=train)

plt.subplot(336)

sns.barplot('Parch', 'Survived', data=train)

plt.subplot(337)

sns.distplot(np.log10(alive['Fare'].dropna().values+1), kde=False, color=surv_col)

sns.distplot(np.log10(dead['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')

plt.plot()
# I did not write this code



nn_output_dim = 2  # output layer dimensionality



# Helper function to evaluate the total loss on the dataset

def calculate_loss(model):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions

    z1 = X.dot(W1) + b1

    a1 = np.tanh(z1)

    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss

    corect_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(corect_logprobs)

    # Add regulatization term to loss (optional)

    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1./num_examples * data_loss



# Helper function to predict an output (0 or 1)

def predict(model, x):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation

    z1 = x.dot(W1) + b1

    a1 = np.tanh(z1)

    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)



# This function learns parameters for the neural network and returns the model.

# - nn_hdim: Number of nodes in the hidden layer

# - num_passes: Number of passes through the training data for gradient descent

# - print_loss: If True, print the loss every 1000 iterations

def build_model(nn_hdim, num_passes=20000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these.

    np.random.seed(0)

    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) # 2 vectoren van lengte 3: 1 vector  per node, 3 weights voor  mapping naar 3 nodes in hidden layer

    b1 = np.zeros((1, nn_hdim))

    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)

    b2 = np.zeros((1, nn_output_dim))



    # This is what we return at the end

    model = {}

    #plt.axis([-1.8, 2.5, -1.3, 1.5])

    #plt.ion()



    # Gradient descent. For each batch...

    for i in range(0, num_passes):



        # Forward propagation



        z1 = X.dot(W1) + b1

        a1 = np.tanh(z1)



        z2 = a1.dot(W2) + b2

        exp_scores = np.exp(z2)

        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)



        # Backpropagation

        delta3 = probs



        delta3[range(num_examples), y] -= 1

        dW2 = (a1.T).dot(delta3)

        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))

        dW1 = np.dot(X.T, delta2)

        db1 = np.sum(delta2, axis=0)



        # Add regularization terms (b1 and b2 don't have regularization terms)

        dW2 += reg_lambda * W2

        dW1 += reg_lambda * W1



        # Gradient descent parameter update

        W1 += -epsilon * dW1

        b1 += -epsilon * db1

        W2 += -epsilon * dW2

        b2 += -epsilon * db2



        # Assign new parameters to the model

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}



        # Optionally print the loss.

        # This is expensive because it uses the whole dataset, so we don't want to do it too often.

        if print_loss and i % 2000 == 0:

            print("Loss after iteration %i: %f" % (i, calculate_loss(model)))

            #plot_decision_boundary(lambda x: predict(model, x))

            #plt.pause(0.001)

            # plt.show()

    #plot_decision_boundary(lambda x: predict(model, x))

    return model



def plot_decision_boundary(pred_func):

    # Set min and max values and give it some padding

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# Only train on Sex and Age

# Prepare output

train['Age'][np.isnan(train['Age'])] = np.nanmedian(train.Age)

train['Fare_s']=train['Fare']/max(train['Fare'])

train['Age_s']=train['Age']/max(train['Age'])

train['Pclass_s']=train['Pclass']/max(train['Pclass'])

train['SibSp_s']=train['SibSp']/max(train['SibSp'])



validate = train[int(0.8*len(train)):]

train_nn = train[:int(0.8*len(train))]

print(len(train_nn))

print(len(validate))



X = np.array([[a,b,c,d,e] for a,b,c,d,e in zip(train_nn.Fare_s,train_nn.Age_s,train_nn.Gender,train_nn.Pclass_s,train_nn.SibSp_s)])

validate_params = np.array([[a,b,c,d,e] for a,b,c,d,e in zip(validate.Fare_s,validate.Age_s,validate.Gender,validate.Pclass_s,validate.SibSp_s)])

y = train_nn.Survived.values



nn_input_dim = X.shape[1]  # input layer dimensionality

num_examples = len(X)  # training set size

# Gradient descent parameters (I picked these by hand)

epsilon = 0.005  # learning rate for gradient descent

reg_lambda = 0.1  # regularization strength

nr_hiddenneurons = 5

model = build_model(nr_hiddenneurons, num_passes=20000,print_loss=True)
validate_outcomes = [predict(model,x) for x in validate_params]



nr_test = len(validate_outcomes)



print(sum(validate_outcomes)[0],nr_test)

nr_incorrect = sum([abs(a-b) for a,b in zip(validate.Survived,validate_outcomes)]) 



print((nr_test-nr_incorrect)/nr_test)

#print(np.array([[a,b[0]] for a,b in zip(validate.Survived,validate_outcomes)]))
test['Age'][np.isnan(test['Age'])] = np.nanmedian(train.Age)

test['Fare_s']=test['Fare']/max(train['Fare'])

test['Age_s']=test['Age']/max(train['Age'])

test['Pclass_s']=test['Pclass']/max(train['Pclass'])

test['SibSp_s']=test['SibSp']/max(train['SibSp'])



test_params = np.array([[a,b,c,d,e] for a,b,c,d,e in zip(test.Fare_s,test.Age_s,test.Gender,test.Pclass_s,test.SibSp_s)])



test_outcomes = [predict(model,x)[0] for x in test_params]



submit = pd.DataFrame({'PassengerId' : test.loc[:,'PassengerId'],

                       'Survived':test_outcomes})

print(submit.head())

submit.to_csv("../working/submit.csv", index=False)
