import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/titanic/train.csv")
df.head()
# lets take out first the label

train_y = df['Survived']

train_y.head()
# function to filter the age, sex, fare pclass, sibsp, parch columns

def get_data(data):

    # take only this specific column

    data = data[['Age', 'Sex', 'Fare', 'Pclass', 'SibSp', 'Parch']]

    

    # replace male by 1, female by 0

    data.replace({ 'male' : 1, 'female' : 0 }, inplace=True)

    

    # replace null/nan data by the mean (age and fare columns)

    data['Fare'].fillna(int(data['Fare'].mean()), inplace=True)

    data['Age'].fillna(int(data['Age'].mean()), inplace=True)

    

    # transform into a numpy array

    data = data.to_numpy().astype(float)

    

    # normalize (make sure the data is between -1 and 1)

    for i in range(data.shape[1]):

        data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()

    

    return data
train_x = get_data(df)

print(train_x)
print(train_x.shape)
# same for the labels (contains 0 - 1 if the victim survived or not)

print(train_y.shape)
# the activation function and derivative of the action function

def sigmoid(x):

    return 1/(1+np.exp(-x))



def dsigmoid(x):

    return sigmoid(x) * (1 - sigmoid(x))
# the loss function and its derivative

def loss_fn(y, y_hat):

    return 1/2 * (y - y_hat) ** 2



def dloss_fn(y, y_hat):

    return (y - y_hat)
# number of rows

instances = train_x.shape[0]



# number oof columns

attributes = train_x.shape[1]



# number of hidden node for first layer 

hidden_nodes = 8



# number of hidden node for second layer

hidden_nodes_two = 4



# number of output labels 

output_labels = 1
# Inititate the weights/biases

w1 = np.random.rand(attributes,hidden_nodes)

b1 = np.random.randn(1, hidden_nodes)



w2 = np.random.rand(hidden_nodes,hidden_nodes_two)

b2 = np.random.randn(1, hidden_nodes_two)



w3 = np.random.rand(hidden_nodes_two, output_labels)

b3 = np.random.randn(1, output_labels)



theta = w1, w2, w3, b1, b2, b3
# Neural Network Forward

def forward(x, theta):

    w1, w2, w3, b1, b2, b3 = theta

    

    k = np.dot(x, w1) + b1

    l = sigmoid(k)

    

    m = np.dot(l, w2) + b2

    n = sigmoid(m)

    

    o = np.dot(n, w3) + b3

    p = sigmoid(o)

    

    return k, l, m, n, o, p
# Neural Network Backward

def backward(x, y, sigma, theta):

    k, l, m, n, o, p = sigma

    w1, w2, w3, b1, b2, b3 = theta

    

    # db3 = dloss * dsigm(o) * 1

    # dw3 = dloss * dsigm(o) * n

    

    # db2 = dloss * dsigm(o) * w3 * dsigm(m) * 1

    # dw2 = dloss * dsigm(o) * w3 * dsigm(m) * l

    

    # db1 = dloss * dsigm(o) * w3 * dsigm(m) * w2 * dsigm(k) 

    # dw1 = dloss * dsigm(o) * w3 * dsigm(m) * w2 * dsigm(k) * x

    

    dloss = dloss_fn(p, y)

    dsigm_p = dsigmoid(o)

    dsigm_n = dsigmoid(m)

    dsigm_l = dsigmoid(k)

    

    db3 = dloss * dsigm_p

    dw3 = np.dot(n.T, db3)

    

    db2 = np.dot(db3, w3.T) * dsigm_n

    dw2 = np.dot(l.T, db2)

    

    db1 = np.dot(db2, w2.T) * dsigm_l

    dw1 = np.dot(x, db1)

    

    return dw1, dw2, dw3, db1, db2, db3
# use the avg of the gradients for the derivative of each bias

def avg_bias(grads):

    dw1, dw2, dw3, db1, db2, db3 = grads

    db1 = db1.mean(axis=0)

    db2 = db2.mean(axis=0)

    db3 = db3.mean(axis=0)

    return dw1, dw2, dw3, db1, db2, db3
# Use the SGD in order to optimize the weights and biases

def optimize(theta, grads, lr=0.001):

    dw1, dw2, dw3, db1, db2, db3 = grads

    w1, w2, w3, b1, b2, b3 = theta

    

    w1 -= dw1 * lr

    w2 -= dw2 * lr

    w3 -= dw3 * lr

    b1 -= db1 * lr

    b2 -= db2 * lr

    b3 -= db3 * lr

    

    return w1, w2, w3, b1, b2, b3
# return 1 if the prediction is higher than 0.5

# return 0 if not

def predict(x, theta):

    predict = forward(x, theta)[-1]

    return np.where(predict > 0.5, 1, 0)
# time to train our model

for epoch in range(1000):

    

    for i in range(len(train_x)):

        sigma = forward(train_x[i], theta)

        grads = backward(train_x[i].reshape(6,1), train_y[i], sigma, theta)

        theta = optimize(theta, avg_bias(grads))

    

    if(epoch % 100 == 0):

        print(loss_fn(sigma[-1], train_y[i]).mean())
test_df = pd.read_csv("../input/titanic/test.csv")

test_x = get_data(test_df)
# Get test data predictions

test_preds = predict(test_x, theta)
# Add passengers ids to the test predictions

passenger_ids = test_df['PassengerId'].to_numpy()
# combine passenger ids with the predictions

final_result = np.array(list(map(list, zip(passenger_ids, test_preds))))
# arraay final_result to dataframe

df_final = pd.DataFrame(data=final_result, columns=["PassengerId", "Survived"])



# save the result

df_final.to_csv('submission.csv', index=False)