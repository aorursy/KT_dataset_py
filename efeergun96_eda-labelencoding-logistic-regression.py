import numpy as np # linear algebra 
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/mushrooms.csv")
df.head(10)   # as we can see whole data structured by letter instead of numbers.
df.info()   # we can see that they are all objects. Which is Letters in our case
df.describe()   # as we see, there are some columns with 12 unique rows while some others has only 2 unique rows.
# let's do Label Encoding and do EDA again
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# applying label encoder to whole dataset...
df = df.apply(label_encoder.fit_transform)

# checking the result
df.head(10)                # which seems great.
df.info()  # they are all int64 now and there is no null value which is very good.
y = df["class"].values   # our labels.. okay to eat or poison.

df.drop(["class"],axis=1,inplace=True)  # dropping the lables from the data

x_data = df  # our features..
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x.drop(["veil-color"],axis=1,inplace=True)
x.drop(["veil-type"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   # 20% would be enough
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
def init_weight_and_bias(dimension):
    w = np.full((dimension,1),0.01)          # just creating a dimension sized vector filled with our weight value (0.01)
    b = 0.0                                  # smallest float value is setted to the bias (0.0)
   # print("w::",w)
   # print("b::",b)
    return w,b;

def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))               # implementing the sigmoid function
   # print("y_head::",y_head)
    return y_head;

def forward_and_backward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b             # first phase of actual Computation of the Logistic Regression:: multiplying the weights with corresponding values then adding bias..
    y_head = sigmoid(z)                     # second phase:: Applying the sigmoid on the result of first phase to get result between 0 and 1.
    
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)        # calculating loss and cost is key to optimize since they are the value of fail 
  #  print("loss::",loss)
    cost = (np.sum(loss)) / x_train.shape[1]
  #  print("forw_bck::",cost)
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/ x_train.shape[1]     # applying gradient descent
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients = {"deriv_weight":derivative_weight,"deriv_bias":derivative_bias}  # putting all in dictionary
    
    return cost, gradients;
def update(w,b,x_train,y_train,learning_rate,num_of_iter):
    cost_list = []   # empty arrays to store costs
    index = []
    
    for i in range(num_of_iter):
        cost, gradients = forward_and_backward_propagation(w,b,x_train,y_train)  # do the training as much as iteration given
      #  print("update:: ",cost)
        cost_list.append(cost)      # insert the calculated cost on array
        index.append(i)
        
        w = w - learning_rate*gradients["deriv_weight"]     # set new weights and bias for next iteration
        b = b - learning_rate*gradients["deriv_bias"]
        
    parameters = {"weight":w,"bias":b}    # save all the weights and biases on a dictionary
    plt.plot(index,cost_list)            # draw the plot to visualize (optional)
    plt.show()
    
    return parameters, gradients, cost_list;


def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test) + b)           # do the first and second phase of Computation and store in array z
    y_prediction = np.zeros((1,x_test.shape[1]))  # create empty array to fill by results of z
   # print("y_pred:::", np.zeros((1,x_test.shape[1])))
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0;
        else:
            y_prediction[0,i] = 1;
    
    return y_prediction;    
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_of_iter):
    
    dimension = x_train.shape[0]
    w,b = init_weight_and_bias(dimension)
    
    parameters, gradients, col_list = update(w,b,x_train,y_train,learning_rate,num_of_iter)
    
    y_pred_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_pred_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("train accuracy: {} %".format(100-np.mean(np.abs(y_pred_train-y_train))*100))
    print("test accuracy: {} %".format(100-np.mean(np.abs(y_pred_test-y_test))*100))
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=1,num_of_iter=250)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(x_train.T,y_train.T)

y_head = lr_model.predict(x_test.T)
print("test accuracy: ", lr_model.score(x_test.T,y_test.T))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_head)
import seaborn as sns

plt.figure(figsize=(16,10))
sns.heatmap(cm,annot=True,fmt='.0f')
plt.show()
# I am currently trying to learn and improve my Machine Learning skills.
# Your Comments,Advices and Votes are important for me. Best Regards, Efe.