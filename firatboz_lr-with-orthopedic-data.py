# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pylab import rcParams



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read dataframe

data = pd.read_csv('/kaggle/input/column_2C_weka.csv')

#Let's see the shape of our data

data.shape
data.head()
data.tail()
data.info()
data.describe()
data.rename(columns={"class": "class1"},inplace=True)
data.class1.value_counts()
color=['red' if _=='Abnormal' else 'green' for _ in data.class1];

pd.plotting.scatter_matrix(data, alpha = 0.5,figsize = (20, 20),color=color,marker='*',s=200);
#update class as 0's and 1's ==> 0 if normal 1 otherwise

data.class1 = [1 if _=='Abnormal' else 0 for _ in data.class1]
#Drop dependent variable from x values

x_values=data.drop('class1',axis=1)

#Seperate independent values from features

y_values=data.class1
#Normalize the data between 0 and 1 

x_values=(x_values-x_values.min())/(x_values.max()-x_values.min())
x_values.info()
y_values.value_counts()
x_values.shape
y_values.shape
# Create train and test samples 

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)
x_train.info()
y_train.shape
y_train.value_counts()
x_test.info()
y_test.shape
y_test.value_counts()
# Define the sigmoid function

def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head
# initialize weight and Bias



def initialize_weight_bias(dimension):

    weight=np.full((dimension,1),0.01)

    bias = 0.0

    return weight, bias
# Define Cost Function



def cost_function(y_train,y_head):



    loss_function=((1-y_train)*np.log(1-y_head)+y_train*np.log(y_head))*-1

    cost=np.sum(loss_function)/y_train.shape[0]

    

    return cost
#Define derivative of cost function with respect to weight and biss

def gradients(x_train,y_train,y_head_train):

    weight_derivative=np.dot(x_train.T,(y_head_train-y_train.values.reshape(-1,1)))/x_train.shape[0]

    bias_derivative=np.sum((y_head_train-y_train.values.reshape(-1,1)))/y_train.shape[0]

    return weight_derivative, bias_derivative
#Define forward and backward propagation method. 



def forward_and_backward_propagation(x_train,y_train,learning_rate,number_of_iterations): 

    cost_list=[]

    cost_list_by10=[]

    dimension=x_train.shape[1] # just for plotting purposes to decrease complexity of the plot

    weight,bias=initialize_weight_bias(dimension) # initialize weight and bias only once. 

    

    

    # performs forward and backward propagation n times (n defined by user as number of iterations. )

  

    for i in range(number_of_iterations):

        z=np.dot(x_train,weight)+bias

        y_head_train=sigmoid(z)

        cost=cost_function(y_train.values.reshape(-1,1),y_head_train) # calls cost function to calculate the cost/penalty

        weight_derivative, bias_derivative=gradients(x_train,y_train,y_head_train) # calls gradient function to calculate the derivative of weight & bias

       

        weight=weight-weight_derivative*learning_rate # update weight

        bias=bias-bias_derivative*learning_rate #update bias

        cost_list.append(cost) # append the new cost into the cost_list   

        if i%10==0:

            cost_list_by10.append(cost)

    



    # Draw line chart

    plt.plot(cost_list_by10)

    rcParams['figure.figsize'] = 6,6 # set the size of the plot

    plt.xlabel('iteration/5: actual iteration is 10 times what is denoted ') 

    plt.ylabel('cost')

    plt.title('cost w.r.t iteration')

          

    return weight,bias
# This method acts like a main engine: logistic regression is performed by calling this method.

# Associated methods are called automatically. 



def main_engine(x_train,y_train,x_test,y_test, learning_rate,number_of_iterations):

    prediction_accuracy_train,parameters_train=sub_engine(x_train,y_train, learning_rate,number_of_iterations)

    

    z_test=np.dot(x_test,parameters_train["weight_learnt"])+ parameters_train["bias_learnt"]   #calculate z value ==> sum(wi * xi) + bias

    y_head_test=sigmoid(z_test) # z values are converted into values between 0 and 1 to represent probability. 

    y_head_test=[1 if each_test>0.5 else 0 for each_test in y_head_test] #if y_head value is greater than 0.5 return 1 otherwise 0. 

    prediction_accuracy_test=(1-np.mean(np.abs(y_test-y_head_test)))*100 #calculate prediction accuracy (%)

    return prediction_accuracy_train,prediction_accuracy_test
#Method is used for hyper parameter fine-tuning



def sub_engine(x_train,y_train, learning_rate,number_of_iterations):

    weight,bias=forward_and_backward_propagation(x_train,y_train,learning_rate,number_of_iterations)

    parameters = {"weight_learnt": weight,"bias_learnt": bias}

    

    z_train=np.dot(x_train,weight)+bias #calculate z value ==> sum(wi * xi) + bias

    y_head_train=sigmoid(z_train) # z values are converted into values between 0 and 1 to represent probability. 

    y_head_train=[1 if each>0.5 else 0 for each in y_head_train] #if y_head value is greater than 0.5 return 1 otherwise 0. 

    prediction_accuracy=(1-np.mean(np.abs(y_train-y_head_train)))*100 #calculate prediction accuracy (%)

    return prediction_accuracy,parameters
prediction_accuracy_train,prediction_accuracy_test=main_engine(x_train,y_train,x_test,y_test, 8,550)

print("prediction_accuracy_train is:",prediction_accuracy_train)

print("prediction_accuracy_test is:", prediction_accuracy_test)
#Determine the best performing hyper-parameters

#Method comes up with two hyper parameter suggestions, namely number of iterations & 

#learning rate to maximize prediction accuracy

prediction_list=[]

iteration_list=[]

learning_rate_list=[]



for iteration in range(100,1000,25):

    for learn_rate in range(1,10,1):

        prediction_accuracy_check,parameters_check=sub_engine(x_train,y_train, learn_rate,iteration)

        

        prediction_list.append(prediction_accuracy_check)

        iteration_list.append(iteration)

        learning_rate_list.append(learn_rate)

        

learning_rate_list=pd.DataFrame(learning_rate_list)

prediction_list=pd.DataFrame(prediction_list)

iteration_list=pd.DataFrame(iteration_list)

hyper_parameters_df=pd.concat([iteration_list, learning_rate_list,prediction_list], axis=1, sort=False) 

hyper_parameters_df.columns = ['iteration_no', 'learning_rate','prediction_acc']

print(f'Max prediction accuracy can be reached with the following hyper parameters based on the train data:\n when random state setting is 42\n\n'

      ,hyper_parameters_df.loc[hyper_parameters_df.prediction_acc.idxmax()])

      

from sklearn.linear_model import LogisticRegression

log_res=LogisticRegression(solver='liblinear')

log_res.fit(x_train,y_train)

print('prediction accuracy of Logistic Regression model w/ sci-kit learn is',log_res.score(x_test, y_test)*100)