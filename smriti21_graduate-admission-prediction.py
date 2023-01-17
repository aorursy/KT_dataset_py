

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
admission_data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
admission_data[:10]
correlation_matrix = admission_data.corr(method='pearson')
print(correlation_matrix)
plt.scatter(admission_data['GRE Score'],admission_data['Chance of Admit '])
plt.title('Chance of admit with respect to GRE Score')
plt.xlabel('GRE Score')
plt.ylabel('Chance Of Admit')
plt.scatter(admission_data['TOEFL Score'],admission_data['Chance of Admit '])
plt.title('Chance of admit with respect to TOEFL Score')
plt.xlabel('TOEFL Score')
plt.ylabel('Chance Of Admit')
plt.scatter(admission_data['University Rating'],admission_data['Chance of Admit '])
plt.title('Chance of admit with respect to University Rating')
plt.xlabel('University Rating')
plt.ylabel('Chance Of Admit')
plt.scatter(admission_data['SOP'],admission_data['Chance of Admit '])
plt.scatter(admission_data['LOR '],admission_data['Chance of Admit '])
plt.scatter(admission_data['CGPA'],admission_data['Chance of Admit '])
plt.scatter(admission_data['Research'],admission_data['Chance of Admit '],c='r')
admission_data = pd.get_dummies(admission_data,columns=['University Rating'])
admission_data.head()
processed_admisssion_data = admission_data[:]
# making a copy of data
processed_admisssion_data['GRE Score'] /= 340
processed_admisssion_data['TOEFL Score'] /= 120
processed_admisssion_data['SOP'] /= 5.0
processed_admisssion_data['LOR '] /= 5.0
processed_admisssion_data['CGPA'] /= 10.0
processed_admisssion_data.head()


sample = np.random.choice(processed_admisssion_data.index,size=int(len(processed_admisssion_data)*0.9), replace=False)
train_data , test_data = processed_admisssion_data.iloc[sample],processed_admisssion_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:3])
print(test_data[:3])
features = train_data.drop(columns=['Chance of Admit ','Serial No.'])
targets = train_data['Chance of Admit ']
features_test = test_data.drop(columns=['Chance of Admit ','Serial No.'])
target_test = test_data['Chance of Admit ']
def sigmoid(x):
    return 1/(1+np.exp(-x))
def error_formula(y,output):
    return -y*np.log(output)-(1-y)*np.log(1-output)
def error_term_formula(x, y, out):
    return (y-out)*x


epochs = 1000
learn_rate = 0.5

def train_nn(features,targets,epochs,learn_rate):
    # debugging is easier
    np.random.seed(22)
    n_records , n_features = features.shape
    last_loss = None
    
    # generate weights
    weights = np.random.normal(size=n_features)
    
    #epochs is the number of times an algo runs
    for e in range(epochs):
        for x,y in zip(features.values,targets):
            out = sigmoid(np.dot(x,weights))
            error = error_formula(y,out)
            error_term = error_term_formula(x,y,out)
            weights += learn_rate*error_term /n_records
   
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learn_rate)
        
test_out = sigmoid(np.dot(features_test,weights))
target_test = np.array(target_test)
print(test_out)
print(target_test)
mean_squared_error = np.mean((target_test - test_out)**2)
mean_squared_error
def sigmoid(x):
    return 1/(1+np.exp(-x))
def error_term(y,output):
    return y-output
def error_term_formula(y,output):
    return (y-output)*output*(1-output)

epochs = 2000
learnrate = 0.5

def train_nn(features,targets,epochs,learn_rate):
    # debugging is easier
    np.random.seed(22)
    n_records , n_features = features.shape
    last_loss = None
    
    # generate weights
    weights = np.random.normal(size=n_features)
    
    #epochs is the number of times an algo runs
    for e in range(epochs):
        for x,y in zip(features.values,targets):
            out = sigmoid(np.dot(x,weights))
            error = error_term(y,out)
            error_t = error_term_formula(y,out)
            weights += learnrate*error_t*x
   
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learn_rate)
        
predicted_output = sigmoid(np.dot(features_test,weights))
test = np.array(target_test)
print(predicted_output)
print(test)
mean_squared_error = np.mean((test - predicted_output)**2)
mean_squared_error
figure,axis = plt.subplots()
axis.plot(predicted_output,color='red')
axis.plot(test,color='blue')
axis.set_ylabel('predicted output v/s test output')
