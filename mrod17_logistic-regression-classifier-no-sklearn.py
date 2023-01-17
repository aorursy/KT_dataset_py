import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

#train_data.head(10)
#Looking at the numeric variables

train_data.describe()
train_data["Name"] = train_data["Name"].str.split(',').str[1]

train_data["Name"] = train_data["Name"].str.split('.').str[0]

train_data["Name"] = train_data["Name"].str.strip()
x = train_data.groupby('Name').agg(['count']).index.get_level_values('Name')

x
train_data["Age"] = train_data.groupby("Name").transform(lambda x: x.fillna(x.mean()))['Age']

#changing sex to be 0 or 1 for female & male

train_data['Sex'].replace({'female':0,'male':1},inplace=True)

train_data.head()
train_data_log = train_data.iloc[:,[False,False,True, False,True,True,True,True,False,True,False,False]]

normalized_data_train=(train_data_log-train_data_log.min())/(train_data_log.max()-train_data_log.min())

train_labels_log = train_data.iloc[:,1]

normalized_data_train.head()
def initial_coefs_intercept(data):

    """Function takes a pandas df as input and returns initialized coefficients and intercept"""

    coefficients = []

    intercept = 0

    for i in range(len(data.columns)):

        coefficients.append(0)

    return [coefficients, intercept]



initial_coefs = initial_coefs_intercept(normalized_data_train)[0]

initial_intercept = initial_coefs_intercept(normalized_data_train)[1]

print(initial_coefs)

print(initial_intercept)
#log_odds

def log_odds(data,coefficients,intercept):

    """Takes pandas dataframe, list of coefficients and an intercept value and returns

    an array of the log odds of each feature"""

    return np.dot(data,coefficients) + intercept



l_o= log_odds(normalized_data_train,initial_coefs, initial_intercept)

def sigmoid(log_odds_vars):

    """Takes log odds calculated with the log odds functions and returns the sigmoid transformed values

    restricting the values from 0 to 1"""

    sigmoid_values = 1/(1+np.exp(-log_odds_vars))

    

    return sigmoid_values



sigmoid_vals = sigmoid(l_o)

def log_loss(probabilities, labels):

    """Determines the log loss given a set of sigmoid values (probabilities) and a set of training data labels"""

    #start_time = time.time()

    data_length = len(labels)

    labels = np.array(labels)

    

    left_half = np.dot(labels,np.log(probabilities+.0001)) #including small epsilon so no division by 0

    right_half = np.dot(1-labels,np.log(1-probabilities+.0001))

    loss = (-1/data_length) * (left_half + right_half)



    #print("--- %s seconds ---" % (time.time() - start_time)) 

    return loss

    

#print(log_loss(sigmoid_vals,train_labels_log))
def find_coefficients(data, coefficients, intercept,labels,learning_rate, iterations):

    coefs = coefficients

    for i in range(iterations):

        l_odds = log_odds(data,coefs,intercept)

        sig_vals = sigmoid(l_odds)

        data_transpose = np.transpose(learning_rate * data)

        coefs = np.dot(data_transpose,(labels-sig_vals) * sig_vals*(1-sig_vals)) + coefs

        intercept = intercept + learning_rate * np.dot((labels-sig_vals), (sig_vals*(1-sig_vals)))

    print(coefs, intercept)

    return coefs, intercept

best_coefs= find_coefficients(normalized_data_train,initial_coefs, initial_intercept,train_labels_log,.0005, 50000)
best_coef = best_coefs[0]

best_int = best_coefs[1]



v = sigmoid(log_odds(normalized_data_train,best_coef,best_int))



#print(v)
def find_threshold(sigmoid_vals):

    """Takes sigmoid vals from best coefficients and best intercept and returns the best classifier threshold"""

    predictions = []

    vals = []

    accuracies = []

    

    for num in range(1000):

        vals.append(num/1000)

        accuracy = 0

        for i in v:

            if i > num/1000:

                predictions.append(1)

            else:

                predictions.append(0)

        

        for j in range(len(predictions)):

            if predictions[j] == train_labels_log[j]:

                accuracy += 1

        accuracies.append(accuracy/len(predictions))

        accuracy = 0

        predictions = []

    indx = accuracies.index(max(accuracies))

    print("Best accuracy on training set:")

    print(max(accuracies))

    best_threshold = vals[indx]

    return best_threshold

    

best_thresh = find_threshold(v)

print(best_thresh)
def calculate_precision(sigmoid_vals, threshold, labels):

    "Precision is  True Positives / (True Positives + False Positives)"

    predictions = []

    true_positives = 0

    false_positives = 0

    for i in sigmoid_vals:

        if i > threshold:

            predictions.append(1)

        else:

            predictions.append(0)

    

    

    for i in range(len(labels)):

        if labels[i] == 1 and labels[i] == predictions[i]:

            true_positives += 1

        elif labels[i] == 0 and labels[i] != predictions[i]:

            false_positives += 1

    

    return true_positives/(true_positives + false_positives)

    

print("Precision:")

print(calculate_precision(v, best_thresh, train_labels_log))



    
def calculate_recall(sigmoid_vals, threshold, labels):

    "Precision is  True Positives / (True Positives + False Negatives)"

    predictions = []

    true_positives = 0

    false_negatives = 0

    for i in sigmoid_vals:

        if i > threshold:

            predictions.append(1)

        else:

            predictions.append(0)

    

    

    for i in range(len(labels)):

        if labels[i] == 1 and labels[i] == predictions[i]:

            true_positives += 1

        elif labels[i] == 1 and labels[i] != predictions[i]:

            false_negatives += 1

    

    return true_positives/(true_positives + false_negatives)

    

    

    

print("Recall")          

print(calculate_recall(v, best_thresh, train_labels_log))

    
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")





test_data["Name"] = test_data["Name"].str.split(',').str[1]

test_data["Name"] = test_data["Name"].str.split('.').str[0]

test_data["Name"] = test_data["Name"].str.strip()

test_data['Sex'].replace({'female':0,'male':1},inplace=True)





x = test_data.groupby('Name').agg(['count']).index.get_level_values('Name')

test_data["Age"] = test_data.groupby("Name").transform(lambda x: x.fillna(x.mean()))['Age']





test_data_log = test_data.iloc[:,[False,True,False,True,True,True,True,False,True,False,False]]

normalized_data_test=(test_data_log-test_data_log.min())/(test_data_log.max()-test_data_log.min())

pred_test = sigmoid(log_odds(normalized_data_test,best_coef,best_int))
#looping through sigmoid values and comparing to threshold value. If greater than threshold, predict class 1.

#otherwise predict class 0

classifier = []

for i in range(len(pred_test)):

    if pred_test[i] > best_thresh:

        classifier.append(1)

    else:

        classifier.append(0)
data = {'PassengerId': test_data["PassengerId"].values, 'Survived':classifier} 

df_submission = pd.DataFrame(data)



df_submission.to_csv("submission_log_regression2.csv",index=False)



#Accuracy was 0.758 on testing set