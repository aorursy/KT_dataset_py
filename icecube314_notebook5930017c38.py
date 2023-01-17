# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#Load the training data

df = pd.read_csv('../input/train.csv')
train_labels = df['TARGET'].tolist();

num_unsatisfied = train_labels.count(1);

num_satisfied = train_labels.count(0);

print(str(round(100*num_unsatisfied/(num_satisfied + num_unsatisfied),2)) + "% of customers unsatisfied.")
#Remove the ID feature

del df['ID']

print("ID feature removed.")



#Remove features with variance less than 0.01

k = 0;

for name in list(df.columns.values):

    if name != 'TARGET' and np.var(df[name].tolist()) < 0.01:

        del df[name];

        k += 1;

print(str(k) + " features removed because of low variance.")



#Remove highly correlated features

features = list(df.columns.values);

removed_features = set();

k = 0;

for k1 in range(len(features)-1):

    for k2 in range(k1+1,len(features)-1):

        if features[k2] not in removed_features and abs(df[features[k1]].corr(df[features[k2]])) > 0.99:

            removed_features.add(features[k2])

            k += 1;

            #print(features[k1] + " and " + features[k2] + " are strongly correlated.")

for feature in removed_features:

    del df[feature]

print(str(k) + " features removed because of strong correlation.")
#Find which features are most correlated with the TARGET feature

features = df.columns.values

target_corr = []

for k1 in range(len(features)-1):

    target_corr.append(abs(df['TARGET'].corr(df[features[k1]])))



features_target_corr = zip(features[:-2],target_corr)

sorted_features_target_corr = sorted(features_target_corr,key = lambda pair: pair[1],reverse=True)

sorted_features = [pair[0] for pair in sorted_features_target_corr]

sorted_corr = [pair[1] for pair in sorted_features_target_corr]

for k1 in range(10):

    print(sorted_features[k1] + ": " + str(sorted_corr[k1]))
from sklearn import linear_model

logistic = linear_model.LogisticRegression()



#Make training data into matrix

relevant_features = df.columns.values

relevant_features = relevant_features[:-2]

train_data = df[relevant_features].as_matrix()



##Normalize features

#for k in range(len(train_data[0])):

#    train_data[k] = (train_data[k] - train_data[k].mean())/train_data[k].std()

#train_labels = np.asarray(train_labels)
#Fit simple logistic model (no regularization)

logistic.fit(train_data, train_labels)



#Compute training error

def compute_train_error(logistic0,train_data0,train_labels0):

    train_predict0 = logistic0.predict(train_data0)

    print(str(round(100*sum(train_predict0 != train_labels0)/len(train_labels0),2)) + "% misclassification error.")

    

print("Training set error.")

compute_train_error(logistic,train_data,train_labels)
logistic2 = linear_model.LogisticRegression(class_weight='balanced')

logistic2.fit(train_data,train_labels)



print("Training set error. Balanced classes.")

compute_train_error(logistic2,train_data,train_labels)
#Use only most relevant features

if True:

    top_relevant_features = sorted_features[:2]#:10

    top_train_data = df[top_relevant_features].as_matrix()



    logistic.fit(top_train_data, train_labels)

    print("Training set error.")

    compute_train_error(logistic,top_train_data,train_labels)

    logistic2.fit(top_train_data, train_labels)

    print("Training set error. Balanced classes.")

    compute_train_error(logistic2,top_train_data,train_labels)
#Try using polynomial features

if False:

    L = len(train_data[0])

    poly_train = np.empty([len(train_data),L+L*L])

    for k1 in range(len(train_data)):

        a = train_data[k]

        b = np.kron(a,a)

        poly_train[k] = np.concatenate((a,b),axis=0)



    logistic.fit(poly_train, train_labels)

    print("Training set error. Polynomial features.")

    compute_train_error(logistic,poly_train,train_labels)

    logistic2.fit(poly_train, train_labels)

    print("Training set error. Polynomial features and balanced classes.")

    compute_train_error(logistic2,poly_train,train_labels)
#Decision tree

from sklearn import tree



tree1 = tree.DecisionTreeClassifier()

tree1.fit(train_data, train_labels)

predicted1 = tree1.predict(train_data)



print("Decision tree: " + str(round(100*sum(predicted1 != train_labels)/len(train_labels),2)) + "% misclassification error.")
#Random forest

from sklearn.ensemble import RandomForestClassifier



tree2 = RandomForestClassifier()

tree2.fit(train_data, train_labels)

predicted2 = tree2.predict(train_data)



print("Random forest: " + str(round(100*sum(predicted2 != train_labels)/len(train_labels),2)) + "% misclassification error.")
#Try 5 different training set divisions with 80% used as training data

number_of_samples = len(train_labels)

M = round(number_of_samples*0.2)

for i in range(5):

    test_set = range(i*M,(i+1)*M)

    train_set = set(range(number_of_samples)).difference(set(test_set))

    train_data1 = [train_data[i] for i in train_set]

    train_labels1 = [train_labels[i] for i in train_set]

    test_data1 = [train_data[i] for i in test_set]

    test_labels1 = [train_labels[i] for i in test_set]



    tree1.fit(train_data1,train_labels1)

    predicted1 = tree1.predict(test_data1)

    tree2.fit(train_data1,train_labels1)

    predicted2 = tree2.predict(test_data1)

    

    print("Cross-validation error " + str(i+1) + ".")

    print("Decision tree: " + str(round(100*sum(predicted1 != test_labels1)/len(train_labels),2)) + "% misclassification error.")

    print("Random forest: " + str(round(100*sum(predicted2 != test_labels1)/len(train_labels),2)) + "% misclassification error.")