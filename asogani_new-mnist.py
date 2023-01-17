import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model



train_data = pd.read_csv("../input/train.csv")

test_data  = pd.read_csv("../input/test.csv")
# Partition the data into train and cross-validation



print("Partitioning training data into cross-validation and training...")



cross_validation_data = train_data.sample(frac = 0.2)

train_data_x_cv = train_data.drop(cross_validation_data.index)



print("Length of original training set "+str(len(train_data)))

print("Length of cross-validation set "+str(len(cross_validation_data)))

print("Length of final training set "+str(len(train_data_x_cv)))
print(train_data.columns)

print(test_data.columns)

x_variables = list(train_data.columns)

x_variables.remove('label')

y_variable = 'label'
# Check if some features are always zero

import matplotlib.pyplot as plt

temp_df = pd.DataFrame(train_data[x_variables].sum())

temp_df.columns = ['sum_of_intensities']

#temp_df.reset_index(inplace = True)

temp_df.sort_values(by = ['sum_of_intensities'],ascending = [1],inplace = True)

plt.figure(figsize=[10,50])

temp_df.plot(kind = 'bar')

temp_df.reset_index(inplace = True)

temp_df.columns = ['features','sum_of_intensities']
# Get a list of features for which are always zero and remove them

features_to_remove = list(temp_df.loc[temp_df['sum_of_intensities']==0,'features'])

print("Following are the features that neeed to be removed \n")

print(features_to_remove)

x_variables_copy = x_variables[:]

print("\n These features will now be removed \n")

for f in features_to_remove:

    x_variables_copy.remove(f)
# Neural Netwroks

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

Y = train_data_x_cv[y_variable].as_matrix()

X = train_data_x_cv[x_variables_copy].as_matrix()



print("Training neural nets... ")

my_classifier = MLPClassifier(solver = 'adam',alpha = 0.1,hidden_layer_sizes = (50))

my_classifier.fit(X,Y)

predicted = my_classifier.predict(cross_validation_data[x_variables_copy].as_matrix())



print("Getting Accuracy for CV...")

accu = accuracy_score(cross_validation_data[y_variable].as_matrix(),predicted)

print("Accuracy = "+str(accu))
predicted_test_labels = my_classifier.predict(test_data[x_variables_copy])

data_dict = {'ImageId':range(1,len(predicted_test_labels)+1),'Label':predicted_test_labels}

test_labels = pd.DataFrame(data_dict)

test_labels.to_csv(r'Output_neural_nets.csv')

# classify using a logistic classifier with C = 50

Y = train_data_x_cv[y_variable]

X = train_data_x_cv[x_variables_copy]



print("Training a logistic regression model ... ")

logreg = linear_model.LogisticRegression(penalty = 'l2',C = 50,fit_intercept = True,solver = 'sag',max_iter = 300)

logreg.fit(X,Y)

predicted = logreg.predict(cross_validation_data[x_variables_copy])



print(np.corrcoef(cross_validation_data[y_variable],predicted))

predicted_test_labels = my_classifier.predict(test_data[x_variables_copy])

data_dict = {'ImageId':range(1,len(predicted_test_labels)+1),'Label':predicted_test_labels}

test_labels = pd.DataFrame(data_dict)


