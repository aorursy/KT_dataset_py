import pandas as pd

data = pd.read_csv("../input/UpdatedReFinalFintechData (1).csv")
print(data.head())
# convert decision and sex into boolean values
data['credit_score'] = data['Y= decision'] == "ACCEPTED"
# True if user is male, and False otherwise
data.sex = data.sex == "M"
# print value-space of credit_score to verify it is only boolean
data['credit_score'].unique()
# for each variable to use, states the critical point
decision_points = {'age': 30, 'accountAge': 20, 'balance1': 10000, 'maximumCredit': 20000,
                   'maxAcAge': 50, 'utilization': 50}
# converts each feature into boolean values
for i in decision_points:
	data[i] = data[i] > decision_points[i]
print(data.head())

# features to take into account in the cleaned dataset
select_columns = ['age', 'sex', 'accountAge', 'balance1', 'maximumCredit', 'maxAcAge', 
                  'utilization', 'credit_score']
# for each feature, transforms True to 1 and False to -1
filtered_data = data[select_columns] * 2 - 1
print(filtered_data.head())
import numpy as np

# convert DataFrame to array to shuffle
filtered_data = np.array(filtered_data)
np.random.shuffle(filtered_data)
# convert shuffled matrix to DataFrame again
filtered_data = pd.DataFrame(filtered_data, columns=select_columns)

# select training and testing features having a 80-20 proportion
x_train, x_test = filtered_data[filtered_data.columns.difference(['credit_score'])][:round(len(filtered_data)*0.8)], \
                  filtered_data[filtered_data.columns.difference(['credit_score'])][round(len(filtered_data)*0.8):]

y_train, y_test = pd.DataFrame(filtered_data.credit_score[:round(len(filtered_data)*0.8)], 
                               columns=['credit_score']), \
                  np.array(pd.DataFrame(filtered_data.credit_score[round(len(filtered_data)*0.8):], 
                               columns=['credit_score']).T)[0]

print(x_train.head())
print(y_train.head())
print((len(x_train), len(x_test)))
print((len(y_train), len(y_test)))
print(y_train.credit_score.unique())
from sklearn import tree

# Configuring ML Decision Tree model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train.credit_score.astype('int'))
predicted = clf.predict(x_test)

# print accuracy value 
print(f"Decision Tree Accuracy: {round(sum(predicted == y_test) * 100 / len(y_test), 1)}%")
import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=x_train.columns,  
                         class_names=x_train.columns,  
                         filled=True, rounded=True,  
                         special_characters=True)   
graph = graphviz.Source(dot_data)  
graph.render("treemodel")
graph
from sklearn import svm

# Configuring ML SVM model
clf = svm.SVC()
clf.fit(x_train, y_train.credit_score.astype('int'))
predicted = clf.predict(x_test)

# print accuracy value 
print(f"SVM Accuracy: {round(sum(predicted == y_test) * 100 / len(y_test), 1)}%")
