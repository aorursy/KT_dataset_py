import pandas as pd
import numpy as np
import random as rnd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import KFold
from matplotlib import pyplot

credit_data = pd.read_csv("credit_data.csv")
credit_data
column_headers = credit_data.columns.values.tolist()

column_headers
missing_data_list = []
for i in range(len(column_headers)):
    missing_data_list.append(credit_data[credit_data[column_headers[i]] =="None"].shape[0]/credit_data.shape[0]*100)
missing_df_dict = {'column names':column_headers, 'missing data percent':missing_data_list}
missing_data_df = pd.DataFrame(missing_df_dict)
missing_data_df
credit_data = credit_data.drop(['organization_type', 'seniority', 'vehicle_type'], axis=1)
credit_data.describe()
credit_data.info()
fig = pyplot.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
for i, columns in zip(range(1, 14),credit_data.columns):
    ax = fig.add_subplot(5, 3, i)
    ax.hist(credit_data[columns])
    pyplot.xlabel(columns, fontsize="20")
    pyplot.ylabel('No of Observations', fontsize="17")
genderLabelEncoder = preprocessing.LabelEncoder()
genderLabelEncoder.fit(credit_data["gender"])
credit_data["gender"] = genderLabelEncoder.transform(credit_data["gender"])

educationLabelEncoder = preprocessing.LabelEncoder()
educationLabelEncoder.fit(credit_data["education"])
credit_data["education"] = educationLabelEncoder.transform(credit_data["education"])


occupationLabelEncoder = preprocessing.LabelEncoder()
occupationLabelEncoder.fit(credit_data["occupation"])
credit_data["occupation"] = occupationLabelEncoder.transform(credit_data["occupation"])

houseTypeLabelEncoder = preprocessing.LabelEncoder()
houseTypeLabelEncoder.fit(credit_data["house_type"])
credit_data["house_type"] = houseTypeLabelEncoder.transform(credit_data["house_type"])

maritalStatusLabelEncoder = preprocessing.LabelEncoder()
maritalStatusLabelEncoder.fit(credit_data["marital_status"])
credit_data["marital_status"] = maritalStatusLabelEncoder.transform(credit_data["marital_status"])

credit_data
credit_data_input_columns = credit_data.columns.values.tolist()
credit_data_input_columns.remove("default")
credit_data_input = credit_data[credit_data_input_columns]

credit_data_output = credit_data["default"]
X_train, X_test, y_train, y_test = train_test_split(credit_data_input,credit_data_output , test_size=0.3, random_state=1)
modelGini = DecisionTreeClassifier(max_leaf_nodes=40)
modelGini.fit(X_train,y_train)
y_predicted = modelGini.predict(X_test)
print("Accuracy: %0.2f" % round(metrics.accuracy_score(y_test, y_predicted)*100,2))
modelEntropy = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=40)
modelEntropy.fit(X_train,y_train)
y_predicted_entropy = modelEntropy.predict(X_test)
print("Accuracy: %0.2f" % round(metrics.accuracy_score(y_test, y_predicted_entropy)*100,2))
from sklearn import tree
import pydotplus

dot_data=tree.export_graphviz(modelEntropy,filled=True,rounded=True)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')    

pyplot.imshow(pyplot.imread('tree.png'))
def entropyDecisionTree(max_nodes, accuracy_list):
    modelEntropy = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=max_nodes)
    modelEntropy.fit(X_train,y_train)
    y_predicted_entropy = modelEntropy.predict(X_test)
    accuracy_list.append([max_nodes,metrics.accuracy_score(y_test, y_predicted_entropy)])
accuracy_list = []
for i in range(2,400):
    entropyDecisionTree(i, accuracy_list)

accuracy_df = pd.DataFrame(accuracy_list, columns=["max nodes","accuracy"])
pyplot.plot(accuracy_df["max nodes"], accuracy_df["accuracy"])
pyplot.xlabel("Max Leaf Nodes")
pyplot.ylabel("accuracy")
Score_gini = []
final_x_train = []
final_x_train = []
kfold= KFold(n_splits=5, random_state=1, shuffle=True)
for train,test in kfold.split(credit_data):
    x_train, x_test = credit_data_input.iloc[train], credit_data_input.iloc[test]
    y_train=credit_data_output.iloc[train]
    y_test=credit_data_output.iloc[test]
    
    modelEntropy.fit(x_train,y_train)
    
    y_pred_entropy = modelEntropy.predict(x_test)
    Score_gini.append(metrics.accuracy_score(y_test,y_pred_entropy))
    if Score_gini[len(Score_gini)-1] > Score_gini[len(Score_gini)-2]:
        final_x_train = x_train
        final_y_train = y_train
print("Accuracy: %0.2f (+/- %0.3f)" % (round(np.mean(Score_gini)*100,2), np.std(Score_gini*2)))
Score_gini = []
final_x_train = []
final_x_train = []
kfold= KFold(n_splits=5, random_state=1, shuffle=True)
for train,test in kfold.split(credit_data):
    x_train, x_test = credit_data_input.iloc[train], credit_data_input.iloc[test]
    y_train=credit_data_output.iloc[train]
    y_test=credit_data_output.iloc[test]
    
    modelGini.fit(x_train,y_train)
    
    y_pred_entropy = modelGini.predict(x_test)
    Score_gini.append(metrics.accuracy_score(y_test,y_pred_entropy))
    if Score_gini[len(Score_gini)-1] > Score_gini[len(Score_gini)-2]:
        final_x_train = x_train
        final_y_train = y_train
print("Accuracy: %0.2f (+/- %0.3f)" % (round(np.mean(Score_gini)*100,2), np.std(Score_gini*2)))
