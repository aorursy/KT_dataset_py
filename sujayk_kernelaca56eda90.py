import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score as ass
data= pd.read_csv("../input/training.csv")

data_test = pd.read_csv("../input/testing.csv")
data.head()

data.shape
data.describe()
data_spe = data.iloc[:,1:10].copy()

data_sim_h = data.iloc[:,11:19].copy()

data_sim_s = data.iloc[:,20:28].copy()

cla = data['class']

#data_sim_h['class'] = data['class'].copy()

#data_sim_s['class'] = data['class'].copy()
data_spe_test = data_test.iloc[:,1:10].copy()

data_sim_h_test = data_test.iloc[:,11:19].copy()

data_sim_s_test = data_test.iloc[:,20:28].copy()

cla_test = data_test['class']

#data_sim_h['class'] = data['class'].copy()

#data_sim_s['class'] = data['class'].copy()
from sklearn.preprocessing import LabelEncoder

gle = LabelEncoder()

class_labels = gle.fit_transform(data['class'])

class_mappings = {index: label for index, label in 

                  enumerate(gle.classes_)}

print(class_mappings)



print(class_labels)



class_test_labels = gle.fit_transform(data_test['class'])

class_test_mappings = {index: label for index, label in 

                  enumerate(gle.classes_)}

print(class_test_mappings)



print(class_test_labels)
freq = data['class'].value_counts().to_dict()

freq_test = data_test['class'].value_counts().to_dict()

print(freq)

print(freq_test)
x = data.drop(['class'],axis=1)

x_test = data_test.drop(['class'],axis=1)

print(x_test.head())

print(x.head())
def svm_spe_func(k,c_val):

    svm = SVC(kernel= k, C=c_val, gamma='auto')

    svm.fit(data_spe,class_labels)

    y_spe_pred = svm.predict(data_spe_test)

    #print(y_spe_pred)

    return ass(class_test_labels,y_spe_pred)
def svm_func(k,c_val):

    svm = SVC(kernel= k, C=c_val, gamma='auto')

    svm.fit(x,class_labels)

    y_pred = svm.predict(x_test)

    #print(y_spe_pred.shape)

    return ass(class_test_labels,y_pred)
k = ['linear','rbf','poly']

c = [1,.1,.01,.001]

for k_val in k:

    for c_val in c:

        accuracy = svm_spe_func(k_val,c_val)

        print("Kernel: {0} c: {1} accuracy: {2}".format(k_val,c_val,accuracy))
k = ['linear','rbf','poly']

c = [1,.1,.01,.001]

for k_val in k:

    for c_val in c:

        accuracy = svm_func(k_val,c_val)

        print("Kernel: {0} c: {1} accuracy: {2}".format(k_val,c_val,accuracy))
from sklearn import tree

from sklearn.ensemble import AdaBoostClassifier
clf = tree.DecisionTreeClassifier()

clf = clf.fit(data_spe,class_labels)

y_spe_pred = clf.predict(data_spe_test)

acc = ass(class_test_labels,y_spe_pred)

acc
clf = tree.DecisionTreeClassifier()

#clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=.001)

clf = clf.fit(x,class_labels)

y_pred = clf.predict(x_test)

acc = ass(class_test_labels,y_pred)

acc
import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("data") 

dot_data = tree.export_graphviz(clf, out_file=None, 

                     feature_names=list(x),  

                     class_names=list(class_mappings.values()),  

                     filled=True, rounded=True,  

                     special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
list(data)
list(class_mappings.values())