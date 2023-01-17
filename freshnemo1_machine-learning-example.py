import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier as ABC
%matplotlib inline
data = pd.read_csv("../input/mushrooms.csv")
data.head(10)
labelEncoder = preprocessing.LabelEncoder()
for col in data.columns:
    data[col] = labelEncoder.fit_transform(data[col])

# Splitting test train set, with 20% of the data as the validation set
train, test = train_test_split(data, test_size = 0.2) 
# Train set
train_y = train['class']
train_x = train[[x for x in train.columns if 'class' not in x]]
# Test/Validation set
test_y = test['class']
test_x = test[[x for x in test.columns if 'class' not in x]]

models = [SVC(kernel='rbf', random_state=0), LR(), RF(),LDA(),ABC()]
model_names = ['SVC_rbf', 'Logistic Regression', 'RandomForestClassifier', 'LinearDiscriminantAnalysis','AdaBoostClassifier']
for i, model in enumerate(models):
    model.fit(train_x, train_y)
    print ('The accurancy of ' + model_names[i] + ' is ' + str(accuracy_score(test_y, model.predict(test_x))) )
    print ('The F1_score of ' + model_names[i] + ' is ' + str(f1_score(test_y, model.predict(test_x))) )
train, test = train_test_split(data, test_size = 0.99) #change test_size= 0.99, 0.95, 0.9 to see the result#
# Train set
train_y = train['class']
train_x = train[[x for x in train.columns if 'class' not in x]]
# Test/Validation set
test_y = test['class']
test_x = test[[x for x in test.columns if 'class' not in x]]

models = [SVC(kernel='rbf', random_state=0), LR(), RF(),LDA(),ABC()]
model_names = ['SVC_rbf', 'Logistic Regression', 'RandomForestClassifier', 'LinearDiscriminantAnalysis','AdaBoostClassifier']
for i, model in enumerate(models):
    model.fit(train_x, train_y)
    print ('The accurancy of ' + model_names[i] + ' is ' + str(accuracy_score(test_y, model.predict(test_x))) )
    print ('The F1_score of ' + model_names[i] + ' is ' + str(f1_score(test_y, model.predict(test_x))) )
train, test = train_test_split(data, test_size = 0.2) 
# Train set
train_y = train['class']
train_x = train[[x for x in train.columns if 'class' not in x]]
# Test/Validation set
test_y = test['class']
test_x = test[[x for x in test.columns if 'class' not in x]]
SVCm=SVC()
LRm=LR(C=1.0)#"Please try between 0 and 1"
RFm=RF(n_estimators=100)#"Please try between 1 and 100"
LDAm=LDA(solver='lsqr')#"Please try ‘svd’,‘lsqr’, or ‘eigen’:"
ABCm=ABC(n_estimators=1000)#"Please try between 1 and 1000"
models = [SVCm, LRm, RFm,LDAm,ABCm]
model_names = ['SVC_rbf', 'Logistic Regression', 'RandomForestClassifier', 'LinearDiscriminantAnalysis','AdaBoostClassifier']
resultdict=dict(modelname=[],accuracy=[],f1=[])
for i, model in enumerate(models):
    model.fit(train_x, train_y)
    resultdict['modelname'].append(model_names[i])
    resultdict['accuracy'].append(accuracy_score(test_y, model.predict(test_x)))
    resultdict['f1'].append(f1_score(test_y, model.predict(test_x)))
resultdict1=pd.DataFrame.from_dict(resultdict)
plot=resultdict1.plot.bar(x='modelname',rot=15, subplots=True)
plot