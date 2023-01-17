import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#from google.colab import drive

#drive.mount('/content/gdrive')
data = pd.read_csv('../input/diabetes.csv')
data.head(5)
data.describe()
data.info()
datacorr = data.corr()

sns.heatmap(data[data.columns[:9]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
sns.countplot(data.Outcome)


data.hist(figsize=(10, 15))
data1 = data[(data['BloodPressure']!=0) & (data['BMI']!=0) & (data['Glucose']!=0)]
data1.count()
data1 = data


onlydiabetic = data[(data['Outcome']==1)]

onlydiabetic.head(5)
onlydiabetic.hist(figsize=(10,15))
import sklearn 
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

x = data[data.columns[:8]]

x.head(5)
y= data['Outcome']

y.head(5)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
x_test
model1 = LogisticRegression()

model1.fit(x_train,y_train)

prediction = model1.predict(x_test)

accuracy = metrics.accuracy_score(prediction, y_test)



print("Logistic regression provides an accuracy of ", accuracy)
for param in model1.get_params().keys():

    print(param)
# Create regularization penalty space

penalty = ['l1', 'l2']



# Create regularization hyperparameter space

C = np.logspace(0, 4, 10)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)



clf = GridSearchCV(model1, hyperparameters, cv=5, verbose=0)



best_model = clf.fit(x_train, y_train)



best_model.best_estimator_

model_LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)

model_LR.fit(x_train,y_train)

prediction_LR = model_LR.predict(x_test)

accuracy_LR = metrics.accuracy_score(prediction_LR, y_test)



print("Logistic regression provides an accuracy of ", accuracy_LR)



LogisticRegression()
from yellowbrick.classifier import ConfusionMatrix
# The ConfusionMatrix visualizer taxes a model

cm = ConfusionMatrix(model_LR)



# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model

cm.fit(x_train, y_train)



# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data

# and then creates the confusion_matrix from scikit-learn.

cm.score(x_test, y_test)



# How did we do?

cm.poof()
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import precision_recall_fscore_support as score

from yellowbrick.classifier import ROCAUC

precision_recall_fscore_support(y_test, prediction)
precision, recall, fscore, support = score(y_test, prediction,)



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
# Instantiate the visualizer with the classification model

visualizer = ROCAUC(model1)



visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer.score(x_test, y_test)  # Evaluate the model on the test data

g = visualizer.poof()   
model2_without = svm.SVC()

model2_without.fit(x_train,y_train)

svm_prediction = model2_without.predict(x_test)

svm_accuracy = metrics.accuracy_score(svm_prediction, y_test)



print("SVC provides an accuracy of ", svm_accuracy)
Cs = [0.001, 0.01, 0.1, 1, 10]

gammas = [0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)

grid_search.fit(x_train, y_train)

grid_search.best_params_

  

  
model2 = svm.SVC(C=1,gamma=0.001)

model2.fit(x_train,y_train)

svm_prediction = model2.predict(x_test)

svm_accuracy = metrics.accuracy_score(svm_prediction, y_test)



print("SVC provides an accuracy of ", svm_accuracy)
from yellowbrick.classifier import ConfusionMatrix
# The ConfusionMatrix visualizer taxes a model

cm = ConfusionMatrix(model2)



# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model

cm.fit(x_train, y_train)



# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data

# and then creates the confusion_matrix from scikit-learn.

cm.score(x_test, y_test)



# How did we do?

cm.poof()



cm.confusion_matrix_
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import precision_recall_fscore_support as score

from yellowbrick.classifier import ROCAUC

precision_recall_fscore_support(y_test, svm_prediction)
precision, recall, fscore, support = score(y_test, svm_prediction)



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
# Instantiate the visualizer with the classification model

visualizer = ROCAUC(model2, micro=False, macro=False, per_class=False)



visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer.score(x_test, y_test)  # Evaluate the model on the test data

g = visualizer.poof()   
model3 = RandomForestClassifier()

model3.fit(x_train,y_train)

rf_prediction = model3.predict(x_test)

rf_accuracy = metrics.accuracy_score(rf_prediction, y_test)



print("Random Forest provides an accuracy of ", rf_accuracy)
param_grid = {

    'n_estimators': [200, 700],

    'max_features': ['auto', 'sqrt', 'log2']

}



CV_rfc = GridSearchCV(estimator=model3, param_grid=param_grid, cv= 5)



CV_rfc.fit(x_train,y_train)



CV_rfc.best_estimator_
model_RFC = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

model_RFC.fit(x_train,y_train)

rfc_prediction = model_RFC.predict(x_test)

rfc_accuracy = metrics.accuracy_score(rfc_prediction, y_test)



print("Random Forest provides an accuracy of ", rfc_accuracy)



print("the best score is %s"%CV_rfc.best_score_)  

print("the best parameter value that resulted in the best performance is %s"%CV_rfc.best_estimator_)  

print("the performance over test dats is %s"%CV_rfc.score(x_train, y_train) ) 

from yellowbrick.classifier import ConfusionMatrix
# The ConfusionMatrix visualizer taxes a model

cm = ConfusionMatrix(model_RFC)



# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model

cm.fit(x_train, y_train)



# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data

# and then creates the confusion_matrix from scikit-learn.

cm.score(x_test, y_test)



# How did we do?

cm.poof()
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import precision_recall_fscore_support as score

from yellowbrick.classifier import ROCAUC

precision_recall_fscore_support(y_test, svm_prediction)
precision, recall, fscore, support = score(y_test, rf_prediction)



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
# Instantiate the visualizer with the classification model

visualizer = ROCAUC(model3)



visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer

visualizer.score(x_test, y_test)  # Evaluate the model on the test data

g = visualizer.poof()   
results = []

models = np.array(["Logistic Regression","SVM","Random Forest"])

accuracies = np.round(np.array([accuracy_LR, svm_accuracy,rf_accuracy]),3)



a=np.append(models.reshape(-1,1),accuracies.reshape(-1,1), axis =1 )

res = pd.DataFrame(a,columns= (['Models', 'Accuracies']))

res
d = {'FP': [14,15,19] , 'TN' : [34,46,35]}

FP_TN = pd.DataFrame( d)

FP_TN
result = res.reset_index().merge(FP_TN.reset_index(), how = 'inner', on = 'index')
result.drop(['index'], axis=1, inplace=True)
result
FP_TN.reset_index()
result["Accuracies"] = result.Accuracies.astype(float)

result["Models"] = result.Accuracies.astype(str)



res.dtypes

result[['Models','Accuracies']].dtypes
 

plt.bar(result.Models , result.Accuracies)

plt.xticks(result.Models, ['Logistic Regression','SVM','Random Forest'])

plt.ylim(ymin=0.6)

plt.title('Accuracies')

plt.show()





plt.bar(result.Models, result.FP)

plt.xticks(result.Models, ['Logistic Regression','SVM','Random Forest'])

plt.ylim(ymin=0)

plt.title("Number of who don't have diabetes but are classified as they do")

plt.show()



plt.bar(result.Models, result.TN)

plt.xticks(result.Models, ['Logistic Regression','SVM','Random Forest'])

plt.title("Number of who have diabetes but are classified as they dont")

plt.ylim(ymin=0)

plt.show()
