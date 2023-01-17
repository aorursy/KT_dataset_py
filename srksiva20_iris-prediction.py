# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import sklearn.metrics as me
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.figure_factory as ff

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading Dataset
iris_data = pd.read_csv("../input/Iris.csv")


#Dropping id column 
iris_data=iris_data.drop('Id',axis=1)



#print(iris_data)
#Common Methods

def createdf():
    data = {'Species Type':['Iris-setosa','Iris-versicolor','Iris-virginica']}  
    df = pd.DataFrame(data)
    return df

def createdf_avg():
    data = {'ML Metric':['Precision','Recall','F1 - Score']}  
    df = pd.DataFrame(data)
    return df

def adddata(mlmodel,report,metric,df):
    df.loc [0,mlmodel] = round(report["Iris-setosa"][metric],2)
    df.loc [1,mlmodel] = round(report["Iris-versicolor"][metric],2)
    df.loc [2,mlmodel] = round(report["Iris-virginica"][metric],2)
    return df

def adddata_avg(mlmodel,report,avg_type,df):
    df.loc [0,mlmodel] = round(report[avg_type]["precision"],4)*100
    df.loc [1,mlmodel] = round(report[avg_type]["recall"],4)*100
    df.loc [2,mlmodel] = round(report[avg_type]["f1-score"],4)*100
    return df


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


#DFs
data = {'Metric':['Accuracy']}  
df_accuracy = pd.DataFrame(data)
df_precision = createdf()
df_recall = createdf()
df_f1 = createdf()
df_support = createdf()
df_macro = createdf_avg()
df_weighted = createdf_avg()

#Pair plot among all features given in data set
sns.pairplot(iris_data, hue="Species", height=3, diag_kind="kde")
#Assigning Varibles for Model
y=iris_data.Species
X =iris_data.drop('Species',1)

#Splitting Data for train and test
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=42)
#logistical Regression
model = LogisticRegression(solver='liblinear',multi_class='auto')
model.fit(x_train,y_train)
pred_test_log=model.predict(x_test)
log_report = classification_report(y_test,pred_test_log,output_dict=True)
cnf_matrix = confusion_matrix(y_test,pred_test_log)
plot_confusion_matrix(cnf_matrix,normalize = False, target_names =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                      title='Confusion matrix, without normalization')
plt.figure()
df_accuracy.loc[0,'Logistic'] = round(log_report['accuracy'],4)*100
df_precision = adddata('Logistic',log_report,'precision',df_precision)
df_recall = adddata('Logistic',log_report,'recall',df_recall)
df_f1 = adddata('Logistic',log_report,'f1-score',df_f1)
df_macro = adddata_avg('Logistic',log_report,'macro avg',df_macro)
df_weighted = adddata_avg('Logistic',log_report,'weighted avg',df_weighted)
#Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
pred_test_Dec=model.predict(x_test)
Dec_report = classification_report(y_test,pred_test_Dec,output_dict=True)
cnf_matrix = confusion_matrix(y_test,pred_test_Dec)
plot_confusion_matrix(cnf_matrix,normalize = False, target_names =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                      title='Confusion matrix, without normalization')
plt.figure()
df_accuracy.loc[0,'Decision Tree'] = round(Dec_report['accuracy'],4)*100
df_precision = adddata('DecisionTree',Dec_report,'precision',df_precision)
df_recall = adddata('DecisionTree',Dec_report,'recall',df_recall)
df_f1 = adddata('DecisionTree',Dec_report,'f1-score',df_f1)
df_macro = adddata_avg('DecisionTree',Dec_report,'macro avg',df_macro)
df_weighted = adddata_avg('DecisionTree',Dec_report,'weighted avg',df_weighted)
#SVM
model = svm.SVC(kernel='linear') 
model.fit(x_train,y_train)
pred_test_svm=model.predict(x_test)
svm_report = classification_report(y_test,pred_test_svm,output_dict=True)
cnf_matrix = confusion_matrix(y_test,pred_test_svm)
plot_confusion_matrix(cnf_matrix,normalize = False, target_names =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                      title='Confusion matrix, without normalization')
plt.figure()
df_accuracy.loc[0,'SVM'] = round(svm_report['accuracy'],4)*100
df_precision = adddata('SVM',svm_report,'precision',df_precision)
df_recall = adddata('SVM',svm_report,'recall',df_recall)
df_f1 = adddata('SVM',svm_report,'f1-score',df_f1)
df_macro = adddata_avg('SVM',svm_report,'macro avg',df_macro)
df_weighted = adddata_avg('SVM',svm_report,'weighted avg',df_weighted)
#Random Forest
model =  RandomForestClassifier(n_estimators=10, criterion = 'entropy')
model.fit(x_train,y_train)
pred_test_rf=model.predict(x_test)
rf_report = classification_report(y_test,pred_test_rf,output_dict=True)
cnf_matrix = confusion_matrix(y_test,pred_test_rf)
plot_confusion_matrix(cnf_matrix,normalize = False, target_names =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                      title='Confusion matrix, without normalization')
plt.figure()
df_accuracy.loc[0,'Random Forest'] = round(rf_report['accuracy'],4)*100
df_precision = adddata('RandomForest',rf_report,'precision',df_precision)
df_recall = adddata('RandomForest',rf_report,'recall',df_recall)
df_f1 = adddata('RandomForest',rf_report,'f1-score',df_f1)
df_macro = adddata_avg('RandomForest',rf_report,'macro avg',df_macro)
df_weighted = adddata_avg('RandomForest',rf_report,'weighted avg',df_weighted)
#KNN
model = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
model.fit(x_train,y_train)
pred_test_knn=model.predict(x_test)
knn_report = classification_report(y_test,pred_test_knn,output_dict=True)
cnf_matrix = confusion_matrix(y_test,pred_test_knn)
plot_confusion_matrix(cnf_matrix,normalize = False, target_names =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                      title='Confusion matrix, without normalization')
plt.figure()
df_accuracy.loc[0,'KNN'] = round(knn_report['accuracy'],4)*100
df_precision = adddata('KNN',knn_report,'precision',df_precision)
df_recall = adddata('KNN',knn_report,'recall',df_recall)
df_f1 = adddata('KNN',knn_report,'f1-score',df_f1)
df_macro = adddata_avg('KNN',knn_report,'macro avg',df_macro)
df_weighted = adddata_avg('KNN',knn_report,'weighted avg',df_weighted)
#Naive Bayes
model = GaussianNB()
model.fit(x_train, y_train)
pred_test_nb=model.predict(x_test)
nb_report = classification_report(y_test,pred_test_nb,output_dict=True)
cnf_matrix = confusion_matrix(y_test,pred_test_nb)
plot_confusion_matrix(cnf_matrix,normalize = False, target_names =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                      title='Confusion matrix, without normalization')
plt.figure()
df_accuracy.loc[0,'Naive Bayes'] = round(nb_report['accuracy'],4)*100
df_precision = adddata('NB',nb_report,'precision',df_precision)
df_recall = adddata('NB',nb_report,'recall',df_recall)
df_f1 = adddata('NB',nb_report,'f1-score',df_f1)
df_macro = adddata_avg('NB',nb_report,'macro avg',df_macro)
df_weighted = adddata_avg('NB',nb_report,'weighted avg',df_weighted)
df_accuracy
df_precision
df_recall
df_f1
df_weighted