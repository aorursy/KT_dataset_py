import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import graphviz 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from pandas import read_csv
%matplotlib inline
def plot_decision_tree(a,b):
    """
    http://scikit-learn.org/stable/modules/tree.html
    """
    dot_data = tree.export_graphviz(a, out_file=None, feature_names=b,class_names=['Healthy','Diabetes'],filled=False, rounded=True,special_characters=False)  
    graph = graphviz.Source(dot_data)  
    return graph 

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
dataset = read_csv('../input/train.csv')
dataset=dataset[['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
dataset.head(10)
dataset2 = dataset.iloc[:, :-1]
print("# of Rows, # of Columns: ",dataset2.shape)
print("\nColumn Name           # of Null Values\n")
print((dataset2[:] == 0).sum())
trainingData = read_csv('../input/train.csv') 
trainingData=trainingData[['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
testingData = read_csv('../input/test.csv')
testingData=testingData[['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
trainingFeatures = trainingData.iloc[:, :-1]
trainingLabels = trainingData.iloc[:, -1]
imputer = SimpleImputer(missing_values=0,strategy='median')
trainingFeatures = imputer.fit_transform(trainingFeatures)
trainingFeatures = pd.DataFrame(trainingFeatures)
trainingFeatures.columns=['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
testingData = imputer.transform(testingData)
testingData = pd.DataFrame(testingData)
testingData.columns=['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
print("# of Rows, # of Columns: ",trainingFeatures.shape)
print("\nColumn Name           # of Null Values\n")
print((trainingFeatures[:] == 0).sum())
g = sns.heatmap(trainingFeatures.corr(),cmap="Blues",annot=False)
trainingFeatures.corr()
trainingFeatures2 = trainingFeatures.drop(['Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin','Id'], axis=1)
g = sns.heatmap(trainingFeatures2.corr(),cmap="Blues",annot=False)
print(trainingFeatures2.corr())
model = DecisionTreeClassifier(max_depth=2,min_samples_leaf=2)
X_train, X_test, y_train, y_test = train_test_split(trainingFeatures2, trainingLabels, test_size=0.2, random_state=1)
model.fit(X_train, y_train)
columns = trainingFeatures2.columns
feature_names = trainingFeatures2.columns.values
coefficients = model.feature_importances_.reshape(trainingFeatures2.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Feature']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('\nFeature Importance:\n\n',fullList,'\n')
plot_decision_tree(model,feature_names)
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, trainingFeatures2, trainingLabels, cv=kfold)
print("DecisionTreeClassifier:\n\nCross_Val_Score: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
prediction = model.predict(X_test)
cnf_matrix = confusion_matrix(y_test, prediction)
dict_characters = {0: 'Healthy', 1: 'Diabetes'}
plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')
test = testingData
test = pd.DataFrame(test)
test.columns=['Id','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
test2 = test.drop(['Id','Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin'], axis=1)
my_predictions = model.predict(test2)
Identifier = test.Id.astype(int)
my_submission = pd.DataFrame({'Id': Identifier, 'Outcome': my_predictions})
my_submission.to_csv('my_submission.csv', index=False)
my_submission.head()