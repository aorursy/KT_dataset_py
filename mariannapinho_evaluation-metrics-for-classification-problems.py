%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import metrics, tree

from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#The values used in this example are random and used just to exemplify the matrix



bin_class_matrix = np.array([[15,6],[1,21]])

matrix_flat = bin_class_matrix.flatten()

notes = ['True Positive', 'False Negative', 'False Positive', 'True Negative']

to_show = np.array([[49,70],[70,49]])



labels = [f"{v1}\n{v2}" for v1, v2 in zip(matrix_flat, notes)]

labels = np.asarray(labels).reshape(2,2)



fake_category = ['Class 1','Class 2']

sns.heatmap(to_show, annot=labels, cmap='Paired',

            xticklabels=fake_category, yticklabels=fake_category,

            fmt='',vmin = 0, vmax = 150, cbar = False, linecolor='white', linewidths=0.5)

plt.title("Confusion matrix for binary classification")

plt.ylabel('True labels (Groundtruth)');

plt.xlabel('Predicted labels');
#The values used in this example are random and used just to exemplify the matrix



ter_class_matrix = np.array([[21,6,4],[15,1,11],[1,5,13]])

to_show = np.array([[49,70,70],[70,49,70],[70,70,49]])



matrix_flat = ter_class_matrix.flatten()

labels = [f"{v1}" for v1 in matrix_flat]

labels = np.asarray(labels).reshape(3,3)



fake_category = ['Apples','Pineapples','Bananas']



sns.heatmap(to_show, annot=labels,cmap='Paired',

            xticklabels=fake_category,yticklabels= fake_category,

            fmt='',vmin = 0,vmax = 150,cbar = False,

            linecolor = 'white',linewidths=0.5)



plt.title("Confusion matrix for three class classification")

plt.ylabel('True labels (Groundtruth)');

plt.xlabel('Predicted labels');
#The values used in this example are random and used just to exemplify the curve



equal_prob = [0 for _ in range(20)]

model_prob = np.random.rand(1,20).flatten()

model_prob[:10] = 1 

y_values = np.random.randint(0,2,(1,20)).flatten()

y_values[:10] = 1

y_values[10:15] = 0



model_fpr, model_tpr, _ = roc_curve(y_values, model_prob)

equal_fpr, equal_tpr, _ = roc_curve(y_values, equal_prob)



plt.plot(equal_fpr, equal_tpr, linestyle='--', label='Equal TP and FP');

plt.plot(model_fpr, model_tpr, marker='.', label='Model');



plt.xlabel('False Positive Rate');

plt.ylabel('True Positive Rate');

plt.legend();
iris_dataset = pd.read_csv("/kaggle/input/iris/Iris.csv")

iris_dataset = iris_dataset.drop(labels = ['Id'], axis=1)

iris_dataset.head()
iris_dataset['Species'].value_counts()
iris_dataset.describe()
# Separate training and test



iris_values = iris_dataset.values

X,y = iris_values[:,:-1], iris_values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 42)



#Create model

classifier = tree.DecisionTreeClassifier(random_state=42)

classifier.fit(X_train, y_train)



#Make predictions

predictions = classifier.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, predictions)



categories = ['Iris-setosa','Iris-versicolor','Iris-virginica']

sns.heatmap(conf_matrix,

            annot=True,cmap='YlOrRd',

            xticklabels=categories, cbar=False)



plt.yticks(np.arange(3),categories)

plt.ylabel('True labels');

plt.xlabel('Predicted labels');

plt.title('Confusion matrix of Iris species classification');
print(metrics.classification_report(y_test, predictions, digits=3))
metrics.classification_report(y_test, predictions, digits=3, output_dict=True)