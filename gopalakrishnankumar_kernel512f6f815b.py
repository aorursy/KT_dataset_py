# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load libraries
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from scipy import optimize
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
#%matplotlib inline

#step1 read the filein csv format

data = pd.read_csv("/kaggle/input/diabetes.csv")
print (data.shape)
print (data.describe())
print(data.head())

train,test = train_test_split(data, test_size = 0.4, random_state=30)
target = train["Outcome"]
feature = train[train.columns[0:8]]
feat_names = train.columns[0:8]
target_classes = ['0','1'] 
print(test)

import seaborn as sns
model = DecisionTreeClassifier(max_depth=4, random_state=0)
model.fit(feature,target)
test_input=test[test.columns[0:8]]
expected = test["Outcome"]
print("*******************Input******************")
print(test_input.head(2))
print("*******************Expected******************")
print(expected.head(2))
predicted = model.predict(test_input)
a1 = metrics.classification_report(expected, predicted)
print(a1)
conf = metrics.confusion_matrix(expected, predicted)
print(conf)
print("Decision Tree accuracy: ",model.score(test_input,expected))
dtreescore = model.score(test_input,expected)


label = ["0","1"]
d=sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)
print(d)

#Feature Importance DecisionTreeClassifier
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
print("DecisionTree Feature ranking:")
for f in range(feature.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feat_names[indices[f]], importance[indices[f]]))
plt.figure(figsize=(15,5))
plt.title("DecisionTree Feature importances")
plt.bar(range(feature.shape[1]), importance[indices], color="y", align="center")
plt.xticks(range(feature.shape[1]), feat_names[indices])
plt.xlim([-1, feature.shape[1]])
plt.show()

#Evaluation DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
#pip install randomness
import randomness
fpr,tpr,thres = roc_curve(expected, predicted)
roc_auc = auc(fpr, tpr)
plt.title('DecisionTreeClassifier-Receiver Characteristic Test Data')
plt.plot(fpr, tpr, color='green', lw=2, label='DecisionTree ROC curve (area = %0.2f)' % roc_auc)
plt.legend(loc='upper right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.show()
#%reset
_