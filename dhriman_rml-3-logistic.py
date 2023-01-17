# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Step 1: Import data analysis modules



import numpy as np

import pandas as pd

import os



import pickle



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Step 2 : Data import



#os.chdir(r'C:\Users\dhrimand\Desktop')

input_file = pd.read_csv('/kaggle/input/glass/glass.csv')

input_file.head(5)
input_file.dtypes
input_file.describe()
input_file['Type'].value_counts()
#Step 3: Clean up data

# Use the .isnull() method to locate missing data

missing_values = input_file.isnull()

missing_values.head(5)
#Step 4.1: Visualize the data

# Use seaborn to conduct heatmap to identify missing data

# data -> argument refers to the data to creat heatmap

# yticklabels -> argument avoids plotting the column names

# cbar -> argument identifies if a colorbar is required or not

# cmap -> argument identifies the color of the heatmap

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
#Convert the target feature into a binary feature



input_file['label'] = input_file.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

input_file.head(20)
#Step 5.1: Prepare input X parameters/features and output y



#split dataset in features and target variable





feature_cols = ['Na', 'Mg', 'Al', 'Si','K','Ca','Ba','Fe']

X = input_file[feature_cols] # Features

y = input_file.label # Target variable
# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
# import the class

from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



#

y_pred=logreg.predict(X_test)
# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
#Visualizing Confusion Matrix using Heatmap





class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#Confusion Matrix Evaluation Metrics



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
#ROC Curve



y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()