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
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
# Importing the dataset
dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')
dataset.head()
dataset.info()
# Checking for null values in the dataset
dataset.isnull().values.any()
# Visualising the dataset
plt.figure(figsize=(9,7))
plt.style.use('seaborn-pastel')
labels=['female','male']
sns.set_style("darkgrid")
ax=sns.barplot(x='target',y='age',data=dataset,hue='sex')
h, l = ax.get_legend_handles_labels()
ax.legend(h,labels,title="Gender",loc='upper right')
ax.set_ylabel("Age",fontdict={'fontsize' : 12})
ax.set_xlabel("Target variable: Angiographic disease status",fontdict={'fontsize' : 12})
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (x+ 0.15, y + height + 2.4))
plt.title('Mean age of patients grouped by gender',fontweight="bold")
plt.show()
# Defining the features and the outcome variable
x= dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,[0,3,4,7,9,11]] = sc.fit_transform(x_train[:,[0,3,4,7,9,11]])
x_test[:,[0,3,4,7,9,11]] = sc.transform(x_test[:,[0,3,4,7,9,11]])
print(x_train[0])
print(x_test[0])
## Applying the Logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
## Predicting test results
y_pred = classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results
bins = np.linspace(-1,2,10)
plt.figure(figsize=(8,6))
ax =plt.hist([y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')
plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})
plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})
plt.xlim(-1,2)
plt.xticks([0,1])
plt.ylim(0,len(y_test))
plt.legend(prop={'size': 12})
plt.show()
## Calculating the accuracy score and confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
ac_LogReg = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
## Printing the accuracy score
print(ac_LogReg)
## Printing the confusion matrix
print(cm)
