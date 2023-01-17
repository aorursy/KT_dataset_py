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
#Step 1: Import Library

# Import data analysis modules

import numpy as np

import pandas as pd

import os

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

Glass = pd.read_csv(('/kaggle/input/glass/glass.csv'))

Glass.head(5)

Glass.dtypes
Glass['Type'].unique()
#count of the target variable

sns.countplot(x='Type', data=Glass)
sns.boxplot('Type', 'RI', data =Glass)
missing_values = Glass.isnull()

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
Glass.describe()




Glass['Type'].value_counts()



#The dataset is  unbalanced.  types 1 and 2 have more than 67 % of the glass types
#Data Visualization

#Univariate plots

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

for feat in features:

    skew = Glass[feat].skew()

    sns.distplot(Glass[feat], label='Skew = %.3f' %(skew))

    plt.legend(loc='best')

    plt.show()
#We will check correlation of values using Feature Matrix

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']



mask = np.zeros_like(Glass[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Correlation Matrix',fontsize=25)

sns.heatmap(Glass[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='b',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
#RI- 25% - 1.51 and max - 1.53. Not much of a variance. Could explain the huge number of outliers

#Na,Mg,AI,Si same explanation as above

#K- min - 0.0 and max - 6.21! big difference. Could explain the outliers.

#Ca,Ba,fe same explanation as above

#This column will contain the values of 1 and 0. 

#Bad , Good #This will be split in the following way. #1,3 --> Bad #5,7 --> Excellent 
# glass 1, 2, 3 are Bad glass

# glass 5, 6, 7 are Good glass

Glass['household'] = Glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

Glass.head()
plt.scatter(Glass.Al, Glass.household)

plt.xlabel('Al')

plt.ylabel('household')
# Plot logistic regression line 

sns.regplot(x='Al', y='household', data=Glass, logistic=True, color='b')
plt.scatter(Glass.Ba, Glass.household)

plt.xlabel('Ba')

plt.ylabel('household')

# Plot logistic regression line 

sns.regplot(x='Na', y='household', data=Glass, logistic=True, color='b')
# Plot logistic regression line 

sns.regplot(x='Ba', y='household', data=Glass, logistic=True, color='b')
# Import module to split dataset

from sklearn.model_selection import train_test_split



#Independent variable

X = Glass[['Al','Ba']]

#Dependent variable

y = Glass['household']

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)
# Import model for fitting

from sklearn.linear_model import LogisticRegression



# Create instance (i.e. object) of LogisticRegression

model = LogisticRegression(class_weight='balanced')

output=model.fit(X_train, y_train)

output
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)



#Confusion matrix

results = confusion_matrix(y_test, y_pred)

print(results)



#Accuracy score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy rate : {0:.2f} %".format(100 * accuracy))



#Classification report

report = classification_report(y_test, y_pred)

print(report)
pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)







Ypredict = pickle_model.predict(X_test)

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

y_predict
import matplotlib.pyplot as plt

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_predict)

cnf_matrix

%matplotlib inline

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