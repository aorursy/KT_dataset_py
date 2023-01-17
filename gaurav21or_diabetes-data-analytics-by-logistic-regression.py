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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
#Loading the dataset

diabetes_data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')



#Print the first 5 rows of the dataframe.

diabetes_data.head()
# Descriptive statistics are very useful for initial exploration of the variables

diabetes_data.describe(include='all')
diabetes_data_copy = diabetes_data.copy(deep = True)

diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
# data.isnull() # shows a Diabetes_data_copy with the information whether a data point is null 

# Since True = the data point is missing, while False = the data point is not missing, we can sum them

# This will give us the total number of missing values feature-wise

diabetes_data_copy.isnull().sum()
p = diabetes_data.hist(figsize = (20,20))
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)

diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)

diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)

diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)

diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
#plotting after filling null values

p = diabetes_data_copy.hist(figsize = (20,20))
sns.pairplot(diabetes_data_copy, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")

plt.title("Pairplot of Variables by Outcome")
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diabetes_data.corr(), annot=True,cmap ='YlGnBu')  # seaborn has very simple solution for heatmap
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diabetes_data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
#independent variables

x = diabetes_data_copy[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]



#dependent variables

y = diabetes_data_copy['Outcome']
## Importing stats models for running logistic regression

import statsmodels.api as sm

## Defining the model and assigning Y (Dependent) and X (Independent Variables)

logit_model=sm.Logit(y,x)

## Fitting the model and publishing the results

result=logit_model.fit()

print(result.summary())
X1 = diabetes_data_copy[['Pregnancies','Glucose','BloodPressure']]

logit_model2 = sm.Logit(y,X1)

result2 = logit_model2.fit()

print(result2.summary2())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X1)
X1_scaled = pd.DataFrame(scaler.transform(X1),columns=['Pregnancies', 'Glucose', 'BloodPressure'])

X1_scaled.head()
# checking the balance of the data

diabetes_data_copy['Outcome'].unique()
diabetes_data_copy['Outcome'].value_counts()
#importing train_test_split

from sklearn.model_selection import train_test_split

X1 = diabetes_data_copy[['Pregnancies','Glucose','BloodPressure']]

X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size=0.25,random_state=42, stratify=y)
len(X_train), len(X_test), len(y_train), len(y_test)
#Importing 

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

Lreg = LogisticRegression(solver = 'lbfgs')

Lreg.fit(X_train, y_train.ravel())  #ravel( will return 1D array with all the input-array elements)
y_predict = Lreg.predict(X_test)

y_predict
y_predict_train = Lreg.predict(X_train)

y_predict_train
y_prob_train = Lreg.predict_proba(X_train)[:, 1]

y_prob_train.reshape(1,-1)
y_prob= Lreg.predict_proba(X_test)[:,1]

y_prob.reshape(-1,1)

y_prob
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_predict)

score
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_predict)

pd.crosstab(y_test.ravel(),y_predict.ravel(), rownames=['True'], colnames=['Predicted'], margins=True) # #ravel( will return 1D array with all the input-array elements)
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()

print('true negatives', tn)

print('false positive', fp)

print('false negative', fn)

print('true positive', tp)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
Accuracy = (tp+tn)/(tp+tn+fp+fn)

print('Accuracy {:0.2f}'.format(Accuracy))
Specificity = tn/(tn+fp)

print('Specificity {:0.2f}'.format(Specificity))
Sensitivity = tp/(tp+fn)

print('Sensitivity {:0.2f}'.format(Sensitivity))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

log_ROC_AUC1 = roc_auc_score(y_train, y_predict_train)

fpr1, tpr1, thresholds1 = roc_curve(y_train, y_prob_train)

roc_auc1 = auc(fpr1, tpr1)
plt.figure()

plt.plot(fpr1,tpr1, color = 'blue', label =  'ROC curve (area = %0.2f)'% roc_auc1)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')



plt.legend(loc='lower right')

plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)
print('Area under the roc curve : %f' % roc_auc)
import numpy as np 

i = np.arange(len(tpr)) #index for df

roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr':pd.Series(1-fpr, index=i), 'tf':pd.Series(tpr -(1-fpr), index=i), 'thresholds':pd.Series(thresholds, index=i)})

roc.iloc[(roc.tf-0).abs().argsort()[:1]]
fig, ax = plt.subplots()

plt.plot(roc['tpr'])

plt.plot(roc['1-fpr'], color = 'red')

plt.xlabel('1-false positive rate')

plt.ylabel('true positive rate')

plt.title('receiver operating characteristic')

ax.set_xticklabels([])
from sklearn.preprocessing import binarize

y_predict_class1 = binarize(y_prob.reshape(1, -1),0.341694)[0]

y_predict_class1
confusion_matrix_1 = confusion_matrix(y_test, y_predict_class1)

print(confusion_matrix_1)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict_class1))