# Import Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import pandas_profiling as pp

import seaborn as sns



# Other Libraries

from sklearn.model_selection import train_test_split



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



#Metrices

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import metrics
# Load the Data Set



df = pd.read_csv('../input/creditcard.csv')
# Check the Head of the Data Set

print(df.head())
# Check the name of the Columns

df.columns
# Check the Info of the Data set 

df.info()
# Check the Unique Values of the Data Set

df.nunique()
# Check the NULL Values of the Data 

df.isnull().sum()
# Check the Stats of the Data



df.describe(include ='all')
# Check the Class Imbalance of the Data 



df['Class'].value_counts()
# Histogram Visualtion of the Data



df.hist(bins=10, figsize=(20,15))

plt.show()
# find the number of prom samples so we can down sample our majority to it

yes = len(df[df['Class'] ==1])



# retrieve the indices of the non-prom and prom samples 

yes_ind = df[df['Class'] == 1].index

no_ind = df[df['Class'] == 0].index



# random sample the non-prom indices based on the amount of 

# promulent samples

new_no_ind = np.random.choice(no_ind, yes, \

replace = False)



# merge the two indices together

undersample_ind = np.concatenate([new_no_ind, yes_ind])



# get undersampled dataframe from the merged indices

undersampled_data = df.loc[undersample_ind]
# divide undersampled_data into features and is_promoted label



X = undersampled_data.loc[:, undersampled_data.columns != 'Class']

y = undersampled_data.loc[:, undersampled_data.columns == 'Class']
# Split the Data into Train and Test 



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
# Import Random Forest and Logistic Regression Models



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
# Logistic Regression 



lr = LogisticRegression()



lr.fit(X_train, y_train)



lr_Pred=lr.predict(X_test)



#print(metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, lr_Pred))
#Confusion Matrix

cnf_matrix=metrics.confusion_matrix(y_test, lr_Pred)

cnf_matrix



#Visualize confusion matrix using heat map



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
# Fir Random Forest Model



rf = RandomForestClassifier(n_estimators = 1000)



rf.fit(X_train, y_train)



rf_Pred=rf.predict(X_test)



#print(metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, rf_Pred))
#Confusion Matrix

cnf_matrix=metrics.confusion_matrix(y_test, rf_Pred)

cnf_matrix

#Visualize confusion matrix using heat map



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