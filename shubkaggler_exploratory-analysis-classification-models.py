# Loading packages



import numpy as np

import pandas as pd 

from sklearn import preprocessing

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.svm import SVC

import matplotlib.pyplot as plt



# Setting options for pandas

pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)



%matplotlib inline

plt.rcParams['figure.figsize'] = [15, 10]
# Importing data



data = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
# Analysing dimensions of the data

data.shape
# Preview of data



data.tail(5)
# Summary of data



data.info()
# Datatypes of columns



data.dtypes.sort_values(axis=0)
# Looking at missing values percentage for each column



(data.isnull().sum()/data.shape[0] * 100).sort_values(axis=0, ascending=False, inplace=False, kind='quicksort')
columns_all_missing = (data.isnull().sum()/data.shape[0] * 100).sort_values(axis=0, ascending=False, inplace=False, kind='quicksort').index.to_list()[:5]



# Dropping values with complete missingness

data.drop(columns = columns_all_missing, axis = 1, inplace = True)



# Dropping Patient ID column as it is unlikely to contain insight

data.drop(columns = 'Patient ID',axis = 1, inplace = True)
# Constant value columns



constant_cols = data.columns[data.nunique() <= 1].to_list()

data.drop(columns = constant_cols, axis = 1, inplace = True)
# Target variable



# It has no missingness (above)



# Checking if target is balanced or imbalanced

data['SARS-Cov-2 exam result'].value_counts()



# Target is imbalanced
# List of categorical columns separately



categorical_columns = list(data.select_dtypes(include=['object']).columns)



# Looking at the values in categorical columns, so as to be able to convert them to numerical later on

for each in categorical_columns:

    print(data[each].value_counts())

    print("\n")
# One-hot encoding would further increase dimensionality of data further, so will use integer encoding



data = data.replace({'detected': 1, 'not_detected':0})

data = data.replace({'positive': 1, 'negative': 0})

data = data.replace({'absent': 0, 'present': 1, 'not_done': 2})

data = data.replace({'clear': 0, 'cloudy': 1, 'lightly_cloudy': 2, 'altered_coloring': 3})

data = data.replace({'Não Realizado': 0})

data = data.replace({'normal': 0})

data = data.replace({'<1000': 1000})

data = data.replace({'Ausentes': 0, 'Urato Amorfo --+': 1, 'Urato Amorfo +++': 2, 'Oxalato de Cálcio -++': 3, 'Oxalato de Cálcio +++': 4})

data = data.replace({'yellow': 0, 'light_yellow': 1, 'citrus_yellow': 2, 'orange': 4})
# Columns: Urine - Leukocytes and Urine - pH can be converted to float datatype



data['Urine - Leukocytes'] = data['Urine - Leukocytes'].astype(float)

data['Urine - pH'] = data['Urine - pH'].astype(float)
# Reviewing datatypes again



data.dtypes.sort_values(axis=0)



# Columns are either int or float now
# Filling-in missing values with a different and distant value than present in dataset



data.fillna(-99999, inplace = True)
# Checking missingness



data.isna().sum()



# No missingness now
# Even without further domain knowledge we can still drop columns that would usually not impact diagnosis or the patient being affected



data.drop(columns = ['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis = 1, inplace = True)

# Preparing dataframes for modelling



x = data.drop(columns = 'SARS-Cov-2 exam result', axis = 1, inplace = False)



# The target variable

y = data['SARS-Cov-2 exam result']
column_names = x.columns.to_list()
# Normalising the data

x = preprocessing.normalize(x, norm='max', axis=0).reshape(x.shape)



# Converting to dataframe

x = pd.DataFrame(x, columns = column_names)



# Shuffling the data

x, y = shuffle(x, y, random_state = 100)  # shuffle the data
# Splitting data using 70% as training data and 30% as test data



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100, stratify = y)
# Inspecting the training and test data sets



print(X_test.shape)

print(X_train.shape)

print(y_test.shape)

print(y_train.shape)
# Fitting a Naive Bayes model



# Create a Gaussian Classifier

gnb = GaussianNB()



# Train the model using the training sets

gnb.fit(X_train, y_train)
# Predicting using the naive Bayes model



y_pred = gnb.predict(X_test)
# Analysing quality of classifier 



print("Accuracy of Naive Bayes classifier:",metrics.accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
# Fitting a Random forests classifier



# Create a Gaussian Classifier

rm_clf = RandomForestClassifier(n_estimators = 100, random_state=10)



# Train the model using the training sets y_pred=clf.predict(X_test)

rm_clf = rm_clf.fit(X_train,y_train)
# Making predictions

y_pred_randomforests = rm_clf.predict(X_test)
# Analysing quality of classifier 



print("Accuracy of Random Forests classifier:",metrics.accuracy_score(y_test, y_pred_randomforests ))

print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_randomforests ))
# Support Vector clasifier



svc = SVC(random_state=10)

svc.fit(X_train, y_train)
# Making predictions

y_pred_svc = svc.predict(X_test)
# Analysing quality of classifier 



print("Accuracy of Random Forests classifier:",metrics.accuracy_score(y_test, y_pred_svc ))

print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_svc ))
ax = plt.gca()

metrics.plot_roc_curve(rm_clf, X_test, y_test, ax=ax, alpha=0.8)

metrics.plot_roc_curve(gnb, X_test, y_test, ax=ax, alpha=0.8)

metrics.plot_roc_curve(svc, X_test, y_test, ax=ax, alpha=0.8)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic (ROC)')

plt.legend(loc="lower right")

plt.show()