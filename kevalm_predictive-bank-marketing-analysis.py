import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn import metrics

import statsmodels.formula.api as smf

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder





#Classification Algorithms 

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn import metrics as m

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import roc_auc_score



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/bank.csv", delimiter=";",header='infer')

data.head()
sns.pairplot(data)
data.corr()
sns.heatmap(data.corr())
data.dtypes
data_new = pd.get_dummies(data, columns=['job','marital',

                                         'education','default',

                                         'housing','loan',

                                         'contact','month',

                                         'poutcome'])
#Class column into binary format

data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)
#Successfully converted data into  integer data types

data_new.dtypes
#Whole dataset's shape (ie (rows, cols))

print(data.shape)
#Unique education values

data.education.unique()
#Crosstab to display education stats with respect to y ie class variable

pd.crosstab(index=data["education"], columns=data["y"])
#Education categories and there frequency

data.education.value_counts().plot(kind="barh")
from xgboost import XGBClassifier

classifiers = {

               'Adaptive Boosting Classifier':AdaBoostClassifier(),

               'Linear Discriminant Analysis':LinearDiscriminantAnalysis(),

               'Logistic Regression':LogisticRegression(),

               'Random Forest Classifier': RandomForestClassifier(),

               'K Nearest Neighbour':KNeighborsClassifier(8),

               'Decision Tree Classifier':DecisionTreeClassifier(),

               'Gaussian Naive Bayes Classifier':GaussianNB(),

               'Support Vector Classifier':SVC(),

               }
#Due to one hot encoding increase in the number of columns

data_new.shape
data_y = pd.DataFrame(data_new['y'])

data_X = data_new.drop(['y'], axis=1)

print(data_X.columns)

print(data_y.columns)
log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]

log = pd.DataFrame(columns=log_cols)
import warnings

warnings.filterwarnings('ignore')

rs = StratifiedShuffleSplit(n_splits=2, test_size=0.3,random_state=2)

rs.get_n_splits(data_X,data_y)

for Name,classify in classifiers.items():

    for train_index, test_index in rs.split(data_X,data_y):

        #print("TRAIN:", train_index, "TEST:", test_index)

        X,X_test = data_X.iloc[train_index], data_X.iloc[test_index]

        y,y_test = data_y.iloc[train_index], data_y.iloc[test_index]

        # Scaling of Features 

#         from sklearn.preprocessing import StandardScaler

#         sc_X = StandardScaler()

#         X = sc_X.fit_transform(X)

#         X_test = sc_X.transform(X_test)

        cls = classify

        cls =cls.fit(X,y)

        y_out = cls.predict(X_test)

        accuracy = m.accuracy_score(y_test,y_out)

        precision = m.precision_score(y_test,y_out,average='macro')

        recall = m.recall_score(y_test,y_out,average='macro')

        #roc_auc = roc_auc_score(y_out,y_test)

        f1_score = m.f1_score(y_test,y_out,average='macro')

        log_entry = pd.DataFrame([[Name,accuracy,precision,recall,f1_score,roc_auc]], columns=log_cols)

        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)

        log = log.append(log_entry)

        #metric = metric.append(metric_entry)

        

print(log)

plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')

sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="g")  

plt.show()



#Scroll complete output to view all the accuracy scores and bar graph.
#Divide records in training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=2, stratify=data_y)

print (X_train.shape)

print (X_test.shape)

print (y_train.shape)

print (y_test.shape)
#Create an Logistic classifier and train it on 70% of the data set.

from sklearn import svm

from xgboost import XGBClassifier

clf = LogisticRegression()

clf
#Fiting into model

clf.fit(X_train, y_train)
#Prediction using test data

y_pred = clf.predict(X_test)
#classification accuracy

from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred))
#Predictions

predictions = clf.predict(X_test)
# Imports

from sklearn.metrics import confusion_matrix, classification_report



# Confusion matrix

print(confusion_matrix(y_test, predictions))



# New line

print('\n')



# Classification report

print(classification_report(y_test,predictions))