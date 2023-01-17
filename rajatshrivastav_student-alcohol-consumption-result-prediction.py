

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Importing all the requirerd libraries

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC               # SVM modelling 

from sklearn.metrics import *             # Calculating the metrics for model 

import matplotlib.pyplot as plt           # Data Visualization

import seaborn as sns                     # Data Visualization 

from scipy.stats import chi2_contingency  # Working with categorical data

from scipy.stats import kurtosis, skew    # Calculate Skew and Kurtosis 

from sklearn.preprocessing import LabelEncoder # For Label Encoding 

from sklearn.preprocessing import MinMaxScaler # For Normalization
#import both the train and test data

train = pd.read_csv("../input/School_train_data.csv")

test = pd.read_csv("../input/School_test_user.csv")
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
print(train.shape)

train.head()
print(test.shape)

test.head()
train.info()
test.info()
train.columns
test.columns
data = pd.concat([train,test],join = 'outer')

data.head()

data123 = data
data.shape
#repalce NaN in Result with fail

data['Result'] = data['Result'].replace(np.nan,'FAIL',regex = True)
data.isnull().sum()
data['AVGALC'] = (((data['Dalc'])*5 + (data['Walc'])*2)/7).round()
data = data[['id','school','sex','age','address','famsize','Pstatus','Medu',

       'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',

       'failures', 'schoolsup','famsup', 'paid', 'activities', 'nursery',

       'higher', 'internet','romantic', 'famrel', 'freetime','goout','Dalc',

       'Walc', 'health', 'absences', 'AVGALC','Result']]
display(data)
#data.drop(['AVGALC'], axis=1,inplace = True)
numerical = ['age','absences']
categorical = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu',

       'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',

       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',

       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',

       'Walc', 'health','AVGALC','Result',]
for col in categorical:

    data[col] = data[col].astype('category')
for col in numerical:

    data[col] = data[col].astype('float64')
data.info()
data.describe(include = [np.number])
print('Skewness for numerical columns: ')

print(data.skew())      

print('Kurtosis for numerical columnms: ')

print(data.kurtosis())
def f(x):

    print('Skew: ',skew(x))

    print('Kurtosis:  ',kurtosis(x))

data[numerical].apply(f,axis = 0)
plt.hist(data['absences'],bins =40,density = True)

plt.title('Histogram of Absences',size=14, fontweight='semibold')

plt.show()
sns.distplot(data['absences'], hist=True, kde=True, 

             bins=25, color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 1})
sns.set_style("whitegrid") 

sns.boxplot(data['absences'],orient = 'v')
data['absences'] = np.log(data['absences'] + 1).round(2)
sns.distplot(data['absences'], hist=True, kde=True, 

             bins=25, color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 1})
sns.set_style("whitegrid") 

sns.boxplot(data['absences'],orient = 'v')
factors_paired = [(i,j) for i in categorical for j in categorical]

chi2, p_values = [], []

for f in factors_paired:

    if f[0] != f[1]:

        chitest = chi2_contingency(pd.crosstab(data[f[0]], data[f[1]]))

        chi2.append(chitest[0])

        p_values.append(chitest[1])

    else:

        chi2.append(0)

        p_values.append(0)

        

p1 = np.array(p_values).reshape((30,30))

p1 = pd.DataFrame(p1.round(2), index=categorical, columns=categorical)

p1
data.drop(['id'],axis = 1,inplace = True)
data.corr().unstack().sort_values().drop_duplicates()
#Label Encoding for the values

# target column

tgt_col = ['Result']

# Categorical cols



category_names = data.nunique()[data.nunique() < 7].keys().tolist()

category_names = [x for x in category_names if x not in tgt_col]



#numerical Cols

num_cols = [i for i in data.columns if i not in category_names + tgt_col]



#Binary cols

bin_cols = data.nunique()[data.nunique()==2].keys().tolist()



# Multi-cols

multi_cols = [i for i in category_names if i not in bin_cols]



#label Encoding

le = LabelEncoder()

for i in bin_cols:

    data[i] = le.fit_transform(data[i])

     

# Duplicating cols for multi-value columns

data = pd.get_dummies(data=data, columns=multi_cols)

data.head()
#normalizing features

cont_features = []

for features in data.select_dtypes(include=['float64']):

    cont_features.append(features)

minmax = MinMaxScaler()

data[cont_features] = minmax.fit_transform(data[cont_features].values)
data.head()
cols = [i for i in data.columns if i not in 'Result']

X = data[cols]

Y = pd.DataFrame(data['Result'])
X.columns
Y.columns
X_train = X.iloc[:993,:]

Y_train = Y.iloc[:993,:]

X_test = X.iloc[993:,:]

Y_test = Y.iloc[993:,:]
#Having a Look into the data for modelling

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
data.head()
orgtrain = data.iloc[:993,:]

#Make Predictions on X_test

#We don't have Y_test as we need to predict that

orgtrain.head()

cols = [i for i in orgtrain.columns if i not in 'Result']

X1 = orgtrain[cols]

Y1 = pd.DataFrame(orgtrain['Result'])

print(X1.columns)

print(Y1.columns)
from sklearn.model_selection import train_test_split

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1,Y1,random_state=42,stratify=Y1)

#Having a Look into the data for modelling

print(X_train1.shape)

print(Y_train1.shape)

print(X_test1.shape)

print(Y_test1.shape)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train1, Y_train1)

predicted12 = model.predict(X_test1)

# Accuracy Score

from sklearn import metrics

print("Accuracy Score of Decision Tree Classifier : ",metrics.accuracy_score(Y_test1, predicted12))
#Hyper parameter Tunning

from sklearn.model_selection import GridSearchCV

sample_split_range = range(10,50,100)

param_grid = dict(min_samples_split=sample_split_range)



# Instantiate the grid

grid_dtc = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')

#Fitting model on hyper parameter Tuning

grid_dtc.fit(X_train1,Y_train1)

# Prediction

grid_pred = grid_dtc.predict(X_test1)

# Accuracy Score

print("Accuracy Score of Decision Tree Classifier on Hyper-tuning : ",metrics.accuracy_score(Y_test1, grid_pred))
# Load scikit's random forest classifier library

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')

clf.fit(X_train1, Y_train1)

predicted = clf.predict(X_test1)

from sklearn import metrics

print("Accuracy Score of Decision Tree Classifier : ",metrics.accuracy_score(Y_test1, predicted))

from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train1, Y_train1)

y_pred = svclassifier.predict(X_test1)

print("Accuracy Score of Decision Tree Classifier : ",metrics.accuracy_score(Y_test1, y_pred))
from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf')

svclassifier.fit(X_train1, Y_train1)

y_pred = svclassifier.predict(X_test1)

print("Accuracy Score of Decision Tree Classifier : ",metrics.accuracy_score(Y_test1, y_pred))
from sklearn import svm

from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='linear', C=1)

scores = cross_val_score(clf, X_train1, Y_train1, cv=5)

print(scores)
from sklearn import metrics

scores = cross_val_score(clf, X_train1, Y_train1, cv=5, scoring='f1_macro')

print(scores)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=20)

classifier.fit(X_train1, Y_train1)

y_pred = classifier.predict(X_test1)

print("Accuracy Score of Decision Tree Classifier : ",metrics.accuracy_score(Y_test1, y_pred))
from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=23)

scores = cross_val_score(classifier, X_train1, Y_train1, cv=5)

print(scores)
from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=20)

scores = cross_val_score(classifier, X_train, Y_train, cv=5)

print(scores)
from xgboost import XGBClassifier

from sklearn.metrics import *

clf1 = XGBClassifier()

clf1.fit(X_train1, Y_train1)

y1 = clf1.predict(X_test1)

print("Accuracy: {:.3f}".format(accuracy_score(Y_test1, y1)))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf1, X_train1, Y_train1, cv=5)

scores1 = cross_val_score(clf1,X_train1,Y_train1,cv = 5,scoring='f1_macro')

print(scores)

print(scores1)
from xgboost import XGBClassifier

from sklearn.metrics import *

clf2 = XGBClassifier()

clf2.fit(X_train, Y_train)

y2 = clf2.predict(X_test)

#print("Accuracy: {:.3f}".format(accuracy_score(Y_test1, y2)))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf2, X_train, Y_train, cv=5)

scores2 = cross_val_score(clf2,X_train,Y_train,cv = 5,scoring='f1_macro')

print(scores)

print(scores2)
from sklearn import svm

from sklearn.model_selection import cross_val_score

clf3 = svm.SVC(kernel='linear', C=1)

scores = cross_val_score(clf3, X_train, Y_train, cv=5)

scores3 = cross_val_score(clf3,X_train,Y_train,cv = 5,scoring='f1_macro')

print(scores)

print(scores3)
from sklearn import svm

from sklearn.model_selection import cross_val_score

clf4 = svm.SVC(kernel='rbf', C=1)

scores = cross_val_score(clf4, X_train, Y_train, cv=5)

y_prednew = svclassifier.predict(X_test)

scores4 = cross_val_score(clf4,X_train,Y_train,cv = 5,scoring='f1_macro')

print(scores)

print(scores4)