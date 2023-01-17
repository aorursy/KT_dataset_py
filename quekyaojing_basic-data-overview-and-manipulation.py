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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.io import arff

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer as Imputer

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

import pickle

from sklearn.metrics import precision_score

import itertools

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score



pd.set_option('display.max_columns', 100)
df = pd.read_excel('/kaggle/input/safe-driver-prediction/IT_3.xlsx', na_values=['#NAME?'])
df.describe()
#30250 rows and 17 attribute

df.shape
#check if there are any duplicate values inside(seems to be none)

df.drop_duplicates()

df.shape
#9 attribute need to use pandas dummies, because that are categorical variable

df.info()
# Class count

count_class_1, count_class_0 = df.target.value_counts()



# Divide by class

df_class_0 = df[df['target'] == 0]

df_class_1 = df[df['target'] == 1]



#Under sampling the target value 1 

count_class_1



df.target.value_counts().plot(kind='bar', title='Count (target)');
df_class_1_under = df_class_1.sample(count_class_0, random_state= 2)



df_test_under = pd.concat([df_class_1_under, df_class_0], axis=0)

print('Random under-sampling:')

print(df_test_under.target.value_counts())



df_test_under.target.value_counts().plot(kind='bar', title='Count (target)')
#Assign df_test_under as new dataframe object

df = df_test_under
#Remove this comment if want to select attribute manually

#labels = ["Gender","EngineHP","credit_history","Years_Experience","Marital_Status","Vehical_type","Miles_driven_annually"]



# Remove 'id' and 'target' columns

labels = df.columns[2:]

X = df[labels]

y = df['target']

print(X.head(11))
todummy_list = ['Gender', 'Marital_Status', 'Vehical_type', 'Age_bucket', 'EngineHP_bucket', 'Years_Experience_bucket', 'Miles_driven_annually_bucket', 'credit_history_bucket', 'State']



# Function to dummy all the categorical variables used for modeling

def dummy_df(df, todummy_list):

    for x in todummy_list:

        dummies = pd.get_dummies(df[x], prefix=x,)

        df = df.drop(x, 1)

        df = pd.concat([df, dummies], axis=1)

    return df



X = dummy_df(X, todummy_list)
help(Imputer)
# Impute missing values using Imputer in sklearn.preprocessing



imp = Imputer(missing_values=np.nan, strategy='mean')

imp.fit(X)

X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)



# Now check again to see if you still have missing data

X.isnull().sum().sort_values(ascending=False).head()
# Separate the dataset become 70 percent trainning data and 30 percent testing dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_test.loc[4]
# Try With SVM classification



svmnonlinear_noPca = svm.NuSVC(kernel='rbf',gamma='scale', probability=True)

svmnonlinear_noPca.fit(X_train,y_train)

y_pred = svmnonlinear_noPca.predict(X_test)

confidence = svmnonlinear_noPca.predict_proba(X_test)



accuracy_nonlinearsvm = accuracy_score(y_test, y_pred)

recall_nonlinearsvm = recall_score(y_test, y_pred)

conf_mat_nonlinearsvm = confusion_matrix(y_true=y_test, y_pred=y_pred)

precision_nonlinearsvm = precision_score(y_test, y_pred)

f1_score_nonlinearsvm = f1_score(y_test, y_pred)



print("Accuracy Non-linear: %.2f%%" % (accuracy_nonlinearsvm * 100.0))

print("Recall: ", recall_nonlinearsvm)

print("Precision: ", precision_nonlinearsvm)

print("F1 Score: ", f1_score_nonlinearsvm)

print("Confusion Matric: ", conf_mat_nonlinearsvm)
# Confusion Matrix of the Non-linear SVM with PCA dimension Reduction



from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat_nonlinearsvm)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat_nonlinearsvm, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
# Try Modelling with data preprocessing

# Use MinMaxScaler for normalize our data to range 0 to 1



cs = MinMaxScaler()

# Fit on training set only.

X_train = cs.fit_transform(X_train)

X_test = cs.transform(X_test)



X_train
# Apply PCA dimension reduction technique



pca = PCA(n_components=10)



pca.fit(X_train)

pca.n_components_



X_train = pca.transform(X_train)

X_test = pca.transform(X_test)



X_train
# Try with Non Linear SVM Model with PCA dimension reduction



svmnonlinear_PCA = svm.NuSVC(kernel='poly',gamma='scale', probability=True)

svmnonlinear_PCA.fit(X_train,y_train)

y_pred = svmnonlinear_PCA.predict(X_test)

print(y_pred)



accuracy_nonlinearsvm = accuracy_score(y_test, y_pred)

recall_nonlinearsvm = recall_score(y_test, y_pred)

conf_mat_nonlinearsvm = confusion_matrix(y_true=y_test, y_pred=y_pred)

precision_nonlinearsvm = precision_score(y_test, y_pred)

f1_score_nonlinearsvm = f1_score(y_test, y_pred)



print("Accuracy Non-linear: %.2f%%" % (accuracy_nonlinearsvm * 100.0))

print("Recall: ", recall_nonlinearsvm)

print("Precision: ", precision_nonlinearsvm)

print("F1 Score: ", f1_score_nonlinearsvm)

print("Confusion Matric: ", conf_mat_nonlinearsvm)
# Confusion Matrix of the Non-linear SVM with PCA dimension Reduction



from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat_nonlinearsvm)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat_nonlinearsvm, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()