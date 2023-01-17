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
#Import dataset

train_data=pd.read_csv('/kaggle/input/loan-default-prediction/train_v2.csv.zip')
#Check data shape

train_data.shape
#Check information regarding variables

train_data.info()
#Descriptive stats of the variables

train_data.describe()
#Selecting categorical variables from training set

cat_train_data=train_data.select_dtypes(include=['object'])

print(cat_train_data.head())
#We will drop categorical variables as they have very large values

#Selecting numerical variables

num_data=train_data.select_dtypes(include=['float64', 'int64'])

print(num_data.columns)

print(num_data.head())
#Checking missing values

num_data.isnull().sum()
#Taking care of missing data

#num_data.replace(0,np.nan, inplace=True) 

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy = 'mean')

imputer=imputer.fit(num_data)

num_data_imputed=pd.DataFrame(imputer.transform(num_data))

num_data_imputed.columns=num_data.columns

num_data_imputed.isnull().sum()
#Remove Duplicate variables to avoid double counting of data

num_data_imputed=num_data_imputed.drop_duplicates()

num_data_imputed.shape

num_data_imputed.isnull().sum()
#Drop id variable

num_data_imputed.drop('id', axis=1)
#Find correlation between features drop loss variable

loss=num_data_imputed['loss'].copy()

num_data_imputed.drop(columns='loss', inplace=True)
#Correlation Matrix

corr_matrix = num_data_imputed.corr().abs()
#Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#Find features with correlation greater than 0.8

num_data_imputed_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

len(num_data_imputed_drop)

print('%d columns are highly correlated and to be removed.' % (len(num_data_imputed_drop)))

#Drop features 

X_data_train = num_data_imputed.drop(columns = num_data_imputed_drop)

X_data_train.shape
#Correlation between features and target variable

corr_mat=num_data_imputed.corrwith(loss).sort_values()

print(corr_mat.head(30), '/n')

print(corr_mat.tail(30))

Y_data_train = [column for column in X_data_train.columns if len(X_data_train[column].unique())==1]

print('%d columns have NaN values and to be removed.' % (len(Y_data_train)))
#Drop the highly correlated variables

X_train = X_data_train.drop(columns = Y_data_train)

print('Shape of feature variable: ', X_train.shape)

X_train.isnull().sum()
Y=loss.copy()

Y.value_counts()
Y[Y>0]=1

print(Y.value_counts())
#Creating Y train variable for modelling

Y=Y.copy()
#Splitting data for modelling

from sklearn.model_selection import train_test_split

X_train_lr,X_test_lr,y_train_lr,y_test_lr=train_test_split(X_train, Y, test_size=0.25,random_state=42)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

SC_X = StandardScaler()

X_train_lr = SC_X.fit_transform(X_train_lr)

X_test_lr = SC_X.transform(X_test_lr)
# Convert Y to one-dimensional array

y_train_lr = np.array(y_train_lr).reshape((-1, ))

y_test_lr = np.array(y_test_lr).reshape((-1, ))
#Fitting Logistic Regression to training set

from sklearn.linear_model import LogisticRegressionCV

classifier = LogisticRegressionCV(random_state=0).fit(X_train_lr, y_train_lr)
#Predicting the test results

y_pred=classifier.predict(X_test_lr)

#print('Logistic Regression Performance on training set = %0.0f'%round((classifier.score(X_train_lr,y_train_lr))))

print('Logistic Regression Performance on training set =', classifier.score(X_train_lr,y_train_lr))

print('Logistic Regression Performance on test set =', classifier.score(X_test_lr,y_test_lr))
#Making the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_lr,y_pred)

print(cm)
#Get F1_Score for test set

from sklearn.metrics import f1_score

F1_score_test_LR=round(f1_score(y_test_lr, y_pred, average='macro'),4)

print('F1_Score for test set: ',F1_score_test_LR )
#Random Forest Classification

#Splitting the data for modelling

from sklearn.model_selection import train_test_split

X_train_rf,X_test_rf,y_train_rf,y_test_rf=train_test_split(X_train, Y, test_size=0.25,random_state=0)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

SC_Y = StandardScaler()

X_train_rf = SC_Y.fit_transform(X_train_rf)

X_test_rf = SC_Y.transform(X_test_rf)
# Convert Y to one-dimensional array

y_train_rf = np.array(y_train_rf).reshape((-1, ))

y_test_rf = np.array(y_test_rf).reshape((-1, ))
#Fitting Classifier to the training set

from sklearn.ensemble import RandomForestClassifier

Classifier=RandomForestClassifier(n_estimators=15, criterion='entropy', random_state=42)

Classifier.fit(X_train_rf, y_train_rf)
#Predicting the test results

y_pred_rf=Classifier.predict(X_test_rf)
#Making Confusion Matrix

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test_rf, y_pred_rf)

print(cm)
print('Random Forest Classification Performance on training set =', classifier.score(X_train_rf,y_train_rf))

print('Random Forest Classification Performance on test set = ', classifier.score(X_test_rf,y_test_rf))
from sklearn.metrics import f1_score

F1_score_test_RF=round(f1_score(y_test_rf, y_pred_rf, average='macro'),4)

print('F1_Score for test set: ',F1_score_test_RF )
#Import test dataset

test_data=pd.read_csv('/kaggle/input/loan-default-prediction/test_v2.csv.zip')

test_data.shape
test_data.describe()
test_data.head()
X = X_train

X.columns
test_data.isnull().sum()
test_feat=test_data[X.columns]
test_feat.shape
#Taking care of missing data

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy = 'mean')

imputer=imputer.fit(test_feat)

test_feat_imputed=pd.DataFrame(imputer.transform(test_feat))

test_feat_imputed.columns=test_feat.columns

test_feat_imputed.isnull().sum()
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

test_feat=sc.fit_transform(test_feat_imputed)
# The F1 score of both the models are approximately same so we choose Logistic Regression for prediction
#Predicting the test set

test_predicted=classifier.predict(test_feat)
test_pred_data=pd.DataFrame({'default': test_predicted})
sample_submission  = pd.read_csv("/kaggle/input/loan-default-prediction/sampleSubmission.csv")
sample_submission.id=test_data.id

sample_submission.loss=test_pred_data.default
Submission_LDP=sample_submission

Submission_LDP.head()
Submission_LDP=Submission_LDP.to_csv(index=False)
import os 

os.chdir(r'/kaggle/working')
Submission_LDP.to_csv(r'Submission.csv', index=False)
from IPython.display import FileLink

FileLink(r'Submission.csv')