import pandas as pd
import numpy as np
import math
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


import xgboost
# Load data
data_path = "no-show-appintment-predictor-data.csv"
df = pd.read_csv(data_path)
# Grab a peek at the data
df.head(10)
# Describe the shape of the data sample
print(df.shape)
#Check for duplicated records
sum(df.duplicated())
df.info()
#Display the detailed information of each and every attributes
df.describe().transpose()
#Convert date type column objects to datetime index
column_list = ['ScheduledDay', 'AppointmentDay']
for col in column_list:
  df[col] = pd.to_datetime(df[col])

#After converting, first five rows of the data
df.head(5)
#Target variable('No-show') summary
df['No-show'].value_counts()
#Convert Target variable data values to 1 and 0
Enc = LabelEncoder()
df['No-show']=Enc.fit_transform(df['No-show'])
df['Gender']=Enc.fit_transform(df['Gender'])
#Target variable('No-show') summary
df['No-show'].value_counts()
#Extract the valid age rows
#Check for invalid age rows
df.query('Age < 0')
#Drop the particular invalid row
df.drop(99832, inplace=True)
#Extract the valid age rows
#Check for invalid age rows
df.query('Age < 0')
#Check for Handcap > 1
df.query('Handcap > 1')
#Drop Handcap > 1 rows
df.drop(df.query('Handcap > 1').index, inplace=True)
df.query('Handcap > 1')
#Drop unwanted columns('PatientId','AppointmentID')
df.drop(['PatientId','AppointmentID'], axis = 1)

df.info()
#Rename the column names by converting to lowercase and replace '-' by '_'
df.rename(columns=lambda x: x.lower().replace('-', '_'), inplace=True)

df.columns
#Calculate the date difference betweeb scheduled date and appointment date for future predictions
df['date_gap'] = abs(df.appointmentday - df.scheduledday).astype('timedelta64[D]')

df.columns
df.head(-5)
# Correlation matrix
corr_matrix = df.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corr_matrix, vmax = .8, square = True)
plt.show()
#Encoding data for machine learning modeling
#Creating vars to hold categorical features for one hot encoding
cat_features_for_encoding = ['handcap', 'neighbourhood']
encoded_df = pd.get_dummies(df, columns = cat_features_for_encoding)
encoded_df.head(5)
features = encoded_df.drop(['patientid', 'appointmentid', 'scheduledday','appointmentday','no_show'],axis=1)
labels = encoded_df['no_show']

print(labels.head())
features.head()
#Scalling
stdsc = StandardScaler()
features_std = stdsc.fit_transform(features)
features_std 
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features_std, labels, test_size = 0.2, random_state = 1, stratify = labels)
#Common function to display scores of model for a given train/test data
def modelscores(model, x_train,x_test, y_train, y_test, x_std, y_out):
  scores = cross_val_score(estimator=model, X=x_std, y=y_out, cv = 10, n_jobs = 1)
  print('CV Accuracy : %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
  model.fit(x_train, y_train)
  print('Accuracy: %.3f' % model.score(x_test, y_test))
pipe_lr = make_pipeline(PCA(), LogisticRegression(random_state=0))
print(pipe_lr.get_params().keys())
param_range = [.1, .5, 1, 10]


def lrpca(x_train, y_train):
  n_components = np.arange(7,x_train.shape[1]+1,1)
  param_grid = [{'pca__n_components': n_components, 'logisticregression__C': param_range}]
  grid_search_lr = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='accuracy', cv = 10)

  grid_search_lr.fit(x_train, y_train)
  print('*********Logistic Regression*********')
  print('The best score: %.3f' % (grid_search_lr.best_score_))
  print('The best params: ', grid_search_lr.best_params_)

  lreg_best = grid_search_lr.best_estimator_
  return lreg_best


lr_bestmodel = lrpca(train_features, train_labels)
# Save the model using pickle
file_name_lr = ""
pickle.dump(lr_bestmodel, open(file_name_lr, 'wb'))
# Load the saved model to do the predictions
model_loaded_lr = pickle.load(open(file_name_lr, 'rb'))
modelscores(model_loaded_lr, train_features, test_features, train_labels, test_labels, features_std, labels)
lr_predict = model_loaded_lr.predict(test_features)

from sklearn.metrics import confusion_matrix 
mat = confusion_matrix(test_labels, lr_predict)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False) 
plt.xlabel('true class') 
plt.ylabel('predicted class')
from sklearn import metrics
print("Classification Report : \n\n", metrics.classification_report(lr_predict, test_labels))
forest = RandomForestClassifier(random_state=1)

param_grid = { 'n_estimators':[5,10,25,50], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[2,3,4,5,6,7,8], 'criterion':['gini', 'entropy']}
grid_search_rf = GridSearchCV(estimator=forest, param_grid=param_grid, scoring='accuracy', cv = 10, n_jobs=2)

grid_search_rf = grid_search_rf.fit(train_features, train_labels)
print('*********Random Forest*********')
print('The best score: %.3f' % (grid_search_rf.best_score_))
print('The best params: ', grid_search_rf.best_params_)
print('*******************************')
grid_search_rf_best = grid_search_rf.best_estimator_
file_name_rf = ""
pickle.dump(grid_search_rf_best, open(file_name_rf, 'wb'))
# Load the saved model to do the predictions
model_loaded_rf = pickle.load(open(file_name_rf, 'rb'))
modelscores(model_loaded_rf, train_features, test_features, train_labels, test_labels, features_std, labels)
rf_predict = model_loaded_rf.predict(test_features)

from sklearn.metrics import confusion_matrix 
mat = confusion_matrix(test_labels, rf_predict)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False) 
plt.xlabel('true class') 
plt.ylabel('predicted class')

from sklearn import metrics
print("Classification Report : \n\n", metrics.classification_report(rf_predict, test_labels))
tree_stump = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree_stump, n_estimators=500, random_state=1)

tree_stump = tree_stump.fit(train_features, train_labels)

y_train_pred = tree_stump.predict(train_features)
y_test_pred = tree_stump.predict(test_features)

tree_train = accuracy_score(train_labels, y_train_pred)
tree_test = accuracy_score(test_labels, y_test_pred)

print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
def adaModel(x_train, y_train):
  param_grid = [{'learning_rate':[0.05,0.1,0.2]}]
  grid_search_ada = GridSearchCV(estimator=ada, param_grid=param_grid, scoring='accuracy', cv = 10)

  grid_search_ada.fit(x_train, y_train)
  print('*********For AdaBoost using Tree Trumps*********')
  print('The best score: %.3f' % (grid_search_ada.best_score_))
  print('The best params: ', grid_search_ada.best_params_)

  ada_best = grid_search_ada.best_estimator_
  return ada_best

ada_bestmodel = adaModel(train_features, train_labels)
file_name_ada = ""
pickle.dump(ada_bestmodel, open(file_name_ada, 'wb'))
# Load the saved model to do the predictions
model_loaded_ada = pickle.load(open(file_name_ada, 'rb'))
modelscores(model_loaded_ada, train_features, test_features, train_labels, test_labels, features_std, labels)
ada_predict = model_loaded_ada.predict(test_features)

from sklearn.metrics import confusion_matrix 
mat = confusion_matrix(test_labels, ada_predict)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False) 
plt.xlabel('true class') 
plt.ylabel('predicted class')
from sklearn import metrics
print("Classification Report : \n\n", metrics.classification_report(ada_predict, test_labels))