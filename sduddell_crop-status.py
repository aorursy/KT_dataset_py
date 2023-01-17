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

train_data = pd.read_excel("/kaggle/input/crop-status-classifier/train.xlsx")
test_data = pd.read_excel("/kaggle/input/crop-status-classifier/test.xlsx")

print(train_data.head())
print(train_data.shape)
print(test_data.shape)
print(test_data.head)

target = train_data['Crop_status']
print(target.value_counts())
print(train_data.dtypes)
train_data = train_data.drop("ID", axis=1)
test_data = test_data.drop("ID",axis=1)
#Checking for NULL values
print(train_data["Insects"].value_counts())
print(train_data["Insects"].isnull().values.any())
print(test_data["Insects"].isnull().values.any())

###########################################

print(train_data["Crop"].value_counts())
print(train_data["Crop"].isnull().values.any())
print(test_data["Crop"].isnull().values.any())

###########################################

print(train_data["Soil"].value_counts())
print(train_data["Soil"].isnull().values.any())
print(test_data["Soil"].isnull().values.any())

###########################################

print(train_data["Category_of_Toxicant"].value_counts())
print(train_data["Category_of_Toxicant"].isnull().values.any())
print(test_data["Category_of_Toxicant"].isnull().values.any())

###########################################

print(train_data["Does_count"].value_counts())
print(train_data["Does_count"].isnull().values.any())
print(test_data["Does_count"].isnull().values.any())

#########################################

print(train_data["Number_of_Weeks_Used"].value_counts())
print(train_data["Number_of_Weeks_Used"].isnull().values.any())
print(test_data["Number_of_Weeks_Used"].isnull().values.any())
from sklearn.preprocessing import LabelEncoder
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()

train_data["Crop"] = encoder1.fit_transform(train_data[["Crop"]])
test_data["Crop"] = encoder1.transform(test_data[["Crop"]])
train_data["Soil"] = encoder2.fit_transform(train_data[["Soil"]])
test_data["Soil"] = encoder2.transform(test_data[["Soil"]])

print(train_data.dtypes)
print(test_data.dtypes)
print(train_data["Crop"].value_counts())
print(test_data["Crop"].value_counts())
#Imputing missing values
from sklearn.impute import KNNImputer
impute = KNNImputer(n_neighbors=5, weights="uniform")
train_data["Number_of_Weeks_Used"]= impute.fit_transform(train_data[["Number_of_Weeks_Used"]])
test_data["Number_of_Weeks_Used"]= pd.DataFrame(impute.transform(test_data[["Number_of_Weeks_Used"]]))

#Adding new param dose_count*Num_weeks_used
train_data["Hybrid"] = train_data["Does_count"]*train_data["Number_of_Weeks_Used"]
test_data["Hybrid"] = test_data["Does_count"]*test_data["Number_of_Weeks_Used"]
#Adding new param weeks not used/Num_weeks_used
train_data["Proportion"] = train_data["Number_Weeks_does_not used"]/train_data["Number_of_Weeks_Used"]
test_data["Proportion"] = test_data["Number_Weeks_does_not used"]/test_data["Number_of_Weeks_Used"]
#Imputing missing values
from sklearn.impute import KNNImputer
impute_1 = KNNImputer(n_neighbors=4, weights="uniform")
train_data["Proportion"]= impute_1.fit_transform(train_data[["Proportion"]])
test_data["Proportion"]= pd.DataFrame(impute_1.transform(test_data[["Proportion"]]))

#exponential for cateogry of toxicant
train_data["Exponential"] = np.exp(train_data["Category_of_Toxicant"])
test_data["Exponential"] = np.exp(test_data["Category_of_Toxicant"])
train_data = train_data.drop(["Category_of_Toxicant"], axis=1)
test_data = test_data.drop(["Category_of_Toxicant"], axis=1)
#exponential for cateogry of toxicant
train_data["crop_Exponential"] = np.exp(train_data["Crop"])
test_data["crop_Exponential"] = np.exp(test_data["Crop"])
train_data = train_data.drop(["Crop"], axis=1)
test_data = test_data.drop(["Crop"], axis=1)
#exponential for cateogry of toxicant
train_data["soil_Exponential"] = np.exp(train_data["Soil"])
test_data["soil_Exponential"] = np.exp(test_data["Soil"])
train_data = train_data.drop(["Soil"], axis=1)
test_data = test_data.drop(["Soil"], axis=1)
print(train_data["Number_Weeks_does_not used"].value_counts())
print(train_data["Number_Weeks_does_not used"].isnull().values.any())

##############################################################

print(train_data["Season"].value_counts())
print(train_data["Season"].isnull().values.any())

#############################################################

print(train_data.isnull().values.any())
print(test_data.isnull().values.any())

target = train_data['Crop_status']
print(target.value_counts())
print(train_data["Crop"].unique())
print(train_data["Soil"].unique())
#treating outliers
import seaborn as sns

sns.boxplot(train_data["Insects"])
#Treating outliers
insects = np.array(train_data["Insects"])
for i in range(len(insects)):
    if insects[i]>3500:
        insects[i] = 3500

train_data["Insects"] = insects

insects = np.array(test_data["Insects"])
for i in range(len(insects)):
    if insects[i]>3500:
        insects[i] = 3500

test_data["Insects"] = insects
sns.boxplot(train_data["Category_of_Toxicant"])
sns.boxplot(train_data["Does_count"])
#Treating outliers
dose_count = np.array(train_data["Does_count"])
for i in range(len(dose_count)):
    if dose_count[i]>70:
        dose_count[i] = 70

train_data["Does_count"] = dose_count

#Treating outliers
dose_count = np.array(test_data["Does_count"])
for i in range(len(dose_count)):
    if dose_count[i]>70:
        dose_count[i] = 70

test_data["Does_count"] = dose_count
sns.boxplot(train_data["Number_of_Weeks_Used"])
#Treating outliers
weeks_used = np.array(train_data["Number_of_Weeks_Used"])
for i in range(len(weeks_used)):
    if weeks_used[i]>60:
        weeks_used[i] = 60

train_data["Number_of_Weeks_Used"] = weeks_used

#Treating outliers
weeks_used = np.array(test_data["Number_of_Weeks_Used"])
for i in range(len(weeks_used)):
    if weeks_used[i]>60:
        weeks_used[i] = 60

test_data["Number_of_Weeks_Used"] = weeks_used
sns.boxplot(train_data["Number_Weeks_does_not used"])
#Treating outliers
weeks_not_used = np.array(train_data["Number_Weeks_does_not used"])
for i in range(len(weeks_not_used)):
    if weeks_not_used[i]>40:
        weeks_not_used[i] = 40

train_data["Number_Weeks_does_not used"] = weeks_not_used

#Treating outliers
weeks_not_used = np.array(test_data["Number_Weeks_does_not used"])
for i in range(len(weeks_not_used)):
    if weeks_not_used[i]>40:
        weeks_not_used[i] = 40

test_data["Number_Weeks_does_not used"] = weeks_not_used
sns.boxplot(train_data["Season"])
target = train_data['Crop_status']
print(target.value_counts())
target_0 = []
target_1 = []
target_2 = []

for index,values in enumerate(train_data['Crop_status']):
    if values == 0:
        target_0.append(train_data.iloc[index,:])
    elif values == 1:
        target_1.append(train_data.iloc[index,:])
    else:
        target_2.append(train_data.iloc[index,:])    
        
print(pd.DataFrame(target_0).shape)
print(pd.DataFrame(target_1).shape)
print(pd.DataFrame(target_2).shape)
print(train_data['Crop_status'].value_counts())
target_0_under = target_0
target_1_under = target_1
target_0_df = pd.DataFrame(target_0_under)
target_1_df = pd.DataFrame(target_1_under)
target_2_df = pd.DataFrame(target_2)

Train_final_data = pd.concat([target_0_df,target_1_df ])
print(Train_final_data.shape)
train_data_0_1 = train_data[train_data['Crop_status'] < 2] 
train_data_0_1.shape
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing

#dividing X and Y in data
#X_0_1 = np.array(train_data_0_1.iloc[:, train_data_0_1.columns != 'Crop_status'])
#Y_0_1 = np.array(train_data_0_1.iloc[:, train_data_0_1.columns == 'Crop_status'])
#print("X_data size is ", X.shape, " Y_data size is ", Y.shape)

#min_max_scaler = preprocessing.MinMaxScaler()
#X_normalized = min_max_scaler.fit_transform(X_0_1)
#X_0_1 = pd.DataFrame(X_normalized)


#dividing X and Y in data
X = np.array(train_data.iloc[:, train_data.columns != 'Crop_status'])
Y = np.array(train_data.iloc[:, train_data.columns == 'Crop_status'])
print("X_data size is ", X.shape, " Y_data size is ", Y.shape)

min_max_scaler = preprocessing.MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X)
X = pd.DataFrame(X_normalized)

#ros = RandomOverSampler()
#X_ros, Y_ros = ros.fit_sample(X, Y)
#print("Shape of X_ros is ", X_ros.shape, "Shape of Y_ros is ", Y_ros.shape)

print(pd.DataFrame(Y_ros).value_counts())
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Splitting training and testing data
(X_train, X_Test, Y_train, Y_Test) = train_test_split(X,Y, test_size = 0.2, stratify= Y, random_state = 4)
print(X_train.shape)
print(X_train[1:100])
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=14)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_Test)
accuracy_score(Y_Test, Y_pred)
test_X = np.array(test_data)
X_test_normalized = min_max_scaler.transform(test_X)
X_test_data_normal = pd.DataFrame(X_test_normalized)
print(pd.DataFrame(Y_Test).value_counts())
print(pd.DataFrame(Y_pred).value_counts())
test_data.shape
Y_pred_test = neigh.predict(X_test_data_normal)
pd.DataFrame(Y_pred_test).to_excel("output2.xlsx")  
test_data.head
#Saving K-NN model
from joblib import dump, load
dump(neigh, 'filename.joblib') 
print(test_data.dtypes)
for columns in test_data.columns:
    if (test_data[columns].dtype == "object"):
        test_data[columns] = encoder.transform(test_data[columns])

print(test_data.dtypes)
print(pd.DataFrame(Y_Test).value_counts())
print(pd.DataFrame(Y_pred).value_counts())
np.array(Y_Test)[0]
Y_Test = np.array(Y_Test)
Y_pred= np.array(Y_pred)
count = 0;
for i in range(len(Y_Test)):
    if Y_Test[i] != Y_pred[i]:
        count=count+1
        
print(pd.DataFrame(Y_Test).value_counts())
print(pd.DataFrame(Y_pred).value_counts())
print(count)
Y_train_pred = neigh.predict(X_train)
accuracy_score(Y_train, Y_train_pred)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc.fit(X_train, Y_train)
Y_pred = rfc.predict(X_Test)
accuracy_score(Y_Test, Y_pred)
print(pd.DataFrame(Y_Test).value_counts())
print(pd.DataFrame(Y_pred).value_counts())
from sklearn.tree import DecisionTreeClassifier
DecisionTree = DecisionTreeClassifier(random_state=0)
DecisionTree.fit(X_train, Y_train)
Y_pred_dt = DecisionTree.predict(X_Test)
accuracy_score(Y_Test, Y_pred_dt)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)
Y_pred_svm = clf.predict(X_Test)
accuracy_score(Y_Test, Y_pred_svm)
#Randomforest using cross validation
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42)
# Fit the random search model
rf_random.fit(X_train, Y_train)
rf_random.best_params_
base_model = RandomForestClassifier(n_estimators = 500,min_samples_split = 16,min_samples_leaf = 14, max_features= 'sqrt',max_depth=70, bootstrap= True, random_state = 42)
base_model.fit(X_train, Y_train)
y_pred_regres = base_model.predict(X_Test)
accuracy_score(Y_Test, y_pred_regres)

for i in range(len(Y_Test)):
    if Y_Test[i] != y_pred_regres[i]:
        print(Y_Test[i], y_pred_regres[i] )  
Y_pred_test_rf = base_model.predict(X_test_data_normal)
pd.DataFrame(Y_pred_test_rf).to_csv("output_rf_0_1_2.csv")  
plt.subplot(2, 1, 2)
plt.plot(Y_Test, label='True coef')
plt.plot(y_pred_regres, label='Estimated coef')

print(Y_Test.size)
print(y_pred_regres.size)
importance = base_model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
     print('Feature: %0d, Score: %.5f' % (i,v))
X_train_new = X_train["Insects","Category_of_Toxicant"]
base_model_new = RandomForestClassifier(n_estimators = 32,min_samples_split = 10,min_samples_leaf = 4, max_features= 'auto',max_depth=10, bootstrap= False, random_state = 42)
base_model_new.fit(X_train_new, Y_train)
from xgboost import XGBClassifier

xgbc_model = XGBClassifier(learning_rate=0.001,
                       max_depth = 40, 
                       n_estimators = 300,
                       scale_pos_weight=10)
xgbc_model.fit(X_train, Y_train)
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [100], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(train['QuoteConversion_Flag'], n_folds=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train[features], train["QuoteConversion_Flag"])

#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(test[features])[:,1]

sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = test_probs
sample.to_csv("xgboost_best_parameter_submission.csv", index=False)

y_pred_xgb = xgbc_model.predict(X_Test)
accuracy_score(Y_Test, y_pred_xgb)
from sklearn.ensemble import AdaBoostClassifier
ada_boost = AdaBoostClassifier(n_estimators=500, random_state=0)
ada_boost.fit(X_train, Y_train)
y_pred_ada = ada_boost.predict(X_Test)
accuracy_score(Y_Test, y_pred_ada)
import matplotlib.pyplot as plt
fig= plt.figure(figsize=(10,10))
plt.scatter(train_data["Insects"],train_data["Crop_status"])
fig= plt.figure(figsize=(10,10))
plt.scatter(train_data["Insects"],train_data["Crop_status"])
fig= plt.figure(figsize=(10,10))
plt.scatter(train_data["Does_count"],train_data["Crop_status"])
fig= plt.figure(figsize=(10,10))
plt.scatter(train_data["Number_Weeks_does_not used"],train_data["Crop_status"])
from scipy import stats
all_models_pred = []
for i in range(0,len(Y_Test)):
    all_models = np.array([Y_pred[i],y_pred_regres[i],y_pred_xgb[i],Y_pred_svm[i],y_pred_ada[i]])
    all_models_pred.append(stats.mode(all_models)[0])

accuracy_score(Y_Test, all_models_pred)
import tensorflow as tf
#Defining our DL model
model = tf.keras.models.Sequential()

#input layer
model.add(tf.keras.layers.Reshape((11,),input_shape = (11,)))
model.add(tf.keras.layers.BatchNormalization())

#hidden layers
model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Dense(4))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.LeakyReLU())

#output layer
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))

#compiling the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=["accuracy"])
model.fit(X_train, Y_train, 
          validation_data = (X_Test, Y_Test),
          epochs = 500,
          batch_size = 10000)
#First train the model for targets 1 or 0.If both are less probs, then 2
train_data_0_1 = train_data.loc[(train_data['Crop_status'] == 0) | (train_data['Crop_status'] == 1)]
train_data_0_1['Crop_status'].value_counts()
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Splitting training and testing data
(X_train_0_1, X_Test_0_1, Y_train_0_1, Y_Test_0_1) = train_test_split(train_data_0_1,train_data_0_1['Crop_status'], test_size = 0.2, stratify= train_data_0_1['Crop_status'], random_state = 4)
print(X_train_0_1.shape)
from sklearn.ensemble import RandomForestClassifier
rfc_0_1 = RandomForestClassifier(max_depth=5, random_state=0)
rfc_0_1.fit(X_train_0_1, Y_train_0_1)
Y_pred_0_1 = rfc_0_1.predict_proba(X_Test)
print(pd.DataFrame(Y_pred_0_1).value_counts())
Y_pred_0 = rfc_0_1.predict(X_Test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_Test, Y_pred_0)
