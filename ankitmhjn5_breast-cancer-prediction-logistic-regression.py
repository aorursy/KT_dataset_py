import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
# checking the number of nulls values for each column
data.isnull().sum()
# removing the columns Unnamed: 32 and id
data = data.drop(['id', 'Unnamed: 32'], axis=1)
data.describe()
# counting the number of malignant and benign cases
sns.countplot(data['diagnosis'])
# viewing the radius vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='radius_mean', data=data, ax=ax1)

sns.catplot(x='diagnosis', y='radius_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='radius_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# viewing the texture vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='texture_mean', data=data, ax=ax1)


sns.catplot(x='diagnosis', y='texture_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='texture_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# viewing the texture vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='area_mean', data=data, ax=ax1)


sns.catplot(x='diagnosis', y='area_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='area_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# viewing the texture vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='smoothness_mean', data=data, ax=ax1)


sns.catplot(x='diagnosis', y='smoothness_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='smoothness_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# viewing the texture vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='concavity_mean', data=data, ax=ax1)


sns.catplot(x='diagnosis', y='concavity_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='concavity_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# viewing the texture vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='concave points_mean', data=data, ax=ax1)


sns.catplot(x='diagnosis', y='concave points_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='concave points_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# viewing the texture vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='symmetry_mean', data=data, ax=ax1)


sns.catplot(x='diagnosis', y='symmetry_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='symmetry_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# viewing the texture vs diagnosis
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
sns.catplot(x='diagnosis', y='fractal_dimension_mean', data=data, ax=ax1)


sns.catplot(x='diagnosis', y='fractal_dimension_se', data=data, ax=ax2)

sns.catplot(x='diagnosis', y='fractal_dimension_worst', data=data, ax=ax3)
plt.close(1)
plt.show()
# splitting the data to train and test
train_data, test_data = train_test_split(data, train_size=0.7, test_size=0.3, random_state=100)
y_train = train_data['diagnosis']
X_train = train_data.drop(['diagnosis'], axis=1)
y_train = y_train.apply(lambda row: 1 if row == 'M' else 0)
colsToScale = ['area_mean', 'area_se', 'area_worst', 'texture_mean', 'texture_se', 'texture_worst', 'radius_mean',
              'radius_se', 'radius_worst', 'perimeter_mean', 'perimeter_se', 'perimeter_worst']

# scaling the features
scaler = MinMaxScaler()
X_train[colsToScale] = scaler.fit_transform(X_train[colsToScale])
X_train.describe()
### Building the Logistic Regression model using RFE
log_reg = LogisticRegression()
model = log_reg.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
# predicting the accuracy of the model on train data
train_score = accuracy_score(y_train, y_train_pred)
train_score

# generating the confucsion matrix for the predicted train values
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
train_conf_matrix
TN = train_conf_matrix[0][0]
FN = train_conf_matrix[1][0]
TP = train_conf_matrix[1][1]
FP = train_conf_matrix[0][1]
# sensitivity of train data
sensitivity = TP/(TP+FP)
sensitivity
# specifitcity of train data
specifitcity = TP/(TP+FN)
specifitcity
y_test = test_data['diagnosis']
X_test = test_data.drop(['diagnosis'], axis=1)
y_test = y_test.apply(lambda row: 1 if row == 'M' else 0)

#scaling the test features 
X_test[colsToScale] = scaler.transform(X_test[colsToScale])

# predicting the results
y_test_pred = log_reg.predict(X_test)
# getting the accuracy of the model on test data
test_accuracy = accuracy_score(y_test, y_test_pred)
test_accuracy
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
test_confusion_matrix
TN_test = test_confusion_matrix[1][1]
FN_test = test_confusion_matrix[1][0]
FP_test = test_confusion_matrix[0][1]
TP_test = test_confusion_matrix[1][1]
# specificity of test data
specificity = TP_test/(TP_test + FP_test)
specificity
# sensitivity of test data
sensitivity = TP_test/(TP_test + FN_test)
sensitivity