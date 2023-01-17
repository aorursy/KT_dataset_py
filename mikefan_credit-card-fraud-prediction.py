import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
from sklearn.model_selection import train_test_split # tran and test data split
from sklearn.linear_model import LogisticRegression # Logistic Regression 
from sklearn.svm import SVC #Support Vector Machine 
from sklearn.ensemble import RandomForestClassifier # Random Rorest Classifier 
from sklearn.metrics import roc_auc_score # ROC and AUC 
from sklearn.metrics import accuracy_score # Accuracy 
from sklearn.metrics import recall_score # Recall 
from sklearn.metrics import precision_score # Prescison 
from sklearn.metrics import classification_report # Classification Score Report 
%matplotlib inline 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.sample(10, random_state = 10)
data.info()
# Summary statistics for the variables
data.describe(include = "all")
# Check the missing data status
print("Is the data containing any missing values: {}".format(data.isnull().sum().any()))
data['Class'] = data['Class'].astype('str')
sample_data = data.sample(60000)
plt.style.use('ggplot')
sample_data.hist(figsize = (25, 25), color = 'steelblue', bins = 20)
plt.show()
sns.countplot(x = 'Class', data = data, color = 'steelblue')
plt.title("Transaction Fraud Distribution")
plt.xticks(range(2), ["Normal", "Fraud"])
plt.ylabel("Frequency")
plt.show()
# Print the count for fraud and non-fraud transactions
data["Class"].value_counts(sort = True)
# Time distribution in hours and with Transaction Amount 
plt.subplot(2, 2, 1)
(data['Time']/3600).hist(figsize=(15,15), color = "steelblue", bins = 20)
plt.title("Distribution of Time")
plt.xlabel("Hour")
plt.ylabel("Frequency")

# Transaction amount distribution by hours
plt.subplot(2, 2, 2)
plt.scatter(x = data['Time']/3600, y = data['Amount'], alpha = .8)
plt.title("Distribution of Transaction Amount by Time")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()
# For Normal Transactions
# Look at the time distribution
plt.subplot(2, 2, 1)
(data[data['Class'] == '0']['Time']/3600).hist(figsize=(15, 15), bins = 20, color = 'blue')
plt.title("Time Distribution for Normal Transactions")
plt.xlabel("Hour")
plt.ylabel("Frequency")

# Transaction Amount Distribution
plt.subplot(2, 2, 2)
data[data['Class'] == "0"]['Amount'].hist(bins = [0, 100, 200, 300, 400 ,500, 600, 700, 800, 900, 1000], color = 'blue')
plt.title("Amount Distribution for Normal Transactions")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.xlim(0, 1500)
plt.show()
# For Fraud Transactions
# Look at the time distribution
plt.subplot(2, 2, 1)
(data[data['Class'] == '1']['Time']/3600).hist(figsize=(15, 15), bins = 20)
plt.title("Time Distribution for Normal Transactions")
plt.xlabel("Hour")
plt.ylabel("Frequency")

# Transaction Amount Distribution
plt.subplot(2, 2, 2)
data[data['Class'] == "1"]['Amount'].hist(bins = 20)
plt.title("Amount Distribution for Normal Transactions")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.xlim(0, 1500)
plt.show()
# Correlation Plot for all numeric variables
corr = data.corr()
corr.style.background_gradient()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Make the X and Y inputs
data_x = data.drop('Class', axis = 1)
data_y = data['Class']

# Split the data into training and testing, use 30% data to evaluate the models 
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.3, random_state = 123)

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

print("Training data has {} rows and {} variables".format(train_x.shape[0], train_x.shape[1]))
print("Testing data has {} rows and {} variables".format(test_x.shape[0], test_x.shape[1]))
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

dummy = DummyClassifier(strategy="stratified")
dummy.fit(train_x, train_y)
dummy_pred = dummy.predict(test_x)

# Print test outcome 
print(confusion_matrix(test_y, dummy_pred))
print('\n')
print(classification_report(test_y, dummy_pred))
# Build the model and make predictions using the test data 
logreg = LogisticRegression(random_state = 100)
logreg.fit(train_x, train_y)

pd.DataFrame(logreg.coef_, columns = data.drop('Class', axis = 1).columns)
pred_y = logreg.predict(test_x)
# Print the confusion matrix for the model 
conf_matrix = confusion_matrix(test_y, pred_y)

plt.figure(figsize = (7,7))
sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.title("Accuracy Score: {}".format(accuracy_score(test_y, pred_y)))
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()
# Since the data is highly imbalanced, it is not enough to purly rely on the accuracy score to evaluate the model. Recall and Prescision here plays more important roles
print("The recall score for prediction is {:0.2f}".format(recall_score(test_y, pred_y, pos_label='1')))
print("The prescision score for predion is {:0.2f}".format(precision_score(test_y, pred_y, pos_label='1')))
print("\n")
print(classification_report(test_y, pred_y))
# Print out the Recall-Precision Plot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

plt.figure(figsize = (7,7))
plot_precision_recall_curve(logreg, test_x, test_y)
plt.title("Precision-Recall curve for Logistic Regression Classifier")
# Radial kernel will be used as the kernel function 
from sklearn.svm import SVC 

svc = SVC(kernel='rbf', C = 1, gamma='scale')
svc.fit(train_x, train_y)

pred_y_svc = svc.predict(test_x)
# Print the confusion matrix for the model 
conf_matrix = confusion_matrix(test_y, pred_y_svc)

plt.figure(figsize = (7,7))
sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.title("Accuracy Score: {}".format(accuracy_score(test_y, pred_y_svc)))
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()
# Generate the Recall and Prescision scores for the predictions 
print("The recall score for prediction is {:0.2f}".format(recall_score(test_y, pred_y_svc, pos_label='1')))
print("The prescision score for predion is {:0.2f}".format(precision_score(test_y, pred_y_svc, pos_label='1')))
print("\n")
print(classification_report(test_y, pred_y_svc))
# Print out the Recall-Precision Plot
plt.figure(figsize = (7,7))
plot_precision_recall_curve(svc, test_x, test_y)
plt.title("Precision-Recall curve for Support Vector Classifier")
# Here I will apply GridSearch to find out the best hyperparameters 
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Create the Random Forest Classifier and 
rfc = RandomForestClassifier(random_state = 100)
pprint(rfc.get_params())
# In this case three parameters are tuned: Number of trees used in prediction (n_estimaters), Maximum number of features randomly selected in each decision tree (max_features) and maximum number of levels in decision tree
params = {'n_estimators': [10, 50, 100],
          'max_features': ['auto', 'sqrt'],
          'max_depth': [10, 25, 50]}

# Initiate the grid search model and fit the training data 
grid_search = GridSearchCV(rfc, 
                           param_grid = params, 
                           cv = 5, 
                           n_jobs = -1, 
                           verbose = 2)

# fit the training data
grid_search.fit(train_x, train_y)
grid_search.best_params_
grid_search.best_estimator_
features = data.drop('Class', axis = 1).columns
feature_importances = grid_search.best_estimator_.feature_importances_
feature_indices = np.argsort(feature_importances)
features = [features[i] for i in feature_indices]

plt.figure(figsize = (10,10))
plt.barh(range(len(feature_indices)), feature_importances[feature_indices], color='steelblue')
plt.title('Feature Importances from Random Forest Classifier')
plt.xlabel('Importances')
plt.yticks(range(0, len(features)), features)
plt.show()
# Make predictions on testing data 
pred_y_rfc = grid_search.best_estimator_.predict(test_x)
# Print the confusion matrix
conf_matrix = confusion_matrix(test_y, pred_y_rfc)

plt.figure(figsize = (7,7))
sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.title("Accuracy Score: {}".format(accuracy_score(test_y, pred_y_rfc)))
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()
# Generate the Recall and Prescision scores for the predictions 
print("The recall score for prediction is {:0.2f}".format(recall_score(test_y, pred_y_rfc, pos_label='1')))
print("The prescision score for predion is {:0.2f}".format(precision_score(test_y, pred_y_rfc, pos_label='1')))
print("\n")
print(classification_report(test_y, pred_y_rfc))
# Print out the Recall-Precision Plot
plt.figure(figsize = (7,7))
plot_precision_recall_curve(grid_search.best_estimator_, test_x, test_y)
plt.title("Precision-Recall curve for Random Forest Classifier")