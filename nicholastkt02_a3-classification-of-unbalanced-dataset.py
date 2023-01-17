import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
path = os.path.join(os.getcwd(), "A3-data -csv")
print(path)
data = pd.read_csv("A3-data - csv.csv")
#Data exploration
data.tail()
data = data.rename(columns = {"class": "Class"})
# Brief overview of the data
print(data.shape)
print(data.describe())
# Checking the class of the variables
data.dtypes
# Age contains not only numbers, but also a value 'U', which represents Unknown,
# in which we need to change if we want the entire column as an integer
data['age'] = data['age'].str.replace('U', '7')
data.age.value_counts()
# Changing values of our columns to integers for easier processing
# Gender
data['gender'] = data['gender'].str.replace('F', '1')
data['gender'] = data['gender'].str.replace('M', '2')
data['gender'] = data['gender'].str.replace('E', '3')
data['gender'] = data['gender'].str.replace('U', '4')
data.gender.value_counts()

# Category
data['category'] = data['category'].str.replace('transportation', '1')
data['category'] = data['category'].str.replace('food', '2')
data['category'] = data['category'].str.replace('health', '3')
data['category'] = data['category'].str.replace('wellnessandbeauty', '4')
data['category'] = data['category'].str.replace('fashion', '5')
data['category'] = data['category'].str.replace('barsandrestaurants', '6')
data['category'] = data['category'].str.replace('hyper', '7')
data['category'] = data['category'].str.replace('sportsandtoys', '8')
data['category'] = data['category'].str.replace('tech', '9')
data['category'] = data['category'].str.replace('home', '10')
data['category'] = data['category'].str.replace('hotelservices', '11')
data['category'] = data['category'].str.replace('otherservices', '12')
data['category'] = data['category'].str.replace('contents', '13')
data['category'] = data['category'].str.replace('travel', '14')
data['category'] = data['category'].str.replace('leisure', '15')
data.category.value_counts()
# Changing class to str or integer
data['gender'] = data['gender'].astype(int)
data['merchant'] = data['merchant'].astype(str)
data['age'] = data['age'].astype(int)
data['category'] = data['category'].astype(int)
data['amount'] = data['amount'].astype(int)
#Distribution of the amount
amount = [data['amount'].values]
sns.distplot(amount)
from matplotlib import gridspec
list_remove = ['step', 'customer', 'zipcodeOri', 'zipMerchant', 'merchant']
features = data.loc[:, 'step': 'Class'].columns
features = features.drop(list_remove)
features
plt.figure(figsize = (12, 28*4))
gs = gridspec.GridSpec(28, 1)
for i, c in enumerate(data[features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[c][data.Class == 1], bins = 20)
    sns.distplot(data[c][data.Class == 0], bins = 20)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(c))
plt.show()
# Number of fraud cases within each category
fraud_categories = pd.DataFrame(Counter(data['category'][data['Class'] == 1]), index = [1])
print(fraud_categories)
# Number of valid cases within each category
valid_categories = pd.DataFrame(Counter(data['category'][data['Class'] == 0]), index = [0])
print(valid_categories)
# Side by side comparison of the valid and fraud cases for each category
comparison = pd.concat([fraud_categories, valid_categories])
print(comparison)
data.loc[data['amount'] == 8329, 'category']
data[features].hist(figsize = (20, 20))
plt.show()
# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]


outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Cases: {}'.format(len(data[data['Class'] == 0])))
print("Amount details of fraudulent transaction")
Fraud.amount.describe()
print("Amount details of valid transaction")
Valid.amount.describe()
corr = data[features].corr()
fig = plt.figure(figsize = (12, 9))

annot_kws = {"ha": 'center',"va": 'bottom'}
sns.heatmap(corr, vmax = 1, square = True, cmap = "YlGnBu", annot = True, annot_kws = annot_kws)
plt.show()
# Allocating our response and predict variables
x = data[features].drop(['Class', 'age', 'gender'], axis = 1)
y = data[features]['Class']
x_data = x.values
y_data = y.values
from sklearn.model_selection import train_test_split
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 42)
from sklearn.ensemble import IsolationForest
ifc = IsolationForest(max_samples = len(X_train),
                     contamination = outlier_fraction, 
                      random_state = 1)
ifc.fit(X_train)
scores_pred = ifc.decision_function(X_train)
y_pred = ifc.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
# Reshaping prediction values as in our dataset, 0 as valid and 1 for fraud
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

n_errors = (y_pred != y_test).sum()

n_outliers = len(Fraud)
print("The Model used is {}".format("Isolation Forest"))
acc = accuracy_score(y_test, y_pred)
print("The Accuracy is {}".format(acc))
prec = precision_score(y_test, y_pred)
print("The precision is {}".format(prec))
rec = recall_score(y_test, y_pred)
print("The recall is {}".format(rec))
f1_score = f1_score(y_test, y_pred)
print("The f1-score is {}".format(f1_score))
MCC = matthews_corrcoef(y_test, y_pred)
print("The Matthews correlation coefficient is {}".format(MCC))

LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (20,20))
sns.heatmap(conf_matrix, xticklabels = LABELS,
           yticklabels = LABELS, annot = True, fmt = "d", cmap = "YlGnBu", annot_kws = {"ha": "center", "va": 'center'});
plt.title("Confusion matrix of Isolationn Forest Model")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Classes")
plt.show()
plt.figure(figsize = (9, 7))
print('{}: {}'.format("Isolation Forest", n_errors))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
# predictions
y_pred1 = rfc.predict(X_test)
#Printing the score of the classifiers
n_outliers = len(Fraud)
n_errors = (y_pred1 != y_test).sum()
print("Model used is: Random Forest Classifier")
accuracy = accuracy_score(y_test, y_pred1)
print("Accuracy obtained: {}".format(accuracy))
prec = precision_score(y_test, y_pred1)
print("Precision obtained: {}".format(prec))
rec = recall_score(y_test, y_pred1)
print("Recall obtained: {}".format(rec))
#f1 = f1_score(y_test, y_pred1)
#print("F1 score obtained: {}".format(f1))
MCC = matthews_corrcoef(y_test, y_pred1)
print("Matthews correlation coefficient is {}".format(MCC))

# Printing the confusion matrix
LABELS = ['Valid', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred1)
plt.figure(figsize = (12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt = "d", cmap = "YlGnBu", annot_kws = {"ha":'center', "va":'center'});
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()

# Running our classification matrix
plt.figure(figsize = (9, 7))
print("{}: {}".format("Random Forest Classifier", n_errors))
print(accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
# Shuffling dataset
shuffled_data = data.sample(frac = 1, random_state = 4)
# Seperating the dataset into fraud and valid classes
fraud_data = shuffled_data.loc[shuffled_data['Class'] == 1]
valid_data = shuffled_data.loc[shuffled_data['Class'] == 0]
# Identifying the number of cases in each class
freq = Counter(data['Class'])
print(freq)
# Selecting the same amounts of valid class as in fraud class
valid_data = shuffled_data.loc[shuffled_data['Class'] == 0].sample(n  = 7160, random_state = 42)
# Concatenating both dataframes
normalised_data = pd.concat([fraud_data, valid_data])
plt.figure(figsize = (8, 8))
sns.countplot('Class', data = normalised_data)
plt.title("Balanced Classes")
plt.show()
normalised_data.head()
list_remove2 = ['step', 'customer', 'zipcodeOri', 'zipMerchant', 'merchant']
features2 = normalised_data.loc[:, 'step': 'Class'].columns
features2 = features2.drop(list_remove)
corr2 = normalised_data[features2].corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corr2, vmax = 1, square = True, cmap = "YlGnBu", annot = True)
plt.show()
from imblearn.over_sampling import SMOTE
data.head()
# Selecting only columns that we want
data = data[['age', 'gender', 'category', 'amount', 'Class']]
# Resampling the minority class. Sampling strategy could be "auto" if which class is the minority is unknown
sm = SMOTE(sampling_strategy = "minority", random_state = 7)

# Fitting the sm model 
oversamp_trainX, oversamp_trainy = sm.fit_sample(data.drop('Class', axis = 1), data['Class'])
oversamp_train = pd.concat([pd.DataFrame(oversamp_trainy), pd.DataFrame(oversamp_trainX)], axis = 1)
oversamp_train.columns = data.columns
oversamp_train['amount'] = oversamp_train['amount'].astype(int)
oversamp_train['age'] = oversamp_train['age'].astype(int)
oversamp_train['gender'] = oversamp_train['gender'].astype(int)
oversamp_train['Class'] = oversamp_train['Class'].astype(int)
corr3 = oversamp_train.corr()
fig = plt.figure(figsize = (20, 10))

sns.heatmap(corr3, cmap = "YlGnBu", vmax = 1, square = True, annot = True)
plt.show()
# Allocating our response and predict variables
x2 = oversamp_train.drop(['Class'], axis = 1)
y2 = oversamp_train['Class']
x_data2 = x.values
y_data2 = y.values
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_data2, y_data2, test_size = 0.2, random_state = 42)
# Random Forest Classifier fpr oversampled data
rfc = RandomForestClassifier()
rfc.fit(X_train2, y_train2)
# predictions
y_pred4 = rfc.predict(X_test2)
#Printing the score of the classifiers
n_outliers = len(Fraud)
n_errors = (y_pred4 != y_test2).sum()
print("Model used is: Random Forest Classifier")
accuracy = accuracy_score(y_test2, y_pred4)
print("Accuracy obtained: {}".format(accuracy))
prec = precision_score(y_test2, y_pred4)
print("Precision obtained: {}".format(prec))
rec = recall_score(y_test2, y_pred4)
print("Recall obtained: {}".format(rec))
#f1 = f1_score(y_test, y_pred1)
#print("F1 score obtained: {}".format(f1))
MCC = matthews_corrcoef(y_test2, y_pred4)
print("Matthews correlation coefficient is {}".format(MCC))

# Printing the confusion matrix
LABELS = ['Valid', 'Fraud']
conf_matrix = confusion_matrix(y_test2, y_pred4)
plt.figure(figsize = (12, 9))
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt = "d", cmap = "YlGnBu");
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()

# Running our classification matrix
plt.figure(figsize = (12, 9))
print("{}: {}".format("Random Forest Classifier", n_errors))
print(accuracy_score(y_test2, y_pred4))
print(classification_report(y_test2, y_pred4))
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
data.head()
bbclass = BalancedBaggingClassifier(base_estimator = DecisionTreeClassifier(),
                                   sampling_strategy = "auto",
                                   replacement = False, 
                                   random_state = 0) 

bbclass.fit(X_train, y_train)
y_pred3 = bbclass.predict(X_train)
y_pred3
conf_matrix3 = confusion_matrix(y_train, y_pred3)
fig = plt.figure(figsize = (12, 9))
sns.heatmap(conf_matrix3, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt = "d", cmap = "YlGnBu")

accuracy = accuracy_score(y_train, y_pred3)
print("Accuracy obtained: {}".format(accuracy))
precision = precision_score(y_train, y_pred3)
print("Precision obtained: {}".format(precision))
recall = recall_score(y_train, y_pred3)
print("Recall obtained: {}".format(recall))
mcc = matthews_corrcoef(y_train, y_pred3)
print("Matthews Correlation Coefficient: {}".format(mcc))

# Running our classification matrix
plt.figure(figsize = (9, 7))
print("{}: {}".format("Balanced Bagging Classifier", n_errors))
print(accuracy_score(y_train, y_pred3))
print(classification_report(y_train, y_pred3))
