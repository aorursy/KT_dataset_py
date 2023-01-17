# Importing the basic libraries



import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing the libraries



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import scikitplot as skplt

from keras.utils import to_categorical
# Loading the dataset



dataset = pd.read_csv('/kaggle/input/bank-additional-full.csv', sep = ';', na_values = 'unknown')

dataset
# Analyzing the dataset



print(dataset.shape)

dataset.info()
# Descriptive Statistics



dataset.describe()
# Checking for missing values. The dataset specifies a token 'unknown' which has been used to mark the missing values



dataset.isnull().sum()
# Condition to extract columns that contain the missing values



miss_cond = np.array(dataset.isnull().sum() != 0)

miss_col_ind = np.where(miss_cond)[0]

miss_cols = list(dataset.columns[miss_col_ind])



print(miss_col_ind)

print(miss_cols)
# A list to store all the unique values and their count from the columns that contains missing values



missing_column_freq = []



for i in miss_col_ind:

    missing_column_freq.append(dataset.iloc[:, i].value_counts())

    

missing_column_freq
# Creating a fresh dataset to store only a fragment of the original dataset that needs preprocessing



dataset_miss = dataset[miss_cols]

dataset_miss.isnull().sum()    
# Appending 'Nan' and its count in the missing colum frequency list. 



for i in range(6):

    missing_column_freq[i] = missing_column_freq[i].append(pd.Series({'null' : dataset_miss.isnull().sum()[i]}))



missing_column_freq
# Visualizing all the null values with respect to the legitimate values for the columns that contain 'NaN'



for i in range(6):

    plt.figure(figsize = (18, 5))

    sns.barplot(missing_column_freq[i].index, missing_column_freq[i].values)

    plt.savefig("figure {}".format(i))

    plt.show()
# Fitting an object of Simple Imputer which intends to replace the NaN values with the mode of a particular attribute



from sklearn.impute import SimpleImputer

sim = SimpleImputer(strategy = "most_frequent")

sim.fit(dataset_miss)

print('NaN values shall be replaced by the following values : ', sim.statistics_)
# Replacing the NaN values and double checking the results



dataset_miss = pd.DataFrame(sim.transform(dataset_miss))

dataset_miss.isnull().sum()
# Creating a copy of the original dataset and replacing the NaN fragment with the Preprocessed fragment



dataset1 = dataset.copy()

dataset1.isnull().sum()
# Replacing the NaN fragment with the Preprocessed fragment



dataset1[miss_cols] = dataset_miss

dataset1.isnull().sum()
# Storing all the integer columns in a separate dataset



dataset_with_integers = dataset.select_dtypes(exclude = ['O'])

dataset_with_integers
# Performing EDA (Exploratory Data Analysis)



len = dataset_with_integers.shape[1]

len



cols = list(dataset_with_integers.columns)

cols



for i in range(len):

    plt.figure(figsize = (18, 5))

    plt.hist(dataset_with_integers[cols[i]], bins = 100)

    plt.xlabel(cols[i])

    plt.ylabel('Frequency')

    plt.title('Histogram Analysis for {} variable'.format(cols[i]))

    plt.show()
# Plotting scatter-matrix



pd.plotting.scatter_matrix(dataset1, figsize = (18, 15))

plt.show()
# Plotting correlation matrix to identify important columns



corr_mat = dataset1.corr()

corr_mat_sp = dataset1.corr('spearman')

corr_mat
# Plotting the heatmap



plt.figure(figsize = (18, 8))

sns.heatmap(corr_mat, annot = True)

plt.show()
# Creating a new copy of the preprocessed dataset and dropping the irrelevant columns



dataset2 = dataset1.copy()

dataset2 = dataset2.drop(['duration', 'pdays'], axis = 1)

dataset2
# Performing Feature Scaling



scaler = StandardScaler()

dataset_with_integers = scaler.fit_transform(dataset_with_integers)

dataset_with_integers = pd.DataFrame(dataset_with_integers)

dataset_with_integers
# Performing One Hot Encoding on the categorical variables



dataset_str = dataset2.select_dtypes(include = ['O'])

dataset_str
# Using the get_dummies() function of pandas



dataset_str = pd.get_dummies(dataset_str)

dataset_str
# Concatenating both the preprocessed dataframes with integer and categorical variables respectively



final_dataset = pd.concat([dataset_with_integers, dataset_str], axis = 1)

print(final_dataset.shape)

final_dataset
# Slicing the preprocessed dataset into feature matrix (X) and vector of predictions (y)



X = final_dataset.iloc[:, :-2].values

y = final_dataset.iloc[:, -2:].values

y = np.argmax(y, axis = 1)

X, y
# Splitting the dataset into training set and test set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Applying Logistic Regression



from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver = 'saga', max_iter = 1000)

log_reg.fit(X_train, y_train)
# Analyzing the training set and test set performance



print("Training Set : ", log_reg.score(X_train, y_train))

print("Testing Set : ", log_reg.score(X_test, y_test))
# Applying K-fold Cross Validation



cross_val_score(log_reg, X_train, y_train, cv = 5)
# Getting the decision score for logistic function



y_log_scores = cross_val_predict(log_reg, X_train, y_train, cv = 3, method = 'decision_function')

y_log_scores
# Getting the Precision, Recall, F1 Score



y_log_pred = cross_val_predict(log_reg, X_train, y_train, cv = 3)



print('Precision : ', precision_score(y_train, y_log_pred))

print('Recall : ', recall_score(y_train, y_log_pred))

print('F1 Score : ', f1_score(y_train, y_log_pred))
# Plotting the confusion matrix



skplt.metrics.plot_confusion_matrix(y_train, y_log_pred, figsize=(8, 8))

plt.show()
# Computing the parameters for a Precision Recall Curve



precision, recall, threshold = precision_recall_curve(y_train, y_log_scores)

precision, recall, threshold
# Precision Recall Curve



plt.figure(figsize = (18, 5))

plt.plot(threshold, precision[:-1], c = "r", label = 'Precision')

plt.plot(threshold, recall[:-1], c = "g", label = 'Recall')

plt.xlabel('Threshold')

plt.title('Analyzing precision and recall changes')

plt.legend()

plt.show()
# Plotting Precision vs Recall



plt.figure(figsize = (18, 5))

plt.plot(recall, precision)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision vs Recall')

plt.show()
# Plotting the ROC Curve (Sensitivity vs 1-Specificity)



fpr, tpr, th = roc_curve(y_train, y_log_scores)



plt.figure(figsize = (18, 7))

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate (Sensitivity)')

plt.ylabel('True Positive Rate (1-Specificity)')

plt.title('ROC Curve')

plt.show()
# Generating the AUC Score



roc_auc_score(y_train, y_log_scores)
# Applying the Random Forest Algorithm



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
# Analyzing the training set and test set performance



print("Training Set : ", rf.score(X_train, y_train))

print("Testing Set : ", rf.score(X_test, y_test))
# Applying K-fold Cross Validation



cross_val_score(rf, X_train, y_train, cv = 5)
# Getting the decision score for random forest



y_proba_rf = cross_val_predict(rf, X_train, y_train, cv = 5, method = 'predict_proba')

y_scores_forest = y_proba_rf[:, 1]

y_scores_forest
# Getting the Precision, Recall, F1 Score



y_rf_pred = cross_val_predict(rf, X_train, y_train, cv = 3)



print('Precision : ', precision_score(y_train, y_rf_pred))

print('Recall : ', recall_score(y_train, y_rf_pred))

print('F1 Score : ', f1_score(y_train, y_rf_pred))
# Plotting the confusion matrix



skplt.metrics.plot_confusion_matrix(y_train, y_rf_pred, figsize=(8, 8))

plt.show()
# Computing the parameters for a Precision Recall Curve



precision_rf, recall_rf, threshold_rf = precision_recall_curve(y_train, y_scores_forest)

precision_rf, recall_rf, threshold_rf
# Precision Recall Curve



plt.figure(figsize = (18, 5))

plt.plot(threshold_rf, precision_rf[:-1], c = "r", label = 'Precision')

plt.plot(threshold_rf, recall_rf[:-1], c = "g", label = 'Recall')

plt.xlabel('Threshold')

plt.title('Analyzing precision and recall changes')

plt.legend()

plt.show()
# Plotting Precision vs Recall



plt.figure(figsize = (18, 5))

plt.plot(recall_rf, precision_rf)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision vs Recall')

plt.show()
# Plotting the ROC Curve (Sensitivity vs 1-Specificity)



fpr_forest, tpr_forest, th_forest = roc_curve(y_train, y_scores_forest)



plt.figure(figsize = (18, 7))

plt.plot(fpr_forest, tpr_forest)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate (Sensitivity)')

plt.ylabel('True Positive Rate (1-Specificity)')

plt.title('ROC Curve')

plt.show()
# Generating the AUC Score



roc_auc_score(y_train, y_scores_forest)
# Comparing the ROC Curve for Logistic Regression and Random Forest



plt.figure(figsize = (18, 7))

plt.plot(fpr, tpr, c = 'g')

plt.plot(fpr_forest, tpr_forest, c = 'r')

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate (Sensitivity)')

plt.ylabel('True Positive Rate (1-Specificity)')

plt.legend()

plt.title('ROC Curve')

plt.show()
# Importing the DL libraries



import tensorflow as tf

from tensorflow import keras
# Checking the shape of the dataset



X_train.shape, y_train.shape
# Building the model



model = keras.models.Sequential([

    keras.layers.Dense(128, activation = 'tanh', input_shape = [57, ]),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(64, activation = 'tanh'),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(32, activation = 'tanh'), 

    keras.layers.Dense(32, activation = 'tanh'), 

    keras.layers.Dense(2, activation = 'softmax')

])



model.summary()
# Compiling the model



model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Changing the shape of y



y_train = to_categorical(y_train)

y_train.shape
# Splitting the dataset into training set and validation set



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.05)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
# Fitting the model



history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_valid, y_valid))
# Visualizing the results



pd.DataFrame(history.history).plot(figsize = (12, 6))

plt.gca().set_ylim(0, 1)

plt.grid(True)

plt.show()
# Changing the shape of the test set label



y_test = to_categorical(y_test)

y_test.shape
# Evaluation on the test set



test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test set loss : ', test_loss)

print('Test set accuracy : ', test_accuracy)