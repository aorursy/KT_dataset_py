# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Read csv

df = pd.read_csv("../input/creditcard.csv")

# Explore the features avaliable in our dataframe

df.shape

df.info()

df.head()

df.describe()

print(df.Amount.describe())



# Count the occurrences of fraud and no fraud cases

fnf = df["Class"].value_counts()



# Print the ratio of fraud cases 

print(fnf/len(df))



# Plottingg your data

plt.xlabel("Class")

plt.ylabel("Number of Observations")

fnf.plot(kind = 'bar',title = 'Frequency by observation number',rot=0)
# Plot how fraud and non-fraud cases are scattered 

plt.scatter(df.loc[df['Class'] == 0]['V1'], df.loc[df['Class'] == 0]['V2'], label="Class #0", alpha=0.5, linewidth=0.15)

plt.scatter(df.loc[df['Class'] == 1]['V1'], df.loc[df['Class'] == 1]['V2'], label="Class #1", alpha=0.5, linewidth=0.15,c='r')

plt.show()
import seaborn as sns



fig, ax = plt.subplots(1, 2, figsize=(18,4))



# Plot the distribution of 'Time' feature 

sns.distplot(df['Time'].values/(60*60), ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Time', fontsize=14)

ax[0].set_xlim([min(df['Time'].values/(60*60)), max(df['Time'].values/(60*60))])



sns.distplot(df['Amount'].values, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Amount', fontsize=14)

ax[1].set_xlim([min(df['Amount'].values), max(df['Amount'].values)])



plt.show()
# Seperate total data into non-fraud and fraud cases

df_nonfraud = df[df.Class == 0] #save non-fraud df observations into a separate df

df_fraud = df[df.Class == 1] #do the same for frauds
# Summarize statistics and see differences between fraud and normal transactions

print(df_nonfraud.Amount.describe())

print('_'*25)

print(df_fraud.Amount.describe())



# Import the module

from scipy import stats

F, p = stats.f_oneway(df['Amount'][df['Class'] == 0], df['Amount'][df['Class'] == 1])

print("F:", F)

print("p:",p)
# Plot of high value transactions($200-$2000)

bins = np.linspace(200, 2000, 100)

plt.hist(df_nonfraud.Amount, bins, alpha=1, normed=True, label='Non-Fraud')

plt.hist(df_fraud.Amount, bins, alpha=1, normed=True, label='Fraud')

plt.legend(loc='upper right')

plt.title("Amount by percentage of transactions (transactions \$200-$2000)")

plt.xlabel("Transaction amount (USD)")

plt.ylabel("Percentage of transactions (%)")

plt.show()
# Plot of transactions in 48 hours

bins = np.linspace(0, 48, 48) #48 hours

plt.hist((df_nonfraud.Time/(60*60)), bins, alpha=1, normed=True, label='Non-Fraud')

plt.hist((df_fraud.Time/(60*60)), bins, alpha=0.6, normed=True, label='Fraud')

plt.legend(loc='upper right')

plt.title("Percentage of transactions by hour")

plt.xlabel("Transaction time from first transaction in the dataset (hours)")

plt.ylabel("Percentage of transactions (%)")

plt.show()
# Plot of transactions in 48 hours

plt.scatter((df_nonfraud.Time/(60*60)), df_nonfraud.Amount, alpha=0.6, label='Non-Fraud')

plt.scatter((df_fraud.Time/(60*60)), df_fraud.Amount, alpha=0.9, label='Fraud')

plt.title("Amount of transaction by hour")

plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")

plt.ylabel('Amount (USD)')

plt.legend(loc='upper right')

plt.show()
# Scale "Time" and "Amount"

from sklearn.preprocessing import StandardScaler, RobustScaler

df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))



# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"

df_scaled = df.drop(['Time','Amount'],axis = 1,inplace=False)

df_scaled.head()
# Calculate pearson correlation coefficience

corr = df_scaled.corr() 



# Plot heatmap of correlation

f, ax = plt.subplots(1, 1, figsize=(24,20))

sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})

ax.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=24)
# Define the prep_data function to extrac features 

def prep_data(df):

    X = df.drop(['Class'],axis=1, inplace=False) #  

    X = np.array(X).astype(np.float)

    y = df[['Class']]  

    y = np.array(y).astype(np.float)

    return X,y



# Create X and y from the prep_data function 

X, y = prep_data(df_scaled)
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import BorderlineSMOTE

from imblearn.pipeline import Pipeline # Inorder to avoid testing model on sampled data



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Define the resampling method

undersam = RandomUnderSampler(random_state=0)

oversam = RandomOverSampler(random_state=0)

smote = SMOTE(kind='regular',random_state=0)

borderlinesmote = BorderlineSMOTE(kind='borderline-2',random_state=0)



# resample the training data

X_undersam, y_undersam = undersam.fit_sample(X_train,y_train)

X_oversam, y_oversam = oversam.fit_sample(X_train,y_train)

X_smote, y_smote = smote.fit_sample(X_train,y_train)

X_borderlinesmote, y_borderlinesmote = borderlinesmote.fit_sample(X_train,y_train)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Fit a logistic regression model to our data

model = LogisticRegression()

model.fit(X_train, y_train)



# Obtain model predictions

y_predicted = model.predict(X_test)
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Create true and false positive rates

false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_predicted)



# Calculate Area Under the Receiver Operating Characteristic Curve 

probs = model.predict_proba(X_test)

roc_auc = roc_auc_score(y_test, probs[:, 1])

print('ROC AUC Score:',roc_auc)



# Obtain precision and recall 

precision, recall, thresholds = precision_recall_curve(y_test, y_predicted)



# Calculate average precision 

average_precision = average_precision_score(y_test, y_predicted)



# Define a roc_curve function

def plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)

    plt.plot([0,1],[0,1], linewidth=5)

    plt.xlim([-0.01, 1])

    plt.ylim([0, 1.01])

    plt.legend(loc='upper right')

    plt.title('Receiver operating characteristic curve (ROC)')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()



# Define a precision_recall_curve function

def plot_pr_curve(recall, precision, average_precision):

    plt.step(recall, precision, color='b', alpha=0.2, where='post')

    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    plt.show()



# Print the classifcation report and confusion matrix

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))



# Plot the roc curve 

plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)



# Plot recall precision curve

plot_pr_curve(recall, precision, average_precision)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import BorderlineSMOTE



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Resample your training data

rus = RandomUnderSampler()

ros = RandomOverSampler()

smote = SMOTE(kind='regular',random_state=5)

blsmote = BorderlineSMOTE(kind='borderline-2',random_state=5)



X_train_rus, y_train_rus = rus.fit_sample(X_train,y_train)

X_train_ros, y_train_ros = ros.fit_sample(X_train,y_train)

X_train_smote, y_train_smote = smote.fit_sample(X_train,y_train)

X_train_blsmote, y_train_blsmote = blsmote.fit_sample(X_train,y_train)



# Fit a logistic regression model to our data

rus_model = LogisticRegression().fit(X_train_rus, y_train_rus)

ros_model = LogisticRegression().fit(X_train_ros, y_train_ros)

smote_model = LogisticRegression().fit(X_train_smote, y_train_smote)

blsmote_model = LogisticRegression().fit(X_train_blsmote, y_train_blsmote)



y_rus = rus_model.predict(X_test)

y_ros = ros_model.predict(X_test)

y_smote = smote_model.predict(X_test)

y_blsmote = blsmote_model.predict(X_test)



print('Classifcation report:\n', classification_report(y_test, y_rus))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = y_rus))

print('*'*25)



print('Classifcation report:\n', classification_report(y_test, y_ros))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = y_ros))

print('*'*25)



print('Classifcation report:\n', classification_report(y_test, y_smote))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = y_smote))

print('*'*25)



print('Classifcation report:\n', classification_report(y_test, y_blsmote))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = y_blsmote))

print('*'*25)

# Import the pipeline module we need for this from imblearn

from imblearn.pipeline import Pipeline 



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Define which resampling method and which ML model to use in the pipeline

resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 

model = LogisticRegression() 



# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model

pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])



# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 

pipeline.fit(X_train, y_train) 

y_predicted = pipeline.predict(X_test)



# Obtain the results from the classification report and confusion matrix 

print('Classifcation report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = y_predicted))
# Import the decision tree model from sklearn

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Fit a logistic regression model to our data

model = DecisionTreeClassifier()

model.fit(X_train, y_train)



# Obtain model predictions

y_predicted = model.predict(X_test)



# Calculate average precision 

average_precision = average_precision_score(y_test, y_predicted)



# Obtain precision and recall 

precision, recall, _ = precision_recall_curve(y_test, y_predicted)



# Plot the recall precision tradeoff

plot_pr_curve(recall, precision, average_precision)



# Print the classifcation report and confusion matrix

print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
# Import the pipeline module we need for this from imblearn

from imblearn.pipeline import Pipeline 

from imblearn.over_sampling import BorderlineSMOTE



# Define which resampling method and which ML model to use in the pipeline

resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 

model = DecisionTreeClassifier() 



# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model

pipeline = Pipeline([('SMOTE', resampling), ('Decision Tree Classifier', model)])



# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 

pipeline.fit(X_train, y_train) 

y_predicted = pipeline.predict(X_test)



# Obtain the results from the classification report and confusion matrix 

print('Classifcation report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',  confusion_matrix(y_true = y_test, y_pred = y_predicted))
# Import the Random Forest Classifier model from sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Fit a logistic regression model to our data

model = RandomForestClassifier(random_state=5)

model.fit(X_train, y_train)



# Obtain model predictions

y_predicted = model.predict(X_test)



# Predict probabilities

probs = model.predict_proba(X_test)



# Calculate average precision 

average_precision = average_precision_score(y_test, y_predicted)



# Obtain precision and recall 

precision, recall, _ = precision_recall_curve(y_test, y_predicted)



# Plot the recall precision tradeoff

plot_pr_curve(recall, precision, average_precision)



# Print the classifcation report and confusion matrix

print(accuracy_score(y_test, y_predicted))

print("AUC ROC score: ", roc_auc_score(y_test, probs[:,1]))



print('Classification report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))
# Import the pipeline module we need for this from imblearn

from imblearn.pipeline import Pipeline 

from imblearn.over_sampling import BorderlineSMOTE



# Define which resampling method and which ML model to use in the pipeline



resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 

model = RandomForestClassifier() 



# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model

pipeline = Pipeline([('SMOTE', resampling), ('Random Forest Classifier', model)])



# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 

pipeline.fit(X_train, y_train) 

y_predicted = pipeline.predict(X_test)



# Predict probabilities

probs = model.predict_proba(X_test)



print(accuracy_score(y_test, y_predicted))

print("AUC ROC score: ", roc_auc_score(y_test, probs[:,1]))

# Obtain the results from the classification report and confusion matrix 



print('Classifcation report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n',  confusion_matrix(y_true = y_test, y_pred = y_predicted))
# Import the Random Forest Classifier model from sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Define the model with balanced subsample

model = RandomForestClassifier(bootstrap=True,

                               class_weight={0:1, 1:12}, # 0: non-fraud , 1:fraud

                               criterion='entropy',

                               max_depth=10, # Change depth of model

                               min_samples_leaf=10, # Change the number of samples in leaf nodes

                               n_estimators=20, # Change the number of trees to use

                               n_jobs=-1, 

                               random_state=5)



# Fit your training model to your training set

model.fit(X_train, y_train)



# Obtain the predicted values and probabilities from the model 

y_predicted = model.predict(X_test)



# Calculate probs

probs = model.predict_proba(X_test)



# Calculate average precision 

average_precision = average_precision_score(y_test, y_predicted)



# Obtain precision and recall 

precision, recall, _ = precision_recall_curve(y_test, y_predicted)



# Plot the recall precision tradeoff

plot_pr_curve(recall, precision, average_precision)



# Print the roc auc score, the classification report and confusion matrix

print("auc roc score: ", roc_auc_score(y_test, probs[:,1]))

print('Classifcation report:\n', classification_report(y_test, y_predicted))

print('Confusion matrix:\n', confusion_matrix(y_test, y_predicted))
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



# Define the parameter sets to test

param_grid = {

    'n_estimators': [1, 30], 

    'max_features': ['auto', 'log2'],  

    'max_depth': [4, 8], 

    'criterion': ['gini', 'entropy']

}



# Define the model to use

model = RandomForestClassifier(random_state=5)



# Combine the parameter sets with the defined model

CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)



# Fit the model to our training data and obtain best parameters

CV_model.fit(X_train, y_train)

CV_model.best_params_
from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Build a RandomForestClassifier using the GridSearchCV parameters

model = RandomForestClassifier(bootstrap=True,

                               class_weight = {0:1,1:12},

                               criterion = 'entropy',

                               n_estimators = 30,

                               max_features = 'auto',

                               min_samples_leaf = 10,

                               max_depth = 8,

                               n_jobs = -1,

                               random_state = 5)



# Fit the model to your training data and get the predicted results

model.fit(X_train,y_train)

y_predicted = model.predict(X_test)



# Calculate average precision 

average_precision = average_precision_score(y_test, y_predicted)



# Obtain precision and recall 

precision, recall, _ = precision_recall_curve(y_test, y_predicted)



# Plot the recall precision tradeoff

plot_pr_curve(recall, precision, average_precision)



# Print the roc_auc_score,Classifcation report and Confusin matrix

probs = model.predict_proba(X_test)

print('roc_auc_score:', roc_auc_score(y_test,probs[:,1]))

print('Classification report:\n',classification_report(y_test,y_predicted))

print('Confusion_matrix:\n',confusion_matrix(y_test,y_predicted))
# Import modules 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier



from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Create the training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



# Define the three classifiers to use in the ensemble

clf1 = LogisticRegression(class_weight={0:1,1:15},random_state=5)

clf2 = RandomForestClassifier(class_weight={0:1,1:12},

                              criterion='entropy',

                              max_depth=10,

                              max_features='auto',

                              min_samples_leaf=10, 

                              n_estimators=20,

                              n_jobs=-1,

                              random_state=5)

clf3 = DecisionTreeClassifier(class_weight='balanced',random_state=5)



# Combine the classifiers in the ensemble model

ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')



# Fit the model to your training data and get the predicted results

ensemble_model.fit(X_train,y_train)

y_predicted = ensemble_model.predict(X_test)



# print roc auc score , Classification report and Confusion matrix of the model

print('Classifier report:\n',classification_report(y_test,y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_test,y_predicted))
# Adjust weights within the Voting Classifier



# Define the ensemble model

ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], 

                                  voting='soft', 

                                  weights=[1, 4, 1], 

                                  flatten_transform=True)



# Fit the model to your training data and get the predicted results

ensemble_model.fit(X_train,y_train)

y_predicted = ensemble_model.predict(X_test)



# Calculate average precision 

average_precision = average_precision_score(y_test, y_predicted)



# Obtain precision and recall 

precision, recall, _ = precision_recall_curve(y_test, y_predicted)



# Plot the recall precision tradeoff

plot_pr_curve(recall, precision, average_precision)



# print roc auc score , Classification report and Confusion matrix of the model

print('Classifier report:\n',classification_report(y_test,y_predicted))

print('Confusion matrix:\n',confusion_matrix(y_test,y_predicted))

ensemble_model.estimators_
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize



# Split the data into train set and test set

train,test = train_test_split(df,test_size=0.3,random_state=0)



# Get the arrays of features and labels in train dataset

features_train = train.drop(['Time','Class'],axis=1)

features_train = features_train.values

labels_train = pd.DataFrame(train[['Class']])

labels_train = labels_train.values



# Get the arrays of features and labels in test dataset

features_test = test.drop(['Time','Class'],axis=1)

features_test = features_test.values

labels_test = pd.DataFrame(test[["Class"]])

labels_test = labels_test.values



# Normalize the features in both train and test dataset

features_train = normalize(features_train)

features_test = normalize(features_test)
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix



model = KMeans(n_clusters=2,random_state=0)

model.fit(features_train)

labels_train_predicted = model.predict(features_train)

labels_test_predicted = model.predict(features_test)



# Decide if model predicted label is aligned with true label 

true_negative,false_positive,false_negative,true_positive = confusion_matrix(labels_train,labels_train_predicted).ravel()

reassignflag = true_negative + true_positive < false_positive + false_negative

print(reassignflag)





labels_test_predicted = 1- labels_test_predicted

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score,f1_score

# Calculating confusion matrix for kmeans

print('Confusion Matrix:\n',confusion_matrix(labels_test,labels_test_predicted))



# Scoring kmeans



print('kmeans_precison_score:', precision_score(labels_test,labels_test_predicted))

print('kmeans_recall_score:', recall_score(labels_test,labels_test_predicted))

print('kmeans_accuracy_score:', accuracy_score(labels_test,labels_test_predicted))

print('kmeans_f1_score:',f1_score(labels_test,labels_test_predicted))
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize



# Split the data into train set and test set

train,test = train_test_split(df,test_size=0.3,random_state=0)



# Get the arrays of features and labels in train dataset

features_train = train.drop(['Time','Class'],axis=1)

features_train = features_train.values

labels_train = pd.DataFrame(train[['Class']])

labels_train = labels_train.values



# Get the arrays of features and labels in test dataset

features_test = test.drop(['Time','Class'],axis=1)

features_test = features_test.values

labels_test = pd.DataFrame(test[["Class"]])

labels_test = labels_test.values



# Normalize the features in both train and test dataset

features_train = normalize(features_train)

features_test = normalize(features_test)
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import confusion_matrix



model = MiniBatchKMeans(n_clusters=2,random_state=0)

model.fit(features_train)

labels_train_predicted = model.predict(features_train)

labels_test_predicted = model.predict(features_test)



# Decide if model predicted label is aligned with true label 

true_negative,false_positive,false_negative,true_positive = confusion_matrix(labels_train,labels_train_predicted).ravel()

reassignflag = true_negative + true_positive < false_positive + false_negative

print(reassignflag)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score,f1_score

# Calculating confusion matrix for kmeans

print('Confusion Matrix:\n',confusion_matrix(labels_test,labels_test_predicted))



# Scoring kmeans



print('kmeans_precison_score:', precision_score(labels_test,labels_test_predicted))

print('kmeans_recall_score:', recall_score(labels_test,labels_test_predicted))

print('kmeans_accuracy_score:', accuracy_score(labels_test,labels_test_predicted))

print('kmeans_f1_score:',f1_score(labels_test,labels_test_predicted))
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Make another copy of df and drop the unimportant "Time" feature

data = df.drop(['Time'], axis=1) 



# Use scikit’s StandardScaler on the "Amount" feature

# The scaler removes the mean and scales the values to unit variance

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))



# Create the training and testing sets

X1_train, X1_test = train_test_split(data, test_size=.3, random_state=0)

X1_train = X1_train[X1_train.Class == 0] # train the model on normal transactions

X1_train = X1_train.drop(['Class'], axis=1)



y1_test = X1_test['Class']

X1_test  = X1_test.drop(['Class'], axis=1) #drop the class column





#transform to ndarray

X1_train = X1_train.values

X1_test = X1_test.values

X1_train.shape
import tensorflow as tf

from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers



input_dim = X1_train.shape[1] #num of columns, 29

encoding_dim = 14

hidden_dim = int(encoding_dim / 2)

learning_rate = 1e-5



input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, 

                activation="tanh", 

                activity_regularizer=regularizers.l1(learning_rate))(input_layer)

encoder = Dense(hidden_dim, activation="relu")(encoder)

decoder = Dense(hidden_dim, activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
nb_epoch = 100

batch_size = 128

autoencoder.compile(metrics=['accuracy'],

                    loss='mean_squared_error',

                    optimizer='adam')



checkpointer = ModelCheckpoint(filepath='autoencoder_fraud.h5',

                               save_best_only=True,

                               verbose=0)



tensorboard = TensorBoard(log_dir='./logs',

                          histogram_freq=0,

                          write_graph=True,

                          write_images=True)



history = autoencoder.fit(X1_train, X1_train,

                          epochs=nb_epoch,

                          batch_size=batch_size,

                          shuffle=True,

                          validation_data=(X1_test, X1_test),

                          verbose=1,

                          callbacks=[checkpointer, tensorboard]).history

load_model('autoencoder_fraud.h5')

plt.plot(history['loss'])

plt.plot(history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')
predictions = autoencoder.predict(X1_test)

mse = np.mean(np.power(X1_test - predictions, 2), axis=1)

df_error = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y1_test})

df_error.describe()
# Import modules

from sklearn.metrics import auc, roc_curve,precision_recall_curve

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import recall_score,f1_score,precision_recall_fscore_support



false_positive_rate, true_positive_rate, thresholds = roc_curve(df_error.true_class, df_error.reconstruction_error)

roc_auc = auc(false_positive_rate, true_positive_rate)



# Plot the roc curve 

plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)
precision, recall, thresholds = precision_recall_curve(df_error.true_class, df_error.reconstruction_error)



# Plot recall precision tradeoff

plt.plot(recall, precision, linewidth=5, label='Precision-Recall curve')

plt.title('Recall vs Precision')

plt.xlabel('Recall')

plt.ylabel('Precision')

print(plt.show())



# Plot precision and recall for different thresholds

plt.plot(thresholds, precision[1:], label="Precision",linewidth=5)

plt.plot(thresholds, recall[1:], label="Recall",linewidth=5)

plt.title('Precision and recall for different threshold values')

plt.xlabel('Threshold')

plt.ylabel('Precision/Recall')

plt.legend()

print(plt.show())
# Set a threshold

set_threshold = 5

groups = df_error.groupby('true_class')

fig, ax = plt.subplots()



for name, group in groups:

    ax.plot(group.index, 

            group.reconstruction_error, 

            marker='o', 

            ms=3.5, 

            linestyle='',

            label= "Fraud" if name == 1 else "Nonfraud")

    

ax.hlines(set_threshold, 

          ax.get_xlim()[0], 

          ax.get_xlim()[1], 

          colors="r", 

          zorder=100, 

          label='Threshold')



ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.show()
y_pred = [1 if e > set_threshold else 0 for e in df_error.reconstruction_error.values]

print('Confusion_matrix:\n',confusion_matrix(df_error.true_class, y_pred))