import warnings
warnings.filterwarnings('ignore')
# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_confusion_matrix

%matplotlib inline
pd.pandas.set_option('display.max_columns', None)
# Read data from csv file
df_cc = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

# Print first few rows
df_cc.head()
# Shape of the dataset
df_cc.shape
# The data is stardarized, I will explore them later For now I will look the "normal" columns
df_cc[["Time","Amount","Class"]].describe()
# Explore the Features available in DataFrame
print(df_cc.info())
print()
print(f"There are {df_cc.isnull().sum().max()} NULL values in Dataset")
# The general statistics of frauds and no frauds data
df_fraud = df_cc[df_cc['Class'] == 1]
df_normal = df_cc[df_cc['Class'] == 0]

print("Fraud Transaction Statistics")
print(df_fraud["Amount"].describe())
print("\nNormal Transaction Statistics")
print(df_normal["Amount"].describe())
df_cc['Time'].max()
plt.figure(figsize=(7,5))
sns.distplot(df_cc["Time"])
plt.xlabel('Time (in seconds)')
plt.title('Distribution of Time');
fraud_time = df_cc[df_cc['Class'] == 1]['Time']
no_fraud_time = df_cc[df_cc['Class'] == 0]['Time']

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,5))
bins=50

ax1.hist(fraud_time, bins = bins)
ax1.set_title('Fraud')

ax2.hist(no_fraud_time, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
fig.text(0.04,0.5, 'Number of Transactions', va='center', rotation='vertical')

plt.show()
fraud_amt = df_cc[df_cc['Class'] == 1]['Amount']
no_fraud_amt = df_cc[df_cc['Class'] == 0]['Amount']

plt.subplots(1, 2, figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.distplot(no_fraud_amt)
plt.xlabel('Amount ($)')
plt.title('Distribution of Non-Fraudulent Data Amount')

plt.subplot(1, 2, 2)
sns.distplot(fraud_amt)
plt.xlabel('Amount ($)')
plt.title('Distribution of Fraudulent Data Amount');
plt.subplots(1, 2, figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Class', y='Amount', hue='Class', data=df_cc, showfliers=True)

plt.subplot(1, 2, 2)
sns.boxplot(x='Class', y='Amount', hue='Class', data=df_cc, showfliers=False);
print(f"Fraud Amount Info: \n {fraud_amt.describe()}")
print()
print(f"Non-Fraud Amount Info: \n {no_fraud_amt.describe()}")
# Count the occurences of Fraud and Non-Fraud Cases
occ = df_cc['Class'].value_counts()
print(f"Total NON-FRAUD CASES: {occ[0]}, {(occ[0]/len(df_cc.index)*100):0.3f}%")
print(f"Total FRAUD CASES: {occ[1]}, {(occ[1]/len(df_cc.index)*100):0.3f}%")

plt.bar(x=occ.index, height=occ.values, data=occ, color=['#5976A2', '#CB8866'])
plt.title('Class Distribution');
plt.yscale('log')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
plt.yticks([500,3000,10000,30000,100000,300000], ['500','3K','10K','30K', '100K', '300K'])
plt.show();
plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions Features Correlation Plot (Pearson)')
corr = df_cc.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Blues")
plt.show()
def prep_undersampled_data(df):
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class']==0][:fraud_df.shape[0]]
    undersampled_df = pd.concat([fraud_df, non_fraud_df])
    df_col = [column for column in undersampled_df.columns if column not in ['Time','Class']]
    X = undersampled_df.loc[:, df_col]
    X = np.array(X).astype(np.float)
    y = undersampled_df.loc[:, undersampled_df.columns == 'Class']
    y = np.array(y).astype(np.float).reshape(-1,)
    return X, y
X_under, y_under = prep_undersampled_data(df_cc)
# TSNE
tsne = manifold.TSNE(n_components=2, random_state=2020)
transformed_data = tsne.fit_transform(X_under)
tsne_df = pd.DataFrame(np.column_stack((transformed_data, y_under)), columns=["X","Y","Targets"])
tsne_df.loc[:,"Targets"] = tsne_df.Targets.astype(int)
# PCA
pca = PCA(n_components=2, random_state=2020)
transformed_data = pca.fit_transform(X_under)
pca_df = pd.DataFrame(np.column_stack((transformed_data, y_under)), columns=["X","Y","Targets"])
pca_df.loc[:,"Targets"] = pca_df.Targets.astype(int)
ax, f = plt.subplots(1, 2, figsize=(24,10))

plt.subplot(121)
sns.scatterplot("X","Y", hue='Targets', data=tsne_df)
plt.title('TSNE', fontsize=14)
plt.grid(True)

plt.subplot(122)
sns.scatterplot("X","Y", hue='Targets', data=pca_df)
plt.title('PCA', fontsize=14)
plt.grid(True)

plt.show()
df_cc['normAmount'] = StandardScaler().fit_transform(df_cc['Amount'].values.reshape(-1, 1))
df_cc = df_cc.drop(['Amount'],axis=1)
df_cc.head()
def prep_data(df):
    df_col = [column for column in df.columns if column not in ['Time','Class']]
    X = df.loc[:, df_col]
    X = np.array(X).astype(np.float)
    y = df.loc[:, df.columns == 'Class']
    y = np.array(y).astype(np.float).reshape(-1,)
    return X, y
# Define a function to create a scatter plot of our data and labels
def plot_data(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class: Non-Fraud", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class: Fraud", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()
# Create X and y from our above defined function
X, y = prep_data(df_cc)
# Plot our data by running plot_data function on X and y
plot_data(X, y)
# Define the resampling method
smote_method = SMOTE()

# Create the resampled feature set
X_resampled, y_resampled = smote_method.fit_sample(X, y)

# Plot the resampled data
plot_data(X_resampled, y_resampled)
X_resampled.shape, y_resampled.shape
def compare_plot(X,y,X_resampled,y_resampled, method):
    # Start a plot figure
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # sub-plot number 1, this is our normal data
    c0 = ax1.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0",alpha=0.5)
    c1 = ax1.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1",alpha=0.5, c='r')
    ax1.set_title('Original set')
    
    # sub-plot number 2, this is our oversampled data
    ax2.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=.5)
    ax2.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=.5,c='r')
    ax2.set_title(method)
    
    plt.figlegend((c0, c1), ('Class: Non-Fraud', 'Class: Fraud'), loc='lower center',
                  ncol=2, labelspacing=0.)
    return plt.show()
print(f"Total NON-FRAUD cases in Original Dataset: {pd.Series(y).value_counts()[0]}")
print(f"Total FRAUD cases in Original Dataset: {pd.Series(y).value_counts()[1]}")
print()
print(f"Total NON-FRAUD cases in SMOTE Resampled Dataset: {pd.Series(y_resampled).value_counts()[0]}")
print(f"Total FRAUD cases in SMOTE Resampled Dataset: {pd.Series(y_resampled).value_counts()[1]}")

compare_plot(X, y, X_resampled, y_resampled, method="SMOTE")
def plot_roc_curve(true_y, pred_y):
    """
    Plot the ROC curve along with the curves AUC for a given model. Note make sure true_y and pred_y are from the same model as model_name
    :param model_name: Name of model used for saving plot
    :param true_y: true labels for dataset
    :param pred_y: predicted labels for dataset
    """
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    fpr, tpr, thresholds = roc_curve(true_y, pred_y)
    ax.plot(fpr, tpr, label=f'AUC: {auc(fpr, tpr):.2f}')
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    
    return
# Let's Split Resampled Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)
# Logistic Regression Combined with SMOTE
resampling = SMOTE()
model_lr = LogisticRegression()

pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model_lr)])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
# Print the Classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, predictions))

print(f'ROC-AUC Score: {roc_auc_score(y_test, predictions):.2f}\n')

conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions)

print('Confusion matrix:\n', conf_mat)

tn, fp, fn, tp = conf_mat.ravel()

print()
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")
plot_roc_curve(y_test, predictions)
# Split data into train and test set (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)
# Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Obtain Model Prediction
predictions_lr = model_lr.predict(X_test)
# Print the Classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, predictions_lr))

print(f'ROC-AUC Score: {roc_auc_score(y_test, predictions_lr):.2f}\n')

conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions_lr)

print('Confusion matrix:\n', conf_mat)

tn,fp,fn,tp = conf_mat.ravel()

print()
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")
plot_roc_curve(y_test, predictions_lr)
# Logistic Regression Combined with SMOTE
resampling = BorderlineSMOTE()

pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model_lr)])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
# Print the Classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, predictions))

print(f'ROC-AUC Score: {roc_auc_score(y_test, predictions):.2f}\n')

conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions)

print('Confusion matrix:\n', conf_mat)

tn,fp,fn,tp = conf_mat.ravel()
print()
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")
plot_roc_curve(y_test, predictions)
# Count the Total Number of Observations from the length of y
total_obs = len(y)

# Count the total number of non-fraudulent observations
non_fraud = [i for i in y if i==0]
count_non_fraud = non_fraud.count(0)

# Percentage of Non-Fraud Observations
percentage = count_non_fraud / total_obs * 100
print(f"Percentage of NON-FRAUD observations: {percentage:0.2f}%")
# Random Forest Model
model_rf = RandomForestClassifier(random_state=2020, n_estimators=20)

# Fit the model to our training set
model_rf.fit(X_train, y_train)

# Obtain predictions from the test data
predictions_rf = model_rf.predict(X_test)

# Predict Probabilities
probs = model_rf.predict_proba(X_test)

# Print Accuracy Score
print(f"Accuracy Score : {accuracy_score(y_test, predictions_rf):0.4f}%")
print()
# Print ROC Score
print(f"ROC Score : {roc_auc_score(y_test, probs[:,1])}")
print()
# Print the Classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, predictions_rf))

print(f'ROC-AUC Score: {roc_auc_score(y_test, predictions_rf):.2f}\n')

conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions_rf)

print()

print('Confusion matrix:\n', conf_mat)
tn,fp,fn,tp = conf_mat.ravel()
print()
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")
plot_roc_curve(y_test, predictions_rf)
# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, predictions_rf)
print(f'Average Precision: {average_precision:.3f}%')
# Obtain Precision and Recall 
precision, recall, _ = precision_recall_curve(y_test, predictions_rf)
print(f'Precision: {precision}\nRecall: {recall}')
def plot_pr_curve(recall, precision, average_precision):
    from inspect import signature
    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(f'2-Class Precision-Recall curve: AP = {average_precision:0.2f}')
    return plt.show()
# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
# Define the model with balanced subsample
model = RandomForestClassifier(class_weight='balanced_subsample', random_state=2020, n_estimators=100)

# Fit your training model to your training set
model.fit(X_train, y_train);
# Obtain the predicted values and probabilities from the model 
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

print('\nClassification Report:')

print(classification_report(y_test, predicted))

print(f'ROC-AUC Score: {roc_auc_score(y_test, predicted):.2f}\n')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))

tn,fp,fn,tp = confusion_matrix(y_test, predicted).ravel()
print()
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")
plot_roc_curve(y_test, predicted)
def get_model_results(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, model):
    """
    model: sklearn model (e.g. RandomForestClassifier)
    """
    # Fit your training model to your training set
    model.fit(X_train, y_train)

    # Obtain the predicted values and probabilities from the model 
    predicted = model.predict(X_test)

    print('\nClassification Report:')
    
    print(classification_report(y_test, predicted))

    print(f'ROC-AUC Score: {roc_auc_score(y_test, predicted):.2f}\n')

    print('\nConfusion Matrix:')
    plt.figure()
    cm = confusion_matrix(y_test, predicted)
    
    plot_confusion_matrix(cm, figsize=(4,4), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Normal', 'Fraud'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Fraud'], fontsize=16)
    plt.show()
    plot_roc_curve(y_test, predictions);
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    # Total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies
    
    return positive_frequencies, negative_frequencies
# Computing class frequencies for our training set
freq_pos, freq_neg = compute_class_freqs(y_train)
pos_weights = freq_neg
neg_weights = freq_pos
class_weights = {0: neg_weights, 1:pos_weights}
class_weights
# Change the model options
model = RandomForestClassifier(bootstrap=True,
                               class_weight = class_weights,
                               criterion='entropy',
                               # Change depth of model
                               max_depth=10,
                               # Change the number of samples in leaf nodes
                               min_samples_leaf=10, 
                               # Change the number of trees to use
                               n_estimators=20,
                               n_jobs=-1,
                               random_state=2020)

# Run the function get_model_results
get_model_results(X_train, y_train, X_test, y_test, model)
# Define the paramter sets to test
param_grid = {'n_estimators': [1,30],
              'max_features': ['auto', 'log2'],
              'max_depth': [4, 8, 10, 12],
              'criterion': ['gini', 'entropy']}

# Define the mode to use
model = RandomForestClassifier(random_state=2020)

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)

# Fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)
CV_model.best_params_
# Input the optimal parameters in the model
model = RandomForestClassifier(class_weight=class_weights,
                               criterion='gini',
                               max_depth=12,
                               max_features='auto', 
                               min_samples_leaf=10,
                               n_estimators=30,
                               n_jobs=-1,
                               random_state=2020)
model.fit(X_train, y_train)
# Get results from your model
get_model_results(X_train, y_train, X_test, y_test, model);