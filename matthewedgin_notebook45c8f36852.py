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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
#bring in libraries for potential tests to run
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, confusion_matrix, precision_score, recall_score, accuracy_score, precision_recall_curve, auc,roc_auc_score,roc_curve, classification_report
from scipy.stats import jarque_bera
from scipy.stats import normaltest
from scipy.stats import boxcox
from sklearn.neighbors import KNeighborsClassifier
!pip install sklearn
from sklearn import preprocessing, neighbors, ensemble

import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()
# Checking the size of our data (rows, columns)
print(data.shape)
data.info()
# find percent missing values
data.isna().mean()
from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data['normTime'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)

data.head()
import seaborn as sns
# Calculate correlations
corr = data.corr()
# Heatmap
fig, ax = plt.subplots(figsize=(15,12))
plt.rcParams['font.size'] = 16
ax.set_title('Correlations of Credit Card Fraud')
 
sns.heatmap(corr, ax = ax, cmap = 'coolwarm')
target_count = data.Class.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (Class)');
data['Class'].value_counts(normalize=True) * 100
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Remove 'id' and 'target' columns
labels = data.drop(['Class'], axis=1)

X = labels
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
model = XGBClassifier()
model.fit(X_train[['V28']], y_train)
y_pred = model.predict(X_test[['V28']])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import confusion_matrix

X = labels
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print ("Recall metric in the testing dataset: ", recall_score(y_test, y_pred))
# Class count
count_class_0, count_class_1 = data.Class.value_counts()

# Divide by class
df_class_0 = data[data['Class'] == 0]
df_class_1 = data[data['Class'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.Class.value_counts())

df_test_under.Class.value_counts().plot(kind='bar', title='Count (Class)');
#organize original dataset for testing models at the end
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = ['Class']),data['Class'],test_size = 0.2, random_state = 42, stratify = data['Class'])
#organizing and sorting undersampled dataset for training

# number of fraud cases
fraud = len(data[data['Class'] == 1])

#get indices of non fraud samples
non_fraud_indices = data[data.Class == 0].index

#Random sample non fraud indices
random_indices = np.random.choice(non_fraud_indices,fraud, replace=False)

#find fraud samples
fraud_indices = data[data.Class == 1].index

#concat fraud indices with sample non-fraud ones
under_sample_indices = np.concatenate([fraud_indices,random_indices])

#balance the DF
under_sample = data.loc[under_sample_indices]


# histograms of new data
print(under_sample.Class.value_counts())
under_sample.Class.value_counts().plot(kind='bar', title='Count (Class)');
#setup train and test sets
X_under = under_sample.loc[:,under_sample.columns != 'Class']
y_under = under_sample.loc[:,under_sample.columns == 'Class']
X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under,y_under,test_size = 0.2, random_state = 2)
# example of grid searching key hyperparametres for logistic regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
# define dataset
X = X_under_train
y = y_under_train
# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalty = ['l2', 'l1']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
lr_under = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg')
lr_under.fit(X_under_train,y_under_train)
y_under_pred = lr_under.predict(X_under_test)

# Score the model on it's accuracy using cross validation
lr_score = cross_val_score(lr_under, X_under_test, y_under_test, cv=7)
print(f'Logistic Regression Score: {lr_score.mean(): .3f} +/- {2*lr_score.std(): .4f}')
import seaborn as sns
# Predicting the classes from the test set
y_pred_lr = lr_under.predict(X_under_test)

# Creating a confusion matrix
cm_dt = confusion_matrix(y_under_test, y_pred_lr)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_dt/np.sum(cm_dt), annot=True, cmap='Blues', 
            fmt='.2%', xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y_under_test, y_pred_lr))
# How to optimize hyper-parameters of a DecisionTree model using Grid Search in Python ?
def Snippet_146_Ex_2():
    print('**Optimizing hyper-parameters of a Decision Tree model using Grid Search in Python**\n')


    # importing libraries
    from sklearn import decomposition, datasets
    from sklearn import tree
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    # Loading wine dataset
    data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
    X = X_under_train
    y = y_under_train

    # Creating an standardscaler object
    std_slc = StandardScaler()

    # Creating a pca object
    pca = decomposition.PCA()

    # Creating a DecisionTreeClassifier
    dec_tree = tree.DecisionTreeClassifier()

    # Creating a pipeline of three steps. First, standardizing the data.
    # Second, tranforming the data with PCA.
    # Third, training a Decision Tree Classifier on the data.
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])

    # Creating Parameter Space
    # Creating a list of a sequence of integers from 1 to 30 (the number of features in X + 1)
    n_components = list(range(1,X.shape[1]+1,1))

    # Creating lists of parameter for Decision Tree Classifier
    criterion = ['gini', 'entropy']
    max_depth = [2,4,6,8,10,12]

    # Creating a dictionary of all the parameter options 
    # Note that we can access the parameters of steps of a pipeline by using '__’
    parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

    # Conducting Parameter Optmization With Pipeline
    # Creating a grid search object
    clf_GS = GridSearchCV(pipe, parameters)

    # Fitting the grid search
    clf_GS.fit(X, y)

    # Viewing The Best Parameters
    print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
    print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
    print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
    print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])

   
Snippet_146_Ex_2()
# Import model
from sklearn.tree import DecisionTreeClassifier

# Instantiate (start up) model
dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6, random_state=42)
# Fit model on the training data
dt.fit(X_under_train, y_under_train)

# Score the model on it's accuracy using cross validation
dt_score = cross_val_score(dt, X_under_test, y_under_test, cv=29)
print(f'Decision Tree Score: {dt_score.mean(): .3f} +/- {2*dt_score.std(): .4f}')

# Predicting the classes from the test set
y_pred_dt = dt.predict(X_under_test)

# Creating a confusion matrix
cm_dt = confusion_matrix(y_under_test, y_pred_dt)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_dt/np.sum(cm_dt), annot=True, cmap='Blues', 
            fmt='.2%', xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y_under_test, y_pred_dt))
# Import model
from sklearn.ensemble import RandomForestClassifier
# define dataset
X = X_under_train
y = y_under_train
# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2', 'auto']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Instantiate (start up) model
rf = RandomForestClassifier(max_features = 'sqrt', n_estimators = 1000, random_state=42)
# Fit model on the training data
rf.fit(X_under_train, y_under_train)

# Score the model on it's accuracy using cross validation
rf_score = cross_val_score(rf, X_under_test, y_under_test, cv=3)
print(f'Random Forest Score: {rf_score.mean(): .5f} +/- {2*rf_score.std(): .6f}')
# Predicting the classes from the test set
y_under_pred_rf = rf.predict(X_under_test)

# Creating a confusion matrix
cm_rf = confusion_matrix(y_under_test, y_under_pred_rf)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_rf/np.sum(cm_rf), annot=True, cmap='Blues',
            fmt='.2%', xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y_under_test, y_under_pred_rf))
# define dataset
X = X_under_train
y = y_under_train
# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Import model
from sklearn.svm import SVC

# Instantiate (start up) model
svc = SVC(C = 10, gamma ='scale', kernel = 'rbf', random_state = 42)
# Fit model on the training data
svc.fit(X_under_train, y_under_train)

# Score the model on it's accuracy using cross validation
svc_score = cross_val_score(svc, X_test, y_test, cv=3)
print(f'Support Vector Machine Classifier Score: {svc_score.mean(): .3f} +/- {2*svc_score.std(): .4f}')
# Predicting the classes from the test set
y_pred_svc = svc.predict(X_under_test)

# Creating a confusion matrix
cm_svc = confusion_matrix(y_under_test, y_pred_svc)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_svc/np.sum(cm_svc), annot=True, cmap='Blues',
            fmt='.2%', xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y_under_test, y_pred_svc))
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg')
lr.fit(X_under_train, y_under_train.values.ravel(),)
y_pred_undersample = lr.predict(X_under_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_under_test,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg')
lr.fit(X_under_train,y_under_train.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

# Score the model on it's accuracy using cross validation
lr_score = cross_val_score(lr, X_test, y_test, cv=7)
print(f'Logistic Regression Score: {lr_score.mean(): .3f} +/- {2*lr_score.std(): .4f}')
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
# Instantiate (start up) model
svc = SVC(C = 10, gamma ='scale', kernel = 'rbf', random_state = 42)
# Fit model on the training data
svc.fit(X_under_train, y_under_train)

# Score the model on it's accuracy using cross validation
svc_score = cross_val_score(svc, X_test, y_test, cv=3)
print(f'Support Vector Machine Classifier Score: {svc_score.mean(): .3f} +/- {2*svc_score.std(): .4f}')

# Predicting the classes from the test set
y_pred_svc = svc.predict(X_test.values)

# Creating a confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_svc)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_rf/np.sum(cm_rf), annot=True, cmap='Blues',
            fmt='.2%', xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y_test, y_pred_svc))
# ROC CURVE
lr = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg' )
y_pred_undersample_score = lr.fit(X_under_train,y_under_train.values.ravel()).decision_function(X_under_test.values)

fpr, tpr, thresholds = roc_curve(y_under_test.values.ravel(),y_pred_undersample_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg')
lr.fit(X_train,y_train.values.ravel())
y_pred_undersample = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred_undersample)
np.set_printoptions(precision=2)

# Score the model on it's accuracy using cross validation
lr_score = cross_val_score(lr, X_test, y_test, cv=7)
print(f'Logistic Regression Score: {lr_score.mean(): .3f} +/- {2*lr_score.std(): .4f}')
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
lr = LogisticRegression(C = 0.1, penalty = 'l2', solver = 'newton-cg')
lr.fit(X_under_train,y_under_train.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_under_test.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10,10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i
    
    plt.subplot(3,3,j)
    j += 1
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_under_test,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i)
from itertools import cycle

lr = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg')
lr.fit(X_under_train, y_under_train.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_under_test.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black'])

plt.figure(figsize=(10,10))

j = 1
for i,color in zip(thresholds,colors):
    y_test_predictions_prob = y_pred_undersample_proba[:,1] > i
    
    precision, recall, thresholds = precision_recall_curve(y_under_test,y_test_predictions_prob)
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,
                 label='Threshold: %s'%i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1.1])
    plt.xlim([0.7, 1.0])
    plt.title('Precision-Recall example')
    plt.legend(loc="lower left", prop={'size': 16})
corr = data.corrwith(data['Class']).sort_values(ascending = False)
print(corr.head(10))
corr = data.corrwith(data['Class']).sort_values(ascending = True)
print(corr.head(12))
# Creating a DataFrame with feature importances
impt = pd.DataFrame(zip(X_test.columns, rf.feature_importances_), columns=['Column name', 'Feature Importance'])
impt.sort_values(by='Feature Importance', ascending=False)
print(impt.sort_values(by='Feature Importance', ascending=False).head(17))
print(impt.sort_values(by='Feature Importance', ascending=False).head(17).sum())
#create new dataset containing only the 17 features of interest.

new_data = data[['Class','V14', 'V4', 'V10', 'V12', 'V17',
                'V11', 'V3', 'V16', 'V2', 'V9', 'V21',
                'V19', 'V7', 'normAmount', 'V6', 'V18', 'V26']]
new_data.info()
#create new TT split so that enitre dataset and undersampled one are equal in shape with the 17 features of interest
# Remove 'id' and 'target' columns
labels = data.drop(['Class'], axis=1)

X = labels
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = ['Class']),data['Class'],test_size = 0.2, random_state = 42)
X_test.info()
#create new dataset only containg 17 features of interest

# number of fraud cases
fraud = len(new_data[new_data['Class'] == 1])

#get indices of non fraud samples
non_fraud_indices = new_data[new_data.Class == 0].index

#Random sample non fraud indices
random_indices = np.random.choice(non_fraud_indices,fraud, replace=False)

#find fraud samples
fraud_indices = new_data[new_data.Class == 1].index

#concat fraud indices with sample non-fraud ones
under_sample_indices = np.concatenate([fraud_indices,random_indices])

#balance the DF
under_sample = new_data.loc[under_sample_indices]


# histograms of new data
print(under_sample.Class.value_counts())
under_sample.Class.value_counts().plot(kind='bar', title='Count (Class)');
#setup train and test sets with the 17 features of interest for the undersampling training set and test to check

X2_under = under_sample.loc[:,under_sample.columns != 'Class']
y2_under = under_sample.loc[:,under_sample.columns == 'Class']
X2_under_train, X2_under_test, y2_under_train, y2_under_test = train_test_split(X_under,y_under,test_size = 0.2, random_state = 2)
X2_under.info()
X.info()
# define dataset
X = X2_under_train
y = y2_under_train
# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalty = ['l2', 'l1']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg')
lr.fit(X2_under_train,y2_under_train.values.ravel())
y_pred_undersample = lr.predict(X2_under_test.values)

# Score the model on it's accuracy using cross validation
lr_score = cross_val_score(lr, X2_under_test, y2_under_test, cv=7)
print(f'Logistic Regression Score: {lr_score.mean(): .3f} +/- {2*lr_score.std(): .4f}')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y2_under_test,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
# How to optimize hyper-parameters of a DecisionTree model using Grid Search in Python ?
def Snippet_146_Ex_2():
    print('**Optimizing hyper-parameters of a Decision Tree model using Grid Search in Python**\n')

    # importing libraries
    from sklearn import decomposition, datasets
    from sklearn import tree
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    
    # Loading wine dataset
    data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
    X = X2_under_train
    y = y2_under_train

    # Creating an standardscaler object
    std_slc = StandardScaler()

    # Creating a pca object
    pca = decomposition.PCA()

    # Creating a DecisionTreeClassifier
    dec_tree = tree.DecisionTreeClassifier()

    # Creating a pipeline of three steps. First, standardizing the data.
    # Second, tranforming the data with PCA.
    # Third, training a Decision Tree Classifier on the data.
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])

    # Creating Parameter Space
    # Creating a list of a sequence of integers from 1 to 30 (the number of features in X + 1)
    n_components = list(range(1,X.shape[1]+1,1))

    # Creating lists of parameter for Decision Tree Classifier
    criterion = ['gini', 'entropy']
    max_depth = [2,4,6,8,10,12]

    # Creating a dictionary of all the parameter options 
    # Note that we can access the parameters of steps of a pipeline by using '__’
    parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

    # Conducting Parameter Optmization With Pipeline
    # Creating a grid search object
    clf_GS = GridSearchCV(pipe, parameters)

    # Fitting the grid search
    clf_GS.fit(X, y)

    # Viewing The Best Parameters
    print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
    print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
    print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
    print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])

   
Snippet_146_Ex_2()
# Instantiate (start up) model
dt = DecisionTreeClassifier( criterion = 'entropy', max_depth = 4, random_state=42)
# Fit model on the training data
dt.fit(X2_under_train, y2_under_train)

# Score the model on it's accuracy using cross validation
dt_score = cross_val_score(dt, X2_under_test, y2_under_test, cv=29)
print(f'Decision Tree Score: {dt_score.mean(): .3f} +/- {2*dt_score.std(): .4f}')
import seaborn as sns
# Predicting the classes from the test set
y_pred_dt = dt.predict(X2_under_test)

# Creating a confusion matrix
cm_dt = confusion_matrix(y2_under_test, y_pred_dt)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_dt/np.sum(cm_dt), annot=True, cmap='Blues', 
            fmt='.2%')
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y2_under_test, y_pred_dt))
# define dataset
X = X2_under_train
y = y2_under_train
# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2', 'auto']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Instantiate (start up) model
rf = RandomForestClassifier(max_features = 'auto', n_estimators = 100, random_state=42)
# Fit model on the training data
rf.fit(X2_under_train, y2_under_train)

# Score the model on it's accuracy using cross validation
rf_score = cross_val_score(rf, X2_under_test, y2_under_test, cv=3)
print(f'Random Forest Score: {rf_score.mean(): .5f} +/- {2*rf_score.std(): .6f}')
# Predicting the classes from the test set
y_under_pred_rf = rf.predict(X2_under_test)

# Creating a confusion matrix
cm_rf = confusion_matrix(y2_under_test, y_under_pred_rf)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_rf/np.sum(cm_rf), annot=True, cmap='Blues',
            fmt='.2%')
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y2_under_test, y_under_pred_rf))
# define dataset
X = X2_under_train
y = y2_under_train
# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Instantiate (start up) model
svc = SVC(C = 10, gamma ='scale', kernel = 'rbf', random_state=42)
# Fit model on the training data
svc.fit(X2_under_train, y2_under_train)

# Score the model on it's accuracy using cross validation
svc_score = cross_val_score(svc, X_test, y_test, cv=3)
print(f'Support Vector Machine Classifier Score: {svc_score.mean(): .3f} +/- {2*svc_score.std(): .4f}')
# Predicting the classes from the test set
y_pred_svc = svc.predict(X2_under_test)

# Creating a confusion matrix
cm_svc = confusion_matrix(y2_under_test, y_pred_svc)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_svc/np.sum(cm_svc), annot=True, cmap='Blues',
            fmt='.2%')
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y2_under_test, y_pred_svc))
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = .1, penalty = 'l2', solver = 'newton-cg')
lr.fit(X2_under_train,y2_under_train.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

# Score the model on it's accuracy using cross validation
lr_score = cross_val_score(lr, X_test, y_test, cv=7)
print(f'Logistic Regression Score: {lr_score.mean(): .3f} +/- {2*lr_score.std(): .4f}')
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
# Instantiate (start up) model
svc = SVC(C = 10, gamma ='scale', kernel = 'rbf', random_state = 42)
# Fit model on the training data
svc.fit(X2_under_train, y2_under_train)

# Score the model on it's accuracy using cross validation
svc_score = cross_val_score(svc, X_test, y_test, cv=3)
print(f'Support Vector Machine Classifier Score: {svc_score.mean(): .3f} +/- {2*svc_score.std(): .4f}')

# Predicting the classes from the test set
y_pred_svc = svc.predict(X_test.values)

# Creating a confusion matrix
cm_svc = confusion_matrix(y_test, y_pred_svc)
# Visualizing the confusion matrix as a heatmap
sns.heatmap(cm_svc/np.sum(cm_svc), annot=True, cmap='Blues',
            fmt='.2%')
plt.ylabel('True label')
plt.xlabel('Predicted label');
print ("Recall metric in the testing dataset: ", recall_score(y_test, y_pred_svc))
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V14", data=new_data, palette=colors, ax=axes[0])
axes[0].set_title('V14 vs Class')

sns.boxplot(x="Class", y="V10", data=new_data, palette=colors, ax=axes[1])
axes[1].set_title('V10 vs Class')


sns.boxplot(x="Class", y="V17", data=new_data, palette=colors, ax=axes[2])
axes[2].set_title('V17 vs Class')


sns.boxplot(x="Class", y="V12", data=new_data, palette=colors, ax=axes[3])
axes[3].set_title('V12 vs Class')

plt.show()