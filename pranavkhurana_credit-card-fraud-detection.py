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
data = pd.read_csv("../input/creditcard.csv")
print("Shape of input data: "+str(data.shape))
data.head()
count_classes = pd.value_counts(data['Class'])
count_classes.plot(kind = 'bar')
plt.title("Distribution of target classes")
plt.xlabel("Class")
plt.ylabel("Frequency")
features = ['V%d' % number for number in range(1, 29)]
target = 'Class'
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.gridspec as gridspec
%matplotlib inline

v_features = data.iloc[:,1:29].columns

plt.figure(figsize=(12,8*4))
gs = gridspec.GridSpec(7, 4)
for i, cn in enumerate(data[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn][data.Class == 1], bins=50)
    sns.distplot(data[cn][data.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(cn))
plt.show()
# Removing features which  do not align well with the gaussian curve
# Also amount and time are not used for fitting the model
X = np.matrix(data[features].drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1))
y = np.matrix(data[target])
X.shape
def normalize(X):
    """
    Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
    """
    X -= X.min(axis=0)
    X /= (X.max(axis=0)-X.min(axis=0))

    # for feature in X.columns:
    #     X[feature] -= X[feature].mean()
    #     X[feature] /= X[feature].std()

    return X

X = normalize(X)
X
test_break = 140000

X_test = X[test_break:,:]
y_test = y[:,test_break:]

X = X[:test_break,:]
y = y[:,:test_break]

mu = X.mean(axis=0)
mu = np.squeeze(np.asarray(mu))

cov = np.cov(X,rowvar=0)
print (np.diag(cov))
# print (cov)
from scipy import stats
from scipy.stats import multivariate_normal

p= multivariate_normal.pdf(X, mean=mu, cov=1)

print (p.shape)
print (p)
def select_threshold(pval, yval):  
    
    best_tp=0
    best_fp=0
    best_fn=0
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
#         print "preds->",preds
#         print "pval->",pval
#         print epsilon
#         print yval.shape,preds.shape
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2.0 * precision * recall) / (precision + recall)

#         print ("epsilon=",epsilon)
#         print (tp,fp,fn)
#         print ("f1=",f1)
#         print ("best f1=",best_f1)
#         print ("besttpfp",best_tp,best_fp,best_fn)
        if f1 > best_f1:
            best_tp=tp
            best_fp=fp
            best_fn=fn
#             print "besttpfp",tp,fp,fn
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

epsilon, f1 = select_threshold(p, y)
print ("epsilon and f1(for training data)=",epsilon, f1)

# # Applying the threshold to the data set
def test(X,yval,epsilon):

    pval = multivariate_normal.pdf(X, mean=mu, cov=1)
    
    f1 = 0
    
    preds = pval < epsilon
    tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
    fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
    fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

#     print tp,fp,fn
    precision = tp*1.0 / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2.0 * precision * recall) / (precision + recall)

    yval.shape = (yval.shape[1],1)
#     print(yval.shape,preds.shape)
    
    return f1,precision,preds,recall

f1,precision,preds,recall = test(X_test,y_test,epsilon)

print ('test data results: f1=',f1,' recall = ',recall,' and precision =',precision)
#function for PLOTTING CONFUSION MATRIX

import itertools
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

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


cnf_matrix = confusion_matrix(y_test,preds)

print ('confusion matrix of test dataset = \n',cnf_matrix)

print(classification_report(y_test, preds))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = np.matrix(data[features])
y = np.matrix(data[target])

y=np.squeeze(np.asarray(y))

def normalize(X):

    """
    Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
    """
    X -= X.min(axis=0)
    X /= (X.max(axis=0)-X.min(axis=0))

    # for feature in X.columns:
    #     X[feature] -= X[feature].mean()
    #     X[feature] /= X[feature].std()

    return X


# Define the model
model = LogisticRegression()

# Define the splitter for splitting the data in a train set and a test set
splitter = StratifiedShuffleSplit(y,n_iter=1, test_size=0.5, random_state=0)

# Loop through the splits (only one)
for train_indices, test_indices in splitter:
    # Select the train and test data
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Normalize the data
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    # Fit and predict!
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# And finally: show the results
print(classification_report(y_test, y_pred))
cnf_matrix2 = confusion_matrix(y_test,y_pred)

print ('confusion matrix of test dataset = \n',cnf_matrix2)

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix2
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

# %matplotlib inline
# from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
# from sklearn.ensemble import GradientBoostingRegressor
# # get_some_data is defined in hidden cell above.

# # scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# # this was due to an implementation detail, and a future release will support all model types.
# my_model = GradientBoostingRegressor()
# # fit the model as usual
# my_model.fit(X_train, y_train)

# # Here we make the plot
# my_plots = plot_partial_dependence(my_model,       
#                                    features=[0, 1], # column numbers of plots we want to show
#                                    X=X_train,            # raw predictors data.
#                                    feature_names=['V1', 'V2'], # labels on graphs
#                                    grid_resolution=10)