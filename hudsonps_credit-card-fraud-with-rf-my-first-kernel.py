import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import neighbors
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import scipy
from scipy.stats import spearmanr
from scipy.stats import ks_2samp
#importing the dataset 
rawdf = pd.read_csv("../input/creditcard.csv", delimiter = ",")
rawdf.head()
rawdf.shape
n_fraud = rawdf.Class[rawdf.Class==1].count()
n_legit = rawdf.Class[rawdf.Class==0].count()

print(n_fraud, n_legit)
#Looking at the dataset overall statistics

#We will get into more detail about some of the statistics later. In particular, we care about the variable amount.
rawdf.describe()
PCA_list = rawdf.columns[1:29] #contains the labels V1, V2, ..., V28
PCA_index = np.arange(1,29) #vector with numbers 1, 2, 3, ..., 28
plt.plot(PCA_index[0:10], rawdf[PCA_list[0:10]].std(), 'o')
plt.plot(PCA_index[10:20], rawdf[PCA_list[10:20]].std(), 'o')
plt.plot(PCA_index[20:29], rawdf[PCA_list[20:29]].std(), 'o')
plt.ylabel("standard deviation")
plt.xlabel("PCA component")
plt.xticks(PCA_index)
plt.tight_layout()

#This plot just confirms our expectation that the standard deviation is decreasing for higher PCA components.
#Let's take a look at the first few components to have a feeling for bivariate distributions

PCA_shortlist = list(PCA_list[0:4])
PCA_shortlist.append('Class')
sns.pairplot(rawdf[PCA_shortlist], hue='Class')
plt.show()
#histograms for various PCA components

components = ["V1", "V2", "V3", "V4", "V5"]

for PCA_comp in components:
    
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    rawdf[rawdf.Class == 1][PCA_comp].hist(label="fraud", color="blue", density='True')
    rawdf[rawdf.Class == 0][PCA_comp].hist(label="not fraud", color="orange", density='True')
    plt.title(PCA_comp)
    plt.legend()
    
    plt.subplot(1,2,2)
    sns.kdeplot(rawdf[rawdf.Class == 1][PCA_comp], color='blue')
    sns.kdeplot(rawdf[rawdf.Class == 0][PCA_comp], color='orange')
    plt.tight_layout()
    plt.title(PCA_comp)
components = ["V6", "V8", "V13", "V15", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

for PCA_comp in components:
    
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    rawdf[rawdf.Class == 1][PCA_comp].hist(label="fraud", color="blue", density='True')
    rawdf[rawdf.Class == 0][PCA_comp].hist(label="not fraud", color="orange", density='True')
    plt.title(PCA_comp)
   # sns.kdeplot(rawdf[rawdf.Class == 1][PCA_comp], color="blue")
   # sns.kdeplot(rawdf[rawdf.Class == 0][PCA_comp], color='orange')
    plt.legend()
    
    plt.subplot(1,2,2)
    sns.kdeplot(rawdf[rawdf.Class == 1][PCA_comp], color='blue')
    sns.kdeplot(rawdf[rawdf.Class == 0][PCA_comp], color='orange')
    plt.tight_layout()
    plt.title(PCA_comp)

for component in PCA_list[:3]:
    print("The current component is "+component)
    print(spearmanr(rawdf[component], rawdf["Class"]))

#Null hypothesis: two samples are drawn from the same distribution.
for component in PCA_list[:3]:
        print("The current component is "+component)
        print(ks_2samp(rawdf[rawdf.Class==1][component], rawdf[rawdf.Class==0][component]))
#histogram for transaction amounts below 1000
crit_amount = 1000

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
rawdf[rawdf.Class==1]['Amount'][rawdf.Amount < crit_amount].hist(label="fraud", color="blue", density='True')
plt.legend()
plt.subplot(1,2,2)
rawdf[rawdf.Class==0]['Amount'][rawdf.Amount < crit_amount].hist(label="not fraud", color="orange", density='True')
plt.legend()

plt.figure()
sns.distplot(rawdf[rawdf.Class==1].Amount[rawdf.Amount < crit_amount], label='fraud')
sns.distplot(rawdf[rawdf.Class==0].Amount[rawdf.Amount < crit_amount], label='not fraud')
plt.legend()
rawdf = rawdf[rawdf.Amount>0]
print(rawdf.Amount[rawdf.Class==1].count())
#this is the updated number of fraudulent transactions
print("Non-fraudulent transactions, amount above 1000: "+
      str(rawdf[rawdf.Class==0].Amount[rawdf.Amount > crit_amount].count()))
print("Fraudulent transactions, amount above 1000: "+
       str(rawdf[rawdf.Class==1].Amount[rawdf.Amount > crit_amount].count()))
print(rawdf.Amount.groupby(rawdf.Class).describe())

print(rawdf.Amount.groupby(rawdf.Class).sum())
amounts_fraud = np.asarray(rawdf.Amount[rawdf.Class==1]) #array with the amounts of fraudulent transactions
amounts_fraud.sort() #amount sorted in ascending order
cum_amounts_fraud = np.cumsum(amounts_fraud) #array with the cumulative sum 
#plt.style.use('ggplot')

total = cum_amounts_fraud[-1] #total amount, used to normalize the cumulative sum

plt.figure()
plt.grid()
plt.plot(cum_amounts_fraud/total)
plt.ylabel("Normalized Cumulative amount")
plt.xlabel("Number of transactions (starting from the lowest)")
#A few more useful numbers to have in mind (we did not pick these numbers randomly; they are based on the plot above)

print("Fraction associated with the lowest 300 transactions: "+ 
      str(amounts_fraud[0:300].sum()/total)) 

print("Fraction associated with the highest 100 transactions: "+
      str(amounts_fraud[365:].sum()/total)) 

print("Amount at which the 300th transaction starts: "+
      str(amounts_fraud[300]))

print("Amount lost associated with the 10 highest transactions: "+
      str(amounts_fraud[455:].sum()/total))

print("Amount associated with the 10 most expensive transactions:" )
print(str(amounts_fraud[455:]))
#Let's check if the dataset balance improves in the region starting from amount = 88 and above. 
rawdf.Amount[rawdf.Amount > 88][rawdf.Class==1].count()/rawdf.Amount[rawdf.Amount > 88].count()
df = rawdf[(rawdf.Amount > 88)].copy()
cols_to_drop = ["V6", "V8", "V13", "V15", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", 
                "V27", "V28", "Time", "Amount"]
df.drop(cols_to_drop, inplace=True, axis=1)
df.head()
Y = df.Class 
X = df.drop("Class", axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                stratify=Y, 
                                                test_size=0.3)
#Oversampling 
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(n_jobs=-1)
X_over, Y_over = oversampler.fit_sample(X_train, np.ravel(Y_train))

#undersampling
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler()
X_under, Y_under = undersampler.fit_sample(X_train, np.ravel(Y_train))

#combining both
from imblearn.combine import SMOTEENN
combined = SMOTEENN()
X_comb, Y_comb = combined.fit_sample(X_train, Y_train)
#different classifiers for different strategies
rf = ensemble.RandomForestClassifier(class_weight={0:1,1:100})
rf_over = ensemble.RandomForestClassifier()
rf_under = ensemble.RandomForestClassifier()
rf_comb = ensemble.RandomForestClassifier()

#fitting the four models
rf.fit(X_train, Y_train)
rf_over.fit(X_over, Y_over)
rf_under.fit(X_under, Y_under)
rf_comb.fit(X_comb, Y_comb)

#predictions for the test set associated with the four models
Y_pred = rf.predict(X_test)
Y_over_pred = rf_over.predict(X_test)
Y_under_pred = rf_under.predict(X_test)
Y_comb_pred = rf_comb.predict(X_test)

print("Imbalanced Sample")
print(classification_report(Y_test, Y_pred))

print("Oversampling")
print(classification_report(Y_test, Y_over_pred))

print("Undersampling")
print(classification_report(Y_test, Y_under_pred))

print("Combined")
print(classification_report(Y_test, Y_comb_pred))
#Function to create a nice plot of the confusion matrix
#This function was NOT created by me. It was part of a code an instructor presented during some class
#I am assuming that there is no ownership over it, but I thought I'd make clear that I did not code this function

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest without original sample")

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_over_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest with oversampling")

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_under_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest with undersampling")

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_comb_pred)
plot_confusion_matrix(cm, [0,1])
plt.title("random forest with combined over+under")


#Looking at the most important features.
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',         
                                                                        ascending=True)
feature_importances

#It is important to realize that the precise order of these features may vary from run to run.
#However, some tend to often show up in low tiers, and I eliminated a few based on that.
#One can always go back and add them back to see if they would improve performance.

#In my run, I decided to eliminate the following features:
#Let's start by reducing our space even further
X_train.drop(["V2","V1", "V11", "V18", "V5", "V9", "V7", "V4"], axis=1, inplace=True)
X_test.drop(["V2","V1", "V11", "V18", "V5", "V9", "V7", "V4"], axis=1, inplace=True)
#confirming that we dropped the right columns
X_train.head(3)

param_grid = {"max_depth": [2,3],
              "max_features": [2,3],
              "n_estimators": [80,100,150,200],
              "class_weight": [{0:1, 1:100}]
              }

cv_method = StratifiedKFold(n_splits=5, shuffle=True)

rf_grid = GridSearchCV(estimator = ensemble.RandomForestClassifier(),
                       param_grid = param_grid,
                       cv = cv_method,
                       scoring = 'recall')

#The goal here is to maximize the metric recall.

#using gridsearch to fit
rf_grid.fit(X_train, np.ravel(Y_train))
rf_grid.best_params_
Y_grid_pred = rf_grid.predict(X_test)
print(classification_report(Y_test, Y_grid_pred))

plt.figure(figsize=(6,3))
cm = confusion_matrix(Y_test, Y_grid_pred)
plot_confusion_matrix(cm, [0,1])
Y_probs = rf_grid.predict_proba(X_test)
pos_probs = list(map(lambda l: l[1], Y_probs))
fpr, tpr, thresholds = roc_curve(Y_test, pos_probs, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()