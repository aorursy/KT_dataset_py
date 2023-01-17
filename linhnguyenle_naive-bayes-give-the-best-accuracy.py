%matplotlib inline 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import math
import copy
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
data = pd.read_csv('../input/seattleWeather_1948-2017.csv')
data.head(5)
data.isna().sum()
data[(data.RAIN != True) & (data.RAIN!=False)]
data = data[(data.RAIN == True) | (data.RAIN==False)]
labelencoder = LabelEncoder()
data['RAIN'] = labelencoder.fit_transform(data['RAIN'])
data['MONTH'] = pd.DatetimeIndex(data['DATE']).month
data['DAY'] = pd.DatetimeIndex(data['DATE']).day
data.head()
data.corr()
rain = data[data.RAIN == True]
norain = data[data.RAIN == False]
rain['MONTH'].hist()
norain['MONTH'].hist()
rain['TMAX'].hist()
norain['TMAX'].hist()
sns.pairplot(data=data[['PRCP','TMAX','TMIN','MONTH','DAY']])
sns.boxplot(data=data[['PRCP','TMAX','TMIN','MONTH','DAY']])
dataN = data.copy()
dataN=dataN.drop(dataN[dataN['TMIN']<17 ].index)
dataN=dataN.drop(dataN[(dataN['TMAX']>97.5) | (dataN['TMAX']< 21.5)].index)
dataN=dataN.drop(dataN[(dataN['PRCP']>0.25) | (dataN['PRCP']< -0.15) ].index)
sns.boxplot(data=dataN[['PRCP','TMAX','TMIN','MONTH','DAY']])
X = dataN[['PRCP','TMAX','TMIN','MONTH','DAY']].copy()
y = dataN['RAIN'].copy()
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
scaler = StandardScaler()
Xscale = scaler.fit_transform(X)
XScaletrain,XScaletest,yscaletrain,yscaletest = train_test_split(Xscale,y,test_size=0.2)
# len(ytest[ytest == 1]) /len(ytest)
# len(ytest[ytest == 0]) / len(ytest)
svc = SVC()
svc.fit(Xtrain,ytrain)
ypredS = svc.predict(Xtest)
plt.hist(ypredS)
ypredS
metrics.accuracy_score(ytest,ypredS)
metrics.confusion_matrix(ytest,ypredS)
# TruePositive FalsePositive
# FalseNegative TrueNegative
# sns.heatmap(metrics.confusion_matrix(ytest, ypredS) / len(ytest), cmap='Blues', annot=True)
plot_confusion_matrix(cm = metrics.confusion_matrix(ytest, ypredS),normalize=True, target_names = ['Rain', 'No Rain'],title = "Confusion Matrix Decision Tree")
metrics.roc_auc_score(ytest,ypredS)
svc = SVC()
svc.fit(XScaletrain,yscaletrain)
ypredscale = svc.predict(XScaletest)
metrics.accuracy_score(yscaletest,ypredscale)
metrics.confusion_matrix(yscaletest,ypredscale)
# sns.heatmap(metrics.confusion_matrix(yscaletest, ypredscale) / len(ytest), cmap='Blues', annot=True)
plot_confusion_matrix(cm = metrics.confusion_matrix(yscaletest, ypredscale),normalize=True, target_names = ['Rain', 'No Rain'],title = "Confusion Matrix Decision Tree")
fpr, tpr, threshold = metrics.roc_curve(ytest, ypredS)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(fpr, tpr, color='darkorange', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Seattle Rain ROC Curve')
plt.legend(loc="lower right")
fpr, tpr, threshold = metrics.roc_curve(yscaletest, ypredscale)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(fpr, tpr, color='darkorange', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Seattle Rain ROC Curve')
plt.legend(loc="lower right")
lr = LogisticRegression()
lr.fit(Xtrain,ytrain)
ypredL = lr.predict(Xtest)
plt.hist(ypredL)
metrics.accuracy_score(ytest,ypredL)
metrics.confusion_matrix(ytest,ypredL)
# TruePositive FalsePositive
# FalseNegative TrueNegative
# sns.heatmap(metrics.confusion_matrix(ytest, ypredL) / len(ytest), cmap='Blues', annot=True)
plot_confusion_matrix(cm = metrics.confusion_matrix(ytest,ypredL),normalize=True, target_names = ['Rain', 'No Rain'],title = "Confusion Matrix Decision Tree")
metrics.roc_auc_score(ytest,ypredL)
lr = LogisticRegression()
lr.fit(XScaletrain,yscaletrain)
ypredscale = lr.predict(XScaletest)
metrics.accuracy_score(yscaletest,ypredscale)
metrics.confusion_matrix(yscaletest,ypredscale)
# sns.heatmap(metrics.confusion_matrix(yscaletest, ypredscale) / len(ytest), cmap='Blues', annot=True)
plot_confusion_matrix(cm = metrics.confusion_matrix(ytest,ypredL),normalize=True, target_names = ['Rain', 'No Rain'],title = "Confusion Matrix Decision Tree")
fpr, tpr, threshold = metrics.roc_curve(ytest, ypredL)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(fpr, tpr, color='darkorange', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Seattle Rain ROC Curve')
plt.legend(loc="lower right")
naive = GaussianNB()
naive.fit(Xtrain,ytrain)
ypredB = naive.predict(Xtest)
plt.hist(ypredB)
metrics.accuracy_score(ytest,ypredB)
metrics.confusion_matrix(ytest,ypredB)
# sns.heatmap(metrics.confusion_matrix(ytest, ypredB) / len(ytest), cmap='Blues', annot=True)
plot_confusion_matrix(cm = metrics.confusion_matrix(ytest,ypredB),normalize=True, target_names = ['Rain', 'No Rain'],title = "Confusion Matrix Decision Tree")
metrics.roc_auc_score(ytest,ypredB)
fpr, tpr, threshold = metrics.roc_curve(ytest, ypredB)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(fpr, tpr, color='darkorange', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Seattle Rain ROC Curve')
plt.legend(loc="lower right")
naive = GaussianNB()
naive.fit(XScaletrain,yscaletrain)
ypredscale = naive.predict(XScaletest)
metrics.accuracy_score(yscaletest,ypredscale)
ypred = [ypredS,ypredL,ypredB]
test = ['SVM','Logistic Regression', 'Naive Bayes']
colors = ['blue','red','green']
for i in range(1, 4):
    fpr, tpr, threshold = metrics.roc_curve(ytest, ypred[i-1])
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, color=colors[i-1], label = 'AUC '+test[i-1]+' = %0.2f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Seattle Rain ROC Curve')
    plt.legend(loc="lower right")
