import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np

import pandas as pd

import numpy as np

import random as rnd

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from numpy import genfromtxt

from scipy.stats import multivariate_normal

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score , average_precision_score

from sklearn.metrics import precision_score, precision_recall_curve

%matplotlib inline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)
train_df = pd.read_csv("../input//creditcard.csv")

print(train_df.columns.values)
train_df.head(5)




def estimateGaussian(dataset):

    mu = np.mean(dataset, axis=0)

    sigma = np.cov(dataset.T)

    return mu, sigma



def multivariateGaussian(dataset,mu,sigma):

    p = multivariate_normal(mean=mu, cov=sigma)

    return p.pdf(dataset)





def selectThresholdByCV(probs,gt):

    best_epsilon = 0

    best_f1 = 0

    f = 0

    farray = []

    Recallarray = []

    Precisionarray = []

    epsilons = (0.0000e+00, 1.0527717316e-70, 1.0527717316e-50, 1.0527717316e-24)

    #epsilons = np.asarray(epsilons)

    #step = (probs.max() - probs.min()) / 1000

    #for epsilon in np.arange(probs.min(), probs.max(), step):

    for epsilon in epsilons:

        predictions = (p_cv < epsilon)

        f = f1_score(train_cv_y, predictions, average = "binary")

        Recall = recall_score(train_cv_y, predictions, average = "binary")

        Precision = precision_score(train_cv_y, predictions, average = "binary")

        farray.append(f)

        Recallarray.append(Recall)

        Precisionarray.append(Precision)

        print ('For below Epsilon')

        print(epsilon)

        print ('F1 score , Recall and Precision are as below')

        print ('Best F1 Score %f' %f)

        print ('Best Recall Score %f' %Recall)

        print ('Best Precision Score %f' %Precision)

        print ('-'*40)

        if f > best_f1:

            best_f1 = f

            best_recall = Recall

            best_precision = Precision

            best_epsilon = epsilon    

    fig = plt.figure()

    ax = fig.add_axes([0.1, 0.5, 0.7, 0.3])

    #plt.subplot(3,1,1)

    plt.plot(farray ,"ro")

    plt.plot(farray)

    ax.set_xticks(range(5))

    ax.set_xticklabels(epsilons,rotation = 60 ,fontsize = 'medium' )

    ax.set_ylim((0,1.0))

    ax.set_title('F1 score vs Epsilon value')

    ax.annotate('Best F1 Score', xy=(best_epsilon,best_f1), xytext=(best_epsilon,best_f1))

    plt.xlabel("Epsilon value") 

    plt.ylabel("F1 Score") 

    plt.show()

    fig = plt.figure()

    ax = fig.add_axes([0.1, 0.5, 0.9, 0.3])

    #plt.subplot(3,1,2)

    plt.plot(Recallarray ,"ro")

    plt.plot(Recallarray)

    ax.set_xticks(range(5))

    ax.set_xticklabels(epsilons,rotation = 60 ,fontsize = 'medium' )

    ax.set_ylim((0,1.0))

    ax.set_title('Recall vs Epsilon value')

    ax.annotate('Best Recall Score', xy=(best_epsilon,best_recall), xytext=(best_epsilon,best_recall))

    plt.xlabel("Epsilon value") 

    plt.ylabel("Recall Score") 

    plt.show()

    fig = plt.figure()

    ax = fig.add_axes([0.1, 0.5, 0.9, 0.3])

    #plt.subplot(3,1,3)

    plt.plot(Precisionarray ,"ro")

    plt.plot(Precisionarray)

    ax.set_xticks(range(5))

    ax.set_xticklabels(epsilons,rotation = 60 ,fontsize = 'medium' )

    ax.set_ylim((0,1.0))

    ax.set_title('Precision vs Epsilon value')

    ax.annotate('Best Precision Score', xy=(best_epsilon,best_precision), xytext=(best_epsilon,best_precision))

    plt.xlabel("Epsilon value") 

    plt.ylabel("Precision Score") 

    plt.show()

    return best_f1, best_epsilon

    
v_features = train_df.iloc[:,1:29].columns




plt.figure(figsize=(12,8*4))

gs = gridspec.GridSpec(7, 4)

for i, cn in enumerate(train_df[v_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(train_df[cn][train_df.Class == 1], bins=50)

    sns.distplot(train_df[cn][train_df.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('feature: ' + str(cn))

plt.show()



#rnd_clf = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy',random_state = 0)

#rnd_clf.fit(train_df.iloc[:,1:29],train_df.iloc[:,30]);
#for name, importance in zip(train_df.iloc[:,1:29].columns, rnd_clf.feature_importances_):

#   if importance > 0.020 :

#        print('"' + name + '"'+',')
train_df.drop(['V19','V21','V1','V2','V6','V5','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8', 'Time', 'Amount'], axis =1, inplace = True)
train_df.head(5)
train_strip_v1 = train_df[train_df["Class"] == 1]

train_strip_v0 = train_df[train_df["Class"] == 0]
Normal_len = len (train_strip_v0)

Anomolous_len = len (train_strip_v1)



start_mid = Anomolous_len // 2

start_midway = start_mid + 1



train_cv_v1  = train_strip_v1 [: start_mid]

train_test_v1 = train_strip_v1 [start_midway:Anomolous_len]



start_mid = (Normal_len * 60) // 100

start_midway = start_mid + 1



cv_mid = (Normal_len * 80) // 100

cv_midway = cv_mid + 1



train_fraud = train_strip_v0 [:start_mid]

train_cv    = train_strip_v0 [start_midway:cv_mid]

train_test  = train_strip_v0 [cv_midway:Normal_len]



train_cv = pd.concat([train_cv,train_cv_v1],axis=0)

train_test = pd.concat([train_test,train_test_v1],axis=0)





print(train_fraud.columns.values)

print(train_cv.columns.values)

print(train_test.columns.values)



train_cv_y = train_cv["Class"]

train_test_y = train_test["Class"]



train_cv.drop(labels = ["Class"], axis = 1, inplace = True)

train_fraud.drop(labels = ["Class"], axis = 1, inplace = True)

train_test.drop(labels = ["Class"], axis = 1, inplace = True)

mu, sigma = estimateGaussian(train_fraud)

p = multivariateGaussian(train_fraud,mu,sigma)

p_cv = multivariateGaussian(train_cv,mu,sigma)

p_test = multivariateGaussian(train_test,mu,sigma)
print (p)

print(p_cv)

print(p_test)
fscore, ep= selectThresholdByCV(p_cv,train_cv_y)
print(fscore)

print(ep)
predictions = (p_test < ep)

Recall = recall_score(train_test_y, predictions, average = "binary")    

Precision = precision_score(train_test_y, predictions, average = "binary")

F1score = f1_score(train_test_y, predictions, average = "binary")    

print ('F1 score , Recall and Precision for Test dataset')

print ('Best F1 Score %f' %F1score)

print ('Best Recall Score %f' %Recall)

print ('Best Precision Score %f' %Precision)

ig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(train_test['V14'],train_test['V11'],marker="o", color="lightBlue")

ax.set_title('Anomalies(in red) vs Predicted Anomalies(in Green)')

for i, txt in enumerate(train_test['V14'].index):

        if train_test_y.loc[txt] == 1 :

            ax.annotate('*', (train_test['V14'].loc[txt],train_test['V11'].loc[txt]),fontsize=13,color='Red')

        if predictions[i] == True :

            ax.annotate('o', (train_test['V14'].loc[txt],train_test['V11'].loc[txt]),fontsize=15,color='Green')
predictions = (p_cv < ep)

Recall = recall_score(train_cv_y, predictions, average = "binary")    

Precision = precision_score(train_cv_y, predictions, average = "binary")

F1score = f1_score(train_cv_y, predictions, average = "binary")    

print ('F1 score , Recall and Precision for Cross Validation dataset')

print ('Best F1 Score %f' %F1score)

print ('Best Recall Score %f' %Recall)

print ('Best Precision Score %f' %Precision)

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
cnf_matrix = confusion_matrix(train_cv_y,predictions)



print ('confusion matrix of test dataset = \n',cnf_matrix)



print(classification_report(train_cv_y, predictions))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()