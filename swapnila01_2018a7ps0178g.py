# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

import itertools

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/minor-project-2020/train.csv')
# Compute the correlation matrix

corr = df.corr()



# print(corr)

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(30, 30))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=0.7, vmin= -0.7, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)



plt.show()
df.drop(['id','col_39', 'col_46'], axis=1, inplace = True)
df.head(20)
X = df.drop(['target'], axis=1).values

y = df.target.values
sc = StandardScaler()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
X_train= sc.fit_transform(X_train)
X_val= sc.transform(X_val)
#Hand Picked C after comparing the AUC score on C= 0.1 1 10 5 7 8

lr1 = LogisticRegression(C=8, class_weight='balanced', verbose=5, max_iter=1000)

lr1.fit(X_train, y_train)
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

y_train_pre = lr1.predict(X_train)



cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)



print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))





class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')

plt.show()
y_pred_sample_score = lr1.decision_function(X_val)





fpr, tpr, thresholds = roc_curve(y_val, y_pred_sample_score)



roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
roc_auc
df = pd.read_csv('../input/minor-project-2020/train.csv')
X= sc.fit_transform(X)
lr1 = LogisticRegression(C=8, class_weight='balanced', verbose=5, max_iter=1000)

lr1.fit(X, y)
df_test = pd.read_csv('../input/minor-project-2020/test.csv')

X_test = df_test.drop(['id','col_39','col_46'], axis=1).values
X_test = sc.transform(X_test)
y_test = lr1.predict_proba(X_test)[:,1]
predictions = []

i = 0

for id_num in df_test.id.values:

    predictions.append([id_num, y_test[i]])

    i+=1

    

pd.DataFrame(predictions).to_csv("./predictions.csv", header=['id','target'], index=None)