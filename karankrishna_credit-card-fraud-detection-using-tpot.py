import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 7)

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
data=pd.read_csv('../input/creditcard.csv')

data.head()
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
data.Time[data.Class == 1].describe()
data.Time[data.Class == 0].describe()
data.isnull().sum()
f, ax = plt.subplots(figsize=(15, 15))

corr = data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

f, ax = plt.subplots(figsize=(15, 5))

ax = sns.violinplot(x="Class", y="Amount", data=data, scale="area")
f, ax = plt.subplots(figsize=(15, 5))

ax = sns.violinplot(x="Class", y="Time", data=data, scale="area")
df1 = data[data.Amount <= 1000]

f, ax = plt.subplots(figsize=(15, 5))

ax = sns.violinplot(x="Class", y="Amount", data=df1, scale="area")

df1.head()
f, ax = plt.subplots(figsize=(15, 15))

corr = df1.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

print ("Fraud")

print (df1.Time[df1.Class == 1].describe())

print ()

print ("Normal")

print (df1.Time[df1.Class == 0].describe())
f, ax = plt.subplots(figsize=(15, 5))

ax = sns.violinplot(x="Class", y="Time", data=df1, scale="area",palette="dark")
cols=df1.iloc[:,1:29]

for i in cols.columns:

    f, ax = plt.subplots(figsize=(15, 5))

    sns.violinplot(x="Class", y=i, data=df1, scale="area",palette="Set3")

    plt.show()

    


df = df1.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
from tpot import TPOTClassifier
pipeline_optimizer = TPOTClassifier(generations=5, population_size=10,

                          offspring_size=None, mutation_rate=0.9,

                          crossover_rate=0.1,

                          scoring='accuracy', cv=5,

                          subsample=1.0, n_jobs=1,

                          max_time_mins=None, max_eval_time_mins=5,

                          random_state=None, config_dict=None,

                          warm_start=False,

                          memory=None,

                          periodic_checkpoint_folder=None,

                          early_stop=None,

                          verbosity=3,

                          disable_update_check=False)
Train_Class=df1['Class']

df1.drop(['Class'],inplace=True,axis=1)
pipeline_optimizer.fit(df1,Train_Class)
data.drop(['Class'],inplace=True,axis=1)

pred=pipeline_optimizer.predict(data)
df=pd.read_csv('../input/creditcard.csv')
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
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
# Compute confusion matrix

cnf_matrix = confusion_matrix(df['Class'],pred)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()


