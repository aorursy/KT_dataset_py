import sys

import math

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import scipy



from sklearn.metrics import confusion_matrix,auc,roc_auc_score,roc_curve,recall_score, precision_score, accuracy_score, f1_score

from sklearn.preprocessing import StandardScaler,normalize

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression



import tensorflow as tf

from tensorflow.python.framework import ops



import itertools



plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



dataset = pd.read_csv("../input/creditcard.csv")

df = pd.read_csv("../input/creditcard.csv")

# Any results you write to the current directory are saved as output.
dataset.head()
dataset.describe()
print("Procent całkowitych transakcji oszukańczych")

print(str(dataset["Class"].mean()*100) + '%')
print("Straty spowodowane oszustwem:")

print("Całkowita kwota utracona w wyniku oszustwa")

print(dataset.Amount[dataset.Class == 1].sum())

print("Średnia kwota za transakcję oszukańczą")

print(dataset.Amount[dataset.Class == 1].mean())

print("Porównaj z normalnymi transakcjami:")

print("Całkowita kwota z normalnych transakcji")

print(dataset.Amount[dataset.Class == 0].sum())

print("Średnia kwota za zwykłe transakcje")

print(dataset.Amount[dataset.Class == 0].mean())


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 40



ax1.hist(dataset.Amount[dataset.Class == 1], bins = bins, normed = True, alpha = 0.75, color = 'red')

ax1.set_title('Fraud')



ax2.hist(dataset.Amount[dataset.Class == 0], bins = bins, normed = True, alpha = 0.5, color = 'blue')

ax2.set_title('Not Fraud')



plt.xlabel('Amount')

plt.ylabel('% of Transactions')

plt.yscale('log')

plt.show()
bins = 75

plt.hist(dataset.Time[dataset.Class == 1], bins = bins, normed = True, alpha = 0.75, label = 'Fraud', color = 'red')

plt.hist(dataset.Time[dataset.Class == 0], bins = bins, normed = True, alpha = 0.5, label = 'Not Fraud', color = 'blue')

plt.legend(loc='upper right')

plt.xlabel('Time (seconds)')

plt.ylabel('% of ')

plt.title('Transactions over Time')

plt.show()
Vfeatures = dataset.iloc[:,1:29].columns

print(Vfeatures)
import matplotlib.gridspec as gridspec

import seaborn as sns

bins = 50

plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, V in enumerate(dataset[Vfeatures]):

    ax = plt.subplot(gs[i])

    sns.distplot(dataset[V][dataset.Class == 1], bins = bins, norm_hist = True, color = 'red')

    sns.distplot(dataset[V][dataset.Class == 0], bins = bins, norm_hist = True, color = 'blue')

    ax.set_xlabel('')

    ax.set_title('distributions (w.r.t fraud vs. non-fraud) of feature: ' + str(V))

plt.show()
# heat map of correlation of features

correlation_matrix = df.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()
df.isnull().sum()
df = pd.read_csv("../input/creditcard.csv")



y = np.array(df.Class.tolist())     #classes: 1..fraud, 0..no fraud

df = df.drop('Class', 1)

df = df.drop('Time', 1)     # optional

df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))    #optionally rescale non-normalized column

X = np.array(df.as_matrix())   # features

print("Fraction of frauds: {:.5f}".format(np.sum(y)/len(y)))

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

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

#    else:

#        print('Confusion matrix, without normalization')



#    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()



def show_data(cm, print_res = 0):

    tp = cm[1,1]

    fn = cm[1,0]

    fp = cm[0,1]

    tn = cm[0,0]

    if print_res == 1:

        print('Precision =     {:.3f}'.format(tp/(tp+fp)))

        print('Recall (TPR) =  {:.3f}'.format(tp/(tp+fn)))

        print('Fallout (FPR) = {:.3e}'.format(fp/(fp+tn)))

    return tp/(tp+fp), tp/(tp+fn), fp/(fp+tn)
lrn = LogisticRegression()



skf = StratifiedKFold(n_splits = 5, shuffle = True)

for train_index, test_index in skf.split(X, y):

    X_train, y_train = X[train_index], y[train_index]

    X_test, y_test = X[test_index], y[test_index]

    break



lrn.fit(X_train, y_train)

y_pred = lrn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

if lrn.classes_[0] == 1:

    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])



plot_confusion_matrix(cm, ['0', '1'], )

pr, tpr, fpr = show_data(cm, print_res = 1);
def ROC(X, y, c, r):

#makes cross_validation for given parameters c,r. Returns FPR, TPR (averaged)

    dic_weight = {1:len(y)/(r*np.sum(y)), 0:len(y)/(len(y)-r*np.sum(y))} 

    lrn = LogisticRegression(penalty = 'l2', C = c, class_weight = dic_weight)

    

    N = 5      #how much k-fold

    N_iter = 3    #repeat how often (taking the mean)

    mean_tpr = 0.0

    mean_thresh = 0.0

    mean_fpr = np.linspace(0, 1, 50000)

    



    for it in range(N_iter):

        skf = StratifiedKFold(n_splits = N, shuffle = True)

        for train_index, test_index in skf.split(X, y):

            X_train, y_train = X[train_index], y[train_index]

            X_test, y_test = X[test_index], y[test_index]

         

            lrn.fit(X_train, y_train)

            y_prob = lrn.predict_proba(X_test)[:,lrn.classes_[1]]

            

            fpr, tpr, thresholds = roc_curve(y_test, y_prob)

            mean_tpr += np.interp(mean_fpr, fpr, tpr)

            mean_thresh += np.interp(mean_fpr, fpr, thresholds)

            mean_tpr[0] = 0.0



    mean_tpr /= (N*N_iter)

    mean_thresh /= (N*N_iter)

    mean_tpr[-1] = 1.0

    return mean_fpr, mean_tpr, roc_auc_score(y_test, y_prob), mean_thresh

def plot_roc(X,y, list_par_1, par_1 = 'C', par_2 = 1):



    f = plt.figure(figsize = (12,8));

    for p in list_par_1:

        if par_1 == 'C':

            c = p

            r = par_2

        else:

            r = p

            c = par_2

        list_FP, list_TP, AUC, mean_thresh = ROC(X, y, c, r)      

        plt.plot(list_FP, list_TP, label = 'C = {}, r = {}, TPR(3e-4) = {:.4f}, AUC = {:.4f}'.format(c,r,list_TP[10],AUC));

    plt.legend(title = 'values', loc='lower right')

    plt.xlim(0, 0.001)   #we are only interested in small values of FPR

    plt.ylim(0.5, 0.9)

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.title('ROC detail')

    plt.axvline(3e-4, color='b', linestyle='dashed', linewidth=2)

    plt.show()

    plt.close()
#plot_roc(X,y, [1], 'r', 1)
#plot_roc(X,y, [0.001, 0.01, 0.1, 1, 100], 'C', 1) 
'''

N = np.arange(10,80,2)     # will define threshold

cm = {}     #will store the confusion matrix for different thresholds

for n in N:

    cm[n] = 0.0

lrn = LogisticRegression(penalty = 'l2', C = 1, class_weight = 'balanced')   #'balanced' corresponds to the case r=1

N_Kfold = 5      #how much k-fold

N_iter = 3   #repeat how often (taking the mean)

for it in range(N_iter):

    skf = StratifiedKFold(n_splits = N_Kfold, shuffle = True)

    for train_index, test_index in skf.split(X, y):

        X_train, y_train = X[train_index], y[train_index]

        X_test, y_test = X[test_index], y[test_index]

        lrn.fit(X_train, y_train)



        y_prob = lrn.predict_proba(X_test)[:,lrn.classes_[1]]

        

        for n in N:

            

            thresh = 1 - np.power(10.,-(n/10))  #we want thresholds very close to 1

            # generate the prediction from the probabilities y_prob:

            y_pred = np.zeros(len(y_prob))

            for j in range(len(y_prob)):

                if y_prob[j] > thresh:

                    y_pred[j] = 1

    

            B = confusion_matrix(y_test, y_pred)

            #if the classes are mixed up, remedy that:

            if lrn.classes_[0] == 1:

                B = np.array([[B[1,1], B[1,0]], [B[0,1], B[0,0]]])

            cm[n]+=B

            '''
'''

PR = []      #precision

TPR = []

FPR = []

THRESH = N

for n in N:

    pr, tpr, fpr = show_data(cm[n])

    PR.append(pr)

    TPR.append(tpr)

    FPR.append(-np.log(fpr)/10)



g  = plt.figure(figsize = (12,8))   

plt.plot(THRESH, PR, label = 'Precision')

plt.plot(THRESH, TPR, label = 'Recall (TPR)')

plt.plot(THRESH, FPR, label = '-log(FPR)/10')

plt.axhline(-np.log(3e-4)/10, color='b', linestyle='dashed', linewidth=2)

plt.title('Evaluation of the classifier')

plt.legend( loc='lower right')

plt.xlabel('-log(1-thresh)/log(10)')

plt.ylim(0.55,0.9)

plt.show()

'''
'''

i = 0

while FPR[i] < -np.log(3e-4)/10:

    i+=1

A = cm[THRESH[i]].astype(int)

plot_confusion_matrix(A, ['0', '1'])

show_data(A, print_res = 1);

'''
'''

i = 0

while FPR[i] < -np.log(2e-3)/10:

    i+=1

A = cm[THRESH[i]].astype(int)

plot_confusion_matrix(A, ['0', '1'])

show_data(A, print_res = 1);

'''
'''

i = 0

while FPR[i] < -np.log(3e-4)/10:

    i+=1

A = cm[THRESH[i]].astype(int)

plot_confusion_matrix(A, ['0', '1'])

show_data(A, print_res = 1);

'''
def split_data(dataset,ratio):

    sample=np.random.rand(len(dataset))<ratio

    return(dataset[sample],dataset[~sample])
credit_data = pd.read_csv("../input/creditcard.csv")

col=list(credit_data.columns.values)
def NB_Classify(ratio,drop_var):

    print('Dropped:',drop_var)

    pred_acc=[]

    for i in range(10):

        train,test=split_data(credit_data,ratio)

        clf=GaussianNB()

        clf.fit(train.drop(drop_var,axis=1),train['Class'])

        pred=clf.predict(test.drop(drop_var,axis=1))

        pred_acc.append([pd.crosstab(test['Class'],pred).iloc[1,1]/(pd.crosstab(test['Class'],pred).iloc[1,0]+pd.crosstab(test['Class'],pred).iloc[1,1])])

    

    print(np.mean(pred_acc))
for var in col:

    NB_Classify(0.6,['Class',var])
NB_Classify(0.6,['Class','Time'])
df = pd.read_csv('../input/creditcard.csv')



df = df.drop(['Time','Amount'], axis = 1)

df = df.sample(frac=1)



frauds = df[df['Class'] == 1]

non_frauds = df[df['Class'] == 0][:492]



new_df = pd.concat([non_frauds, frauds])

# Shuffle dataframe rows

new_df = new_df.sample(frac=1, random_state=42)
labels = ['non frauds','fraud']

classes = pd.value_counts(new_df['Class'], sort = True)

classes.plot(kind = 'bar', rot=0)

plt.title("Dystrybucja klasy transakcji")

plt.xticks(range(2), labels)

plt.xlabel("Class")

plt.ylabel("Częstotliwość")
features = new_df.drop(['Class'], axis = 1)

labels = pd.DataFrame(new_df['Class'])



feature_array = features.values

label_array = labels.values
X_train,X_test,y_train,y_test = train_test_split(feature_array,label_array,test_size=0.20)



# normalize: skaluje wektory wejściowe indywidualnie do normy jednostki (długość wektora).

X_train = normalize(X_train)

X_test=normalize(X_test)
neighbours = np.arange(1,25)

train_accuracy =np.empty(len(neighbours))

test_accuracy = np.empty(len(neighbours))



for i,k in enumerate(neighbours):

    #Skonfigurujemy klasyfikator knn z k sąsiadami

    knn=KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree",n_jobs=-1)

    

    #Doposowanie modelu

    knn.fit(X_train,y_train.ravel())

    

    #Oblicz dokładność na zestawie treningowym

    train_accuracy[i] = knn.score(X_train, y_train.ravel())

    

    #Oblicz dokładność na zestawie testowym

    test_accuracy[i] = knn.score(X_test, y_test.ravel()) 
plt.title('k-NN  Liczba sąsiadów')

plt.plot(neighbours, test_accuracy, label='Testing Accuracy')

plt.plot(neighbours, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
idx = np.where(test_accuracy == max(test_accuracy))

x = neighbours[idx]



knn=KNeighborsClassifier(n_neighbors=x[0],algorithm="kd_tree",n_jobs=-1)

knn.fit(X_train,y_train.ravel())
knn_predicted_test_labels=knn.predict(X_test)



knn_accuracy_score  = accuracy_score(y_test,knn_predicted_test_labels)

knn_precison_score  = precision_score(y_test,knn_predicted_test_labels)

knn_recall_score    = recall_score(y_test,knn_predicted_test_labels)





print("")

print("K-Nearest Neighbours")

print("Scores")

print("Accuracy -->",knn_accuracy_score)

print("Precison -->",knn_precison_score)

print("Recall -->",knn_recall_score)



print(classification_report(y_test,knn_predicted_test_labels))
import seaborn as sns

LABELS = ['Normal', 'Fraud']

conf_matrix = confusion_matrix(y_test, knn_predicted_test_labels)

plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()