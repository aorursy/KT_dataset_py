# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline



import seaborn as sns

sns.set_style("dark")



from sklearn import preprocessing

from scipy.stats import skew, boxcox



import warnings

warnings.filterwarnings("ignore")

# Utilities-related functions

def now():

    tmp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return tmp



def my_file_read(file):

    df = pd.read_csv(file)

    print("{}: {} has {} observations and {} columns".format(now(), file, df.shape[0], df.shape[1]))

    print("{}: Column name checking::: {}".format(now(), df.columns.tolist()))

    return df



# Self-defined function to read dataframe and find the missing data on the columns and # of missing

def checking_na(df):

    try:

        if (isinstance(df, pd.DataFrame)):

            df_na_bool = pd.concat([df.isnull().any(), df.isnull().sum(), (df.isnull().sum()/df.shape[0])*100],

                                   axis=1, keys=['df_bool', 'df_amt', 'missing_ratio_percent'])

            df_na_bool = df_na_bool.loc[df_na_bool['df_bool'] == True]

            return df_na_bool

        else:

            print("{}: The input is not panda DataFrame".format(now()))



    except (UnboundLocalError, RuntimeError):

        print("{}: Something is wrong".format(now()))

        
raw_data = my_file_read("../input/PS_20174392719_1491204439457_log.csv")
print(checking_na(raw_data))
print(raw_data.head(5))

print(raw_data.describe())

print(raw_data.info())
print(raw_data.type.value_counts())



f, ax = plt.subplots(1, 1, figsize=(8, 8))

raw_data.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8))

plt.show()
ax = raw_data.groupby(['type', 'isFraud']).size().plot(kind='bar')

ax.set_title("# of transaction which are the actual fraud per transaction type")

ax.set_xlabel("(Type, isFraud)")

ax.set_ylabel("Count of transaction")

for p in ax.patches:

    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
ax = raw_data.groupby(['type', 'isFlaggedFraud']).size().plot(kind='bar')

ax.set_title("# of transaction which is flagged as fraud per transaction type")

ax.set_xlabel("(Type, isFlaggedFraud)")

ax.set_ylabel("Count of transaction")

for p in ax.patches:

    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

tmp = raw_data.loc[(raw_data.type == 'TRANSFER'), :]



a = sns.boxplot(x = 'isFlaggedFraud', y = 'amount', data = tmp, ax=axs[0][0])

axs[0][0].set_yscale('log')

b = sns.boxplot(x = 'isFlaggedFraud', y = 'oldbalanceDest', data = tmp, ax=axs[0][1])

axs[0][1].set(ylim=(0, 0.5e8))

c = sns.boxplot(x = 'isFlaggedFraud', y = 'oldbalanceOrg', data=tmp, ax=axs[1][0])

axs[1][0].set(ylim=(0, 3e7))

d = sns.regplot(x = 'oldbalanceOrg', y = 'amount', data=tmp.loc[(tmp.isFlaggedFraud ==1), :], ax=axs[1][1])

plt.show()

from statsmodels.tools import categorical



# 1. Keep only interested transaction type ('TRANSFER', 'CASH_OUT')

# 2. Drop some columns

# 3. Convert categorical variables to numeric variable

tmp = raw_data.loc[(raw_data['type'].isin(['TRANSFER', 'CASH_OUT'])),:]

tmp.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

tmp = tmp.reset_index(drop=True)

a = np.array(tmp['type'])

b = categorical(a, drop=True)

tmp['type_num'] = b.argmax(1)



print(tmp.head(3))
def correlation_plot(df):

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(111)

    cmap = cm.get_cmap('jet', 30)

    cax = ax1.imshow(df.corr(), interpolation = "nearest", cmap = cmap)

    ax1.grid(True)

    plt.title("Correlation Heatmap")

    labels = df.columns.tolist()

    ax1.set_xticklabels(labels, fontsize=13, rotation=45)

    ax1.set_yticklabels(labels, fontsize=13)

    fig.colorbar(cax)

    plt.show()

    

correlation_plot(tmp)



# Alternatively, we can use quick seaborn

# plot the heatmap

sns.heatmap(tmp.corr())
ax = tmp.type.value_counts().plot(kind='bar', title="Transaction type", figsize=(6,6))

for p in ax.patches:

    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))



plt.show()



ax = pd.value_counts(tmp['isFraud'], sort = True).sort_index().plot(kind='bar', title="Fraud transaction count")

for p in ax.patches:

    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()))

    

plt.show()
tmp['amount_boxcox'] = preprocessing.scale(boxcox(tmp['amount']+1)[0])



figure = plt.figure(figsize=(16, 5))

figure.add_subplot(131) 

plt.hist(tmp['amount'] ,facecolor='blue',alpha=0.75) 

plt.xlabel("Transaction amount") 

plt.title("Transaction amount ") 

plt.text(10,100000,"Skewness: {0:.2f}".format(skew(tmp['amount'])))



figure.add_subplot(132)

plt.hist(np.sqrt(tmp['amount']), facecolor = 'red', alpha=0.5)

plt.xlabel("Square root of amount")

plt.title("Using SQRT on amount")

plt.text(10, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmp['amount']))))



figure.add_subplot(133)

plt.hist(tmp['amount_boxcox'], facecolor = 'red', alpha=0.5)

plt.xlabel("Box cox of amount")

plt.title("Using Box cox on amount")

plt.text(10, 100000, "Skewness: {0:.2f}".format(skew(tmp['amount_boxcox'])))



plt.show()
tmp['oldbalanceOrg_boxcox'] = preprocessing.scale(boxcox(tmp['oldbalanceOrg']+1)[0])



figure = plt.figure(figsize=(16, 5))

figure.add_subplot(131) 

plt.hist(tmp['oldbalanceOrg'] ,facecolor='blue',alpha=0.75) 

plt.xlabel("old balance originated") 

plt.title("Old balance org") 

plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmp['oldbalanceOrg'])))





figure.add_subplot(132)

plt.hist(np.sqrt(tmp['oldbalanceOrg']), facecolor = 'red', alpha=0.5)

plt.xlabel("Square root of oldBal")

plt.title("SQRT on oldbalanceOrg")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmp['oldbalanceOrg']))))



figure.add_subplot(133)

plt.hist(tmp['oldbalanceOrg_boxcox'], facecolor = 'red', alpha=0.5)

plt.xlabel("Box cox of oldBal")

plt.title("Box cox on oldbalanceOrg")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmp['oldbalanceOrg_boxcox'])))



plt.show()
tmp['newbalanceOrg_boxcox'] = preprocessing.scale(boxcox(tmp['newbalanceOrig']+1)[0])



figure = plt.figure(figsize=(16, 5))

figure.add_subplot(131) 

plt.hist(tmp['newbalanceOrig'] ,facecolor='blue',alpha=0.75) 

plt.xlabel("New balance originated") 

plt.title("New balance org") 

plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmp['newbalanceOrig'])))





figure.add_subplot(132)

plt.hist(np.sqrt(tmp['newbalanceOrig']), facecolor = 'red', alpha=0.5)

plt.xlabel("Square root of newBal")

plt.title("SQRT on newbalanceOrig")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmp['newbalanceOrig']))))



figure.add_subplot(133)

plt.hist(tmp['newbalanceOrg_boxcox'], facecolor = 'red', alpha=0.5)

plt.xlabel("Box cox of newBal")

plt.title("Box cox on newbalanceOrig")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmp['newbalanceOrg_boxcox'])))



plt.show()
tmp['oldbalanceDest_boxcox'] = preprocessing.scale(boxcox(tmp['oldbalanceDest']+1)[0])



figure = plt.figure(figsize=(16, 5))

figure.add_subplot(131) 

plt.hist(tmp['oldbalanceDest'] ,facecolor='blue',alpha=0.75) 

plt.xlabel("Old balance desinated") 

plt.title("Old balance dest") 

plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmp['oldbalanceDest'])))





figure.add_subplot(132)

plt.hist(np.sqrt(tmp['oldbalanceDest']), facecolor = 'red', alpha=0.5)

plt.xlabel("Square root of oldBalDest")

plt.title("SQRT on oldbalanceDest")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmp['oldbalanceDest']))))



figure.add_subplot(133)

plt.hist(tmp['oldbalanceDest_boxcox'], facecolor = 'red', alpha=0.5)

plt.xlabel("Box cox of oldbalanceDest")

plt.title("Box cox on oldbalanceDest")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmp['oldbalanceDest_boxcox'])))



plt.show()
tmp['newbalanceDest_boxcox'] = preprocessing.scale(boxcox(tmp['newbalanceDest']+1)[0])



figure = plt.figure(figsize=(16, 5))

figure.add_subplot(131) 

plt.hist(tmp['newbalanceDest'] ,facecolor='blue',alpha=0.75) 

plt.xlabel("newbalanceDest") 

plt.title("newbalanceDest") 

plt.text(2,100000,"Skewness: {0:.2f}".format(skew(tmp['newbalanceDest'])))





figure.add_subplot(132)

plt.hist(np.sqrt(tmp['newbalanceDest']), facecolor = 'red', alpha=0.5)

plt.xlabel("Square root of newbalanceDest")

plt.title("SQRT on newbalanceDest")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(tmp['newbalanceDest']))))



figure.add_subplot(133)

plt.hist(tmp['newbalanceDest_boxcox'], facecolor = 'red', alpha=0.5)

plt.xlabel("Box cox of newbalanceDest")

plt.title("Box cox on newbalanceDest")

plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(tmp['newbalanceDest_boxcox'])))



plt.show()
print("The fraud transaction of the filtered dataset: {0:.4f}%".format((len(tmp[tmp.isFraud == 1])/len(tmp)) * 100))
tmp.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'amount', 'type'], axis=1, inplace=True)



X = tmp.ix[:, tmp.columns != 'isFraud']

y = tmp.ix[:, tmp.columns == 'isFraud']
# Number of data points in the minority class

number_records_fraud = len(tmp[tmp.isFraud == 1])

fraud_indices = tmp[tmp.isFraud == 1].index.values



# Picking the indices of the normal classes

normal_indices = tmp[tmp.isFraud == 0].index



# Out of the indices we picked, randomly select "x" number (x - same as total fraud)

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)

random_normal_indices = np.array(random_normal_indices)



# Appending the 2 indices

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

under_sample_data = tmp.iloc[under_sample_indices, :]



X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'isFraud']

y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'isFraud']



# Showing ratio

print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.isFraud == 0])/len(under_sample_data))

print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.isFraud == 1])/len(under_sample_data))

print("Total number of transactions in resampled data: ", len(under_sample_data))
from sklearn.model_selection import train_test_split



# Whole dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



print("Number transactions train dataset: ", format(len(X_train),',d'))

print("Number transactions test dataset: ", format(len(X_test), ',d'))

print("Total number of transactions: ", format(len(X_train)+len(X_test), ',d'))



# Undersampled dataset

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample

                                                                                                   ,y_undersample

                                                                                                   ,test_size = 0.3

                                                                                                   ,random_state = 0)

print("")

print("Number transactions train dataset: ", format(len(X_train_undersample),',d'))

print("Number transactions test dataset: ", format(len(X_test_undersample),',d'))

print("Total number of transactions: ", format(len(X_train_undersample)+len(X_test_undersample),',d'))
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.metrics import confusion_matrix, precision_score, precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 





def printing_Kfold_scores(x_train_data, y_train_data, kfoldnum, c_array):

    # define K-Fold

    fold = KFold(len(y_train_data), kfoldnum,shuffle=False) 



    results_table = pd.DataFrame(index = range(len(c_array),3), columns = ['C_parameter','Mean recall score', 'Mean precision score'])

    results_table['C_parameter'] = c_array



    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]

    j = 0

    for c_param in c_array:

        print('-------------------------------------------')

        print('C parameter: ', c_param)

        print('-------------------------------------------')

        print('')



        recall_accs = []

        precision_accs = []

        for iteration, indices in enumerate(fold,start=1):



            # Call the logistic regression model with a certain C parameter

            lr = LogisticRegression(C = c_param, penalty = 'l1')



            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model

            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]

            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())



            # Predict values using the test indices in the training data

            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)



            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter

            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)

            recall_accs.append(recall_acc)

            

            precision_acc = precision_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)

            precision_accs.append(precision_acc)

            print("Iteration {}: recall score = {:.4f}, precision score = {:.4f}".format(iteration, recall_acc, precision_acc))



        # The mean value of those recall scores is the metric we want to save and get hold of.

        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)

        results_table.ix[j, 'Mean precision score'] = np.mean(precision_accs)

        j += 1

        print('')

        print('Mean recall score {:.4f}'.format(np.mean(recall_accs)))

        print('Mean precision score {:.4f}'.format(np.mean(precision_accs)))

        print('')



    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    

    # Finally, we can check which C parameter is the best amongst the chosen.

    print('*********************************************************************************')

    print('Best model to choose from cross validation is with C parameter = ', best_c)

    print('*********************************************************************************')

    

    return best_c
c_param_range = [0.001, 0.01, 0.1, 1, 10, 100]

k_fold = 5

best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample, k_fold, c_param_range)
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
lr = LogisticRegression(C = best_c, penalty = 'l1')

lr.fit(X_train_undersample,y_train_undersample.values.ravel())

y_pred_undersample = lr.predict(X_test_undersample.values)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)



print("Recall metric in the testing dataset: {0:.4f}".format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()


lr = LogisticRegression(C = best_c, penalty = 'l1')

lr.fit(X_train_undersample,y_train_undersample.values.ravel())

y_pred = lr.predict(X_test.values)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
# ROC CURVE

lr = LogisticRegression(C = best_c, penalty = 'l1')

y_pred_undersample_score = lr.fit(X_train_undersample,y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)



fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(),y_pred_undersample_score)

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
print(lr)

print(lr.intercept_ )

print(lr.coef_)

print(X.columns.tolist())