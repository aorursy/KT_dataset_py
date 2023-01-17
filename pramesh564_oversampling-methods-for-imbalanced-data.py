# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
cc_df = pd.read_csv('../input/creditcard.csv')
cc_df.info()

cc_df.describe()

cc_df.head()
sns.pairplot(cc_df,hue='Class')
fig, axes = plt.subplots(1,2,figsize=(20,10))

sns.scatterplot(x='V3', y='V1', hue='Class', data=cc_df, ax=axes[0])

sns.scatterplot(x='Time', y='Amount', hue= 'Class', data=cc_df, ax=axes[1])
for column in cc_df.columns:

    a = column + ': ' + str(cc_df[column].isnull().sum())

    print(a)

    print('\n')
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

cc_df['Amount'] = ss.fit_transform(cc_df['Amount'].values.reshape(-1,1))
from sklearn.model_selection import train_test_split

X = cc_df.drop(['Class','Time'],axis=1)

y = cc_df['Class']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=109)

#type(X_train)
sns.countplot(cc_df['Class'])

cc_df['Class'].value_counts()
from sklearn.utils import resample



(y_train==0).sum()

X_train_ones = X_train[y_train == 1]

X_train_zeros = X_train[y_train == 0]

X_train_ones = resample(X_train_ones,replace=True,n_samples=len(X_train_zeros))

X_train_randomOversampled = pd.concat([X_train_ones,X_train_zeros], ignore_index=True)

y_train_zeros, y_train_ones = y_train[X_train_zeros.index], y_train[X_train_ones.index]

y_train_randomOversampled = pd.concat([y_train_ones,y_train_zeros], ignore_index=True)



# Now to try SMOTE

from imblearn.over_sampling import SMOTE



smote = SMOTE()

X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)

X_train_smote_df = pd.DataFrame(X_train_smote,columns=X.columns)

y_train_smote_df = pd.DataFrame(y_train_smote,columns=['Class'])



sum(y_train_smote_df['Class']==1), sum(y_train_smote_df['Class']==0)
#Undersampling

X_train_zeros = X_train[y_train==0].reset_index(drop=True)



import sklearn

X_train_ones = X_train[y_train==1]

X_train_zeros_sampled = X_train_zeros.iloc[sklearn.utils.random.sample_without_replacement(len(X_train_zeros),len(X_train_ones))]



X_train_undersampled = pd.concat([X_train_ones,X_train_zeros_sampled],axis=0).reset_index(drop=True)

y_train_undersampled = pd.DataFrame(np.concatenate((np.ones(len(X_train_ones)),np.zeros(len(X_train_zeros_sampled))),axis=0)).reset_index(drop=True)



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.metrics import recall_score, f1_score

seven_foldCV_forC = KFold(n_splits=7)



list_of_C = [0.001, 0.01, 0.1, 1, 10, 100]



for C in list_of_C:

    list_of_recall_scores = []

    for train_indices, test_indices in seven_foldCV_forC.split(X_train_smote_df):

        lreg_model = LogisticRegression(C=C, solver='lbfgs')

        lreg_model.fit(X_train_smote_df.iloc[train_indices],y_train_smote_df.iloc[train_indices].values.ravel())

        predictions = lreg_model.predict(X_train_smote_df.iloc[test_indices])

        recall_sc = recall_score(predictions,y_train_smote_df.iloc[test_indices])

        list_of_recall_scores.append(recall_sc)

    print('Mean Recall score for C: {} is {}'.format(C, np.mean(list_of_recall_scores)) )

    

        
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



models = {#'NaiveBayes': MultinomialNB(),

          'RandomForestClassifier': RandomForestClassifier(n_estimators=400),

          'LogisticRegression': LogisticRegression(C=10),

          'SupportVectorClassifier': SVC(),

          'DecisionTreeClassifier': DecisionTreeClassifier()}
from sklearn.metrics import roc_auc_score

def FindAUC(model, X_train_in, y_train_in, X_test_in, y_test_in):

    model_fit = model.fit(X_train_in,y_train_in)

    try:

        y_pred_score = model_fit.decision_function(X_test_in)

        return roc_auc_score(y_test_in, y_pred_score)

    except:

        y_pred_score = model_fit.predict_proba(X_test_in)    

        return roc_auc_score(y_test_in, y_pred_score[:,-1])
    FindAUC(models['LogisticRegression'],X_train_smote_df,y_train_smote_df, X_test, y_test)

auc_list_smote, auc_list_random, auc_list_under = [], [], []

for model in models:

    print('Now checking {}'.format(model))

    auc_list_smote.append(FindAUC(models[model],X_train_smote_df,y_train_smote_df, X_test, y_test))

    auc_list_random.append(FindAUC(models[model],X_train_randomOversampled,y_train_randomOversampled, X_test, y_test))

    auc_list_under.append(FindAUC(models[model],X_train_undersampled,y_train_undersampled, X_test, y_test))

    print(auc_list_smote)

    print('\n')

#pd.DataFrame(index=[models.keys], columns=['AUC SMOTE', 'AUC RANDOM'])
comp_table = pd.DataFrame(index=[list(models.keys())], columns=['AUC SMOTE', 'AUC RANDOM', 'AUC UNDERSAMPLED'])

comp_table['AUC SMOTE'] = auc_list_smote

comp_table['AUC RANDOM OVERSAMPLED'] = auc_list_random

comp_table['AUC UNDERSAMPLED'] = auc_list_under

comp_table