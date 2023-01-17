# Importing the required packages
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn import metrics, preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
# Loading the required dataset
data = pd.read_csv('../input/PL_XSELL.csv')
data.head()
data.dtypes
data.describe().T
df = data.drop(['CUST_ID','ACC_OP_DATE','AGE_BKT','random'],axis=1)

df.head()
df['GENDER'] = df['GENDER'].map({'M':0,'F':1,'O':2})
df['OCCUPATION'].value_counts()
df['OCCUPATION'] = df['OCCUPATION'].map({'SAL':0,'PROF':1,'SENP':2,'SELF-EMP':3})
df['ACC_TYPE'].value_counts()
df['ACC_TYPE'] = df['ACC_TYPE'].map({'SA':0,'CA':1})
df.dtypes
df.describe().T
df.isnull().sum()
df.boxplot()
(['BALANCE','SCR','ATM_AMT_DR','ATM_CHQ_DR','AMT_NET_DR','AMT_MOB_DR','AMT_L_DR','AVG_AMT_PER_ATM_TXN',

  'AVG_AMT_PER_CSH_WDL_TXN','AVG_AMT_PER_CHQ_TXN','AVG_AMT_PER_NET_TXN','AVG_AMT_PER_ATM_TXN'])
sns.boxplot(df['BALANCE'])
# Since the domain is banking domain, and the outliers are only present in money and transaction related columns,

# we will keep the outliers and proceed further
df['GENDER'] = df['GENDER'].astype('category')

df['OCCUPATION'] = df['OCCUPATION'].astype('category')

df['ACC_TYPE'] = df['ACC_TYPE'].astype('category')

df['FLG_HAS_CC'] = df['FLG_HAS_CC'].astype('category')

df['FLG_HAS_ANY_CHGS'] = df['FLG_HAS_ANY_CHGS'].astype('category')

df['FLG_HAS_NOMINEE'] = df['FLG_HAS_NOMINEE'].astype('category')

df['FLG_HAS_OLD_LOAN'] = df['FLG_HAS_OLD_LOAN'].astype('category')

df['TARGET'] = df['TARGET'].astype('category')
# EDA



plt.hist(x = df['AGE'],rwidth=0.9)

plt.show()
plt.hist(x = df['HOLDING_PERIOD'],rwidth=0.9,bins=10)

plt.show()
plt.hist(x = df['LEN_OF_RLTN_IN_MNTH'],rwidth=0.9,bins=25)

plt.show()
plt.hist(x = df['NO_OF_L_CR_TXNS'],rwidth=0.9,bins=20)

plt.show()
plt.hist(x = df['NO_OF_L_DR_TXNS'],rwidth=0.9,bins=20)

plt.show()
#From the histograms we can see that

#The frequency distribution for AGE shows that the targeted customers are highest in the age group between 26–30.

#The Holding Period (Ability to hold money in the account) and 

# length of relationship with the bank are more or less evenly distributed.

#The customers had most of their credit transactions in the range between 0–15 and debit transactions less than 10.
sns.boxplot(x=df['AGE'])
sns.boxplot(x=df['HOLDING_PERIOD'])
sns.boxplot(x=df['LEN_OF_RLTN_IN_MNTH'])
sns.boxplot(x=df['NO_OF_L_CR_TXNS'])
sns.boxplot(x=df['NO_OF_L_DR_TXNS'])
# From the box plots, we can visualize and infer:



# Box plot shows the following median values for the numeric variables: 

# age around 38 years, holding period i.e. ability to hold money in the account of 15 months, 

# Length of relationship with bank at 125 months, No. of credit transactions 10, and no. of debit transactions = 5.



# There are many outliers for the variables no. of credit transactions and no. of debit transactions.
sns.countplot(x='GENDER',data=df) # O - Male, 1 - Female 2 - Others
sns.countplot(x='OCCUPATION',data=df) # SAL - 0, PROF - 1, SENP - 2, SELF-EMP - 3
sns.countplot(x='FLG_HAS_CC',data=df)
sns.countplot(x='FLG_HAS_NOMINEE',data=df)
sns.countplot(x='FLG_HAS_OLD_LOAN',data=df)
sns.countplot(x='FLG_HAS_ANY_CHGS',data=df)
sns.countplot(x='ACC_TYPE',data=df) # SA - 0, CA - 1
sns.countplot(x='TARGET',data=df) # 0 - Not Responded, 1 - Responded for Personal Loan
# We can infer from the bar plots that

# Nearly 3/4th of the customers targeted in the loan campaign belonged to male gender.

# Salaried and Professional class form majority of the targeted customers.

# A quarter of the customers had a credit card

# equal proportion of customers in the dataset have an old loan or did not have one.

# More number of customers have savings account
sns.boxplot(x='TARGET',y='AGE',data=df)
sns.boxplot(x='TARGET',y='LEN_OF_RLTN_IN_MNTH',data=df)
sns.boxplot(x='TARGET',y='HOLDING_PERIOD',data=df)
# When we look at the data visualization from bivariate analysis of numeric variables 

# against the categorical target variable, we get the following insights:

# AGE vs Target (TARGET: responded to loan campaign = 1; Did not respond to loan campaign = 0)

# The median age of customers who responded to the campaign is slightly higher than the age of those who didn’t respond. 

# There is not much differentiation though between the two classes based on Age, an inference we also draw from 

# Length of Relationship with Bank in Months vs TARGET class.

# Customers who had lesser median holding period (Ability to hold money in the account) 

# of around 10 months are the ones who had responded to the personal loan campaign.
# plt.stackplot(x='GENDER',y='TARGET',)
# Split x and y

x = df.drop('TARGET',axis = 1)

y = df['TARGET']
# Split into train and test



x_std = preprocessing.StandardScaler().fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

x_train1,x_test1,y_train1,y_test1 = train_test_split(x_std,y,test_size=0.3,random_state=0)
# Instantiating the classification models



log_model = LogisticRegression()

gini_model = DecisionTreeClassifier(max_depth=5)

ent_model = DecisionTreeClassifier(max_depth=5,criterion='entropy')

knn_model = KNeighborsClassifier()

nbg_model = GaussianNB()

nbb_model = BernoulliNB()

nbm_model = MultinomialNB()

rf_gini_model = RandomForestClassifier(n_estimators=10,max_depth=5,criterion='gini')

rf_ent_model = RandomForestClassifier(n_estimators=10,max_depth=5,criterion='entropy')

svm_model = SVC()
models = []



models.append(('Logistic Regression Model',log_model))

models.append(('Decision Tree Model-Gini',gini_model))

models.append(('Decision Tree Model-Entropy',ent_model))

models.append(('KNN Model',knn_model))

models.append(('Random Forest Model-Gini',rf_gini_model))

models.append(('Random Forest Model-Entropy',rf_ent_model))

models.append(('Naive Bayes Model-Gaussian',nbg_model))

models.append(('Naive Bayes Model-Bernoulli',nbb_model))

models.append(('Naive Bayes Model-Multinomial',nbm_model))

models.append(('SVM Model',svm_model))
# Getting the AUC Scores, Bias and Variance errors



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



results = []

names = []



for name,model in models:

    kfold = KFold(n_splits = 5, shuffle = True, random_state = 2)

    if model == knn_model:

        cv_results = cross_val_score(model,x_std,y,cv = kfold, scoring = 'roc_auc')

    else:

        cv_results = cross_val_score(model,x,y,cv = kfold, scoring = 'roc_auc')

    results.append(cv_results)

    names.append(name)

    msg = '\n%s:\n\n\t\t %s (%f) %s (%f) %s (%f) ' % (name,"AUC Score:",np.mean(cv_results),",Bias Error:",

                                          1-np.mean(cv_results),",Variance Error:",np.var(cv_results, ddof = 1))

    print(msg)
# Printing the train and test accuracy for all the classification models



for model in models:

    

    if model[1] == knn_model:

        model[1].fit(x_train1,y_train1)

        print('Training Accuracy:',model[0],model[1].score(x_train1,y_train1))

        print('Test Accuracy:',model[0],model[1].score(x_test1,y_test1))

    

    else:

        model[1].fit(x_train,y_train)

        print('Training Accuracy:',model[0],model[1].score(x_train,y_train))

        print('Test Accuracy:',model[0],model[1].score(x_test,y_test))
# Getting the training and testing accuracy in a dataframe



Accuracy_Scores=pd.DataFrame([[log_model.score(x_train,y_train),log_model.score(x_test,y_test)],

                             [gini_model.score(x_train,y_train),gini_model.score(x_test,y_test)],

                             [ent_model.score(x_train,y_train),ent_model.score(x_test,y_test)],

                             [knn_model.score(x_train1,y_train1),knn_model.score(x_test1,y_test1)],

                             [rf_gini_model.score(x_train,y_train),rf_gini_model.score(x_test,y_test)],

                             [rf_ent_model.score(x_train,y_train),rf_ent_model.score(x_test,y_test)],

                             [nbg_model.score(x_train,y_train),nbg_model.score(x_test,y_test)],

                             [nbb_model.score(x_train,y_train),nbb_model.score(x_test,y_test)],

                             [nbm_model.score(x_train,y_train),nbm_model.score(x_test,y_test)],

                             [svm_model.score(x_train,y_train),svm_model.score(x_test,y_test)]],

                             columns=["Training Accuracy","Testing Accuracy"]

                               ,index= ["Logistic Regression","Decision Tree - Gini",

                                        "Decision Tree - Entropy","KNN Model",

                                     "Random Forest Model - Gini","Random Forest Model - Entropy",

                                        "Naive Bayes - Gaussian","Naive Bayes - Bernoulli",

                                       "Naive-Bayes - Multinomial","SVM Model"])
Accuracy_Scores
# From the above dataframe, we can conclude that the testing accuracy obtained using SVM Model is better than other

# classification models and using SVM Model we can predict the outcome of whether the customer accepts personal loan

# or not much better and accurate.