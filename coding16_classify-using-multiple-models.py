import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
adult=pd.read_csv('../input/adult-census-income/adult.csv', na_values=["?"])

print(adult.head())
adult_new=adult.dropna(axis=0)

print(adult_new.head())

print('Dimensions:',adult.shape)
col=adult_new.columns

data_type=adult_new.dtypes

uniq=adult_new.nunique()



print("\n%30s  %10s   %10s\n " % ("Column Name", "Data Type", "Unique Values"))

for i in range(len(adult_new.columns)):

    print("%30s  %10s   %10s " % (col[i],data_type[i],uniq[i]))



print("\nDimensions:",adult_new.shape[0],'rows and ',adult_new.shape[1],'columns')


adult_new['income'].replace({'<=50K':0,'>50K':1},inplace=True)

adult_new=adult_new.drop('education.num',axis=1)
from collections import Counter

occupatn=dict(Counter(adult_new['occupation'])).keys()

print('Occupation types:','\n',list(occupatn),'\n')

race=dict(Counter(adult_new['race'])).keys()

print('Race types:','\n',list(race),'\n')

relation=dict(Counter(adult_new['relationship'])).keys()

print('Relation types:','\n',list(relation),'\n')

educate=dict(Counter(adult_new['education'])).keys()

print('Education levels:','\n',list(educate),'\n')

marital=dict(Counter(adult_new['marital.status'])).keys()

print('Marital status levels:','\n',list(marital),'\n')

work=dict(Counter(adult_new['workclass'])).keys()

print('Workclass levels:','\n',list(work),'\n')

country=dict(Counter(adult_new['native.country'])).keys()

print('Native countries:','\n',list(country),'\n')
import seaborn as sns

#get correlations of each features in dataset

corrmat = adult_new.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(adult_new[top_corr_features].corr(),annot=True,cmap="twilight_shifted_r")
import scipy.stats as stats

a=['age','capital.loss','capital.gain','hours.per.week','fnlwgt']

for i in a:

    print(i,':',stats.pointbiserialr(adult_new['income'],adult_new[i])[0])

adult_new=adult_new.drop('fnlwgt',axis=1)

adult_new.dtypes
categorical_cols = adult_new.columns[adult_new.dtypes==object].tolist()

categorical_cols


def cross_tab(obs1=[]):

    observed=pd.crosstab(obs1,adult_new['income'])

    val=stats.chi2_contingency(observed)

    return(val[1])
alpha=0.01

df=adult_new.drop('income',axis=1)

count=0

attributes2=[]

for i in categorical_cols:

    p_value=cross_tab(adult_new[i])

    if p_value<=alpha:

        count+=1

        attributes2.append(i)

print('Number of attributes contributing:',count,'\n')

print(attributes2)
pd.crosstab(adult.relationship,adult_new['income'])
categorical_cols
adult_new1=pd.get_dummies(adult_new,columns=categorical_cols)

adult_new1.head()
adult_new1.columns
from sklearn.preprocessing import MinMaxScaler

columns_to_scale = ['age', 'capital.gain', 'capital.loss', 'hours.per.week']

mms = MinMaxScaler()

min_max_scaled_columns = mms.fit_transform(adult_new1[columns_to_scale])

#processed_data = np.concatenate([min_max_scaled_columns, adult_new], axis=1)

adult_new1['age'],adult_new1['capital.gain'],adult_new1['capital.loss'],adult_new1['hours.per.week']=min_max_scaled_columns[:,0],min_max_scaled_columns[:,1],min_max_scaled_columns[:,2],min_max_scaled_columns[:,3]

adult_new1.head()
category=adult_new1.columns[adult_new1.dtypes!=object].tolist()[5:]

#category

alpha=0.01

#df=adult_new.drop('income',axis=1)

count=0

features=[]

for i in category:

    p_value=cross_tab(adult_new1[i])

    if p_value<=alpha:

        count+=1

        features.append(i)

        #print(i,' has a relation')

        #print('p-value for ',i,' is ',cross_tab(adult_new[i]),'\n')

print('Number of contributing attributes:',count,'\n')

print(features)
features.append('age')

features.append('capital.gain')

features.append('capital.loss')

features.append('hours.per.week')

features.append('income')
adult_new1[features].head()

import seaborn as sns

#get correlations of each features in dataset

corrmat = adult_new1[features].corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(adult_new1[top_corr_features].corr(),annot=True,cmap="twilight_shifted_r")
f,ax=plt.subplots(1,2,figsize=(18,10))

#plt.figure(figsize=(7,10))

income1=adult_new1['income'].value_counts()

ax[0].pie(income1,explode=(0,0.05),autopct='%1.1f%%',startangle=90,labels=['<=50K','>50K'])

ax[0].set_title('Income Share')

ax[1]=sns.countplot(x='income',data=adult_new1,palette='pastel')

ax[1].legend(labels=['<=50K','>50K'])

ax[1].set(xlabel="INCOME CATEGORIES")

ax[1].set(ylabel='COUNT OF THE CATEGORIES')

ax[1].set_title('COUNT OF THE TWO LEVELS')



for p in ax[1].patches:

    ax[1].annotate(p.get_height(),(p.get_x()+0.3,p.get_height()+500))
plt.figure(figsize=(10,10))

ax=sns.countplot(x='income',hue='sex',data=adult_new,palette='Set1')

ax.set(xlabel='INCOME 50')

ax.set(ylabel='COUNT WITH AGE')

ax.set_title('INCOME WITH RESPECT TO SEX')

for p in ax.patches:

    ax.annotate(p.get_height(),(p.get_x()+0.15,p.get_height()+200))
f, ax = plt.subplots(figsize=(15, 8))

ax = sns.countplot(x="income", hue="race", data=adult_new, palette="Set1")

ax.set_title("FREQUENCY DISTRIBUTION OF INCOME WITH RESPECT TO AGE")

ax.set(xlabel='INCOME RANGE',ylabel='FREQUENCY OF AGES')



for p in ax.patches:

    ax.annotate(p.get_height(),(p.get_x()+0.05,p.get_height()+200))

plt.show()
f, ax = plt.subplots(figsize=(12, 8))

ax = sns.countplot(x="workclass", hue="income", data=adult_new, palette="Set2")

ax.set_title("FREQUENCY DISTRIBUTION OF WORKCLASS WITH RESPECT TO INCOME")

ax.set(xlabel='WORKCLASS RANGE',ylabel='FREQUENCY OF WORKCLASS')

ax.legend(labels=['<=50K','>50K'],loc='upper right',fontsize='large')

plt.show()
#adult1=sns.load_dataset("adult.csv")

g = sns.FacetGrid(adult, col="occupation")

g.map(sns.countplot,'sex',alpha=0.7)

plt.figure(figsize=(10,10))

#sns.regplot(x='hours.per.week', y='fnlwgt',data=adult_new);

x=adult_new['hours.per.week']

plt.hist(x,bins=8,histtype='step')

plt.ylabel('FREQUENCY')#,xlabel='Hours per week')

plt.xlabel('HOURS PER WEEK')

plt.title('HISTOGRAM OF HOURS PER WEEK')

import statistics as stat

plt.axvline(stat.mode(x),color='red')

plt.show()
f,ax=plt.subplots(1,2,figsize=(15,10))



less=adult_new[adult_new['income']==0]

age_mode1=stat.mode(less.age)

more=adult_new[adult_new['income']!=0]

age_mode2=stat.mode(more.age)

#ax.axvline(age_mode1,age_mode2)

print('Maximum people around age ',age_mode1,' earn <=50K \n')

print('Maximum people around age ',age_mode2,' earn >50K \n')

ax[0].hist(less['age'],bins=15,histtype='step',color='green')

ax[0].set(xlabel='AGE RANGE',ylabel='FREQUENCY OF AGE')

ax[0].set_title('AGE DISTRIBUTION FOR INCOME <=50K')

ax[0].axvline(age_mode1,color='red')

ax[1].hist(more['age'],bins=8,histtype='step',color='red')

ax[1].set(xlabel='AGE RANGE',ylabel='FREQUENCY OF AGE')

ax[1].set_title('AGE DISTRIBUTION FOR INCOME >50K')

ax[1].axvline(age_mode2,color='black')

plt.show()
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix,accuracy_score

chi2=adult_new1[features]

def train_print(clf,x_test,y_test):

    predictions = clf.predict(x_test)

    print('Precision report:\nprecision \t\t\t recall \t\t\t f-score \t\t\t support\n',

          precision_recall_fscore_support(y_test, predictions)[0],'\t',

          precision_recall_fscore_support(y_test, predictions)[1],

          '\t',precision_recall_fscore_support(y_test, predictions)[2],'\t',

          precision_recall_fscore_support(y_test, predictions)[3],'\n')

    print('Confusion matrix:\n',confusion_matrix(y_test, predictions),'\n')

    print('Accuracy score:',accuracy_score(y_test, predictions)*100,'\n')
from sklearn.linear_model import LogisticRegression



x = chi2.drop('income', axis=1)

y = chi2['income']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

X_train, Y_train = SMOTE().fit_sample(x_train, y_train)



logmodel = LogisticRegression()

logmodel.fit(X_train, Y_train)



train_print(logmodel,x_test,y_test)
# predict probabilities

lr_probs = logmodel.predict_proba(x_test)

#print(lr_probs)

# keep probabilities for the positive outcome only

lr_probs = lr_probs[:, 1]

#print(lr_probs)

ns_probs = [0 for _ in range(len(y_test))]

#print(ns_probs)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# calculate scores

ns_auc = roc_auc_score(y_test, ns_probs)

lr_auc = roc_auc_score(y_test, lr_probs)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='Logistic: ROC AUC=%.3f' % (lr_auc))

# axis labels

plt.title('ROC CURVE')

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.tree import DecisionTreeClassifier

x = chi2.drop('income', axis=1)

y = chi2['income']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

x_train, y_train = SMOTE().fit_sample(x_train, y_train)



# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=8,max_depth=10)



# Train Decision Tree Classifer

clf.fit(x_train,y_train)



train_print(clf,x_test,y_test)
# predict probabilities

dt_probs = clf.predict_proba(x_test)

# keep probabilities for the positive outcome only

dt_probs1 = dt_probs[:, 1]

#lr_probs2 = lr_probs[:,0]

ns_probs = [0 for _ in range(len(y_test))]



# calculate scores

ns_auc = roc_auc_score(y_test, ns_probs)

dt_auc = roc_auc_score(y_test, dt_probs1)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Deciison Tree: ROC AUC=%.3f' % (dt_auc))
# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs1)

# plot the roc curve for the model

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(dt_fpr, dt_tpr, linestyle='--',marker='*',label='Deciison Tree: ROC AUC=%.3f' % (dt_auc))

# axis labels

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier

model = GaussianNB()



x = chi2.drop('income', axis=1)

y = chi2['income']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) # 70% training and 30% test

x_train, y_train = SMOTE().fit_sample(x_train, y_train)



# Train the model using the training sets

gnb = model.fit(x_train,y_train)



train_print(gnb,x_test,y_test)
# predict probabilities

nb_probs = model.predict_proba(x_test)

# keep probabilities for the positive outcome only

nb_probs1 = nb_probs[:, 1]

#lr_probs2 = lr_probs[:,0]

ns_probs = [0 for _ in range(len(y_test))]



# calculate scores

ns_auc = roc_auc_score(y_test, ns_probs)

nb_auc = roc_auc_score(y_test, nb_probs1)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Naive Bayes: ROC AUC=%.3f' % (nb_auc))

# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs1)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(nb_fpr, nb_tpr, linestyle='--',marker='*',label='Naive Bayes: ROC AUC=%.3f' % (nb_auc))

# axis labels

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()

from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier(min_samples_split=30)



x = chi2.drop('income', axis=1)

y = chi2['income']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

x_train, y_train = SMOTE().fit_sample(x_train, y_train)



# Train the model using the training sets

rf.fit(x_train,y_train)



train_print(rf,x_test,y_test)
# predict probabilities

rf_probs = rf.predict_proba(x_test)

# keep probabilities for the positive outcome only

rf_probs1 = rf_probs[:, 1]

#lr_probs2 = lr_probs[:,0]

ns_probs = [0 for _ in range(len(y_test))]



# calculate scores

ns_auc = roc_auc_score(y_test, ns_probs)

rf_auc = roc_auc_score(y_test, rf_probs1)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Random Forest: ROC AUC=%.3f' % (rf_auc))

# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs1)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(rf_fpr, rf_tpr, linestyle='--',marker='*',label='Random Forest: ROC AUC=%.3f' % (rf_auc))

# axis labels

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()

plt.figure(figsize=(15,10))

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='Logistic: ROC AUC=%.3f' % (lr_auc))

plt.plot(dt_fpr, dt_tpr, linestyle='--',marker='*',label='Deciison Tree: ROC AUC=%.3f' % (dt_auc))

plt.plot(nb_fpr, nb_tpr, linestyle='--',marker='*',label='Naive Bayes: ROC AUC=%.3f' % (nb_auc))

plt.plot(rf_fpr, rf_tpr, linestyle='--',marker='*',label='Random Forest: ROC AUC=%.3f' % (rf_auc))

# axis labels

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

plt.title('ROC CURVES')

# show the legend

plt.legend()

# show the plot

plt.show()