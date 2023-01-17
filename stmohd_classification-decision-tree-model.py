import numpy as np # linear algebra

import pandas as pd # data processing



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import boxcox,skew,norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

#reading the dataset



bank_data = pd.read_csv('../input/PL_XSELL.csv')
bank_data.shape
bank_data.head()
bank_data.drop(['random','CUST_ID'],axis=1,inplace=True)
bank_data.info()#checking data types
bank_data.isna().sum()#checking null values
bank_data.describe(include='all').transpose()#descriptive statistics
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

bank_data['GENDER'] = le.fit_transform(bank_data['GENDER'])

bank_data['AGE_BKT'] = le.fit_transform(bank_data['AGE_BKT'])

bank_data['OCCUPATION'] = le.fit_transform(bank_data['OCCUPATION'])

bank_data['ACC_TYPE'] = le.fit_transform(bank_data['ACC_TYPE'])

bank_data['ACC_OP_DATE'][:3]
type(bank_data['ACC_OP_DATE'][0])
# converting into pd datetime format
bank_data['ACC_OP_DATE']=pd.to_datetime(bank_data['ACC_OP_DATE'])
type(bank_data['ACC_OP_DATE'][0])
bank_data.head(3)
plt.subplots(figsize=(16,10))

sns.heatmap(bank_data.corr())
bank_data.columns

#Univariate analysis
bank_data.TARGET.value_counts()

bank_data.TARGET.value_counts().plot(kind='bar')
bank_data.GENDER.value_counts().plot(kind='bar')
sns.boxplot(bank_data.AGE)
sns.distplot(bank_data.AGE)
sns.countplot(bank_data['ACC_OP_DATE'])
bank_data.OCCUPATION.value_counts().plot(kind='bar')
#occupation wise interested in new loan  policy

bank_data.OCCUPATION[bank_data.TARGET==1].value_counts().plot(kind='bar')

print(bank_data.OCCUPATION[bank_data.TARGET==1].value_counts())
(bank_data.OCCUPATION[bank_data.TARGET==1].value_counts()/bank_data.OCCUPATION.value_counts())*100
#ACC_TYPE wise interested in new loan  policy

bank_data.ACC_TYPE[bank_data.TARGET==1].value_counts().plot(kind='bar')

print(bank_data.ACC_TYPE[bank_data.TARGET==1].value_counts())

(bank_data.ACC_TYPE[bank_data.TARGET==1].value_counts()/bank_data.ACC_TYPE.value_counts())*100
# Gender wise interest in new loan policy

bank_data.GENDER[bank_data.TARGET==1].value_counts().plot(kind='bar')

print(bank_data.GENDER[bank_data.TARGET==1].value_counts())

# GENDER wise  interest in new loan 

(bank_data.GENDER[bank_data.TARGET==1].value_counts()/bank_data.GENDER.value_counts())*100
# bank_data['ACC_OP_DATE'] = le.fit_transform(bank_data['OCCUPATION'])
bank_data.columns
##Age Bucket interested in new policy

bank_data.AGE_BKT[bank_data.TARGET==1].value_counts().plot(kind='bar')

print(bank_data.AGE_BKT[bank_data.TARGET==1].value_counts())

# % of each Age Bucket groups interested in new loan 

(bank_data.AGE_BKT[bank_data.TARGET==1].value_counts()/bank_data.AGE_BKT.value_counts())*100
# Age Bucket 3 is more interested(17%) in new loan policy than other age groups
sns.scatterplot(bank_data['ACC_OP_DATE'],bank_data.TARGET,alpha=0.1)
#scaling data

x = bank_data.drop(['ACC_OP_DATE','TARGET'],axis=1)

y =bank_data.TARGET
sc= StandardScaler()

sc.fit_transform(x)
from sklearn.model_selection import train_test_split
#splitting data into 70% btrain and 30% test



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#CART Algotrithm
from sklearn.tree import DecisionTreeClassifier

import sklearn.metrics as metrics
dct = DecisionTreeClassifier()
dct.fit(x_train,y_train)
#train set accuracy score

dct.score(x_train,y_train)
#test set accuracy score

dct.score(x_test,y_test)
#check with cross validation
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(dct,x_test,y_test,cv=5,scoring='accuracy')
cross_score
#mean accuracy score

print(cross_score.mean())
predit = dct.predict(x_test)#predictd values
#confusion metrics

cm=metrics.confusion_matrix(y_test,predit)
print(cm)
cr=metrics.classification_report(y_test,predit)
print(cr)
#roc curve



def roccurve(y_values, y_preds_proba):

    fpr, tpr, _ = metrics.roc_curve(y_values, y_preds_proba)

    xx = np.arange(101) / float(100)

    aur = metrics.auc(fpr,tpr)

    plt.xlim(0, 1.0)

    plt.ylim(0, 1.25)

    plt.plot([0.0, 0.0], [0.0, 1.0], color='green', linewidth=8)

    plt.plot([0.0, 1.0], [1.0, 1.0], color='green', label='Perfect Model', linewidth=4)

    plt.plot(xx,xx, color='blue', label='Random Model')

    plt.plot(fpr,tpr, color='red', label='CART Model')

    plt.title("ROC Curve - AUR value ="+str(aur))

    plt.xlabel('% False positives')

    plt.ylabel('% True positives')

    plt.legend()

    plt.show()









dct_test_pred_proba = dct.predict_proba(X=x_test)

roccurve(y_values=y_test, y_preds_proba=dct_test_pred_proba[:,1])