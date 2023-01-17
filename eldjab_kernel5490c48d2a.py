# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd

import math

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import resample



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

#Metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score, recall_score





import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/FinScope2016.csv", sep=",", header=0)

data=data.drop_duplicates()

data.shape
data = data.rename(columns={'A1':'Province', 'A6':'Zone','C3':'Sex','H2A':'Borrower','H2C':'Pay_back'})

print("--------Provinces--------")

print(data['Province'].value_counts(dropna=False))

print("--------Zones---------")

print(data['Zone'].value_counts(dropna=False))

print("-------Sex----------")

print(data['Sex'].value_counts(dropna=False))

print("--------Borrows money in the last 12 months--------")

print(data['Borrower'].value_counts(dropna=False))
##Selecting borrowers only

data=data[data.Borrower=='yes']

data.shape
data['Pay_back']=data['Pay_back'].fillna('No')

data['Pay_back'].value_counts(dropna=False)
lab_yes_no={'yes': 1, 'No':0}

data['Pay_back']=data['Pay_back'].map(lab_yes_no).astype(int)

data['Pay_back'].value_counts(dropna=False)
data[['Sex','Pay_back']].groupby(['Sex'], as_index=False).mean()
data[['Province','Pay_back']].groupby(['Province'], as_index=False).mean().sort_values(by='Pay_back',ascending=False)
data[['Zone','Pay_back']].groupby(['Zone'],as_index=False).mean()
print("---------Level of education-------------")

display(data['C4A'].value_counts())

print("---------Marital status-------------")

display(data['C4B'].value_counts())
data['education']=data['C4A']

data['education']=data['education'].replace(['No formal education','Vocational training'],'no education')

data['education']=data['education'].replace(['Primary 1-3','Primary 4-6'],'primary')

data['education']=data['education'].replace(['Secondary 1-3','Secondary 4-6'],'secondary')

data['education']=data['education'].replace(['University or other higher education',"Don't know"],'university')

data[['education','Pay_back']].groupby(['education'],as_index=False).mean()
data['Marital']=data['C4B']

data['Marital']=data['Marital'].replace('Never married','Single')

data['Marital']=data['Marital'].replace(['Married','Living together'],'Married')

data['Marital']=data['Marital'].replace(['Divorced/Separated','Widowed'],'Separated')

data[['Marital','Pay_back']].groupby(['Marital'],as_index=False).mean()
print("---------Problem of getting meal-------------")

display(data['C5A'].value_counts())

print("---------Problem of getting treatment/medecine-------------")

display(data['C5B'].value_counts())

print("---------Problem of getting cash income-------------")

display(data['C5D'].value_counts())
data['poverty1']=data['C5A']

data['poverty1']=data['poverty1'].replace('Refused','A few times')



data['poverty2']=data['C5B']

data['poverty2']=data['poverty2'].replace("Don't know",'Never')

data['poverty2']=data['poverty2'].replace("A few times,",'A few times')

data['poverty2']=data['poverty2'].replace("Many times,",'Many times')



data['poverty3']=data['C5D']

data['poverty3']=data['poverty3'].replace(["Don't know","Refused"],"A few times")

data['poverty3']=data['poverty3'].replace("A few times,",'A few times')

data['poverty3']=data['poverty3'].replace("Many times,",'Many times')
data[['poverty1','Pay_back']].groupby(['poverty1'],as_index=False).mean()
data[['poverty2','Pay_back']].groupby(['poverty2'],as_index=False).mean()
data[['poverty3','Pay_back']].groupby(['poverty3'],as_index=False).mean()
print('----------------Owning a cell---------------')

display(data['C14B.1'].value_counts(dropna=False))

print('--------------Dwelling ownership-------------')

display(data['C7'].value_counts(dropna=False))

print('----------------Income----------------')

display(data['N4A'].value_counts(dropna=False))
data = data.rename(columns={'C14B.1':'Own_cell'})

data['Own_cell']=data['Own_cell'].replace('Own,','Yes')

data['Own_cell']=data['Own_cell'].replace(" Household doesn't own",'No')

data['Own_cell']=data['Own_cell'].fillna('No' )

data[['Own_cell','Pay_back']].groupby(['Own_cell'],as_index=False).mean()
data['House_ownership']=data['C7']

data['House_ownership']=data['House_ownership'].replace([' You/your household rent this dwelling ,','You own this dwelling',\

                                                        'A member/other members of the household (not you) own this dwelling',\

                                                        'You own this dwelling together with someone else'],'Yes')

data['House_ownership']=data['House_ownership'].replace(['The dwelling is provided to you/your household rent free','Does not know'],\

                                                       'No')

data[['House_ownership','Pay_back']].groupby(['House_ownership'],as_index=False).mean()
data['Income']=data['N4A']

data['Income']=data['Income'].replace(['1,500 Rwf or less','1,501-    3,000 Rwf','3,001-    5,000 Rwf','5,001-    7,000 Rwf',\

                                      '7,001-  10,000 Rwf','10,001-  15,000 Rwf','15,001-  20,000 Rwf'],'less than 20000')

data['Income']=data['Income'].replace(['20,001-  25,000 Rwf','25,001-  30,000 Rwf','30,001-  40,000 Rwf',\

                                      '40,001-  50,000 Rwf'],'from 20001 to 50000')

data['Income']=data['Income'].replace(['50,001-100,000 Rwf','More than 100, 000 Rwf','Irregular/seasonal'],'more than 50000')



##Treatment of non responding

data.loc[(data.education=='no education') & (data.Income.isnull()),'Income']='less than 20000'

data.loc[(data.education=='primary') & (data.Income.isnull()),'Income']='less than 20000'

data.loc[(data.education=='secondary') & (data.Income.isnull()),'Income']='from 20001 to 50000'

data.loc[(data.education=='university') & (data.Income.isnull()),'Income']='more than 50000'



data[['Income','Pay_back']].groupby(['Income'],as_index=False).mean()
print('-------------Farming----------------')

display(data['M1'].value_counts(dropna=False))

print('-------------Reason for borrwing money----------------')

display(data['H5'].value_counts(dropna=False))

print('-------------Self employment----------------')

display(data['N2A.05'].value_counts(dropna=False))
data = data.rename(columns={'M1':'Farming'})

data[['Farming','Pay_back']].groupby(['Farming'],as_index=False).mean()
data['Borrow_reason']=data['H5']

data['Borrow_reason']=data['Borrow_reason'].replace(['For business/investment','Farming expenses such as seeds, fertiliser',\

                                                    'Buying livestock','Buying farming equipment/implements'],'Direct Invest')

data['Borrow_reason']=data['Borrow_reason'].replace(['Education or school fees','Building/improving dwelling',\

                                                    'Buying land/dwelling'],'indirect Invest')

data['Borrow_reason']=data['Borrow_reason'].replace(['Living expenses when you did not have money',\

                                                    'Medical expenses/medical emergencies',\

                                                    'An emergency other than medical','Paying off other debt',\

                                                    'Funeral expenses','Other specify'],'Other')

data['Borrow_reason']=data['Borrow_reason'].fillna("Na")

data[['Borrow_reason','Pay_back']].groupby(['Borrow_reason'], as_index=False).mean()
data['Borrow_reason']=data['Borrow_reason'].replace(['Direct Invest','indirect Invest'],'Invest')

data['Borrow_reason']=data['Borrow_reason'].replace(['Other','Na'],'Other')

data[['Borrow_reason','Pay_back']].groupby(['Borrow_reason'], as_index=False).mean()
data['Self_emp']=data['N2A.05']

data['Self_emp']=data['Self_emp'].fillna('No')

data[['Self_emp','Pay_back']].groupby(['Self_emp'],as_index=False).mean()
data = data.rename(columns={'G3.1.A': 'Saving_at_Bank','G3.3.A': 'Saving_at_MobMoney','G3.12.A': 'Saving_Products'})

data[['Pay_back','Saving_at_MobMoney']].groupby(['Saving_at_MobMoney'], as_index=False).mean()
data[['Pay_back','Saving_Products']].groupby(['Saving_Products'], as_index=False).mean()
data[['Pay_back','Saving_at_Bank']].groupby(['Saving_at_Bank'], as_index=False).mean()
print('-----------Bank-------------')

display(data['H4.01.A'].value_counts())

print('-----------Microfinance institutions-------------')

display(data['H4.02.A'].value_counts())

print('-------------Mobile Money---------------')

display(data['H4.03.A'].value_counts())

print('-----------SACCO-------------')

display(data['H4.04.A'].value_counts())

print('-----------Government-------------')

display(data['H4.05.A'].value_counts())
#Those who borrow money from Bank,MFI,mobile money, sacco or government

data['Formal_borrow']=0

data.loc[(data['H4.01.A']=='yes')|(data['H4.02.A']=='yes')|(data['H4.03.A']=='yes')|(data['H4.04.A']=='yes')|(data['H4.05.A']=='yes'),'Formal_borrow']=1

data[['Pay_back','Formal_borrow']].groupby(['Formal_borrow'], as_index=False).mean()
df=data[['Pay_back','Income','Self_emp','Farming','Province','Zone','education','House_ownership','Own_cell','poverty1','poverty2',\

         'poverty3','Marital','Saving_at_Bank','Saving_at_MobMoney','Saving_Products','Borrow_reason','Formal_borrow','Sex']]
yn={'Yes': 1, 'No': 0}

pov={'Never': 0, 'A few times': 1, 'Many times': 2}



##Socio Demographic variables

df['Province']=df['Province'].map({'Eastern Province': 0,'Southern Province': 1,'Western Province': 2,'Northern Province': 3,'Kigali City': 4})

df['Zone']=df['Zone'].map({'Rural': 0,'Urban': 1}).astype(int)

df['education']=df['education'].map({'no education': 0,'primary': 1,'secondary': 2, 'university': 3})

df['Marital']=df['Marital'].map({'Single': 0, 'Married': 1, 'Separated':2})

df['Sex']=df['Sex'].map({'Male':1, 'Female':0})



##Employment and borrowing reason

df['Borrow_reason']=df['Borrow_reason'].map({'Invest':1, 'Other':0})

df['Farming']=df['Farming'].map({'Your household is only involved in farming and no-one in the household has any other work': 0,'Your household is involved in farming AND other work': 1,'Your household is NOT involved in farming at all': 2})

df['Self_emp']=df['Self_emp'].map(yn)



##Richest variables

df['Income']=df['Income'].map({'less than 20000': 0, 'from 20001 to 50000': 1, 'more than 50000': 2})

df['House_ownership']=df['House_ownership'].map(yn)

df['Own_cell']=df['Own_cell'].map(yn)



##Poverty variables

df['poverty1']=df['poverty1'].map(pov)

df['poverty2']=df['poverty2'].map(pov)

df['poverty3']=df['poverty3'].map(pov)



##Saving habit variables

df['Saving_at_Bank']=df['Saving_at_Bank'].map(yn)

df['Saving_at_MobMoney']=df['Saving_at_MobMoney'].map(yn)

df['Saving_Products']=df['Saving_Products'].map(yn)



df.head()
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=2)

print('Train data :', df_train.shape, 'Test data :', df_test.shape)
X=df.drop('Pay_back', axis=1)

Y=df['Pay_back']

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2)

X_train=df_train.drop('Pay_back', axis=1)

Y_train=df_train['Pay_back']

X_test=df_test.drop('Pay_back', axis=1)

Y_test=df_test['Pay_back']



print()

print("========= Train and Test Sets Splits ==========")

print("X_train shape: {}".format(X_train.shape)) 

print("Y_train shape: {}".format(Y_train.shape))

print("X_test shape: {}".format(X_test.shape))

print("Y_test shape: {}".format(Y_test.shape))
display(df_train['Pay_back'].value_counts(dropna=False))

display(df_test['Pay_back'].value_counts(dropna=False))
##Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

precision_log = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_log = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_log = round(logreg.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(logreg, X, Y, cv=cv)  

cv_log = round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_log)) 

print("Recall: {}".format(recall_log))

print("Accuracy score: {}".format(acc_log))

print("Cross validation mean score: {}".format(cv_log))
##Support Vector Machine

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

precision_svc = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_svc = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_svc = round(svc.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(svc, X, Y, cv=cv) 

cv_svc = round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_svc)) 

print("Recall: {}".format(recall_svc))

print("Accuracy score: {}".format(acc_svc))

print("Cross validation mean score: {}".format(cv_svc))
##KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

precision_knn = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_knn = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_knn = round(knn.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(knn, X, Y, cv=cv) 

cv_knn = round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_knn)) 

print("Recall: {}".format(recall_knn))

print("Accuracy score: {}".format(acc_knn))

print("Cross validation mean score: {}".format(cv_knn))
##Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

precision_gaussian = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_gaussian = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(gaussian, X, Y, cv=cv) 

cv_gaussian = round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_gaussian)) 

print("Recall: {}".format(recall_gaussian))

print("Accuracy score: {}".format(acc_gaussian))

print("Cross validation mean score: {}".format(cv_gaussian))
##Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10)

rfc.fit(X_train, Y_train)

Y_pred=rfc.predict(X_test)

precision_rfc = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_rfc = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_rfc=round(rfc.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(rfc, X, Y, cv=cv) 

cv_rfc = round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_rfc)) 

print("Recall: {}".format(recall_rfc))

print("Accuracy score: {}".format(acc_rfc))

print("Cross validation mean score: {}".format(cv_rfc))
##Perceptron

from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

precision_perceptron = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_perceptron = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(perceptron, X, Y, cv=cv) 

cv_perceptron = round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_perceptron)) 

print("Recall: {}".format(recall_perceptron))

print("Accuracy score: {}".format(acc_perceptron))

print("Cross validation mean score: {}".format(cv_perceptron))
##Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

precision_linear_svc = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_linear_svc = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(linear_svc, X, Y, cv=cv) 

cv_linear_svc = round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_linear_svc)) 

print("Recall: {}".format(recall_linear_svc))

print("Accuracy score: {}".format(acc_linear_svc))

print("Cross validation mean score: {}".format(cv_linear_svc))
##Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

precision_sgd = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_sgd = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_sgd = round(sgd.score(X_test, Y_test) * 100, 2)



scores=cross_val_score(sgd, X, Y, cv=cv) 

cv_sgd=round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_sgd)) 

print("Recall: {}".format(recall_sgd))

print("Accuracy score: {}".format(acc_sgd))

print("Cross validation mean score: {}".format(cv_sgd))
##Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

precision_decision_tree = round(precision_score(Y_test, Y_pred, average='macro')*100, 2)

recall_decision_tree = round(recall_score(Y_test, Y_pred,average='macro')*100, 2)

acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)



scores = cross_val_score(decision_tree, df.drop('Pay_back', axis=1), df['Pay_back'], cv=5)



scores=cross_val_score(decision_tree, X, Y, cv=cv) 

cv_decision_tree=round(np.mean(scores)*100, 2)



print()

print("========= Model Evaluation ==========")

print("Precision: {}".format(precision_decision_tree)) 

print("Recall: {}".format(recall_decision_tree))

print("Accuracy score: {}".format(acc_decision_tree))

print("Cross validation mean score: {}".format(cv_decision_tree))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_rfc, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'CV Score': [cv_svc, cv_knn, cv_log, 

              cv_rfc, cv_gaussian, cv_perceptron, 

              cv_sgd, cv_linear_svc, cv_decision_tree]})

models.sort_values(by='CV Score', ascending=False)
coeff_df = pd.DataFrame(df_train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)