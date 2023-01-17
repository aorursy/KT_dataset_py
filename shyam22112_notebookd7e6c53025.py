#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")
df.head()
df.info()
df.columns
df[['Class','Amount']].head()
df['Amount'].max()
df.loc[2]
df.loc[5665,'Amount']
df.loc[[123,45356],['Amount','Class']]
df[df['Amount']==25691.16]['Class']
df.Time[df.Class == 1]
print ("Fraud")

print (df.Time[df.Class == 1].describe())

print ()

print ("Normal")

print (df.Time[df.Class == 0].describe())
df.describe()
df['Class'].value_counts()
df['Amount'].sum()
df.Amount[df.Class == 1]
print ("Fraud")

print (df.Amount[df.Class == 1].describe())

print ()

print ("Normal")

print (df.Amount[df.Class == 0].describe())
fig,(ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14,4))



bins = 10



ax1.hist(df.Amount[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Amount[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.yscale('log')

plt.show()
Count_Normal_transacation = len(df[df["Class"]==0]) # normal transaction are repersented by 0

Count_Fraud_transacation = len(df[df["Class"]==1]) # fraud by 1

Count_Normal_transacation
Count_Fraud_transacation
Percentage_of_Normal_transacation = Count_Normal_transacation/(Count_Normal_transacation+Count_Fraud_transacation)

print("percentage of normal transacation is",Percentage_of_Normal_transacation*100)

Percentage_of_Fraud_transacation= Count_Fraud_transacation/(Count_Normal_transacation+Count_Fraud_transacation)

print("percentage of fraud transacation",Percentage_of_Fraud_transacation*100)
from sklearn.preprocessing import StandardScaler

df['Amount_n']= StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df.head()
df['Time_H']= df['Time']/3600
sns.jointplot(df['Time_H'], df['Class'])
df.drop(["Time_H"],axis=1,inplace=True)

df.head()
df.drop(["Time","Amount"],axis=1,inplace=True)
df.head()
X= df.iloc[:, df.columns != 'Class']

y= df.iloc[:, df.columns == 'Class']   
X.head()
y.head()
fraud_count = len(df[df.Class == 1])

fraud_indices = df[df.Class == 1].index

normal_indices = df[df.Class == 0].index



r_normal_indices = np.random.choice(normal_indices, fraud_count, replace = False) # random 



undersample_indices = np.concatenate([fraud_indices,r_normal_indices])

undersample_data = df.iloc[undersample_indices,:]



X_undersample = undersample_data.iloc[:, undersample_data.columns != 'Class']

y_undersample = undersample_data.iloc[:, undersample_data.columns == 'Class']

from sklearn.model_selection import train_test_split

X_tr, X_test, y_tr, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

X_tr_u, X_test_u, y_tr_u, y_test_u = train_test_split(X_undersample,y_undersample,test_size = 0.3,random_state = 0)

                
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_tr_u,y_tr_u)
predictions = logmodel.predict(X_test_u)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_u,predictions))
print(confusion_matrix(y_test_u,predictions))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_tr_u,y_tr_u)
predictions_dt = dtree.predict(X_test_u)
print(classification_report(y_test_u,predictions_dt))
print(confusion_matrix(y_test_u,predictions_dt))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_tr_u, y_tr_u)
rfc_pred = rfc.predict(X_test_u)
print(confusion_matrix(y_test_u,rfc_pred))
print(classification_report(y_test_u,rfc_pred))
from sklearn.svm import SVC
model_svm = SVC()
model_svm.fit(X_tr_u,y_tr_u)
predictions_svm = model_svm.predict(X_test_u)
print(confusion_matrix(y_test_u,predictions_svm))
print(classification_report(y_test_u,predictions_svm))
print("length of training data",len(df))

print("length of normal data",len(df[df["Class"]==0]))

print("length of fraud  data",len(df[df["Class"]==1]))
# ok Now we have a traing data

X_tr["Class"]= y_tr["Class"] # combining class with original data

data_train = X_tr.copy() # for naming conevntion

print("length of training data",len(data_train))

# Now make data set of normal transction from train data

normal_data = data_train[data_train["Class"]==0]

print("length of normal data",len(normal_data))

fraud_data = data_train[data_train["Class"]==1]

print("length of fraud data",len(fraud_data))
# Now start oversamoling of training data 

# means we will duplicate many times the value of fraud data

for i in range (355): # the number is choosen by myself on basis of nnumber of fraud transaction

    normal_data= normal_data.append(fraud_data)

ovs_data = normal_data.copy() 

print("length of oversampled data is ",len(ovs_data))

print("Number of normal transcation in oversampled data",len(ovs_data[ovs_data["Class"]==0]))

print("No.of fraud transcation",len(ovs_data[ovs_data["Class"]==1]))

print("Proportion of Normal data in oversampled data is ",len(ovs_data[ovs_data["Class"]==0])/len(ovs_data))

print("Proportion of fraud data in oversampled data is ",len(ovs_data[ovs_data["Class"]==1])/len(ovs_data))
ovs_data.head()
X_oversample = ovs_data.iloc[:, ovs_data.columns != 'Class']

y_oversample = ovs_data.iloc[:, ovs_data.columns == 'Class']

X_tr_o, X_test_o, y_tr_o, y_test_o = train_test_split(X_oversample,y_oversample,test_size = 0.3,random_state = 0)
print(len(X_oversample))
print(len(X_tr_o))
logmodel_ovs = LogisticRegression()

logmodel_ovs.fit(X_tr_o,y_tr_o)

predictions_log_ovs = logmodel_ovs.predict(X_test_o)

print(classification_report(y_test_o,predictions_log_ovs))

print(confusion_matrix(y_test_o,predictions_log_ovs))
dtree_ovs = DecisionTreeClassifier()

dtree_ovs.fit(X_tr_o,y_tr_o)

predictions_dt_ovs = dtree_ovs.predict(X_test_o)

print(classification_report(y_test_o,predictions_dt_ovs))

print(confusion_matrix(y_test_o,predictions_dt_ovs))
rfc_ovs = RandomForestClassifier(n_estimators=100)

rfc_ovs.fit(X_tr_o,y_tr_o)

predictions_rfc_ovs = rfc_ovs.predict(X_test_o)

print(classification_report(y_test_o,predictions_rfc_ovs))

print(confusion_matrix(y_test_o,predictions_rfc_ovs))
from imblearn.over_sampling import SMOTE 

oss = SMOTE(random_state=0)
columns = df.columns

#columns1 =y_tr.columns
columns
df.columns
os = SMOTE(random_state=0)
df.columns
data_train_X,data_test_X,data_train_y,data_test_y=train_test_split(X,y,test_size = 0.3, random_state = 0)

columns = data_train_X.columns
# now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y

os_data_X,os_data_y=os.fit_sample(data_train_X,data_train_y)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns )

os_data_y= pd.DataFrame(data=os_data_y,columns=["Class"])

# we can Check the numbers of our data

print("length of oversampled data is ",len(os_data_X))

print("Number of normal transcation in oversampled data",len(os_data_y[os_data_y["Class"]==0]))

print("No.of fraud transcation",len(os_data_y[os_data_y["Class"]==1]))

print("Proportion of Normal data in oversampled data is ",len(os_data_y[os_data_y["Class"]==0])/len(os_data_X))

print("Proportion of fraud data in oversampled data is ",len(os_data_y[os_data_y["Class"]==1])/len(os_data_X))
rfc_smote = RandomForestClassifier(n_estimators=100)

rfc_smote.fit(os_data_X,os_data_y)

predictions_rfc_ovs = rfc_smote.predict(X_test)

print(classification_report(y_test,predictions_rfc_ovs))

print(confusion_matrix(y_test,predictions_rfc_ovs))