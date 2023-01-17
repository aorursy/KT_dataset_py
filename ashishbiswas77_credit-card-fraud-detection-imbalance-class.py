# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
import matplotlib

#Checking  Versions of libraries used

#print("Pandas version: Pandas {}".format(pd.__version__)) # Same way of printing as bellow

print(f"Pandas version: Pandas {pd.__version__}")
print(f"Numpy version: Pandas {np.__version__}")
print(f"Matplotlib version: Pandas {matplotlib.__version__}")
print(f"Seaborn version: Pandas {sns.__version__}")

#Magicfunctions for In-Notebook Display
%matplotlib inline

# Setting seabon style
sns.set(style='darkgrid', palette='deep')
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
# Displaying all the columns
pd.options.display.max_rows=100
pd.options.display.max_columns=100
# Displaying all the rows
pd.set_option('display.max_rows',1000)
df.head(2)
df.tail(2)
# Number of rows and
m,n=df.shape
print(f'Total rows :  {m}')
print(f'Total Columns :  {n}')
# Name of the Columns
df.columns
# Levels of Classification
df.Class.value_counts()
df.info()
df.isnull().sum()
#df.isnull().sum().max()
#df.isnull().any()
#df.isnull().any().sum()
df[['Time','Amount','Class']].describe().T
print(f'Count of Fraud and Non Fraud Transaction: \n{df.Class.value_counts()}')
print('\n')
print(f'Percentage of Fraud and Non Fraud Transaction: \n{df.Class.value_counts(normalize=True)}')
plt.figure(figsize=(4,4), dpi=100)
df["Class"].value_counts().plot(kind = 'pie', autopct='%1.1f%%', fontsize = 20)
plt.title("Fraudulent and Non-Fraudulent Distribution",color='b', fontsize = 15,fontweight='bold')
plt.legend(["Non-Fraud", "Fraud"])
plt.show()
plt.figure(figsize=(6,3),dpi=100)
sns.distplot(df['Amount'], bins=1)
plt.title('Distribution of Transaction Amount', fontweight='bold')
plt.show()
plt.figure(figsize=(6,3), dpi=100)
sns.boxplot(y='Amount', data=df)
plt.title('Boxplot of Transaction Amount', fontweight='bold')
plt.show()
plt.figure(figsize=(6,3),dpi=100)
sns.boxplot(x='Class',y='Amount', data=df)
plt.title('Transaction Amount for Non-Fraud and Fraud Transactions', fontweight='bold')
plt.show()
plt.figure(figsize=(6,3),dpi=100)
sns.distplot(df[df['Class']==1].Amount, bins=1)
plt.title("Distribution of Fraudulent Transaction Amount", fontweight='bold')
plt.show()
plt.figure(figsize=(6,3),dpi=100)
sns.distplot(df[df['Class']==0].Amount,bins=1)
plt.title("Distribution of Non-Fraudulent Transaction Amount", fontweight='bold')

plt.show()
df[(df['Class']==1)&(df['Amount']<=10)].Amount.value_counts().head(10)
l1=len(df[(df['Class']==1)&(df['Amount']<=10)].Amount)
print(f'Total count of Fraudulent Transaction where Transaction Amount up to 10: {l1}')
l2=len(df[(df['Class']==1)].Amount)
print(f'Total count of Fraudulent Transaction: {l2}')
p=(l1/l2)*100
print(f'Percentage of Fraudulent Transaction where Transaction Amount up to 10: {p}')
# Maximum transaction value for Fraudelent Transaction
print(f"Maximum transaction value for Fraudulent Transaction: {df[df['Class']==1].Amount.max()}")
fig , axs = plt.subplots(nrows = 2 , ncols = 3 , figsize = (15,10))
fig.suptitle('Fraudulent and Non Fraudulent Distribution for Different Transacton Amount',fontsize = 20,color='b', fontweight='bold')


df[(df['Amount']<=1)].Class.value_counts().plot(kind = 'pie', autopct='%1.2f%%', fontsize = 20, ax=axs[0,0])
axs[0,0].set_title("Transaction Amount up to 1",color='b', fontsize = 15, fontweight='bold')

df[(df['Amount']>1)&(df['Amount']<500)].Class.value_counts().plot(kind = 'pie', autopct='%1.2f%%', fontsize = 20, ax=axs[0,1])
axs[0,1].set_title("Transaction Amount >1 and <500",color='b', fontsize = 15, fontweight='bold')

df[(df['Amount']>=500)&(df['Amount']<1000)].Class.value_counts().plot(kind = 'pie', autopct='%1.2f%%', fontsize = 20, ax=axs[0,2])
axs[0,2].set_title("Transaction Amount >500 and <1000",color='b', fontsize = 15, fontweight='bold')

df[(df['Amount']>=1000)&(df['Amount']<1500)].Class.value_counts().plot(kind = 'pie', autopct='%1.2f%%', fontsize = 20, ax=axs[1,0])
axs[1,0].set_title("Transaction Amount >1000 and <1500",color='b', fontsize = 15, fontweight='bold')

df[(df['Amount']>=1500)&(df['Amount']<2000)].Class.value_counts().plot(kind = 'pie', autopct='%1.2f%%', fontsize = 20, ax=axs[1,1])
axs[1,1].set_title("Transaction Amount >1500 and <2000",color='b', fontsize = 15, fontweight='bold')

df[(df['Amount']>=2000)].Class.value_counts().plot(kind = 'pie', autopct='%1.2f%%', fontsize = 20, ax=axs[1,2])
axs[1,2].set_title("Transaction Amount >2000",color='b', fontsize = 15, fontweight='bold')

plt.show()
# quartile values
q1,q3=np.percentile(df['Amount'],[25,75])

# Inter quartile range
iqr=q3-q1

#upper and lower bound
lower=q1-(iqr*1.5)
upper=q3+(iqr*1.5)

print(f"Q1: {q1}")
print(f"Q3: {q3}")
print(f"IQR: {iqr}")
print(f"Lower_Bound: {upper}")
print(f"Upper_Bound: {lower}")

# percentage of data removed
l=len(df)-len(df[df['Amount']<upper])
r=(len(df)-(len(df[df['Amount']<upper])))/len(df)*100
print(f"Percentage of data removed: {r}")
print(f"Number of rows deleted: {l}")
# Fraud and non Fraud distribution in the data
df['Class'].value_counts(normalize=True)
# Fraud and non Fraud distribution after outliar remuval
df[df['Amount']<upper].Class.value_counts(normalize=True)
# Fraud and non Fraud distribution in the removed data
df[df['Amount']>=upper].Class.value_counts(normalize=True)
x1=len(df[(df['Amount']>=upper)&(df['Class']==1)])
print(f'Number of Fraudulent Transaction Removed : {x1}')
x2=len(df[(df['Class']==1)].Amount)
print(f'Total count of Fraudulent Transaction: {x2}')
p1=(x1/x2)*100
print(f'Percentage Fraudulent Transaction Removed: {p1}')
# Maximum Amount Value of Class == 1 or Fraud Claumn
df[df['Class']==1 ]['Amount'].max()
# Function - Maximum Transaction Amount as Threshold for outliar removal
def outliar_removal(max_val):
    print('***  Removing Outlier using Maximum Transaction Amount as Threshold  ***')
    print("Number of Outliar Value Removed: {}".format(len(df[(df['Class']==0) & (df['Amount']>max_val)])))
    print("Proportion of data lost: {}".format((len(df)-len(df[(df['Class']==0) & (df['Amount']<max_val)]))/len(df)*100))
          
          
#Remove outliars
    print('\n')
    print('Distribution of Fraudulent and Non-Fraudulent Class:')
    temp_df=df[df['Amount']<max_val]
    print(temp_df.Class.value_counts(normalize=True))
# Applying Function to remove outliar 
outliar_removal(df[(df['Class']==1) & (df['Amount'])].Amount.max())
# Removing Outliar
d=df[df['Amount']<2125.87]
#Reseting index after removing outliers
d.reset_index(drop = True , inplace = True)
plt.figure(figsize=(6,3),dpi=100)
sns.boxplot(x='Class',y='Amount', data=d)
plt.title('Transaction Amount for Non-Fraud and Fraud Transactions', fontweight='bold')
plt.show()
# Maximum duration of data recording
d.Time.max()/(60*60)
plt.figure(figsize=(10,6))
plt.title("Distribution of Transactions in two days duration", fontsize=15,fontweight='bold')
sns.distplot(d.Time, bins=100, color='k')
plt.show()
# converting Time into hours
d['Time_Hours']=d['Time']/(60*60)
d.columns
d.head()

fig , axs = plt.subplots(nrows = 1 , ncols = 2 , figsize = (15,5))
fig.suptitle('Fraudulent and Non Fraudulent Transaction Distribution',fontsize = 20,color='b', fontweight='bold')


sns.distplot(d[d['Class']==0]['Time_Hours'].values , color = 'g' , ax = axs[0])
axs[0].set_title("Non-Fraudulent Transaction",color='g', fontsize = 15, fontweight='bold')

sns.distplot(d[d['Class']==1]['Time_Hours'].values , color = 'r' , ax = axs[1])
axs[1].set_title("Fraudulent Transaction",color='r', fontsize = 15, fontweight='bold')

plt.show()
plt.figure(figsize=(10,5))
plt.title('Fraudulent and Non Fraudulent Transaction Distribution in Two Days',fontsize = 20, color='k', fontweight='bold')
sns.distplot(d[d.Class==0].Time_Hours.values,color='g')
sns.distplot(d[d.Class==1].Time_Hours.values, color='r')
plt.show()
# Ploting data in 0-24 hrs time frame
plt.figure(figsize=(10,5))
plt.title('Fraudulent and Non Fraudulent Transaction Distribution in first Day',fontsize = 20, color='k', fontweight='bold')
sns.distplot(d[d.Class==0].Time_Hours.values,color='g', bins=100)
sns.distplot(d[d.Class==1].Time_Hours.values, color='r',bins=100)
plt.xlim([0,24])
plt.show()
# Ploting data in 25-48 hrs time frame
plt.figure(figsize=(10,5))
plt.title('Fraudulent and Non Fraudeulent Transaction Distribution in second Day',fontsize = 20, color='k', fontweight='bold')
sns.distplot(d[d.Class==0].Time_Hours.values,color='g', bins=100)
sns.distplot(d[d.Class==1].Time_Hours.values, color='r',bins=100)
plt.xlim([25,48])
plt.show()
plt.figure(figsize=(10,5))
sns.distplot(d[d.Class==0].Amount,bins=100,color='g')
sns.distplot(d[d.Class==1].Amount,bins=100,color='r')
plt.show()
 # Importing library
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
d['Std_Amount']=ss.fit_transform(d['Amount'].values.reshape(-1,1))
d.columns
plt.figure(figsize=(10,5))
sns.distplot(d[d.Class==0].Std_Amount,bins=100,color='g')
sns.distplot(d[d.Class==1].Std_Amount,bins=100,color='r')
plt.show()
from sklearn.preprocessing import MinMaxScaler

n = MinMaxScaler()
d['Mm_Amount'] = n.fit_transform(d['Amount'].values.reshape(-1,1))
d.columns
plt.figure(figsize=(10,5))
sns.distplot(d[d.Class==0].Mm_Amount,bins=100,color='g')
sns.distplot(d[d.Class==1].Mm_Amount,bins=100,color='r')
plt.show()
#Log Transformation
d['Log_Amount'] = np.log(d.Amount + 0.01)
d.columns
plt.figure(figsize=(10,5))
sns.distplot(d[d.Class==0].Log_Amount,bins=100,color='g')
sns.distplot(d[d.Class==1].Log_Amount,bins=100,color='r')
plt.show()
plt.figure(figsize=(15,12))

# Let's explore the Amount by Class and see the distribuition of Amount transactions
plt.subplot(221)
ax = sns.boxplot(x ="Class",y="Amount",data=d)
ax.set_title("Distribution Before Log Transform", fontsize=16, fontweight='bold')
ax.set_xlabel("Non-Fraud vs Fraud", fontsize=12, fontweight='bold')
ax.set_ylabel("Amount", fontsize = 12, fontweight='bold')

plt.subplot(222)
ax1 = sns.boxplot(x ="Class",y="Std_Amount", data=d)
ax1.set_title("Distribution After Standard Scaler", fontsize=16, fontweight='bold')
ax1.set_xlabel("Non-Fraud vs Fraud", fontsize=12, fontweight='bold')
ax1.set_ylabel("Amount(Standard Scaler)", fontsize = 12, fontweight='bold')

plt.subplot(223)
ax2 = sns.boxplot(x ="Class",y="Mm_Amount", data=d)
ax2.set_title("Distribution After Min-Max Scaler", fontsize=16, fontweight='bold')
ax2.set_xlabel("Non-Fraud vs Fraud", fontsize=12, fontweight='bold')
ax2.set_ylabel("Amount(Min-Max)", fontsize = 12, fontweight='bold')

plt.subplot(224)
ax3 = sns.boxplot(x ="Class",y="Log_Amount", data=d)
ax3.set_title("Distribution After Log Transform", fontsize=16, fontweight='bold')
ax3.set_xlabel("Non-Fraud vs Fraud", fontsize=12, fontweight='bold')
ax3.set_ylabel("Amount(Log)", fontsize = 12, fontweight='bold')

plt.show()
# Plotting a Correlation plot between cloumns
plt.figure(figsize = (10,10))
plt.title("Correlation Heat Map", fontsize=20, color='r', fontweight='bold')
corr_matrix = d.corr()
sns.heatmap(corr_matrix)
plt.show()
d.hist(figsize = (20,20))
plt.show()
#data=pd.read_csv('processed.csv')
data=d # Creating a coppy of the data set
y=data.Class
data.columns
X=data.drop(columns=['Time','Class','Time_Hours','Std_Amount','Mm_Amount','Amount'])
X.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
pd.Series(y_train).value_counts()
plt.figure(figsize=(4,4), dpi=100)
pd.Series(y_train).value_counts().plot(kind = 'pie', autopct='%1.1f%%', fontsize = 20)
plt.title("Fraudulent and Non-Fraudulent Distribution",color='b', fontsize = 15, fontweight='bold')
plt.legend(["Non-Fraud", "Fraud"])
plt.show()
# Importing Logistic Regression Library
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

# Creating method for logistic regression
logreg = LogisticRegression() 
logreg.fit(X_train, y_train)  
y_pred=logreg.predict(X_test)
y_pred[:5]
# Importing library for confusion matrix
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)
print('Distribution of Test Data:')
pd.Series(y_test).value_counts()
print('Confusion Matrix:')
cnf_matrix
plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='BuPu', fmt='d')
plt.title("Confusion_Matrix", fontsize=16, color='b',fontweight="bold")
plt.ylabel('Predicted', color='b',fontweight="bold")
plt.xlabel('Actual', color='b',fontweight="bold")
plt.show()
# Fraudlent Class prediction (Recall Score of class-1)
print(f'Recall Score of Class 1: {cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]):.4f}')
# Accuracy Score
print(f'Accuracy Score: {accuracy_score(y_pred, y_test):.4f}')
# Precision Score
print(f'Precision Score: {precision_score(y_test , y_pred):.4f}')
# Recall Score
print(f'Recall Score: {recall_score(y_test, y_pred):.4f}')
# F1 Score, harmonic mean of precision (PRE) and recall (REC)
print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)
y_pred_proba[:5]
from sklearn.metrics import roc_auc_score
print(f'ROC_AUC Score: {roc_auc_score(y_test , y_pred):.4f}')
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
# Importing Libraries
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter # counter takes values returns value_counts dictionary
from sklearn.datasets import make_classification

#X, y = make_classification(n_classes=2) #, class_sep=2,weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s'  %Counter(y))

rus = RandomUnderSampler(random_state=42)

X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
pd.Series(y_res).value_counts()
plt.figure(figsize=(4,4), dpi=100)
pd.Series(y_res).value_counts().plot(kind = 'pie', autopct='%1.1f%%', fontsize = 20)
plt.title("Fraudulent and Non-Fraudulent Distribution",color='b', fontsize = 15, fontweight='bold')
plt.legend(["Non-Fraud", "Fraud"])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=0)
# Importing Logistic Regression Library
from sklearn.linear_model import LogisticRegression

# Creating method for logistic regression
logreg = LogisticRegression() 
logreg.fit(X_train, y_train)  
y_pred=logreg.predict(X_test)
y_pred[:5]
# Importing library for confusion matrix
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)
print('Distribution of Test Data:')
pd.Series(y_test).value_counts()
print('Confusion Matrix:')
cnf_matrix
plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='BuPu', fmt='d')
plt.title("Confusion_Matrix", fontsize=16, color='b',fontweight="bold")
plt.ylabel('Predicted', color='b',fontweight="bold")
plt.xlabel('Actual', color='b',fontweight="bold")
plt.show()
# Fraudlent Class prediction (Recall Score of class-1)
print(f'Recall Score of Class 1: {cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]):.4f}')
# Accuracy Score
print(f'Accuracy Score: {accuracy_score(y_pred, y_test):.4f}')
# Precision Score
print(f'Precision Score: {precision_score(y_test , y_pred):.4f}')
# Recall Score
print(f'Recall Score: {recall_score(y_test, y_pred):.4f}')
# F1 Score, harmonic mean of precision (PRE) and recall (REC)
print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)
y_pred_proba[:5]
from sklearn.metrics import roc_auc_score
print(f'ROC_AUC Score: {roc_auc_score(y_test , y_pred):.4f}')
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
# Importing Library
from imblearn.over_sampling import RandomOverSampler
print('Original dataset shape %s' % Counter(y))
random_state = 42

rus = RandomOverSampler(random_state=random_state)

X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

pd.Series(y_res).value_counts()
plt.figure(figsize=(4,4), dpi=100)
pd.Series(y_res).value_counts().plot(kind = 'pie', autopct='%1.1f%%', fontsize = 20)
plt.title("Fraudlent and Non-Fraudlent Distribution",color='b', fontsize = 15, fontweight='bold')
plt.legend(["Non-Fraud", "Fraud"])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=0)
# Importing Logistic Regression Library
from sklearn.linear_model import LogisticRegression

# Creating method for logistic regression
logreg = LogisticRegression() 
logreg.fit(X_train, y_train)  
y_pred=logreg.predict(X_test)
y_pred[:5]
# Importing library for confusion matrix
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)
print('Distribution of Test Data:')
pd.Series(y_test).value_counts()
print('Confusion Matrix:')
cnf_matrix
plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='BuPu', fmt='d')
plt.title("Confusion_Matrix", fontsize=16, color='b',fontweight="bold")
plt.ylabel('Predicted', color='b',fontweight="bold")
plt.xlabel('Actual', color='b',fontweight="bold")
plt.show()
# Fraudlent Class prediction (Recall Score of class-1)
print(f'Recall Score of Class 1: {cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]):.4f}')
# Accuracy Score
print(f'Accuracy Score: {accuracy_score(y_pred, y_test):.4f}')
# Precision Score
print(f'Precision Score: {precision_score(y_test , y_pred):.4f}')
# Recall Score
print(f'Recall Score: {recall_score(y_test, y_pred):.4f}')
# F1 Score, harmonic mean of precision (PRE) and recall (REC)
print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)
y_pred_proba[:5]
from sklearn.metrics import roc_auc_score
print(f'ROC_AUC Score: {roc_auc_score(y_test , y_pred):.4f}')
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
# Importing Library
from imblearn.over_sampling import SMOTE, ADASYN
print('Original dataset shape %s' % Counter(y))

rus = SMOTE(random_state=42)

X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
pd.Series(y_res).value_counts()
plt.figure(figsize=(4,4), dpi=100)
pd.Series(y_res).value_counts().plot(kind = 'pie', autopct='%1.1f%%', fontsize = 20)
plt.title("Fraudlent and Non-Fraudlent Distribution",color='b', fontsize = 15)
plt.legend(["Non-Fraud", "Fraud"])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=0)
# Importing Logistic Regression Library
from sklearn.linear_model import LogisticRegression

# Creating method for logistic regression
logreg = LogisticRegression() 
logreg.fit(X_train, y_train)  
y_pred=logreg.predict(X_test)
y_pred[:5]
# Importing library for confusion matrix
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)
print('Distribution of Test Data:')
pd.Series(y_test).value_counts()
print('Confusion Matrix:')
cnf_matrix
plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='BuPu', fmt='d')
plt.title("Confusion_Matrix", fontsize=16, color='b',fontweight="bold")
plt.ylabel('Predicted', color='b',fontweight="bold")
plt.xlabel('Actual', color='b',fontweight="bold")
plt.show()
# Fraudlent Class prediction (Recall Score of class-1)
print(f'Recall Score of Class 1: {cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]):.4f}')
# Accuracy Score
print(f'Accuracy Score: {accuracy_score(y_pred, y_test):.4f}')
# Precision Score
print(f'Precision Score: {precision_score(y_test , y_pred):.4f}')
# Recall Score
print(f'Recall Score: {recall_score(y_test, y_pred):.4f}')
# F1 Score, harmonic mean of precision (PRE) and recall (REC)
print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)
y_pred_proba[:5]
from sklearn.metrics import roc_auc_score
print(f'ROC_AUC Score: {roc_auc_score(y_test , y_pred):.4f}')
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
print('Original dataset shape %s' % Counter(y))

rus = ADASYN(random_state=42)

X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
pd.Series(y_res).value_counts()
plt.figure(figsize=(4,4), dpi=100)
pd.Series(y_res).value_counts().plot(kind = 'pie', autopct='%1.1f%%', fontsize = 20)
plt.title("Fraudulent and Non-Fraudulent Distribution",color='b', fontsize = 15, fontweight='bold')
plt.legend(["Non-Fraud", "Fraud"])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=0)
# Importing Logistic Regression Library
from sklearn.linear_model import LogisticRegression

# Creating method for logistic regression
logreg = LogisticRegression() 
logreg.fit(X_train, y_train)  
y_pred=logreg.predict(X_test)
y_pred[:5]
# Importing library for confusion matrix
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)
print('Distribution of Test Data:')
pd.Series(y_test).value_counts()
print('Confusion Matrix:')
cnf_matrix
plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='BuPu', fmt='d')
plt.title("Confusion_Matrix", fontsize=16, color='b',fontweight="bold")
plt.ylabel('Predicted', color='b',fontweight="bold")
plt.xlabel('Actual', color='b',fontweight="bold")
plt.show()
# Fraudulent Class prediction (Recall Score of class-1)
print(f'Recall Score of Class 1: {cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]):.4f}')
# Accuracy Score
print(f'Accuracy Score: {accuracy_score(y_pred, y_test):.4f}')
# Precision Score
print(f'Precision Score: {precision_score(y_test , y_pred):.4f}')
# Recall Score
print(f'Recall Score: {recall_score(y_test, y_pred):.4f}')
# F1 Score, harmonic mean of precision (PRE) and recall (REC)
print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)
y_pred_proba[:5]
from sklearn.metrics import roc_auc_score
print(f'ROC_AUC Score: {roc_auc_score(y_test , y_pred):.4f}')
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
from sklearn.decomposition import PCA # SVD , t-SNE , Linear Discrimant Analysis
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X_res)
plt.figure(figsize=(10,6), dpi=100)

plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_res== 0), cmap='coolwarm', label='No Fraud', linewidths=2)
plt.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_res == 1), cmap='coolwarm', label='Fraud', linewidths=2)
plt.show()
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_under))

rus = RandomOverSampler(random_state=42)
X_over, y_over = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_over))
rus = SMOTE(random_state=42)
X_smote, y_smote = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_smote))
# Importing Libraries
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Creating object and running algorithm
dte = DecisionTreeClassifier()
dte.fit( X_train, y_train )

# Predicting Test Data
y_pred = dte.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.3, random_state=0)

# Creating object and running algorithm
dte = DecisionTreeClassifier()
dte.fit( X_train, y_train )

# Predicting Test Data
y_pred = dte.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=0)

# Creating object and running algorithm
dte = DecisionTreeClassifier()
dte.fit( X_train, y_train )

# Predicting Test Data
y_pred = dte.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=0)

# Creating object and running algorithm
dte = DecisionTreeClassifier()
dte.fit( X_train, y_train )

# Predicting Test Data
y_pred = dte.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Creating object and running algorithm
rfc = RandomForestClassifier()
rfc.fit( X_train, y_train )


# Predicting Test Data
y_pred = rfc.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.3, random_state=0)

# Creating object and running algorithm
rfc = RandomForestClassifier()
rfc.fit( X_train, y_train )


# Predicting Test Data
y_pred = rfc.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=0)

# Creating object and running algorithm
rfc = RandomForestClassifier()
rfc.fit( X_train, y_train )


# Predicting Test Data
y_pred = rfc.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=0)

# Creating object and running algorithm
rfc = RandomForestClassifier()
rfc.fit( X_train, y_train )


# Predicting Test Data
y_pred = rfc.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Creating object and running algorithm
kn=KNeighborsClassifier()
kn.fit( X_train, y_train )


# Predicting Test Data
y_pred = kn.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.3, random_state=0)

# Creating object and running algorithm
kn=KNeighborsClassifier()
kn.fit( X_train, y_train )


# Predicting Test Data
y_pred = kn.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()

# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=0)

# Creating object and running algorithm
kn=KNeighborsClassifier()
kn.fit( X_train, y_train )


# Predicting Test Data
y_pred = kn.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()

# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=0)

# Creating object and running algorithm
kn=KNeighborsClassifier()
kn.fit( X_train, y_train )


# Predicting Test Data
y_pred = kn.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()

from sklearn.model_selection import GridSearchCV
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=0)
# DecisionTree Classifier
tree_params = {"criterion" :['gini',"entropy"],
               "splitter" : ['best','random'],
               "max_features" : ["auto","sqrt", "log2"]}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# tree best estimator
tree_clf = grid_tree.best_estimator_
tree_clf
# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3, random_state=0)

# Creating object and running algorithm
dte = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
dte.fit( X_train, y_train )

# Predicting Test Data
y_pred = dte.predict(X_test)

# Prediction Accuracy
print('\n')
print('*** Looking at Performance Measures ***')
print(f'Accuracy Score: {accuracy_score(y_pred , y_test)}')


# Confusion Matrix
cnf_matrix=confusion_matrix(y_test, y_pred)

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc}')

print('\n')
print(classification_report(y_test, y_pred))
print('\n')
plt.figure(figsize=(15,6))

#Ploting Confusion Matrix
plt.subplot(121)
ax = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
ax.set_xlabel("Actual", fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted", fontsize = 12, fontweight='bold')

# Ploting ROC AUC Curve
plt.subplot(122)
ax1 = plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC_AUC Curve',fontsize=16, fontweight='bold')
plt.show()