# import the required librairies
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
rcParams['figure.figsize']= 14,8
RANDOM_SEED = 42
LABELS =['Normal','Fraud']
import warnings
warnings.filterwarnings('ignore')
# Loading the data from file
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.shape
df['Class'].value_counts()
# Cheack if any null values present or not
df.isnull().sum()
sns.countplot(df['Class'])
plt.xlabel('Class')
plt.ylabel("Frequency")
plt.show()
fraud = df[df['Class']==1]
normal = df[df['Class']==0]
fraud.Amount.describe()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();
# Taking sample of data from large dataset
df1 =  df.sample(frac=0.1,random_state=1)
df1.shape
Fraud = df1[df1['Class']==1]
Normal = df1[df1['Class']==0]

outlier_fraction = len(Fraud)/float(len(Normal))
outlier_fraction
Fraud.shape
Normal.shape
cor = df1.corr()
top_cor_features = cor.index
plt.figure(figsize=(20,20))

g = sns.heatmap(df[top_cor_features].corr(),annot=True,cmap='RdYlGn')
X = df.drop('Class',axis=1)
y = df['Class']
X.shape
y.shape
# Creating model with Isolation Forest Algorithm
model = IsolationForest(n_estimators=100,max_samples=len(X),contamination = outlier_fraction,verbose=0)
model.fit(X)
y_pred = model.predict(X)
y_pred[y_pred ==1] = 0
y_pred[y_pred ==-1] = 1
acc = accuracy_score(y,y_pred)
print("Accuray of the model:- {}".format(acc))
