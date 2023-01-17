import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv(r"../input/creditcardfraud/creditcard.csv")
# Grab a peek at the data 
df.head()
#describe information about dataset
df.info()
df.describe()
# Determine number of missing values in dataset
df.isnull().sum()

# Determine number of fraud cases in dataset
df['Class'].value_counts()
sns.countplot(df["Class"],data=df)
plt.title("Class Distrubution",fontsize=14)
# Seperate total data into non-fraud and fraud cases
fraud = df[df.Class == 0] #save non-fraud df observations into a separate df
normal = df[df.Class == 1] #do the same for frauds
print("Amount details of the fraudulent transaction") 
fraud.Amount.describe()
print("details of valid transaction") 

normal.Amount.describe() 
# plot the histogram of each parameter
df.hist(figsize = (20, 20))
plt.show()
sns.scatterplot(x="Amount",y="Time",data=df,hue="Class")
#Correlation matrix
corrmat=df.corr()
f,ax=plt.subplots(figsize=(50,30))
sns.heatmap(corrmat,vmax=.8,square=True,cbar=True,annot=True)
#Dividing the data into inputs parameters and outputs value format
X=df.drop("Class",axis=1)
y=df['Class']
# Using Skicit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,f1_score
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
len(X_train)
len(y_test)
#Let us run Logistic regression and evaluate the performance metrics
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
pred=log.predict(X_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(f1_score(y_test,pred))
#Let us run RandomForestClassifier and evaluate the performance metrics
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier()
random.fit(X_train,y_train)
pred1=random.predict(X_test)
print(accuracy_score(y_test,pred1))
print(confusion_matrix(y_test,pred1))
print(f1_score(y_test,pred1))
fraud.shape
normal.shape
from imblearn.under_sampling import NearMiss
nm=NearMiss()
x_us,y_us=nm.fit_sample(X,y)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_us,y_us)
pred5=log.predict(X_test)
print(confusion_matrix(y_test,pred5))
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier()
random.fit(x_us,y_us)
pred6=random.predict(X_test)
print(confusion_matrix(y_test,pred6))
from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler()
x_os,y_os=os.fit_sample(X,y)
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier()
random.fit(x_os,y_os)
pred8=random.predict(X_test)
print(confusion_matrix(y_test,pred8))
print(f1_score(y_test,pred8)*100)
