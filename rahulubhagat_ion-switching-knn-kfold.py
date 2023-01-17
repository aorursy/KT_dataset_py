#Performing some basic analysis on data,data cleaning
import pandas as pd
import numpy as np
from sklearn import linear_model
#Reading Tain dataset
#ion=pd.read_csv('../input/Ion-Switching-knn/train.csv')
ion=pd.read_csv('../input/train.csv')
df=pd.DataFrame(ion)
df.info()
df.isnull()
df.isnull().sum()
df.describe()

#Reading Test Dataset
#Testdata=pd.read_csv("../input/Ion-Switching-knn/test.csv")
Testdata=pd.read_csv("../input/test.csv")
df2=pd.DataFrame(Testdata)

#Reading sample_submission dataset
#sample_submission=pd.read_csv('../input/Ion-Switching-knn/sample_submission.csv')
sample_submission=pd.read_csv('../input/sample_submission.csv')
#sample_submission=pd.read_csv('')
df3=pd.DataFrame(sample_submission)




#Analysing relationship between various coloumn using pairplot
df["open_channels"].replace(1,"Opened",inplace=True)
df["open_channels"].replace(0,"Closed",inplace=True)
import seaborn as sns
sns.pairplot(df);
#Plotting scattered plot
import seaborn as sns
import matplotlib.pyplot as plt
fig_dims = (10, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.scatterplot(x='time',y='signal',hue='open_channels',ax=ax,data=df)
#analysing signal if Monotonic or not
sr=df['signal']
print(sr) 
sr.is_monotonic
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
ax=sns.boxplot(x='signal',y='open_channels',data=df)
#ax= sns.swarmplot(x='signal',y='open_channels',data=df, color="grey")
#Plotting Boxplot for signal
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
ax=sns.boxplot(df['signal'],color="b",linewidth=3)
#ax=sns.swarmplot(df['signal'],color="y")
print("Following Data Is for Signal column: \n")
Q1 = df['signal'].quantile([0.25])
Q3=df['signal'].quantile([0.75])
print(" \nQ1",Q1)
print(" \nQ3",Q3)
type(Q3)
IQR=float(Q3)-float(Q1)
print("\nIQR for Signal: ",IQR)
UL=Q3+1.5*IQR
LL=Q1-1.5*IQR

print("\nUpper Limit",UL)
print("Lower Limit",LL)

c_2outliers=df[((df["signal"]<=float(LL)) |(df["signal"]>=float(UL)))]
print()
print("\n\n",c_2outliers.info())
c_2outliers.sort_values("signal",ascending=False)

df_signalclean=df[((df["signal"]>float(LL)) &(df["signal"]<float(UL)))]
df_signalclean.info()
print("Following Data Is for ion signal column After cleaning outliers: \n")
QC1 = df_signalclean["signal"].quantile([0.25])
QC3=df_signalclean["signal"].quantile([0.75])
QC2=df_signalclean["signal"].quantile([0.50])
Q2=df_signalclean["signal"].quantile([0.50])
print(" \nQC1",QC1)
print(" \nQC2",QC2)
print(" \nQC3",QC3)
print(" \nQ1",Q1)
print(" \nQ2",Q2)
print(" \nQ3",Q3)

type(QC3)
IQRC=float(QC3)-float(QC1)
print("\nIQR after cleaning for signal: ",IQRC)
print("\nIQR before cleaning for signal: ",IQR)
ULC=QC3+1.67*IQRC
LLC=QC1-1.67*IQRC

print("\nUpper Limit after cleaning: ",ULC)
print("\nUpper Limit beforer cleaning: ",UL)
print("\nLower Limit after cleaning: ",LLC)
print("\nLower Limit beforer cleaning: ",LL)

c_2outliers_2nd_pass=df_signalclean[((df_signalclean["signal"]<=float(LLC)) |(df_signalclean["signal"]>=float(ULC)))]
print()
print("\n\nOutlayer",c_2outliers_2nd_pass.info())
c_2outliers_2nd_pass.sort_values("signal",ascending=False)
#Boxplot for signal before removing outliers
plt.figure(figsize=(16,4))
sns.boxplot(x="signal",data=df,linewidth=4)
#sns.swarmplot(x="signal",data=df,color="y")
#Boxplot for signal After removing outliers
plt.figure(figsize=(12,4))
sns.boxplot(x="signal",data=df_signalclean,color="g",linewidth=4,whis=1.65)
#sns.swarmplot(x="signal",data=df_signalclean,color="r")
plt.figure(figsize=(16,4))


#print("Original data is shown with yellow colour")
sns.boxplot(x="signal",data=df,color="b",linewidth=4)
#sns.swarmplot(x="signal",data=df,color="y",size=5)

#print("Cleaned Data is shown with red colour")
sns.boxplot(x="signal",data=df_signalclean,color="w",linewidth=4,whis=1.65)
#sns.swarmplot(x="signal",data=df_signalclean,color="r")


#plt.legend()
plt.show()
plt.figure(figsize=(12,4))
sns.distplot(df["signal"])
from scipy import stats
df_Z_Outliers_clean=df[np.abs((stats.zscore(df[["signal"]])))<3]
print(df_Z_Outliers_clean.info())
plt.figure(figsize=(15,4))
sns.boxplot(x="signal",data=df_Z_Outliers_clean,color="g",linewidth=4,whis=1.7)
#sns.swarmplot(x="signal",data=df_Z_Outliers_clean,color="r")
#Truncating the time values to floor
df["open_channels"].replace(1,"Opened",inplace=True)
df["open_channels"].replace(0,"Closed",inplace=True)
#floor gets the rounded down (truncated) values of column in dataframe
df['time'] = df['time'].apply(np.floor)
df['time']

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='open_channels', data=df)
plt.xticks(rotation=90)
df["open_channels"].value_counts()
#df["open_channels"].value_counts("Closed")
#pd.pivot_table(df,index=["time"])
#df.groupby('time')['open_channels'].value_counts()
total=df.groupby("time")["open_channels"].count()
total
#df.groupby(["time", "signal"])["open_channels"].value_counts()

openchannels=df.groupby('time')['open_channels'].value_counts()
openchannels
#Plotting graph of time vs no opened and closed channels
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(100,5))
df.groupby('time')['open_channels'].value_counts().plot(ax=ax,kind='bar')




#Plotting a Signal Distribution
import seaborn as sns
sns.set_style("whitegrid")
ax = sns.distplot(df['signal'])
#Plotting bar Graph for time,signal and openchannels showing closed  and opened channels
import seaborn as sns
fig_dims = (15, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x='time',y='signal',hue='open_channels',ax=ax,data=df)
#sns.barplot(y='signal',hue='open_channels',ax=ax,data=df)
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
from sklearn import datasets
import pandas as pd
ion=pd.read_csv('../input/train.csv')
df=pd.DataFrame(ion)
df["open_channels"].replace("Opened",1,inplace=True)
df["open_channels"].replace("Closed",0,inplace=True)
y=df['open_channels']
X=df[['time','signal']]
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=50)#Taking cluster size of 50 for sampling
classifier.fit(Xtrain,ytrain)

y_pred=classifier.predict(Xtest)
y_pred

#Applying model on test dataset
import pandas as pd
Testdata=pd.read_csv("../input/test.csv")
df2=pd.DataFrame(Testdata)
X2=df2[['time','signal']]
#y_pred2=classifier.predict(X2[2000000,])
y_pred2=classifier.predict(X2)
y_pred2


# Prinnting R square value for Train dataset
print('R square:',classifier.score(X,y))
# Prinnting R square value for Test dataset

print('R square:',classifier.score(X2,y_pred2))
#calculating  values for test dataset
from sklearn import metrics
print('mean absolute error:',metrics.mean_absolute_error(df3["open_channels"],y_pred2))
print('mean squared error:',metrics.mean_squared_error(df3["open_channels"],y_pred2))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(df3["open_channels"],y_pred2)))
#calcuating values for train datset
print("now")
#error estimation
from sklearn import metrics
print('mean absolute error:',metrics.mean_absolute_error(ytest,y_pred))
print('mean squared error:',metrics.mean_squared_error(ytest,y_pred))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(ytest,y_pred)))

#Calculating Test score i.e accuracy for train dataset
from sklearn.metrics import classification_report
print("Test Score:{:.6f}".format(np.mean(y_pred==ytest)))

#Calculating Test score i.e accuracy for test dataset
from sklearn.metrics import classification_report
df3["open_channels"].replace("Opened",1,inplace=True)
df3["open_channels"].replace("Closed",0,inplace=True)
print("Test Score:{:.6f}".format(np.mean(y_pred2==df3['open_channels'])))

#verifying test score by confusion_matrix for tain dataset
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(ytest,y_pred)
print(confusion_matrix)

a=195193+144+202+1416
b=195193+1416
c=b/a
print(c)
#plotting confusion Matrix for train dataset
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
sn.set(font_scale=1.4) # for label size
sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16}) # font size

plt.show()
#calculating error for k  values between 1 and 40 for train dataset
from sklearn.neighbors import KNeighborsClassifier
error=[]
for i in range (1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain,ytrain)
    pred_i=knn.predict(Xtest)
    error.append(np.mean(pred_i != ytest))
print(error)
#plotting error for train dataset
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.plot(range(1,50),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize='10')
plt.title('Error rate k value ')
plt.xlabel('k value')
plt.ylabel('mean error')
#Using K fold sklearn for train dataset
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

ion=pd.read_csv('../input/train.csv')
df=pd.DataFrame(ion)
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
# Evaluate using a train and a test set
y=df['open_channels']
X=df[['time','signal']]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20,shuffle=True,random_state=50000)
model = LogisticRegression()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (result*100.0))

y_pred=model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)
print("for Train datset")
from sklearn import metrics
print("Test Score:{:.6f}".format(np.mean(y_pred==y_test)))

#for test dataset
print("for test dataset")
y_pred2=model.predict(X2)
from sklearn import metrics
print("Test Score:{:.6f}".format(np.mean(y_pred2==df3["open_channels"])))

#K-fold cross validation
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=50000)
model_kfold = LogisticRegression()
results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 

#Performing Ridge  and validation curve 
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
ion=pd.read_csv('../input/train.csv')
df=pd.DataFrame(ion)

y=df['open_channels']
X=df[['time','signal']]
train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
                                              np.logspace(-7, 3, 3),
                                              cv=5)
train_scores




#plotting confusion Matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
sn.set(font_scale=1.4) # for label size
sn.heatmap(train_scores, annot=True, annot_kws={"size": 16}) # font size

plt.show()
valid_scores

#plotting confusion Matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
sn.set(font_scale=1.4) # for label size
sn.heatmap(valid_scores, annot=True, annot_kws={"size": 16}) # font size

plt.show()
#plotting accuracy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from scipy.special import expit

# and plot the result
y=df['open_channels']
X=df['signal']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20,shuffle=True,random_state=50000)
clf = LogisticRegression()
clf.fit(X_train[:, np.newaxis], y_train)
#clf.fit(X_train, y_train)
plt.figure(1, figsize=(6, 4))
plt.clf()
plt.scatter(X, y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)

ols = linear_model.LinearRegression()
ols.fit(X[:, np.newaxis], y)
#ols.fit(X, y)


plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='medium')
plt.tight_layout()
plt.show()
#As the data in open_channels arte not monotonic although we can fit it by using Iostonic regression
#Plotting linear regression as well as isotonic regression as signals are not monotonic

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

X=df['signal']
y=df['open_channels']  

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20)     

# Fit IsotonicRegression and LinearRegression models
ir = IsotonicRegression()  
y_ = ir.fit_transform(X, y)



#for isotonic regression
y_pred=ir.predict(Xtest)
y_pred

import matplotlib.pyplot as plt
plt.plot(y_pred)

from sklearn.metrics import classification_report
print("Test Score:{:.6f}".format(np.mean(y_pred==ytest)))

#for linear regression
import numpy as np
#X=df['time'].reset_index().values.ravel().view(dtype=[('index', int), ('time', float)])
#X=df[['time','signal']]
#converting x to array as x should be two dimensional for linear regression
#X=df[['signal','time']].to_numpy()
X=df['signal']
y=df['open_channels']  
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20)  
Xtest=Xtest.to_numpy()
#Xtest.reshape(-1, 1)
lr = LinearRegression(normalize=True)
lr.fit(X[:, np.newaxis], y)  # x needs to be 2d for LinearRegression 

#lr.fit(X, y)


#print(Xtest)

#Xtest.reshape(-1, 1)
#print(Xtest)
X[:, np.newaxis]
y_pred=lr.predict(Xtest[:, np.newaxis])
#y_pred=lr.predict(Xtest)
y_pred

import matplotlib.pyplot as plt
plt.plot(y_pred)

from sklearn.metrics import classification_report
print("Test Score:{:.6f}".format(np.mean(y_pred==ytest)))
# Plotting linear and Isotonic regression to observe the data

segments = [[[i, y[i]], [i, y_[i]]] for i in range(len(X))]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(np.full(len(X), 0.5))

fig = plt.figure()
plt.plot(X, y, 'r.', markersize=12)
plt.plot(X, y_, 'b.-', markersize=12)
plt.plot(X, lr.predict(X[:, np.newaxis]), 'b-')
#plt.plot(X, lr.predict(X), 'b-')

plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()
import pandas as pd
import numpy as np
from sklearn import linear_model
ion=pd.read_csv('../input/sample_submission.csv')
df3=pd.DataFrame(ion)
df3.info()
#df3.isnull()
#df3.isnull().sum()
#df3.describe()
#Truncating the time values to floor
df3["open_channels"].replace(1,"Opened",inplace=True)
df3["open_channels"].replace(0,"Closed",inplace=True)
#floor gets the rounded down (truncated) values of column in dataframe
df3['time'] = df3['time'].apply(np.floor)
df3['time']

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='open_channels', data=df3)
plt.xticks(rotation=90)
df3["open_channels"].value_counts()
#df3["open_channels"].value_counts("Closed")
#pd.pivot_table(df,index=["time"])
#df.groupby('time')['open_channels'].value_counts()
total=df3.groupby("time")["open_channels"].count()
total
#df.groupby(["time", "signal"])["open_channels"].value_counts()

openchannels=df3.groupby('time')['open_channels'].value_counts()
openchannels
#Plotting graph of time vs no opened and closed channels
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(100,5))
df3.groupby('time')['open_channels'].value_counts().plot(ax=ax,kind='bar')

#Truncating the time values to floor
df2["open_channels"]=y_pred2   #Assigning values of y_pred2 predicted  by model 
df2["open_channels"].replace(1,"Opened",inplace=True)
df2["open_channels"].replace(0,"Closed",inplace=True)
#floor gets the rounded down (truncated) values of column in dataframe
df2['time'] = df2['time'].apply(np.floor)
df2['time']
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='open_channels', data=df2)
plt.xticks(rotation=90)
df2["open_channels"].value_counts()
#df3["open_channels"].value_counts("Closed")
#pd.pivot_table(df,index=["time"])
#df.groupby('time')['open_channels'].value_counts()
total=df2.groupby("time")["open_channels"].count()
total
#df.groupby(["time", "signal"])["open_channels"].value_counts()

#openchannels=df2.groupby('time')['open_channels'].value_counts()
openchannels=df2.groupby('time')['open_channels'].value_counts()
openchannels
#Plotting graph of time vs no opened and closed channels for test Dataset i.e df2
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(100,5))
df2.groupby('time')['open_channels'].value_counts().plot(ax=ax,kind='bar')
import pandas as pd
import numpy as np
from sklearn import linear_model
ion=pd.read_csv('../input/sample_submission.csv')
df3=pd.DataFrame(ion)
df3.info()

#generating a Sample csv file for submission
import pandas as pd
import numpy as np
from sklearn import linear_model
df4=df3.iloc[:,0:1]
df4['open_channels'] = pd.DataFrame(y_pred2)
df4['time'] = df4['time'].map(lambda x: '%2.4f' % x)
df4['time']
#df4.info()
#df4.to_csv(outfile, index=False, header=True, float_format='%11.6f')
df4.to_csv (r'sample_submission1.csv', index = False, header=True,float_format='%11.6f')