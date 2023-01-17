# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))







# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



from sklearn import metrics



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

#reading CSV

empdf=pd.read_csv("../input/HR-Employee-Attrition.csv")

empdf1=empdf.copy()

empdf1.head()
empdf1.shape

retained = empdf1[empdf1.Attrition == 'Yes']

retained.shape
retained = empdf1[empdf1.Attrition == 'No']

retained.shape
empdf1.groupby('Attrition').mean()
sns.boxplot(x="Attrition", y= "MonthlyIncome", data=empdf1)

plt.show()
sns.boxplot(x='Attrition',y='DailyRate',data=empdf1)

plt.show()
sns.boxplot(x='Attrition',y='DistanceFromHome',data=empdf1)

plt.show()
empdf1.Attrition.value_counts()
empdf1.info()
empdf1.isna().sum()
empdf1.duplicated().sum()
#Extracting the numerical columns

num_col=empdf1.select_dtypes(include=np.number).columns

num_col
#Extracting the categorical columns

cat_col=empdf1.select_dtypes(exclude=np.number).columns



cat_col
for i in cat_col:

    print(empdf1[i].value_counts())
#one hot encoding for category column

encoded_cat_col=pd.get_dummies(empdf1[cat_col])

encoded_cat_col.head()
#Concat Category column & numerical column

empdf1_ready_model=pd.concat([empdf1[num_col],encoded_cat_col],axis=1)

empdf1_ready_model.head()
#performing Label encoding so that dataframe gets updated

le=LabelEncoder()

for i in cat_col:

    empdf1[i]=le.fit_transform(empdf1[i])

empdf1.head()
#verification of data count after encoding

left=empdf1[empdf1.Attrition== 1]

left.shape
#verification of data count after encoding

retained = empdf1[empdf1.Attrition== 0]

retained.shape
empdf1.describe().T
plt.figure(figsize=(50,20))

sns.heatmap(empdf1.corr(),vmin=-1,vmax=1,center=0,annot=True)

#ax=sns.heatmap(empdf1.corr(),annot=True)

#plt.show(ax)
#finding the correlation

empdf1.corr()
#split the data in to Train & Test

X=empdf1.drop(columns='Attrition')

y=empdf1['Attrition']

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=1)



print("Train X :", train_X.shape)

print("Test X :",test_X.shape)

print("Train y :",train_y.shape)

print("Test y :",test_y.shape)





#Building the Logistic Model

model=LogisticRegression()



model.fit(train_X,train_y)
#Predicting the Y value from the train set and test set

train_y_predict=model.predict(train_X)

train_y_predict[0:5]



test_y_predict=model.predict(test_X)

metrics.accuracy_score(train_y_predict,train_y)
test_y_prediction=model.predict(test_X)
df_confusion=metrics.confusion_matrix(test_y,test_y_prediction)

df_confusion
#plotting the confusion matrix

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df_confusion,cmap=cmap)

    plt.title('Confusion Matrix')

    plt.colorbar()

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

plot_confusion_matrix(df_confusion)


print(classification_report(test_y, test_y_prediction))
Log_ROC_AUC=roc_auc_score(test_y,model.predict(test_X))

fpr,tpr,thresholds=roc_curve(test_y,model.predict_proba(test_X)[:,1])

plt.figure()

plt.plot(fpr,tpr,label='Logistic Regression  (area=0.2%f)' % Log_ROC_AUC)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()