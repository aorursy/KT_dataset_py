# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd, numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os

print(os.listdir("../input"))

traindf = pd.read_csv('../input/predict-the-churn-for-customer-dataset/Train File.csv',index_col='customerID')

traindf.head(10)

#Numerical fields- Senior Citizen, tenure,monthly charges, total charges
traindf.info()
#Finding Churn Percentage in dataset train

percent=traindf['Churn'].value_counts(sort = True)

color=['green','red']

labels=['No','Yes']

plt.pie(percent,colors=color,labels=labels,autopct='%1.1f%%')

plt.show()
traindf.describe()
traindf.isnull().sum(axis=0)

#Plotting bar for categorical variables

traindf.gender.value_counts(normalize=True).plot(kind='bar')

traindf.SeniorCitizen.value_counts(normalize=True).plot(kind='bar')
traindf.Partner.value_counts(normalize=True).plot(kind='bar')
traindf.Dependents.value_counts(normalize=True).plot(kind='bar')

traindf.tenure.value_counts(normalize=True).plot(kind='bar')
traindf.PhoneService.value_counts(normalize=True).plot(kind='bar')
traindf.MultipleLines.value_counts(normalize=True).plot(kind='bar')
traindf.InternetService.value_counts(normalize=True).plot(kind='bar')

traindf.Contract.value_counts(normalize=True).plot(kind='bar')

traindf.PaymentMethod.value_counts(normalize=True).plot(kind='bar')
pairs = sns.pairplot(traindf, hue='Churn')

pairs.fig.set_size_inches(15,15)

plt.show()
#Totalcharges seems to be not numeric. so making Totalcharges to numeric. Using errors=’coerce’. It will replace all non-numeric values with NaN.

traindf['TotalCharges'] = pd.to_numeric(traindf['TotalCharges'], errors = 'coerce')

traindf.loc[traindf['TotalCharges'].isna()==True]

#All values of TotalCharges are blank for Tenure 0. So instead of NAN, replace it with 0.

traindf[traindf['TotalCharges'].isna()==True] = 0
#Categorical Variables: gender,Partner,Dependents,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,

#DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,Churn

traindf= pd.get_dummies(traindf,columns=['gender','Partner','Dependents','PhoneService', 'MultipleLines','StreamingTV',

       'StreamingMovies','Contract','PaperlessBilling','InternetService','PaymentMethod','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','Churn'],drop_first=True)

traindf.info()
#Find correlation to figure out which customer feature need to include in churn model

corrs=traindf.corr()
corrs
fig,ax=plt.subplots(figsize=(20,20))

sns.heatmap(corrs,annot=True,cmap="Reds",annot_kws={"size":8},ax=ax)  
#As TotalCharges highly correlated with Tenure and Monthly Charges; drop TotalCharges variable.

traindf.pop('TotalCharges')
from sklearn.model_selection import train_test_split

train, test = train_test_split(traindf, test_size = 0.25)



train_y = train['Churn_No']

test_y = test['Churn_No']



train_x = train

train_x.pop('Churn_No')

test_x = test

test_x.pop('Churn_No')

test_x
train.shape, test.shape
#Precision tells us how many churned users did our classifier predicted correctly. On the other side, recall tell us how many churned users it mi

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report



logreg = LogisticRegression()

logreg.fit(X=train_x, y=train_y)



logreg.coef_
train_y_pred = logreg.predict(train_x)

test_y_pred = logreg.predict(test_x)
from sklearn.metrics import accuracy_score

accuracy_score(train_y, train_y_pred)
accuracy_score(test_y, test_y_pred)
preds=logreg.predict(test_x[train_x.keys()])

preds=1-preds

df_pred=pd.DataFrame()

df_pred




# To get the weights of all the variables

weights = pd.Series(logreg.coef_[0], index=train.columns.values)

weights.sort_values(ascending = False)
train.info()
test.info()
from sklearn.metrics import confusion_matrix
confusion_matrix(train_y, train_y_pred)
accuracy_score(test_y, test_y_pred)
confusion_matrix=confusion_matrix(test_y, test_y_pred)
#Precision tells us how many churned users did our classifier predicted correctly. On the other side, recall tell us how many churned users it mi

print('Intercept: ' + str(logreg.intercept_))

print('Regression: ' + str(logreg.coef_))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(test_x, test_y)))

print(classification_report(test_y, test_y_pred))



confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))

heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)

plt.ylabel('True label', fontsize = 14)

plt.xlabel('Predicted label', fontsize = 14)
traindf['Churn_Yes'].value_counts()
traindf['Churn_No'].value_counts()
df_test = pd.read_csv('../input/predict-the-churn-for-customer-dataset/Test File.csv')
df_concat
data

data.index.names = ['customerID']
df_test.info()
data=data.rename(columns={'Churn_No': 'Churn'})

data.info()


Churn=pd.DataFrame(test_y_pred)

submission=pd.concat([traindf['customerID'],Churn],axis=1)

submission=submission.set_index("customerID", inplace = False)

submission=submission.rename(columns={0:'Churn'})

submission['Churn'] = np.where(submission['Churn']==1, 'Yes', 'No')

submission.to_csv('jis.csv')