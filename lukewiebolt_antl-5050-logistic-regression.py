# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_sas('/kaggle/input/buy-test/buytest.sas7bdat', format = 'sas7bdat', encoding = 'Latin1')

df
df.shape
df.isnull().sum()/len(df)*100
df.describe()
plt.hist(df.AGE, bins = 10)

plt.title('Histogram of Age')

plt.show()
df['AGE'].value_counts().sort_index().plot.line()

plt.title('Age of Customers')

plt.ylabel('Count')

plt.xlabel('Age')

plt.show()
label = ['Yes', 'No']

df['MARRIED'].value_counts().plot.pie(labels = label,autopct='%1.1f%%')

plt.gca().set_aspect('equal')

plt.title('Married Status')

plt.ylabel('')

plt.show()
df['SEX'].value_counts().plot.bar(color = 'blue')

plt.xlabel('Gender')

plt.ylabel('Count')

plt.title('Gender of Customers')

plt.show()
label = ['Yes', 'No']

df['COA6'].value_counts().plot.pie(labels = label,autopct='%1.1f%%')

plt.gca().set_aspect('equal')

plt.title('Address Change in Past 6 Months')

plt.ylabel('')

plt.show()
label = ['Yes', 'No']

df['DISCBUY'].value_counts().plot.pie(labels = label,autopct='%1.1f%%')

plt.gca().set_aspect('equal')

plt.title('Discount Buyer')

plt.ylabel('')

plt.show()
label = ['Yes', 'No']

df['RETURN24'].value_counts().plot.pie(labels = label,autopct='%1.1f%%')

plt.gca().set_aspect('equal')

plt.title('Returned in Past 24 Months')

plt.ylabel('')

plt.show()
label = ['Yes', 'No']

df['OWNHOME'].value_counts().plot.pie(labels = label,autopct='%1.1f%%')

plt.gca().set_aspect('equal')

plt.title('Homeowners')

plt.ylabel('')

plt.show()
label = ['Yes', 'No']

df['RESPOND'].value_counts().plot.pie(labels = label,autopct='%1.1f%%')

plt.gca().set_aspect('equal')

plt.title('Response to Mailing Test')

plt.ylabel('')

plt.show()
df['LOC'].value_counts().plot.bar(color = 'green')

plt.title('Location of Residence Codes')

plt.show()
df['CLIMATE'].value_counts().plot.bar(color = 'orange')

plt.title('Climate Codes')

plt.show()
df['ORGSRC'].value_counts().plot.bar(color = 'navy')

plt.title('Source within Org')

plt.show()
print('Avg Age',round(df['AGE'].mean(),1))

print('Avg Income',round(df['INCOME'].mean(),1))

print('Avg FICO Score', round(df['FICO'].mean(),1))

print('Avg Value of Purchases over 2 Years', round(df['VALUE24'].mean(),1))
df.head()
a = df['AGE']>=50

print(round(a.sum()/len(a)*100,0), '% of customers are older than 50')
#Replace the Female and Male with with 0 and 1 indicators to feed into the model#

df['SEX'].replace(['F','M'],[1,0],inplace=True)
df.head()
df['LOC'].unique()
loc_dummies = pd.get_dummies(df['LOC'],prefix = 'LOC', prefix_sep = '_', drop_first = True)

df = df.drop('LOC',axis = 1)

df = df.join(loc_dummies)

df 
df['ORGSRC'].unique()
orgsrc_dummies = pd.get_dummies(df['ORGSRC'],prefix = 'orgsrc', prefix_sep = '_', drop_first = True)

df = df.drop('ORGSRC',axis = 1)

df = df.join(orgsrc_dummies)

df 
df['AGE'].fillna(df['AGE'].mean(), inplace = True)

df['INCOME'].fillna(df['INCOME'].mean(), inplace = True)

df['FICO'].fillna(df['FICO'].mean(), inplace = True)

df['OWNHOME'].fillna(df['OWNHOME'].mean(), inplace = True)
df['SEX'].fillna(2, inplace = True)

df['MARRIED'].fillna(2, inplace = True)
df['Above50'] = df['AGE']>=50

df['Above50'].replace([True,False],[1,0],inplace=True)
df.loc[(df.RESPOND == 1) & (df.Above50 == 1), 'Target'] = 1

df['Target'].fillna(0, inplace = True)
print(df['Target'].sum(), 'of the customers in our data set are above 50 and responded to the mailing offer')
df.columns
#Dataframe that only includes the filtered 1 for target

df2_target = df[df.Target == 1]

print(df2_target.shape)
#Dataframe that only includes the 0's in Target variable

df2_target_0 = df[df.Target == 0]

df2_target_0
#Now we need to take a random sample of 150 to 175 from our dataframe with those who are not 50 above and purchased#

df2_0 = df2_target_0.sample(n = 175)

df2_0
#Time to combined these two datasets back together

frame = [df2_target, df2_0]

df3 = pd.concat(frame)

df3
df3.drop(columns = ['ID', 'RESPOND', 'AGE', 'Above50'], inplace = True)

df3
df3.columns
from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

pax_data = df3.loc[:, ('INCOME', 'SEX', 'MARRIED', 'FICO', 'OWNHOME', 'BUY6', 'BUY12',  

                     'BUY18', 'VALUE24', 'DISCBUY','RETURN24', 'COA6', 'PURCHTOT', 'LOC_C', 

                       'LOC_D','orgsrc_D', 'orgsrc_I','orgsrc_O', 'orgsrc_P', 'orgsrc_R', 'orgsrc_U')].values



scaler = StandardScaler()

scaler.fit(pax_data)



y = df3.iloc[:,34].values

X = scaler.transform(pax_data)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
import statsmodels.api as sm

logit_model=sm.Logit(y_train, X_train)

result=logit_model.fit()

print(result.summary())
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression  classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix')

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()