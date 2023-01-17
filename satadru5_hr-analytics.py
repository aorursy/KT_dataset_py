# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read data

data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()
data.info()
data1=data.copy()
data.head(2)
corr=data1.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')

corr
corr_left=pd.DataFrame(corr['left'].drop('left'))

corr_left
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

##r col in data.columns.values:

data1['sales']=le.fit_transform(data1['sales'])

data1['salary']=le.fit_transform(data1['salary'])



data1.head(2)     
g = sns.FacetGrid(data, col = 'left')

g.map(sns.boxplot, 'time_spend_company')
sns.barplot(x = 'time_spend_company', y = 'left', data = data)
sns.factorplot(x='time_spend_company' ,y='left',data=data)
sns.factorplot(x = 'number_project', y = 'left', data = data)
sns.barplot(x='number_project',y='left',data=data)
sns.distplot(data['average_montly_hours'])
sns.factorplot(x='average_montly_hours',y='left',data=data)
sns.barplot(x='time_spend_company', y = 'left', hue = 'salary', data = data)
g = sns.FacetGrid(data, hue="left",aspect=4)

g.map(sns.kdeplot,'satisfaction_level',shade= True)

g.set(xlim=(0, data['satisfaction_level'].max()))

g.add_legend()
k=sns.factorplot(x='satisfaction_level',y='left',data=data)

k.map(sns.kdeplot,'satisfaction_level',shade= True)
data['satisfaction_range'] = pd.cut(data['satisfaction_level'], 3)

data[['satisfaction_range', 'left']].groupby(['satisfaction_range']).mean()
sns.barplot('salary', 'left', data = data)

sns.plt.title('Left over Salary (bar plot)')

sns.factorplot('salary','left', data = data, size = 5)

sns.plt.title('Left over Salary (factor plot)')
#promotion_last_5years

sns.barplot('promotion_last_5years', 'left', data = data)

sns.plt.title('Left over promotion_last_5years (barplot)')

sns.factorplot('promotion_last_5years','left',order=[0, 1], data=data,size=5)

sns.plt.title('Left over promotion_last_5years (factorplot)')

#it seems people who are promoted in last 5 years are less likely to leave than those who are not.

#Therefore we can confidently say, if someone get promoted, he is much less likely to leave.
#we can combine promotion_last_5years to salary to see what happens

promoted = data[data['promotion_last_5years'] == 1]

not_promoted = data[data['promotion_last_5years'] == 0]



#separate employee into promoted and not_promoted groups



fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



sns.barplot('salary', 'left', data = promoted, ax=axis1)



sns.barplot('salary', 'left', data = not_promoted, ax=axis2)



axis1.set_title('Promoted')



axis2.set_title('Not Promoted')
#Sales

sns.barplot('sales','left',order=[0, 1, 2, 3, 4, 5, 6], data=data1)

sns.plt.title('Left over Sales')
#Let's look at our dataset again

data1.head()
from sklearn.preprocessing import scale



data2=scale(data1)

data2=pd.DataFrame(data1)

data2.head(3)
#Train-Test split

from sklearn.model_selection import train_test_split

label = data2.pop('left')

data_train, data_test, label_train, label_test = train_test_split(data2, label, test_size = 0.2, random_state = 42)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(data_train, label_train)

logis_score_train = logis.score(data_train, label_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(data_test, label_test)

print("Testing score: ",logis_score_test)
coeff_df = pd.DataFrame(data2.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Correlation"] = pd.Series(logis.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
#decision tree

from sklearn.ensemble import RandomForestClassifier

dt = RandomForestClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
#kNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(data_train, label_train)

knn_score_train = knn.score(data_train, label_train)

print("Training score: ",knn_score_train)

knn_score_test = knn.score(data_test, label_test)

print("Testing score: ",knn_score_test)
#SVM

from sklearn.svm import SVC

svm = SVC()

svm.fit(data_train, label_train)

svm_score_train = svm.score(data_train, label_train)

print("Training score: ",svm_score_train)

svm_score_test = svm.score(data_test, label_test)

print("Testing score: ",svm_score_test)
#random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(data_train, label_train)

rfc_score_train = rfc.score(data_train, label_train)

print("Training score: ",rfc_score_train)

rfc_score_test = rfc.score(data_test, label_test)

print("Testing score: ",rfc_score_test)
#Model comparison

models = pd.DataFrame({

        'Model'          : ['Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],

        'Training_Score' : [logis_score_train, svm_score_train, knn_score_train, dt_score_train, rfc_score_train],

        'Testing_Score'  : [logis_score_test, svm_score_test, knn_score_test, dt_score_test, rfc_score_test]

    })

models.sort_values(by='Testing_Score', ascending=False)