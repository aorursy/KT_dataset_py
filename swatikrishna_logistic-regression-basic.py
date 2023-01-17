# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Classification with Logistic Regression



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

d_data=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')



count_no_dia = len(d_data[d_data['Outcome']==0])

count_dia = len(d_data[d_data['Outcome']==1])

pct_of_no_dia = count_no_dia/(count_no_dia+count_dia)

#print("percentage of people with no diabetes is", pct_of_no_dia*100)

pct_of_dia = count_dia/(count_no_dia+count_dia)

#print("percentage of people with diabetes is", pct_of_dia*100)

d_data.describe()

d_data['Outcome'].unique()

d_data['Outcome'].value_counts()

d_data.groupby('Outcome').mean()



#bp vs diabetes

#pd.crosstab(d_data.BloodPressure,d_data.Outcome).plot(kind='bar',figsize=(15,15))

#plt.title('BP vs Diabetes')

#plt.xlabel('BP')

#plt.ylabel('Frequency of Diabetes')

#plt.savefig('bp vs diabetes')



data_copy = d_data.copy(deep = True)

data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace = True)

data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace = True)

data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace = True)

data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace = True)

data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace = True)





sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(data_copy.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])

y=data_copy.Outcome

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

#DATA CLEAN AND SPLIT



#implementing logistic regression



#RFE(Recursive feature elimination)





from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 20)

rfe = rfe.fit(X_train,y_train.values.ravel())

print(rfe.support_)

print(rfe.ranking_)





logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_predictions=logreg.predict(X_test)



from sklearn.metrics import confusion_matrix



confusion_matrix=confusion_matrix(y_test,y_predictions)

#print(confusion_matrix)

#result [[140  27],[ 42  47]]

#140+47 correct predictions(True positves+true negatives)

#27+42 incorrect predictions(False positives+false negatives)



from sklearn.metrics import classification_report

print(classification_report(y_test, y_predictions))
