# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ins_data = pd.read_csv("../input/auto-insurance-claims-data/insurance_claims.csv")

ins_data.head()
ins_data.drop('_c39', inplace=True, axis = 1)
ins_data.describe()
print ("Rows     : " ,ins_data.shape[0])

print ("Columns  : " ,ins_data.shape[1])

print ("\nFeatures : \n" ,ins_data.columns.tolist())

print ("\nMissing values :  ", ins_data.isnull().sum().values.sum())

print ("\nUnique values :  \n",ins_data.nunique())
ins_data.duplicated()
ins_data.dtypes
for col in [ 'policy_state', 'policy_csl', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship', 'incident_date',

            'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city', 'incident_location',

             'property_damage',  'police_report_available','vehicle_claim', 'auto_make', 'auto_model', 'fraud_reported']:

    ins_data[col] = ins_data[col].astype('category')
ins_data['policy_bind_date'] = pd.to_datetime(ins_data['policy_bind_date'], format='%Y-%m-%d')
ins_data.dtypes
ins_data.head(2)
ins_data['policy_number'].value_counts().sort_values()

#policy numbers seems unique and hence dropping this feature
ins_data.drop('policy_number', axis = 1, inplace =True)
ins_data.replace('?','No info')
ins_data.groupby(['insured_sex'])['fraud_reported'].count().sort_values().plot(kind='bar')
import matplotlib.pyplot as plt

plt.hist(ins_data.total_claim_amount)

plt.title("Histogram of total amount claimed by the customers")

plt.xlabel("Amount claimed")

plt.ylabel("Number of customers")
import matplotlib.pyplot as plt

plt.hist(ins_data.age)

plt.title("Histogram of age of the customers")

plt.xlabel("Age of the customers")

plt.ylabel("Number of customers")
ins_data.groupby(['insured_sex','age']).fraud_reported.count().plot(kind='bar',figsize=(30,5))
import seaborn as sns

sns.countplot(ins_data['fraud_reported']);
ins_data.groupby(['fraud_reported','age']).age.count()
plt.bar(ins_data['age'], ins_data['fraud_reported']== 'Y')

plt.title("Age of the customers")

plt.xlabel("Age of the customers")

plt.ylabel("Fraud_reported")
plt.bar(ins_data['age'], ins_data['fraud_reported']== 'N')

plt.title("Age of the customers")

plt.xlabel("Age of the customers")

plt.ylabel("Fraud_reported")
abc = ins_data.groupby([ 'auto_model'])['fraud_reported'].count().plot(kind='bar',figsize=(30, 5), rot=90)
import seaborn as sns

sns.set(style="darkgrid")



ax = sns.countplot(x="age", data=ins_data)
ins_data.groupby(['policy_state']).fraud_reported.value_counts().plot(kind='bar',figsize=(20, 5), rot=90)

plt.xlabel("State")

plt.ylabel("Fraud Reported")

plt.title("Distribution of fraud cases w.r.t states")

plt.show()
#which areas suspecting what type of outages

pd.crosstab(ins_data['auto_model'],ins_data['vehicle_claim'])
#count of frauds reported w.r.t auto model

pd.crosstab(ins_data['auto_model'],ins_data['fraud_reported'])
ins_data.shape
X = ins_data.iloc[:,:-1]

Y = ins_data.iloc[:, -1]





categorical_data = X.select_dtypes(exclude="number")

categorical_cols = categorical_data.columns



X = X.drop(categorical_cols, axis=1)

X.shape
from sklearn.model_selection import train_test_split

one_hot_data = pd.get_dummies(categorical_data)

X = X.join(one_hot_data)



Xcols = X.columns

ycols = Y



x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, stratify=Y)
x_train.shape, y_train.shape
x_test.shape,y_test.shape,
x_train.columns
categorical_cols
x_train = x_train.drop(['policy_bind_date','auto_year'], axis = 1)

x_test = x_test.drop(['policy_bind_date','auto_year'], axis = 1)
num_cols = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',

       'umbrella_limit', 'insured_zip',  'incident_hour_of_the_day','capital-loss','capital-gains'

       'number_of_vehicles_involved',  'bodily_injuries',

       'witnesses',  'total_claim_amount','injury_claim', 'property_claim']
from sklearn.preprocessing import StandardScaler

 

print('scaling numerical columns')

scaler = StandardScaler(num_cols)

x_train= scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)

print("Done with standard scaling, Proceed.....All the best :)")
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import  cross_val_score 

Logistic_model = LogisticRegression(random_state=200)

Logistic_model.fit( x_train,y_train)



y_pred1_train = Logistic_model.predict(x_train)

y_pred1_test = Logistic_model.predict(x_test)



from sklearn.metrics import accuracy_score

print("Accuracy on Train using Logistic Model is:", accuracy_score(y_train, y_pred1_train))

print("Accuracy on Test using Logistic Model is:", accuracy_score(y_test, y_pred1_test))



scores = cross_val_score(Logistic_model, x_train,y_train, cv=5, scoring='accuracy')

print("Max cross validation score on logistic model",max(scores))

print("Mean cross validation score on logistic model",np.mean(scores))

print("Min cross validation score on logistic model",min(scores))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score



DT_clf = DecisionTreeClassifier(criterion='entropy', max_depth=800,max_features=15,min_samples_split=2,min_samples_leaf=1,class_weight='balanced')



# Train Decision Tree Classifer

DT_clf = DT_clf.fit(x_train, y_train)



#Predict the response for train and validation dataset

y_pred2_train = DT_clf .predict(x_train)

y_pred2_test = DT_clf.predict(x_test)





conf_matrix = confusion_matrix(y_pred2_test, y_test)

plot_confusion_matrix(conf_matrix)



precision = precision_score(y_pred2_test, y_test, pos_label='Y')

recall = recall_score(y_pred2_test, y_test, pos_label='Y')





print("on complete data using DT:\n")

print("Accuracy on Train using Decision Tree Model is:", accuracy_score(y_train, y_pred2_train))

print("Accuracy on Test using Decision Tree Model is:", accuracy_score(y_test, y_pred2_test))

scores = cross_val_score(DT_clf,x_train, y_train, cv=5, scoring='accuracy')

print("Max cross validation score on DT model",max(scores))

print("Mean cross validation score on DTmodel",np.mean(scores))

print("Min cross validation score on DT model",min(scores))
from imblearn import under_sampling, over_sampling

from imblearn.over_sampling import SMOTE

smote=SMOTE("minority")

x_smote,y_smote = smote.fit_sample(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score



DT_clf_smote = DecisionTreeClassifier(criterion='entropy', max_depth=800,max_features=15,min_samples_split=2,min_samples_leaf=1,class_weight='balanced')



# Train Decision Tree Classifer

DT_clf_smote = DT_clf_smote.fit(x_smote, y_smote)



#Predict the response for train and validation dataset

y_pred2_train_smote = DT_clf_smote .predict(x_smote)

y_pred2_test_smote = DT_clf_smote.predict(x_test)





conf_matrix_smote = confusion_matrix(y_pred2_test_smote, y_test)

plot_confusion_matrix(conf_matrix_smote)



precision = precision_score(y_pred2_test_smote, y_test, pos_label='Y')

recall = recall_score(y_pred2_test_smote, y_test, pos_label='Y')





print("on complete data using DT:\n")

print("Accuracy on Train using Decision Tree Model is:", accuracy_score(y_smote, y_pred2_train_smote))

print("Accuracy on Test using Decision Tree Model is:", accuracy_score(y_test, y_pred2_test_smote))

scores = cross_val_score(DT_clf,x_smote, y_smote, cv=5, scoring='accuracy')

print("Max cross validation score on DT model",max(scores))

print("Mean cross validation score on DTmodel",np.mean(scores))

print("Min cross validation score on DT model",min(scores))
from sklearn.ensemble import RandomForestClassifier

RFC_model = RandomForestClassifier(random_state = 1, max_features='sqrt',oob_score=True, n_estimators = 1200)

RFC_model.fit(x_smote, y_smote)

y_pred3_trainSMOTE = RFC_model.predict(x_smote)



y_pred3_testSMOTE = RFC_model.predict(x_test)

print("ON SMOTE dataset:\n")



conf_matrix_smote = confusion_matrix(y_pred3_testSMOTE, y_test)

plot_confusion_matrix(conf_matrix_smote)



precision = precision_score(y_pred3_testSMOTE, y_test, pos_label='Y')

recall = recall_score(y_pred3_testSMOTE, y_test, pos_label='Y')





print("on complete data using random Forest:\n")

print("Accuracy on Train using random Forest Model is:", accuracy_score(y_smote, y_pred3_trainSMOTE))

print("Accuracy on Test using random Forest Model is:", accuracy_score(y_test, y_pred3_testSMOTE))

scores = cross_val_score(DT_clf,x_smote, y_smote, cv=5, scoring='accuracy')

print("Max cross validation score on random Forest model",max(scores))

print("Mean cross validation score on random Forestmodel",np.mean(scores))

print("Min cross validation score on random Forest model",min(scores))

print("oob score", RFC_model.oob_score_)