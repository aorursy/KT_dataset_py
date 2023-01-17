import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
#Importing data from kaggle space
data=pd.read_csv("../input/german-credit-details-updated/german_credit_data.csv")
print(data)
data[data['Checking_account'].isna()]
data[data['Saving_accounts'].isna()]
#Checking the null values in different columns
print(data['Checking_account'].isna().sum())
data['Saving_accounts'].isna().sum()
#Plotting default as categorical value
sns.countplot(x='default', data=data)
#Printing the headers of the dataset
print(data.columns)

#Visualizing relation between Sex and number of defaults
pd.crosstab(data.Sex, data.default).plot(kind='bar')
#Plotting impact of job types on default
pd.crosstab(data.Job, data.default).plot(kind='bar')
#Demonstrating relation between housing and the type of house person live in
pd.crosstab(data.Housing, data.default).plot(kind='bar')
#Distribution of customers among various Age groups in the data
data.Age.hist()
plt.title('Number of Customers among various age Groups')
plt.xlabel('Age')
plt.ylabel('Occurence in Dataset')
plt.show()
data.groupby('Sex').mean()
data.groupby('Job').mean()
data.groupby('Housing').mean()
data.head()
#Randomly sampling the data for training_set and test_set 
data_rand=data.sample(frac=1, random_state=1)
training_index=round(len(data_rand)*0.8)
training_set=data_rand[:training_index].reset_index(drop=True)
test_set=data_rand[training_index:].reset_index(drop=True)
one_hot_encoded_training_data=pd.get_dummies(training_set)
one_hot_encoded_test_data=pd.get_dummies(test_set)
final_train, final_test=one_hot_encoded_training_data.align(one_hot_encoded_test_data, join='left', axis=1)
print(final_train.columns)
import statsmodels.api as sm
model_fin=sm.GLM.from_formula("default~ Age+Job+Duration+Sex_female+Credit_amount+Sex_male+Housing_free+Housing_own+Housing_rent+Saving_accounts_little+Saving_accounts_moderate+Saving_accounts_quite_rich+Saving_accounts_rich+Checking_account_little+Checking_account_moderate+Checking_account_rich+Purpose_business+Purpose_car+Purpose_domestic_appliances+Purpose_education+Purpose_furniture+Purpose_radio+Purpose_repairs+Purpose_vacation", family=sm.families.Binomial(), data=one_hot_encoded_training_data)
result=model_fin.fit()
print(result.summary())
from sklearn.linear_model import LogisticRegression 
y_train=one_hot_encoded_training_data['default']

#list of significant fields
sig=['Duration','Sex_female','Sex_male','Housing_own','Saving_accounts_little','Checking_account_little','Checking_account_moderate','Checking_account_rich']
x_train=one_hot_encoded_training_data[sig]
logreg=LogisticRegression()
logreg.fit(x_train, y_train)
#Testing for the test set

y_test=one_hot_encoded_test_data['default']
x_test=one_hot_encoded_test_data[sig]
#Confusion Matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np

#Converting y_test from dataframe to array type for confusion matrix
y_test=np.array(y_test)

con_matrix=metrics.confusion_matrix(y_test, y_pred)
print(con_matrix)

print(classification_report(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
