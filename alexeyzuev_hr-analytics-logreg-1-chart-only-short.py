import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

from sklearn.utils import resample

from sklearn.preprocessing import StandardScaler, QuantileTransformer

from sklearn.model_selection import train_test_split

from sklearn import ensemble, model_selection, metrics 

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.impute import SimpleImputer

import seaborn as sns
data = pd.read_csv("../input/hr-analytics-case-study/general_data.csv",sep=",")

emp_surv = pd.read_csv("../input/hr-analytics-case-study/employee_survey_data.csv",sep=",")

man_surv = pd.read_csv("../input/hr-analytics-case-study/manager_survey_data.csv",sep=",")

in_time = pd.read_csv("../input/hr-analytics-case-study/in_time.csv",sep=",")

out_time = pd.read_csv("../input/hr-analytics-case-study/out_time.csv",sep=",")





y = data.Attrition

y = np.where(y.values == 'Yes', 1, 0)



# Drop constant features and EmpID - it's in index.

data.drop(['Attrition','EmployeeID', 'StandardHours', 'EmployeeCount', 'Over18' ], axis = 1, inplace = True)



# Add surveys results to the main dataframe

surveys = pd.concat([man_surv, emp_surv], axis=1).drop('EmployeeID',axis=1 )

data = pd.concat([data, surveys], axis=1)



# In and Out time data

out_time.drop(['Unnamed: 0'], axis = 1,inplace =True )

in_time.drop(['Unnamed: 0'], axis = 1,inplace =True )

out_time.fillna(0,inplace =True)

in_time.fillna(0,inplace =True)

in_time = in_time.astype('datetime64[ns]')

out_time = out_time.astype('datetime64[ns]') 



# Calculate time at the office

time_diff = out_time - in_time



# Average office hours 

means_all = pd.DataFrame(round((time_diff.mean(axis = 1)/  np.timedelta64(60, 'm')),2))

means_all.columns = ['Average_office_hours']

data = pd.concat([data, means_all], axis=1)



# Split between categorical and numeric columns

columns = data.columns

numeric_columns  = data._get_numeric_data().columns

categorical_columns = list(set(columns) - set(numeric_columns))



# Transform numeric columns using log transformation

data[numeric_columns] = (data[numeric_columns] + 1).transform(np.log)





# Change categorical features to dummies and scale data. 

# I dont use LabelEncoder since it's better to keep different feature values in different cells

# otherwise the model could misunderstand data as 1 < 2 < 3 (Singe vs Married vs Divorced)

X = pd.get_dummies(data = data, columns = categorical_columns)

X_col = X.columns

# Fill Nan

my_imputer = SimpleImputer(strategy = 'median')

X = my_imputer.fit_transform(X)



# Split dataset

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Scale data.

Scaler_S = StandardScaler()



X_train = Scaler_S.fit_transform(X_train)

X_test = Scaler_S.transform(X_test)





clf = LogisticRegression().fit(X_train, y_train)

y_pred = clf.predict(X_test)



print('Accuracy score of Logistic Regression', accuracy_score(y_test,y_pred), '\n')

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
features_importance = pd.concat([pd.Series(X_col), pd.Series(clf.coef_[0])], axis=1)

features_importance.columns = ['Feature', 'Importance']

most_importnant_features = pd.concat([features_importance.nlargest(5, 'Importance'),

                                      features_importance.nsmallest(5, 'Importance').sort_values(by='Importance', ascending=False)])

plt.figure(figsize=(10,6))

sns.set_style("whitegrid")

ax = sns.barplot(y="Feature", x="Importance", data=most_importnant_features, palette= 'muted')

ax.set_title('Top and bottom 5 factors impacting attrition\n', fontsize=15)

ax.set(xlabel='<---Negative correlation------------------------------Positive correlation--->', ylabel='')

plt.tight_layout()
print('Job Satisfaction for all employees = ',round(surveys.JobSatisfaction.mean(),2))

print('Job Satisfaction for Attrition[0] = ',round(surveys.JobSatisfaction[y==0].mean(),2))

print('Job Satisfaction for Attrition[1] = ',round(surveys.JobSatisfaction[y==1].mean(),2))

print()

print('Environment Satisfaction for all employees = ',round(surveys.EnvironmentSatisfaction.mean(),2))

print('Environment Satisfaction for Attrition[0] = ',round(surveys.EnvironmentSatisfaction[y==0].mean(),2))

print('Environment Satisfaction for Attrition[1] = ',round(surveys.EnvironmentSatisfaction[y==1].mean(),2))
