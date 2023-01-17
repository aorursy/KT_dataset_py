# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd, xgboost # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#gd stands for General Data
gd = pd.read_csv("/kaggle/input/hr-analytics-case-study/general_data.csv")
sd = pd.read_csv("/kaggle/input/hr-analytics-case-study/employee_survey_data.csv")
md = pd.read_csv("/kaggle/input/hr-analytics-case-study/manager_survey_data.csv")
#fill Null values from survey with Zero
sd.fillna(value=0, inplace=True)
md.fillna(value=0, inplace=True)
#Join survey result with general data
gd=gd.join(sd, on=['EmployeeID'], how='inner', lsuffix='_caller', rsuffix='_other')
gd=gd.join(md, on=['EmployeeID'], how='inner', lsuffix='_caller', rsuffix='_other')
gd.head()
print("Did not leave: ", gd.Attrition.value_counts()['No']/len(gd)*100,"%")
print("Left: ", gd.Attrition.value_counts()['Yes']/len(gd)*100,"%")
print(gd.dtypes)
#Age to Attrition relationship
age_att=gd[gd['Attrition']== 'Yes'].groupby(["Age"])['Attrition'].count()
plt.figure(figsize=(16,4))
plt.title('Attrition to Age')
plt.xlabel('Age')
plt.ylabel('Attrition')
age_att.plot(figsize=(15, 6))
plt.show();
career_att=gd[(gd['Attrition'] == 'Yes') & (gd['YearsAtCompany'] >= 4)].groupby(["YearsSinceLastPromotion"])['Attrition'].count()
plt.figure(figsize=(16,4))
plt.title('Attrition to Career Path')
plt.xlabel('Years Since Last Promotion')
plt.ylabel('Attrition')
career_att.plot(figsize=(15, 6))
plt.show();
sns.countplot(x = "Gender",data=gd)
plt.show()
sns.countplot(x = "Attrition",data=gd,hue="Gender")
plt.show()
xs=gd.groupby(["Gender"])['MonthlyIncome'].mean()
plt.title('Income')
plt.xlabel('Gender')
plt.ylabel('Pay Mean')
plt.ylim([0,80000])
plt.bar(xs.index, xs)
plt.show();
Age18_21=gd[gd['Age']<21]
Age21_25=gd[(gd['Age']>=21) & (gd['Age']<25)]
Age25_35=gd[(gd['Age']>=25) & (gd['Age']<35)]
Age35_45=gd[(gd['Age']>=35) & (gd['Age']<45)]
Age45_55=gd[(gd['Age']>=45) & (gd['Age']<55)]
Age55_65=gd[(gd['Age']>=55)]
Age=['18-21', '21-25', '25-35', '35-45', '45-55', '55-65']
Income=[Age18_21['MonthlyIncome'].mean(), Age21_25['MonthlyIncome'].mean(), Age25_35['MonthlyIncome'].mean(), Age35_45['MonthlyIncome'].mean(), Age45_55['MonthlyIncome'].mean(), Age55_65['MonthlyIncome'].mean()]
d={'Age':Age,'Income':Income}
df=pd.DataFrame(data=d)
sns.barplot(x = "Age",data=df, y="Income")
plt.show()
sns.countplot(x = "Attrition",data=gd,hue="Education")
plt.show()
features = ['EmployeeID', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
X = gd[features].copy()
Obj_Type = (X.dtypes == 'object')
object_cols = list(Obj_Type[Obj_Type].index)
print("Categorical variables:")
print(object_cols)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(gd['Attrition'])
for col in object_cols:
    #make NAN as 0 Catgory Variable
    X[col] = label_encoder.fit_transform(X[col].fillna('0'))    
#Let us have a look at the Data now
X.head()    
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
# Imputation removed column names; put them back
imputed_X.columns = X.columns
imputed_X.head()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(imputed_X, y, train_size=0.8, test_size=0.2,
                                                      random_state=43)
classifier = xgboost.XGBClassifier(learning_rate=0.1, n_estimators=169, max_depth=9,
 min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.8)
classifier.fit(X_train, y_train)
# predict the labels on validation dataset
predictions = classifier.predict(X_valid)
f1_score = metrics.f1_score(y_valid,predictions)
print ("Result of XGB: ", f1_score)
d={'EmployeeId':X_valid['EmployeeID'],'Predicted Attrition':predictions, 'Attrition':y_valid}
df=pd.DataFrame(data=d)
df.to_csv("test_predictions.csv", index=False)
classifier.feature_importances_
featuress = pd.DataFrame({'Feature':features, 'Importance':classifier.feature_importances_})
featuress = featuress.sort_values('Importance', ascending=False).reset_index().drop('index',axis=1)
featuress
plt.figure(figsize=(15, 6))
base_color = sns.color_palette()[0]
sns.barplot(x = "Importance", data=featuress, y="Feature", color = base_color)
plt.show()