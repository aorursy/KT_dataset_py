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
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv') 
combine = [df] 
mpl.style.use("ggplot")
print("Done")
df.head(10)
df.dtypes
df.describe()
df.corr()
df.describe(include='all')
df.info
df.replace("?",np.nan,inplace= True) 
df.head()
missing_data = df.isnull() 
missing_data.tail()
for col in missing_data.columns.values.tolist(): 
    print(col) 
    print(missing_data[col].value_counts()) 
    print('-'*20)
print("Total Employees in each department :")
df.Department.value_counts().head() 
df['Department'].value_counts().head().plot(kind='bar',figsize=(10,6)) 
plt.ylabel("No. of employees") 
plt.xlabel("Department")
plt.title("Comparision of total no. of employees vs Department")
df_copy = df.copy() 
df_copy.head()

df_t = df_copy[['EmployeeNumber','MonthlyIncome']]
df_t.set_index('EmployeeNumber',inplace=True) 
df_t.head()

count,bin_edges=np.histogram(df_t) 
print(count) 
print(bin_edges)
df_t.plot(kind='hist', figsize=(8, 5),color='chartreuse') 
plt.xlabel("Monthly Salary")

df_b=df[['Education', 'MonthlyIncome']].groupby(['Education'], as_index=False).mean().sort_values(by='MonthlyIncome', ascending=False) 
df_b.set_index('Education',inplace=True) 
df_b.head()
df_b.plot(kind='bar',figsize=(10,6),color='Rybgm') 
plt.ylabel("Average Monthly income") 
plt.title("Comparison of average monthly income by education. \n\n Education: 1.'Below College', 2.'College', 3.'Bachelor', 4.'Master', 5.'Doctor'")
df_s=df[['YearsAtCompany', 'PercentSalaryHike']].groupby(['YearsAtCompany'], as_index=False).mean().sort_values(by='PercentSalaryHike', ascending=False) 
df_s.head()
df_s.plot(kind='scatter', x='YearsAtCompany', y='PercentSalaryHike', figsize=(10, 6), color='c')

plt.title('Comparison of PercentSalaryHike by YearsAtCompany.')
plt.xlabel('YearsAtCompany')
plt.ylabel('PercentSalaryHike')

plt.show()
for dataset in combine:
    dataset['Attrition'] = dataset['Attrition'].map( {'Yes': 1, 'No': 0} ).astype(int)

df.head()
for dataset1 in combine:
    dataset1['OverTime'] = dataset1['OverTime'].map( {'Yes': 1, 'No': 0} ).astype(int)

df.head()
for dataset2 in combine:
    dataset2['Gender'] = dataset2['Gender'].map( {'Female': 1, 'Male': 0} ).astype(int)

df.head()
sns.regplot(x='Age',y='Attrition',data=df) 
plt.ylim(0,)
df[['Age','Attrition']].corr() 
#weak linear relationship
df[['EducationField', 'Attrition']].groupby(['EducationField'], as_index=False).mean().sort_values(by='Attrition', ascending=False)
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['DistanceFromHome'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['Age'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['Education'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['EnvironmentSatisfaction'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['JobInvolvement'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['OverTime'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['TotalWorkingYears'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#YearsSinceLastPromotion 
pearson_coef, p_value = stats.pearsonr(df['YearsSinceLastPromotion'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['YearsWithCurrManager'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['NumCompaniesWorked'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['YearsInCurrentRole'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['MonthlyIncome'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['JobLevel'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
pearson_coef, p_value = stats.pearsonr(df['Gender'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
df= df[['JobInvolvement','Age','MonthlyIncome','YearsInCurrentRole','YearsWithCurrManager','TotalWorkingYears','HourlyRate','OverTime','Attrition']] 
df['Attrition'] = df['Attrition'].astype('int') 
df.head()
x = np.asarray(df[['JobInvolvement','MonthlyIncome','YearsInCurrentRole','YearsWithCurrManager','TotalWorkingYears','HourlyRate','OverTime','Age']]) 

y = np.asarray(df['Attrition'])
from sklearn import preprocessing 
x = preprocessing.StandardScaler().fit(x).transform(x) 
x[:5]
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4) 
print('Train set:  ', x_train.shape, y_train.shape) 
print('Test set: ', x_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)  
LR
yhat = LR.predict(x_test)
yhat
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred
jaccard_similarity_score(y_test, y_pred)