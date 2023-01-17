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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/ibm-attrition/IBM-Attrition.csv')
data.head()
data.isna().sum()
data.info()
data.select_dtypes(include= np.object).head()
data.select_dtypes(include= np.object).nunique()
data[data.select_dtypes(include=np.object).columns[0]].head()
data.select_dtypes(include= np.int64).head()
# data distribution
data.select_dtypes(include= np.int64).hist(figsize = (12,12))
plt.tight_layout()
plt.show()
data.head()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'DailyRate', data = data, hue = 'Attrition')
plt.title('Age vs Dailtrate based on attrition')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'StandardHours', data = data, hue = 'Attrition')
plt.title('Age vs StandardHours based on attrition')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'TotalWorkingYears', data = data, hue = 'Attrition')
plt.title('Age vs TotalWorkingYears based on attrition')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'YearsAtCompany', data = data, hue = 'Attrition')
plt.title('Age vs YearsAtCompany based on attrition')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'MonthlyRate', data = data, hue = 'Attrition')
plt.title('Age vs MonthlyRate based on attrition')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'MonthlyIncome', data = data, hue = 'Attrition')
plt.title('Age vs MonthlyIncome based on attrition')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'PercentSalaryHike', data = data, hue = 'Attrition')
plt.title('Age vs PercentSalaryHike based on attrition')
plt.show()
plt.figure(figsize= (15,5))
sns.scatterplot(x = 'Age', y = 'YearsSinceLastPromotion', data = data, hue = 'Attrition')
plt.title('Age vs YearsSinceLastPromotion based on attrition')
plt.show()
data.select_dtypes(include= np.object).columns
data.groupby('BusinessTravel')['Attrition'].value_counts()
# for plotting this is the best way
pd.crosstab(data['BusinessTravel'], data['Attrition']).plot(kind = 'bar', figsize = (15,5))
plt.title('BusinessTravel wise Attrition rate')
plt.show()
pd.crosstab(data['Department'], data['Attrition']).plot(kind = 'bar', figsize = (15,5))
plt.title('Department wise Attrition rate')
plt.show()
pd.crosstab(data['EducationField'], data['Attrition']).plot(kind = 'bar', figsize = (15,5))
plt.title('EducationField wise Attrition rate')
plt.show()
pd.crosstab(data['Gender'], data['Attrition']).plot(kind = 'bar', figsize = (15,5))
plt.title('Gender wise Attrition rate')
plt.show()
pd.crosstab(data['JobRole'], data['Attrition']).plot(kind = 'bar', figsize = (15,5))
plt.title('JobRole wise Attrition rate')
plt.show()
pd.crosstab(data['MaritalStatus'], data['Attrition']).plot(kind = 'bar', figsize = (15,5))
plt.title('MaritalStatus wise Attrition rate')
plt.show()
pd.crosstab(data['OverTime'], data['Attrition']).plot(kind = 'bar', figsize = (15,5))
plt.title('Overtime wise Attrition rate')
plt.show()
