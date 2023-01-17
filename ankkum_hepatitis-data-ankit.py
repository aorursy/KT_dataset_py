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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/hepatitis-csv/hepatitis.csv')
data.head()
#We See there is '?' in dataset. lets replace with NaN which we can transform later.

data = data.replace('?', np.NaN)
data.head(10)
print(data.dtypes)
#Calculating the Nan values in each columns
data.isnull().sum()
#Calculating Average Steriod level, malaise, fatigue and replacing the missing value with mean
avg_steriod = data['STEROID'].astype(float).mean()
data['STEROID'].replace(np.NaN,avg_steriod, inplace = True)

avg_fatigue = data['FATIGUE'].astype(float).mean()
data['FATIGUE'].replace(np.NaN,avg_fatigue,inplace = True)

avg_malaise = data['MALAISE'].astype(float).mean()
data['MALAISE'].replace(np.NaN,avg_malaise,inplace = True)

avg_anorexia = data['ANOREXIA'].astype(float).mean()
data['ANOREXIA'].replace(np.NaN,avg_anorexia,inplace = True)

avg_liver_big = data['LIVER BIG'].astype(float).mean()
data['LIVER BIG'].replace(np.NaN,avg_liver_big,inplace = True)

avg_liver_firm = data['LIVER FIRM'].astype(float).mean()
data['LIVER FIRM'].replace(np.NaN,avg_liver_firm,inplace = True)

avg_spleen = data['SPLEEN PALPABLE'].astype(float).mean()
data['SPLEEN PALPABLE'].replace(np.NaN,avg_spleen,inplace = True)

avg_spiders = data['SPIDERS'].astype(float).mean()
data['SPIDERS'].replace(np.NaN,avg_spiders,inplace = True)

avg_ascites = data['ASCITES'].astype(float).mean()
data['ASCITES'].replace(np.NaN,avg_ascites,inplace = True)

avg_varices = data['VARICES'].astype(float).mean()
data['VARICES'].replace(np.NaN,avg_varices,inplace = True)

avg_bilirubin = data['BILIRUBIN'].astype(float).mean()
data['BILIRUBIN'].replace(np.NaN,avg_bilirubin,inplace = True)

avg_sgot = data['SGOT'].astype(float).mean()
data['SGOT'].replace(np.NaN,avg_sgot,inplace = True)

avg_alphos = data['ALK PHOSPHATE'].astype(float).mean()
data['ALK PHOSPHATE'].replace(np.NaN,avg_alphos,inplace = True)

avg_albumin = data['ALBUMIN'].astype(float).mean()
data['ALBUMIN'].replace(np.NaN,avg_albumin,inplace = True)

avg_protime = data['PROTIME'].astype(float).mean()
data['PROTIME'].replace(np.NaN,avg_protime,inplace = True)
#tO CONFIRM THERE IS NO MISSING VAUE
data.isnull().sum()
import seaborn as sns
%matplotlib inline
data.corr()
sns.regplot(x = 'Class', y = 'SEX', data = data)
data['Class'].corr(data['SEX'])

# correlation between the class and sex
fig = plt.figure(figsize= (10,10))
fig,axs = sns.boxplot(x = 'BILIRUBIN',y = 'AGE', hue = 'SEX',width = 2,data = data)
# Categorical plots
data.describe()
data.describe(include = ['object'])
# Grouping

grp_one = data[['AGE','BILIRUBIN','SGOT','ALBUMIN','PROTIME']]
grp_one
# Avg Outcomes

grp_one = grp_one.groupby(['BILIRUBIN'], as_index = False).mean()
grp_one
grp_combine = data[['BILIRUBIN','SGOT','AGE']]
grp_results = grp_combine.groupby(['BILIRUBIN','SGOT'], as_index = False).mean()
grp_results
data.corr()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(data['Class'], data['AGE'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
# Since the p value is 0.006, there is a moderate and significant correlation 
pearson_coef,p_value = stats.pearsonr(data['Class'],data['SEX'])
p_value
# p_value - 0.031 , Class and SEX has moderate and significant correlation
# coefficient heat map
sns.heatmap(data.corr(), annot = True)
sns.set(rc={'figure.figsize':(15, 15)})