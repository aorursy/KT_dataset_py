# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import plot



tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    

             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    

             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    

             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    

             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  



for i in range(len(tableau20)):    

    r, g, b = tableau20[i]    

    tableau20[i] = (r / 255., g / 255., b / 255.) 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/h-1b-visa/h1b_kaggle.csv')
#  Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    #start_mem = df.memory_usage().sum() / 1024**2

    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    #end_mem = df.memory_usage().sum() / 1024**2

    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
df = reduce_mem_usage(df)

df.head()
df.describe()
df.info()
print("The shape of the dataset is : {}".format(df.shape))
print("There were around {} applications for H-1B Visa from 2011 to 2016.".format(df.shape[0]))
df.CASE_STATUS.value_counts()
plt.figure(figsize=(10,7))

df.CASE_STATUS.value_counts().plot(kind='barh',  color=tableau20)

df.sort_values('CASE_STATUS')

plt.title("NUMBER OF APPLICATIONS")

plt.show()
df.YEAR.value_counts().plot(kind = 'bar',color=tableau20)
df.columns
plt.figure(figsize=(10,7))



ax1 = df['EMPLOYER_NAME'][df['YEAR'] == 2011].groupby(df['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title = "Top 10 Applicants in 2016",

                                                                                                                           color=tableau20)

ax1.set_label("")

plt.show()
plt.figure(figsize=(10,7))



ax2 = df['EMPLOYER_NAME'][df['YEAR'] == 2016].groupby(df['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title='Top 10 Applicants in 2016'

                                                                                                                             ,color=tableau20)

ax2.set_ylabel("")

plt.show()
plt.figure(figsize=(10,7))



ax3 = df['EMPLOYER_NAME'].groupby([df['EMPLOYER_NAME']]).count().sort_values(ascending=False).head(10).plot(kind = 'barh', title = 'Top 10 Applicants from 2011 to 2016'

                                                                                                           ,color=tableau20)

ax3.set_ylabel("")

plt.show()
top_emp = list(df['EMPLOYER_NAME'][df['YEAR'] >= 2015].groupby(df['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).index)



byempyear = df[['EMPLOYER_NAME', 'YEAR', 'PREVAILING_WAGE']][df['EMPLOYER_NAME'].isin(top_emp)]



byempyear = byempyear.groupby([df['EMPLOYER_NAME'], df['YEAR']])
plt.figure(figsize=(12,7))



markers=['o','v','^','<','>','d','s','p','*','h','x','D','o','v','^','<','>','d','s','p','*','h','x','D']



for company in top_emp:

    tmp = byempyear.count().loc[company]

    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,marker=markers[top_emp.index(company)])

plt.xlabel("Year")

plt.ylabel("Number of Applications")

plt.legend()

plt.title('Number of Applications of Top 10 Applicants')

plt.show()
plt.figure(figsize=(12,7))



for company in top_emp:

    tmp = byempyear.mean().loc[company]

    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,marker=markers[top_emp.index(company)])

plt.xlabel("Year")

plt.ylabel("Average Salary offered (USD)")

plt.legend()

plt.title('Average Salary of Top 10 Applicants')

plt.show()
df.head()
plt.figure(figsize=(10,12))

df.JOB_TITLE.value_counts().nlargest(20).plot(kind = 'barh', title = "Top 20 Job Titles",color=tableau20)

plt.show()
plt.figure(figsize=(12,7))

sns.set(style="whitegrid")

g = sns.countplot(x = 'FULL_TIME_POSITION', data = df)

plt.title("NUMBER OF APPLICATIONS MADE FOR THE FULL TIME POSITION")

plt.ylabel("NUMBER OF PETITIONS MADE")

plt.show()
df.head()
df = df[df['PREVAILING_WAGE'] <= 500000]

by_emp_year = df[['EMPLOYER_NAME', 'YEAR', 'PREVAILING_WAGE']][df['EMPLOYER_NAME'].isin(top_emp)]

by_emp_year = by_emp_year.groupby([df['EMPLOYER_NAME'],df['YEAR']])
df.PREVAILING_WAGE.max()
## Checking for null values

df.isnull().sum()
df['SOC_NAME'] = df['SOC_NAME'].fillna(df['SOC_NAME'].mode()[0])
df.CASE_STATUS.value_counts()
df['CASE_STATUS'] = df['CASE_STATUS'].map({'CERTIFIED' : 0, 'CERTIFIED-WITHDRAWN' : 1, 'DENIED' : 2, 'WITHDRAWN' : 3, 

                                           'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED' : 4, 'REJECTED' : 5, 'INVALIDATED' : 6})
df.head()
df.FULL_TIME_POSITION.value_counts()
df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].map({'N' : 0, 'Y' : 1})

df.head()
df['SOC_NAME'].value_counts()
import sys

df['SOC_NAME1'] = 'others'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('computer','software')] = 'it'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('chief','management')] = 'manager'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('mechanical')] = 'mechanical'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('database')] = 'database'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('sales','market')] = 'scm'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('financial')] = 'finance'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('public','fundraising')] = 'pr'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('education','law')] = 'administrative'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('auditors','compliance')] = 'audit'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('distribution','logistics')] = 'scm'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('recruiters','human')] = 'hr'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('agricultural','farm')] = 'agri'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('construction','architectural')] = 'estate'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('forencsic','health')] = 'medical'

df['SOC_NAME1'][df['SOC_NAME'].str.contains('teachers')] = 'education'
df.head()
df.columns
df = df.drop(['Unnamed: 0', 'EMPLOYER_NAME', 'SOC_NAME','JOB_TITLE','WORKSITE', 'lon','lat'], axis = 1)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(df.SOC_NAME1)

# print list(le.classes_)

df['SOC_N']=le.transform(df['SOC_NAME1'])
df.head()
df = df.drop(['SOC_NAME1'], axis=1)
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
df.columns
x = df.drop(['CASE_STATUS'], axis=1) # Independent variables

y = df['CASE_STATUS'] # Dependent variables
x.columns
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
from sklearn.preprocessing import OneHotEncoder

x_train_encode = pd.get_dummies(x_train)

x_test_encode = pd.get_dummies(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

LogReg = LogisticRegression()

LogReg.fit(x_train_encode, y_train)

y_pred = LogReg.predict(x_test_encode)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))