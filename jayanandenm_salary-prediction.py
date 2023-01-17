import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data = pd.read_csv('../input/glassdoor_jobs.csv')

print(data.shape)

data.head()
# Dropping Columns 

data = data.drop('Unnamed: 0', axis=1)
data = data[data['Salary Estimate'] != '-1']
data['hourly_pay'] = data['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0 )

data['employer_prov'] = data['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0 )
salary = data['Salary Estimate'].apply(lambda x: x.split('(')[0])

k_dollar_rep = salary.apply(lambda x: x.replace('K', '').replace('$', ''))

salary_range = k_dollar_rep.apply(lambda x: x.replace('Per Hour', '').replace('Employer Provided Salary:',''))
data['min_sal'] = salary_range.apply(lambda x: int(x.split('-')[0]))

data['max_sal'] = salary_range.apply(lambda x: int(x.split('-')[1]))

data['avg_sal'] = (data.min_sal + data.max_sal)/2
data['company'] = data.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-4], axis = 1)

# data.company.head()
# data.Location.head()
data['cities'] = data['Location'].apply(lambda x: x.split(', ')[0])

data['job_states'] = data['Location'].apply(lambda x: x.split(', ')[1])

# data.states.value_counts() # We have one unique abbrevated state as Los Angeles

data['job_states'] = data['job_states'].apply(lambda x: 'LA' if x =='Los Angeles' else x)

# data.states.value_counts()
# data.Headquarters.head()
# data['headquartes_state'] = data['Headquarters'].apply(lambda x: x.split(', ')[-1])

# print(data.job_states.head())

# print(data['headquartes_state'].head())

# data['same_state'] = data.apply(lambda x: 1 if x.job_states == x.headquartes_state else 0, axis =1)

# data.same_state.head()
data['same_state'] = data.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)
data['age'] = data['Founded'].apply(lambda x: x if x<1 else 2020-x )
# print(data.columns)

# print(data.shape)
# yn is yes(1) or no(0) 

# Python

data['python_yn'] = data['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

# print('Python count:\n',data.python_yn.value_counts())



# R studio

data['Rstudio_yn'] = data['Job Description'].apply(lambda x: 1 if ('r studio' in x.lower() or 'r-studio' in x.lower()) else 0)

# print('R or R studio count:\n',data.Rstudio_yn.value_counts())



# Excel

data['excel_yn'] = data['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

# print('Excel count:\n',data.excel_yn.value_counts())



# AWS

data['aws_yn'] = data['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# print('AWS count:\n',data.aws_yn.value_counts())



# Spark

data['spark_yn'] = data['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# print('Spark count:\n',data.spark_yn.value_counts())
# dict(data['Job Title'].value_counts())
def title_simp(title):

    '''Makes the given below set of jobs as title other that these will be nil which can later changed'''

    

    if 'data scientist' in title.lower():

        return 'data scientist'

    elif 'data engineer' in title.lower():

        return 'data engineer'

    elif 'analyst' in title.lower():

        return 'analyst'

    elif 'machine learning' in title.lower():

        return 'mle' #Machine Learning Engineer

    elif 'manager' in title.lower():

        return 'manager' 

    elif 'director' in title.lower():

        return 'director'

    else:

        return 'na'

    

    

def seniority(title):

    '''Considers only senior and junior if any other then returns NaN can be changed later'''

    

    seniors = ['senior', 'sr', 'lead', 'principal']

    

    for senior in seniors:

        if senior in title.lower():

            return 'sr'

    if 'jr' in title.lower() or 'junior' in title.lower():

        return 'jr'

    else:

        return 'na'
data['job_simp'] = data['Job Title'].apply(title_simp)

data['seniority_lvl'] = data['Job Title'].apply(seniority)



data.job_simp.value_counts()
# data.seniority_lvl.value_counts()
data['job_des'] = data['Job Description'].apply(lambda x: len(x))

# data.job_des
# data.Competitors.value_counts()
data['comp_num'] = data['Competitors'].apply(lambda x: 0 if x == 0 else len(x.split(',')))
data.comp_num.value_counts()
data['min_sal'] = data.apply(lambda x: (x.min_sal*50*52)/1000 if x.hourly_pay == 1 else x.min_sal, axis=1)

data['max_sal'] = data.apply(lambda x: (x.max_sal*50*52)/1000 if x.hourly_pay == 1 else x.max_sal, axis=1)
data.to_csv('Salary_cleaned.csv', index=False)
df = pd.read_csv('Salary_cleaned.csv')
df.head()
df.describe()
# Lets see the distribution of all and consider which for hist and to normalize

# df.hist()
# Hist plot for visualizing distribution

hist_ls = ['Rating', 'age', 'avg_sal', 'job_des']

for feature in hist_ls:

    print('Histogram of ', feature)

    plt.hist(df[feature])

    plt.show()
# cols = df.select_dtypes(include=np.number).columns.tolist() # use this visualize and take only the needed ones

cols = ['Rating', 'age', 'avg_sal', 'job_des']

for col in cols:

    df.boxplot(column= col)

    plt.show()
df[['Rating', 'age', 'avg_sal', 'job_des', 'comp_num']].corr()
sns.set()

sns.heatmap(df[['Rating', 'age', 'avg_sal', 'job_des', 'comp_num']].corr(), annot=True, fmt='.1f')

plt.yticks(rotation = 0)
df_cat = df[['Location', 'Headquarters', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'company', 'cities', 'job_states', 'same_state', 'python_yn',

       'Rstudio_yn', 'excel_yn', 'aws_yn', 'spark_yn', 'job_simp',

       'seniority_lvl']]
for feature in df_cat.columns:

    if len(df[feature].unique()) < 15:

        count = df[feature].value_counts()

    #     print('Plot for {} total : {}'.format(feature, len(count)))

        sns.barplot(x=count.index , y= count, data = df)

        plt.xticks(rotation=90)

        plt.show()
for feature in df_cat[['Location', 'Headquarters', 'Industry', 'company', 'cities', 'job_states']].columns:

    count = df[feature].value_counts()[:15] # We can just view the Top 15 in all Series

#     print('Plot for {} total : {}'.format(feature, len(count)))

    sns.barplot(x=count.index , y= count, data = df)

    plt.xticks(rotation=90)

    plt.show()  
# Using pivot_table() function

pd.pivot_table(df, index='job_simp', values='avg_sal')
# Avg salary based on job title and the seniority level

pd.pivot_table(df, index=['job_simp', 'seniority_lvl'], values='avg_sal')
# Checking for data scientist avg salary per state

pd.pivot_table(df[df.job_simp == 'data scientist'], index='job_states', values='avg_sal').sort_values('avg_sal', ascending=False)
df.head()
df.columns
train = df[['Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'hourly_pay', 'employer_prov',

       'company', 'cities', 'job_states', 'same_state', 'age', 'python_yn',

       'Rstudio_yn', 'excel_yn', 'aws_yn', 'spark_yn', 'job_simp',

       'seniority_lvl', 'job_des', 'comp_num']]
X = pd.get_dummies(train)

X.shape
X.head()

X.info()
y = df['avg_sal']
from sklearn.model_selection import train_test_split as tts

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=140)

model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_absolute_error(y_test, prediction)
accuracy = model.score(X_test, y_test)

round(accuracy*100, 2)
# parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}



# gs = GridSearchCV(model,parameters,scoring='neg_mean_absolute_error',cv=3)

# gs.fit(X_train,y_train)



# gs.best_score_
# gs.best_estimator_
model = RandomForestRegressor(n_estimators=140)

model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

round(accuracy*100, 2)