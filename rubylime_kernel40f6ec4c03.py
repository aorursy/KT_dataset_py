import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
responses = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)
def num_of_years(year_range):
    year_range = str(year_range).replace("+", "")
    return int(year_range.split("-")[-1])

def salary(salary_range):
    if("I do not wish" in salary_range):
        return None
    else:
        salary_range = str(salary_range).replace("+", "").replace(",", "")
        salary_range = int(float(salary_range.split("-")[-1]))
        return salary_range

responses.drop(responses.index[0:1], inplace=True)
responses.rename(columns={'Time from Start to Finish (seconds)':'Q0'}, inplace=True)
# For some reason, dropna didn't work for me so I set the nan's to a high number of years to filter out below
responses['Q8'].replace(np.nan, 500, inplace=True)

responses['num_of_years'] = responses['Q8'].apply(num_of_years)

responses.dropna(subset=['Q9'], inplace=True)

responses['salary'] = responses['Q9'].apply(salary)

# For some reason, dropna didn't work for me so I set the nan's to a high salary to filter out below
responses['salary'].replace(np.nan, 2000000, inplace=True)

responses = responses[responses['num_of_years']<500]
responses = responses[responses['salary']<2000000]
sns.lmplot(x='num_of_years',y='salary',data=responses,hue='Q1')
python = responses[(responses['Q17']=="Python")]
r = responses[(responses['Q17']=="R")]
python.describe()
r.describe()
responses.dropna(subset=['Q23'], inplace=True)
responses[['salary', 'Q23']].head()
a4_dims = (50, 40)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(ax=ax, x='Q23',y='salary',data=responses,estimator=np.std)
a4_dims = (50, 40)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(ax=ax, x='Q24',y='salary',hue='Q1', data=responses,estimator=np.std)
a4_dims = (50, 40)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(ax=ax, x='Q26',y='salary',hue='Q1', data=responses,estimator=np.std)
responses.dropna(subset=['Q32'], inplace=True)
a4_dims = (50, 40)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(ax=ax, x='Q32',y='salary', data=responses,estimator=np.std)






