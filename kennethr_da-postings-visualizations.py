import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv")

data
data.info()

# 1 null in company name
data['Job Title'].nunique()

                          
top10_job_titles= data['Job Title'].value_counts().sort_values(ascending = False).head(10)

plt.pie(top10_job_titles, autopct='%.2f',pctdistance=1.3)

plt.title('Top 5 Most Popular Job Postings',fontweight ='bold')

plt.legend(list(dict(top10_job_titles).keys()), loc='upper right')

plt.show()

data['Founded'].value_counts()

year_freq=data.groupby(['Founded']).size()[1:,]

year=data['Founded'].unique()

year = np.delete(pen, np.argwhere(pen==-1))

year = np.sort(year)

plt.plot(year,year_freq)

plt.show()
data.columns
df = data



# filter the jobs by city

for i in range(df.shape[0]): # remove state

    df['Location'].iloc[i] = df['Location'].iloc[i][:-3]



data_analyst = df['Job Title'] == 'Data Analyst'

da_indicies = df['Job Title'][data_analyst].index



da_data = df.iloc[da_indicies]



da_data
da_salary = np.sort(da_data['Salary Estimate'].unique())



da_salary
top5_da_salary = np.append(da_salary[-4:],'$110K-$190K (Glassdoor est.)')

top5_da_salary
da_salary_dict = {}

for salary in top5_da_salary:

    true_false_cases = da_data['Salary Estimate'] == salary

    indicies = da_data['Salary Estimate'][true_false_cases].index

    

    da_salary_dict[salary] = indicies

da_salary_dict # dict of DAs at different salary estimates
top5_sal_df = pd.DataFrame() 



for key in da_salary_dict.keys():

    top5_sal_df = top5_sal_df.append(df.iloc[da_salary_dict[key]])

top5_sal_df


location_salary = pd.crosstab(top5_sal_df['Location'],top5_sal_df['Salary Estimate'])

location_salary.plot(kind='bar',width=2,figsize=(15,8))

plt.show()



top5_sal_df['Salary Estimate'].value_counts().plot(kind='barh',

                                                   color = ['black', 'green','blue','red','purple'])

plt.show()

# word cloud for only data analyst positions



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



text = str(np.ravel(da_data['Job Description']))



wordcloud = WordCloud(max_font_size=50, max_words=10000, background_color="black").generate(text)

plt.figure(figsize=(10,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# word cloud for all positions



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



text = str(np.ravel(data['Job Description']))



wordcloud = WordCloud(max_font_size=50, max_words=10000, background_color="black").generate(text)

plt.figure(figsize=(10,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()