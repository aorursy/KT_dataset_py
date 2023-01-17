import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib_venn import venn2

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/multipleChoiceResponses.csv')
df.shape
# Here list of question

pd.set_option('display.max_colwidth', -1, 'display.max_rows', -1)

# print(df.iloc[0,:])

pd.set_option('display.max_colwidth', 50, 'display.max_rows', 60)

df.iloc[0, :].to_excel('ques.xlsx')
# remove question row

df = df.drop(0)
gender = df.Q1.value_counts()[:2]

plt.figure(figsize=(5, 5))

gender.plot.pie(autopct='%.2f%%', cmap=plt.cm.Pastel1, textprops=dict(size=13))

plt.title('Gender')

plt.ylabel('')

plt.show()
pivoted = pd.crosstab(df.Q2, df.Q1).iloc[:, :2]

sns.set(style='white')

pivoted.plot.bar(stacked=True, figsize=(10, 5))

plt.title('Ages')

plt.show()
country = df.Q3.value_counts().to_frame()

sns.barplot(country.Q3[:15], country.index[:15])

plt.title('Country')

plt.show()

print(country[:2])

print('{} from indonesia'.format(int(country.loc['Indonesia'])))
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

edu_level = df.Q4.value_counts().to_frame()

sns.barplot(edu_level.Q4, edu_level.index, ax=ax[0])

ax[0].title.set_text('Education Level')

edu_major = df.Q5.value_counts().to_frame()

sns.barplot(edu_major.Q5, edu_major.index, ax=ax[1])

ax[1].title.set_text('Education Major')

plt.show()
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

job_role = df.Q6.value_counts().to_frame()

sns.barplot(job_role.Q6, job_role.index, ax=ax[0])

ax[0].title.set_text('Job Role')

job_industry = df.Q7.value_counts().to_frame()

sns.barplot(job_industry.Q7, job_industry.index, ax=ax[1])

ax[1].title.set_text('Job Industry')

plt.show()
year_exp = df.Q8.value_counts().to_frame()

# for sorting year of experience

l = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30 +']

d = {j: i for i, j in enumerate(l)}

year_exp['s'] = year_exp.index.map(d)

year_exp = year_exp.sort_values('s').drop('s', axis=1)

sns.barplot(year_exp.index, year_exp.Q8)

plt.title('Year of Experience')

plt.ylabel('')

plt.show()
df['Q9'] = df['Q9'].replace({'I do not wish to disclose my approximate yearly compensation': 'not disclose'})



salaries = df.Q9.value_counts().to_frame()

l2 = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',

     '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',

     '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000',

     '300-400,000', '400-500,000', '500,000+',

     'I do not wish to disclose my approximate yearly compensation']

d2 = {j: i for i, j in enumerate(l2)}

salaries['s'] = salaries.index.map(d2)

salaries = salaries.sort_values('s').drop('s', axis=1)

plt.figure(figsize=(10, 5))

sns.barplot(salaries.index, salaries.Q9)

plt.xticks(rotation=70)

plt.ylabel('')

plt.title('Salaries')

plt.show()
df['Q4'] = df['Q4'].replace({'Some college/university study without earning a bachelor’s degree': 'College no degree', 'No formal education past high school': 'No formal education'})



injob = pd.crosstab(df.Q9[df.Q3 == 'Indonesia'], df.Q4[df.Q3 == 'Indonesia'])

injob = df[df.Q3=='Indonesia'].groupby(['Q9', 'Q4']).size().to_frame().reset_index()

injob['s1'] = injob.Q9.map(d2)

# l3 = ['I prefer not to answer', 'Professional degree', 'Doctoral degree', 'Master’s degree', 'Bachelor’s degree', 'Some college/university study without earning a bachelor’s degree', 'No formal education past high school']

# d3 = {j: i for i, j in enumerate(l3)}

# injob['s2'] = injob.Q4.map(d3)

injob = injob.sort_values(['s1'])#.drop(['s1'], axis=1)

sns.relplot(data=injob, x='Q9', y='Q4', size=0, sizes=(20, 2000), aspect=2)

plt.xticks(rotation=45)

plt.ylabel('')

plt.title('In my Country')

plt.show()

# injob
salary_vs_year_exp = pd.crosstab(df.Q9, df.Q8)

salary_vs_year_exp['s'] = salary_vs_year_exp.index.map(d2)

salary_vs_year_exp = salary_vs_year_exp.sort_values('s').drop('s', axis=1).T

salary_vs_year_exp['s'] = salary_vs_year_exp.index.map(d)

salary_vs_year_exp = salary_vs_year_exp.sort_values('s').drop('s', axis=1).T.reset_index()

salary_vs_year_exp.style.background_gradient(cmap='coolwarm', axis=1)
applied_ML = df.groupby('Q10').size()

applied_ML.plot.pie(autopct='%.2f%%', cmap=plt.cm.Pastel1, textprops=dict(size=10))

plt.title('Applied ML on business')

plt.ylabel('')

plt.show()
activity = pd.concat([

    df.Q11_Part_1.value_counts(),

    df.Q11_Part_2.value_counts(),

    df.Q11_Part_3.value_counts(),

    df.Q11_Part_4.value_counts(),

    df.Q11_Part_5.value_counts(),

    df.Q11_Part_6.value_counts(),

    df.Q11_Part_7.value_counts()]).to_frame()

activity['%'] = round(activity / df.shape[0] * 100, 2)

activity.columns = ['count', '%']

activity.sort_values('%', ascending=False)
tools = df.Q12_MULTIPLE_CHOICE.value_counts().to_frame()

sns.barplot(tools.Q12_MULTIPLE_CHOICE, tools.index)

plt.title('Tools')

plt.xlabel('')

plt.show()
cloud = pd.concat([

    df.Q15_Part_1.value_counts(),

    df.Q15_Part_2.value_counts(),

    df.Q15_Part_3.value_counts(),

    df.Q15_Part_4.value_counts(),

    df.Q15_Part_5.value_counts(),

    df.Q15_Part_6.value_counts()])

sns.barplot(cloud, cloud.index)

plt.title('Cloud Service')

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

lang = df.Q17.value_counts().to_frame()

rec = df.Q18.value_counts().to_frame()

sns.barplot(lang.Q17, lang.index, ax=ax[0])

sns.barplot(rec.Q18, rec.index, ax=ax[1])

ax[0].title.set_text('Programming Language most used')

ax[1].title.set_text('Programming Language recomended')

plt.xlabel('')

plt.show()

lang_rec = pd.concat([lang[:2], rec[:2]], axis=1)

lang_rec.columns = ['Most Used', 'Recomended']

print(lang_rec)
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

venn = venn2(subsets = (df[df.Q16_Part_1 == 'Python'].shape[0], df[df.Q16_Part_2 == 'R'].shape[0], df[(df.Q16_Part_1 == 'Python') & (df.Q16_Part_2 == 'R')].shape[0]), set_labels = ('Python', 'R'), ax=ax[0])

ax[0].title.set_text('Python vs R')



use_python = df.Q16_Part_1.copy()

use_python = use_python.map({'Python': 'Use Python'}).fillna('Not use Python').value_counts()

use_python.plot.pie(cmap=plt.cm.Pastel1, ax=ax[1], autopct='%.2f%%', textprops=dict(size=13))

ax[1].title.set_text('Using Python')



rec_python =df.Q18.apply(lambda x: 'Python' if x=='Python' else 'Other').value_counts()

rec_python.plot.pie(cmap=plt.cm.Pastel1, ax=ax[2], autopct='%.2f%%')

ax[2].title.set_text('Recomend Python')

plt.ylabel('')

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

lib = df.Q20.value_counts().to_frame()

sns.barplot(lib.Q20, lib.index, ax=ax[0])

vis = df.Q22.value_counts().to_frame()

sns.barplot(vis.Q22, vis.index, ax=ax[1])

ax[0].title.set_text('ML Library')

ax[1].title.set_text('Visualization Library')

plt.show()
datatype = pd.concat([

    df.Q31_Part_1.value_counts(),

    df.Q31_Part_2.value_counts(),

    df.Q31_Part_3.value_counts(),

    df.Q31_Part_4.value_counts(),

    df.Q31_Part_5.value_counts(),

    df.Q31_Part_6.value_counts(),

    df.Q31_Part_7.value_counts(),

    df.Q31_Part_8.value_counts(),

    df.Q31_Part_9.value_counts(),

    df.Q31_Part_10.value_counts(),

    df.Q31_Part_11.value_counts(),

    df.Q31_Part_12.value_counts()]).to_frame().sort_values(0, ascending=False)

sns.barplot(datatype[0], datatype.index)

plt.title('Data Type')

plt.xlabel('')

plt.show()
work = list()

work.append(pd.DataFrame(df.Q34_Part_1[df.Q34_Part_1.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

work.append(pd.DataFrame(df.Q34_Part_2[df.Q34_Part_2.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

work.append(pd.DataFrame(df.Q34_Part_3[df.Q34_Part_3.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

work.append(pd.DataFrame(df.Q34_Part_4[df.Q34_Part_4.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

work.append(pd.DataFrame(df.Q34_Part_5[df.Q34_Part_5.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

work.append(pd.DataFrame(df.Q34_Part_6[df.Q34_Part_6.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

work[0]['Work'] = 'Gathering Data'

work[1]['Work'] = 'Cleaning Data'

work[2]['Work'] = 'Visualizing Data'

work[3]['Work'] = 'Model Building'

work[4]['Work'] = 'Production Setup'

work[5]['Work'] = 'Find Insight'

work = pd.concat(work)

sns.boxplot(data=work, x='Portion', y='Work')

plt.title('Portion of Activity')

plt.show()
learn = list()

learn.append(pd.DataFrame(df.Q35_Part_1[df.Q34_Part_1.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

learn.append(pd.DataFrame(df.Q35_Part_2[df.Q34_Part_2.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

learn.append(pd.DataFrame(df.Q35_Part_3[df.Q34_Part_3.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

learn.append(pd.DataFrame(df.Q35_Part_4[df.Q34_Part_4.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

learn.append(pd.DataFrame(df.Q35_Part_5[df.Q34_Part_5.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

learn.append(pd.DataFrame(df.Q35_Part_6[df.Q34_Part_6.notnull()].tolist(), columns=['Portion'], dtype=np.float64))

learn[0]['Source'] = 'Self-taught'

learn[1]['Source'] = 'Online Course'

learn[2]['Source'] = 'Work'

learn[3]['Source'] = 'University'

learn[4]['Source'] = 'Kaggle'

learn[5]['Source'] = 'Other'

learn = pd.concat(learn)

sns.boxplot(data=learn, x='Portion', y='Source')

plt.title('Learning Source')

plt.show()
online = df.Q37.value_counts().to_frame()

sns.barplot(online.Q37, online.index)

plt.title('Online Course')

plt.show()
proclaim = df.Q26.value_counts()

ax = sns.barplot(proclaim, proclaim.index)

plt.xlabel('')



total = proclaim.sum()

for i in ax.patches:

    ax.text(i.get_width() - 500, i.get_y() + 0.5, round(i.get_width()/total*100,2))

plt.show()