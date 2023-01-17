import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



multiple_choice_responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv", low_memory=False)

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")
multiple_choice_responses['Q3'].replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom',

                                       inplace = True)
genders = multiple_choice_responses[['Q2']].drop(multiple_choice_responses.index[0])

gender_distr = genders[(genders['Q2'] == 'Female') | (genders['Q2'] == 'Male')].groupby('Q2').agg({'Q2':'count'})



fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.set_title('Gender Distribution')

ax.set(xlabel='Gender', ylabel='Respondents')

sns.set(style='whitegrid')

ax = sns.barplot(['Female', 'Male'], list(gender_distr['Q2']), ax=ax)

plt.show()
demographics = multiple_choice_responses[['Q1','Q2','Q3']].drop(multiple_choice_responses.index[0])

demographics = demographics[(demographics['Q2'] == 'Female') | (demographics['Q2'] == 'Male')]

demographics = demographics.rename(columns={'Q1':'Age_Group', 'Q2':'Gender', 'Q3':'Country'})
loc = pd.crosstab(demographics['Country'], demographics['Gender'])

loc['Female'] = loc['Female'].astype(float)

loc['Male'] = loc['Male'].astype(float)



loc = loc.sort_values(by=['Female'], ascending=False)

fig, ax = plt.subplots(1,1,figsize=(15,30))

ax.set_title('Female Respondents by Location')

ax.set(xlabel='Counts')

sns.set(style='whitegrid')

ax = sns.barplot(loc['Female'], loc.index, ax=ax, palette="vlag")

plt.show()



for i in loc.index:

    female = loc.at[i, 'Female']

    male = loc.at[i, 'Male']

    total = loc.at[i, 'Female'] + loc.at[i, 'Male']

    loc.at[i, 'Female'] = round(female / total,4)



loc = loc.drop(['Male'], axis = 1)

loc = loc.sort_values(by=['Female'], ascending=False)



fig, ax = plt.subplots(1,1,figsize=(15,30))

ax.set_title('Female Respondents by Location (Percentage)')

ax.set(xlabel='Percentage')

sns.set(style='whitegrid')

ax = sns.barplot(list(loc['Female']), loc.index, ax=ax, palette="vlag")

plt.show()
age = pd.crosstab(demographics['Age_Group'], demographics['Gender'])

age['Female'] = age['Female'].astype(float)

age['Male'] = age['Male'].astype(float)



totals = demographics.groupby('Age_Group').agg({'Gender':'count'})

fig, ax = plt.subplots(1,1,figsize=(15,5))

ax.set_title('Age Group Distribution')

ax.set(xlabel='Age Group', ylabel='Respondents')

sns.set(style='whitegrid')

ax = sns.barplot(totals.index, list(totals['Gender']), ax=ax, palette="rocket")

plt.show()





for i in age.index:

    female = age.at[i, 'Female']

    male = age.at[i, 'Male']

    total = age.at[i, 'Female'] + age.at[i, 'Male']

    age.at[i, 'Female'] = round(female / total,4)



age = age.drop(['Male'], axis=1)
fig, ax = plt.subplots(1,1,figsize=(15,5))

ax.set_title('Female Respondents by Age Group (Percentage)')

ax.set(ylabel='Percentage')

sns.set(style='whitegrid')

ax = sns.barplot(age.index, list(age['Female']), ax=ax, palette="rocket")

plt.show()
women = multiple_choice_responses[multiple_choice_responses['Q2']=='Female'].copy()

men = multiple_choice_responses[multiple_choice_responses['Q2']=='Male'].copy()
jobs = women.groupby('Q5', as_index=False).agg({'Q1':'count'})

jobs = jobs.rename(columns={'Q1':'Count', 'Q5':'Job_Title'})

jobs = jobs.sort_values(by=['Count'], ascending = False)



fig, ax = plt.subplots(1,1,figsize=(10,10))

ax.set_title('Job Titles (Women only)')

sns.set(style='whitegrid')

ax = sns.barplot('Count', 'Job_Title', ax=ax, palette="deep", data=jobs)

plt.show()
stem = ['Data Scientist', 'Data Analyst', 'Software Engineer', 'Research Scientist', 

        'Data Engineer', 'DBA/Database Engineer', 'Statistician']

unclear = ['Business Analyst', 'Product/Project Manager']



jobs['Job_Field'] = [ 'STEM' if x in stem else 'Unclear' if x in unclear else x for x in jobs['Job_Title'] ]

field = jobs.groupby('Job_Field', as_index=False).agg({'Count':'sum'})

field = field.sort_values(by=['Count'], ascending=False)

field['Count'] = field['Count'].astype(float)





total = field['Count'].sum()



for i in field.index:

    count = field.at[i, 'Count']

    field.at[i, 'Count'] = round(count / total,4)



field = field.rename(columns={'Count':'Percentage'})



fig, ax = plt.subplots(1,1,figsize=(10,8))

ax.set_title('Job Fields (Women only)')

sns.set(style='whitegrid')

ax = sns.barplot('Percentage', 'Job_Field', ax=ax, palette="deep", data=field)

plt.show()
education = women.groupby('Q4', as_index = False).agg({'Q1':'count'})

education = education.rename(columns={'Q1':'Count','Q4':'Education'}).sort_values(by=['Count'], ascending = False)

education.replace("Some college/university study without earning a bachelor’s degree", "Some college", inplace = True)

education.replace("No formal education past high school", "High School", inplace = True)

education.replace("I prefer not to answer", "No answer", inplace = True)







postgrad = ["Master’s degree", "Doctoral degree", "Professional degree"]

no_edu = ["Some college", "High School"]

education['Category'] = [ 'Postgrad' if x in postgrad else 'No Degree' if x in no_edu 

                         else 'Undergrad' if x == "Bachelor’s degree" else x for x in education['Education'] ]



edu_cat = education.groupby('Category', as_index = False).agg({'Count':'sum'})

edu_cat['Count'] = edu_cat['Count'].astype(float)

total = edu_cat['Count'].sum()



for i in edu_cat.index:

    count = edu_cat.at[i, 'Count']

    edu_cat.at[i, 'Count'] = round(count / total,4)

    

edu_cat = edu_cat.rename(columns={'Count':'Percentage'}).sort_values(by=['Percentage'], ascending = False)





fig, ax = plt.subplots(1,1,figsize=(10,8))

ax.set_title('Degrees')

sns.set(style='whitegrid')

ax = sns.barplot('Count', 'Education', ax=ax, palette="deep", data=education)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.set_title('Degree Categories')

sns.set(style='whitegrid')

ax = sns.barplot('Percentage', 'Category', ax=ax, palette="deep", data=edu_cat)

plt.show()
low_salary = women[women['Q10']=='$0-999']

lows = low_salary.groupby('Q5', as_index = False).agg({'Q1':'count'})

lows = lows.rename(columns={'Q1':'Count', 'Q5':'Occupation'})

lows = lows.sort_values(by=['Count'], ascending = False)



fig, ax = plt.subplots(1,1,figsize=(10,8))

ax.set_title('Low Salaries: Occupation')

sns.set(style='whitegrid')

ax = sns.barplot('Count', 'Occupation', ax=ax, palette="deep", data=lows)

plt.show()
def find_number(x):    

    if x[0] == '>':

        return 500001

    elif x[0] == '$':

        return 0

    else:

        x = x.strip().replace(',','')

        return int(x[0 : x.find('-')])



salaries = women.groupby('Q10', as_index = False).agg({'Q1':'count'})

salaries = salaries.rename(columns={'Q10':'Range', 'Q1':'Count'})



salaries['Range_Start'] = salaries.Range.apply(lambda x: find_number(x))



salaries = salaries.sort_values(by=['Range_Start'])

salaries = salaries[salaries['Range_Start'] != 0]



fig, ax = plt.subplots(1,1,figsize=(20,8))

ax.set_title('Salaries')

sns.set(style='whitegrid')

ax = sns.barplot('Range', 'Count', ax=ax, palette="deep", data=salaries)

ax.set_xticklabels(labels=salaries['Range'], rotation=30)

plt.show()
women['Q4'].replace("No formal education past high school", "High School", inplace = True)

women['Q4'].replace("Some college/university study without earning a bachelor’s degree", "Some college", inplace = True)

no_degree = women[(women['Q4']=='High School') | (women['Q4']=='Some college')]



no_degree_sal = no_degree.groupby('Q10', as_index=False).agg({'Q1':'count'})

no_degree_sal = no_degree_sal.rename(columns={'Q10':'Range', 'Q1':'Count'})



no_degree_sal['Range_Start'] = no_degree_sal.Range.apply(lambda x: find_number(x))

no_degree_sal = no_degree_sal.sort_values(by=['Range_Start'])



fig, ax = plt.subplots(1,1,figsize=(20,8))

ax.set_title('Salaries - No Degree')

sns.set(style='whitegrid')

ax = sns.barplot('Range', 'Count', ax=ax, palette="deep", data=no_degree_sal)

ax.set_xticklabels(labels=salaries['Range'], rotation=30)

plt.show()



no_degree_job = no_degree.groupby('Q5', as_index=False).agg({'Q1':'count'})

no_degree_job = no_degree_job.rename(columns={'Q5':'Education', 'Q1':'Count'})

no_degree_job = no_degree_job.sort_values(by=['Count'], ascending = False)



fig, ax = plt.subplots(1,1,figsize=(10,10))

ax.set_title('Job Title - No Degree')

sns.set(style='whitegrid')

ax = sns.barplot('Count', 'Education', ax=ax, palette="deep", data=no_degree_job)

plt.show()
women['Q15'].replace('I have never written code', 'None', inplace=True)

women['Q15'].replace(np.nan, 'No answer', inplace = True)



code = women.groupby('Q15', as_index = False).agg({'Q1':'count'})

code = code.rename(columns={'Q1':'Count', 'Q15':'Coding Experience'})

code = code.sort_values(by=['Count'], ascending = False)



fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.set_title('Coding Experience')

sns.set(style='whitegrid')

ax = sns.barplot('Count', 'Coding Experience', ax=ax, palette="deep", data=code)

plt.show()
beginner = ['< 1 years', '1-2 years']

mid = ['3-5 years', '5-10 years']

expert = ['10-20 years', '20+ years']



code['Level'] = [ 'Beginner' if x in beginner else 'Intermediate' if x in mid 

                 else 'Expert' if x in expert else x for x in code['Coding Experience']]



code_lvl = code.groupby('Level').agg({'Count':'sum'})

order = ['None','Beginner','Intermediate','Expert','No answer']

code_lvl = code_lvl.reindex(order).reset_index()



total = code_lvl['Count'].sum()





fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.set_title('Coding Level - Women')

sns.set(style='whitegrid')

ax = sns.barplot('Level', 'Count', ax=ax, palette="deep", data=code_lvl, estimator=lambda x: sum(x)/total*100.0)

ax.set(ylabel='Percentage')

plt.show()
men['Q15'].replace('I have never written code', 'None', inplace=True)

men['Q15'].replace(np.nan, 'No answer', inplace = True)



code_m = men.groupby('Q15', as_index = False).agg({'Q1':'count'})

code_m = code_m.rename(columns={'Q1':'Count', 'Q15':'Coding Experience'})

code_m = code_m.sort_values(by=['Count'], ascending = False)



code_m['Level'] = [ 'Beginner' if x in beginner else 'Intermediate' if x in mid 

                 else 'Expert' if x in expert else x for x in code_m['Coding Experience']]



code_lvl_m = code_m.groupby('Level').agg({'Count':'sum'})

code_lvl_m = code_lvl_m.reindex(order).reset_index()



total_m = code_lvl_m['Count'].sum()



fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.set_title('Coding Level - Men')

sns.set(style='whitegrid')

ax = sns.barplot('Level', 'Count', ax=ax, palette="deep", data=code_lvl_m, estimator=lambda x: sum(x)/total_m*100.0)

ax.set(ylabel='Percentage')

plt.show()
exploring = "We are exploring ML methods (and may one day put a model into production)"

no = "No (we do not use ML methods)"

women['Q8'] = ["Don't know" if x == "I do not know" else "No" if x == no 

              else "Exploring" if x == exploring else "Yes" for x in women['Q8']]

men['Q8'] = ["Don't know" if x == "I do not know" else "No" if x == no 

              else "Exploring" if x == exploring else "Yes" for x in men['Q8']]
ml = women.groupby('Q8', as_index = False).agg({'Q1':'count'})

ml_m = men.groupby('Q8', as_index = False).agg({'Q1':'count'})

ml = ml.rename(columns={'Q8':'Machine_Learning', 'Q1':'Count'})

ml_m = ml_m.rename(columns={'Q8':'Machine_Learning', 'Q1':'Count'})



total = ml['Count'].sum()

total_m = ml_m['Count'].sum()



fig, ax = plt.subplots(1,2,figsize=(18,5))

sns.set(style='whitegrid')

sns.barplot('Machine_Learning', 'Count', ax=ax[0], palette="deep", data=ml, estimator=lambda x: sum(x)/total*100.0)

sns.barplot('Machine_Learning', 'Count', ax=ax[1], palette="deep", data=ml_m, estimator=lambda x: sum(x)/total_m*100.0)

for a in ax:

    a.set(xlabel='Machine Learning', ylabel='Percentage')

ax[0].set_title('ML Usage - Women')

ax[1].set_title('ML Usage - Men')

plt.show()
responses = women['Q1'].count()

responses_m = men['Q1'].count()

base = 'Q18_Part_'

counts = dict()

counts_m = dict()



palette = dict()

colours = sns.color_palette('deep',11)



for i in range(1,12):

    col = base + str(i)

    count = women[col].count()

    count_m = men[col].count()

    lang = women[col].dropna().values[0]

    counts[lang] = count

    counts_m[lang] = count_m

    palette[lang] = colours[i-1]





result = pd.DataFrame.from_dict(counts, orient='index').reset_index().rename(columns={'index':'Language', 0:'Count'})

result = result.sort_values(by=['Count'], ascending = False)

result_m = pd.DataFrame.from_dict(counts_m, orient='index').reset_index().rename(columns={'index':'Language', 0:'Count'})

result_m = result_m.sort_values(by=['Count'], ascending = False)



fig, ax = plt.subplots(1,2,figsize=(18,10))

sns.set(style='whitegrid')

sns.barplot(result['Count'], result['Language'],  ax=ax[0], palette=palette, data=result, 

            estimator=lambda x: sum(x)/responses*100.0)

sns.barplot(result_m['Count'], result_m['Language'], ax=ax[1], palette=palette, data=result_m, 

            estimator=lambda x: sum(x)/responses_m*100.0)

for a in ax:

    a.set(ylabel='', xlabel='Percentage')

ax[0].set_title('Languages - Women')

ax[1].set_title('Languages - Men')

plt.show()
python = women[women['Q18_Part_1']=='Python'].copy()



python['Q15'] = [ 'New' if x == '< 1 years' else 'Beginner' if x == '1-2 years' else 'Proficient' for x in python['Q15']]

py_exp = python.groupby('Q15').agg({'Q1':'count'})



py_order = ['New', 'Beginner', 'Proficient']

py_exp = py_exp.reindex(py_order).reset_index()

py_exp = py_exp.rename(columns={'Q15':'Experience', 'Q1':'Count'})



total_py = py_exp['Count'].sum()



fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.set_title('Coding Level - Female Python Users')

sns.set(style='whitegrid')

ax = sns.barplot('Experience', 'Count', ax=ax, palette="deep", data=py_exp, estimator=lambda x: sum(x)/total_py*100.0)

ax.set(ylabel='Percentage')

plt.show()
for i in python.index:

    count = 0

    for j in range(2,11):

        lang = python.at[i, (base + str(j))]

        if pd.isnull(lang) == False:

            count += 1

    python.at[i, 'Q18_Count'] = count

    

python['Multi_Language'] = [ '> 5' if x >= 5.0 else '2-4' if x > 2.0 

                            else 0 if x == 0.0 else 1 if x == 1.0 else 2 for x in python['Q18_Count']]

others = python.groupby(['Multi_Language', 'Q15']).agg({'Q1':'count'}).rename(columns={'Q1':'Count'})

others = pd.DataFrame(data={'Count': others['Count']}, index=others.index).reset_index()



py_only = others[others['Multi_Language']==0].drop(['Multi_Language'],axis=1).rename(columns={'Q15':'Experience'})

total_py_only = py_only['Count'].sum()



fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.set_title('Coding Level - Python only (Female)')

sns.set(style='whitegrid')

ax = sns.barplot('Experience', 'Count', ax=ax, palette="deep", data=py_only, estimator=lambda x: sum(x)/total_py_only*100.0,

                order=py_order)

ax.set(ylabel='Percentage')

plt.show()
cat_order = ['New', 'Beginner', 'Proficient']



sns.set(style='whitegrid')

g = sns.catplot(x='Multi_Language', y='Count', col='Q15', data=others, kind='bar', col_order=cat_order)

titles = ["New", "Beginner", "Proficient"]

for ax, title in zip(g.axes.flat, titles):

    ax.set_title(title)

    ax.set(xlabel = 'Languages in Addition to Python')

plt.show()