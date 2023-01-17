import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', None) 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')

df[:2]
def salary_range_str_to_ranges(salary):

    if '-' not in str(salary):

        return None, None

    lower, upper = salary.split(',')[0].split('-')

    return int(lower) * 1000, int(upper) * 1000



def simplify_data_scientist_data(df):

    df = df[['Q4', 'Q6', 'Q8', 'Q9', 'Q11_Part_4', 'Q11_Part_5', 'Q16_Part_1']].rename(columns=dict(

        Q4='education_degree', 

        Q6='current_role', 

        Q8='experience_years', 

        Q9='salary_range_usd', 

        Q11_Part_4='apply_ml_in_new_areas', 

        Q11_Part_5='do_research_in_ml', 

        Q16_Part_1='uses_python'

    ))

    df['lower_bound_salary_usd'], df['upper_bound_salary_usd'] = zip(*df['salary_range_usd'].map(salary_range_str_to_ranges))

    df['lower_bound_salary_usd'] = df['lower_bound_salary_usd'].astype(float)

    df['upper_bound_salary_usd'] = df['upper_bound_salary_usd'].astype(float)

    df['apply_ml_in_new_areas'] = df['apply_ml_in_new_areas'].map(pd.notnull)

    df['do_research_in_ml'] = df['do_research_in_ml'].map(pd.notnull)

    df = df[pd.notnull(df.lower_bound_salary_usd)]

    return df[df.current_role == 'Data Scientist']
pl_df = simplify_data_scientist_data(df[df.Q3 == 'Poland'])

pl_df
us_df = simplify_data_scientist_data(df[df.Q3 == 'United States of America'])

us_df
world_df = simplify_data_scientist_data(df)

world_df
pl_df[

    (pl_df.experience_years == '4-5') | (pl_df.experience_years == '5-10')

]
len(us_df[

    (us_df.apply_ml_in_new_areas == True) | (us_df.do_research_in_ml == True)

][

    (us_df.experience_years == '4-5') | (us_df.experience_years == '5-10')

])
us_df[

        (us_df.apply_ml_in_new_areas == True) | (us_df.do_research_in_ml == True)

    ][

        (us_df.experience_years == '4-5') | (us_df.experience_years == '5-10')

    ][['lower_bound_salary_usd', 'upper_bound_salary_usd']].mean()
pd_stats = pd.DataFrame(dict(

    poland=pl_df.groupby('experience_years').size(),

    world=world_df.groupby('experience_years').size(),

)).fillna(0)

pd_stats = pd_stats.iloc[pd_stats.index.str.extract('(\d+)', expand=False).astype(int).argsort()]

(pd_stats / pd_stats.sum(axis=0) * 100).plot(kind='bar')