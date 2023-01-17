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
df = pd.read_csv("/kaggle/input/adult-census-income/adult.csv")
race_count = df['race'].value_counts()



print("Number of each race:\n", race_count) 
average_age_men = round(df.loc[df['sex'] == 'Male', 'age'].mean(), 1)



print("Average age of men:", average_age_men)
percentage_bachelors = round(len(df[df['education'] == 'Bachelors']) / len(df) * 100, 1)



print(f"Percentage with Bachelors degrees: {percentage_bachelors}%")
# with and without `Bachelors`, `Masters`, or `Doctorate`

higher_education = df[df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]





# percentage with salary >50K

higher_education_rich = round(

        len(higher_education[higher_education["income"] == ">50K"]) / len(higher_education) * 100, 1)





print(f"Percentage with higher education that earn >50K: {higher_education_rich}%")
lower_education = df[~df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]



lower_education_rich = round(

        len(lower_education[lower_education["income"] == ">50K"]) / len(lower_education) * 100, 1)



print(f"Percentage without higher education that earn >50K: {lower_education_rich}%")
min_work_hours = df["hours.per.week"].min()

print(f"Min work time: {min_work_hours} hours/week")
num_min_workers = len(df[df['hours.per.week'] == min_work_hours])



rich_percentage = round(len(df[(df['hours.per.week'] == min_work_hours) & (df['income'] == '>50K')]) / num_min_workers * 100, 1)

 

print(f"Percentage of rich among those who work fewest hours: {rich_percentage}%")
highest_earning_country = (df.loc[df['income'] == ">50K", 'native.country'].value_counts() / df['native.country'].value_counts()).fillna(0).sort_values(ascending=False).index[0]

highest_earning_country_percentage = round(len(df[(df['native.country'] == highest_earning_country) & (df['income'] == '>50K')]) / len(df[df['native.country'] == highest_earning_country]) * 100, 1)

    

print("Country with highest percentage of rich:", highest_earning_country)

print(f"Highest percentage of rich people in country: {highest_earning_country_percentage}%")
top_IN_occupation = df[(df['income'] == ">50K") & (df['native.country'] == "India")]["occupation"].value_counts().index[0]

print("Top occupations in India:", top_IN_occupation)