import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import csv

import seaborn as sns



fig_size = (12, 8)
df_in_out = pd.read_csv("../input/aac_intakes_outcomes.csv")

df_in = pd.read_csv("../input/aac_intakes.csv")

df_out = pd.read_csv("../input/aac_outcomes.csv")
print(df_in_out.shape)

print(df_in.shape)

print(df_out.shape)

df = df_in_out
df.isnull().sum()
df = df.drop(['outcome_subtype'], axis = 1)
df.columns
df.groupby('animal_type')['animal_id_outcome'].count()
dogs   = df[df.animal_type=='Dog']

cats   = df[df.animal_type=='Cat']

others = df[(df['animal_type'] != 'Dog') & (df['animal_type'] != 'Cat')]

print("dogs: "   + str(dogs['animal_id_outcome'].count()))

print("cats: "   + str(cats['animal_id_outcome'].count()))

print("others: " + str(others['animal_id_outcome'].count()))
x = pd.DataFrame({'intake': dogs['sex_upon_intake'].value_counts(), 'outcome': dogs['sex_upon_outcome'].value_counts()}, index = dogs['sex_upon_intake'].unique())

ax = x.plot.bar(rot=0, figsize = fig_size)
dogs['intake_type'].value_counts()
stray_dogs = dogs[dogs['intake_type'] == 'Stray']



# Top 20

stray_dogs['breed'].value_counts().head(20)
stray_dogs['breed'].value_counts().head(20).plot.bar(figsize = fig_size)
dogs['outcome_type'].value_counts()
a_dogs = dogs[dogs.outcome_type == 'Adoption']

r_dogs = dogs[dogs.outcome_type == 'Return to Owner']

a_r_dogs = dogs[(dogs.outcome_type == 'Adoption') | (dogs.outcome_type == 'Return to Owner')]



# Top 20

a_dogs['breed'].value_counts().head(20)
r_dogs['breed'].value_counts().head(20)
a_r_dogs['breed'].value_counts().head(20)
x = pd.DataFrame({'adopted': a_dogs['breed'].value_counts().head(9), 'returned': r_dogs['breed'].value_counts().head(9), 'abandoned': stray_dogs['breed'].value_counts().head(9)}, index = a_r_dogs['breed'].value_counts().head(10).index.values)

ax = x.plot.bar(rot=90, figsize = fig_size)
dogs_year_in  = dogs.groupby(['intake_year'])

dogs_year_out = dogs.groupby(['outcome_year'])
# ref 1

# https://www.theguardian.com/us-news/2015/may/27/chimpanzee-animals-rights-new-york-court

# ref 2

# https://www.avma.org/Events/pethealth/Pages/default.aspx#May



dogs_year_in['intake_month'].value_counts().sort_index().plot.bar(figsize = fig_size)
dogs_year_out['outcome_month'].value_counts().sort_index().plot.bar(figsize = fig_size)
dogs_in_2015  = dogs[dogs.intake_year == 2015].groupby(['intake_year'])['intake_month'].value_counts().sort_index()

dogs_out_2015  = dogs[dogs.outcome_year == 2015].groupby(['outcome_year'])['outcome_month'].value_counts().sort_index()



# dogs_2015_g_in  = dogs_in_2015.plot.bar(figsize = fig_size)

# dogs_2015_g_out  = dogs_out_2015.plot.bar(figsize = fig_size)



x = pd.DataFrame({'intake': dogs_in_2015, 'outcome': dogs_out_2015})

ax = x.plot.bar(rot=90, figsize = fig_size)
dogs["intake_date"]  = pd.to_datetime(dogs["intake_datetime"]).dt.date

dogs["outcome_date"] = pd.to_datetime(dogs["outcome_datetime"]).dt.date

dogs[(dogs.intake_year == 2015) & (dogs.intake_month == 5)]['intake_date'].value_counts().sort_index().plot.bar(figsize = fig_size)
dogs[(dogs.outcome_year == 2015) & (dogs.outcome_month == 5)]['outcome_date'].value_counts().sort_index().plot.bar(figsize = fig_size)
x = pd.DataFrame({'intake': cats['sex_upon_intake'].value_counts(), 'outcome': cats['sex_upon_outcome'].value_counts()}, index = cats['sex_upon_intake'].unique())

ax = x.plot.bar(rot=0, figsize = fig_size)
cats['intake_type'].value_counts()
stray_cats = cats[cats['intake_type'] == 'Stray']



# Top 20

stray_cats['breed'].value_counts().head(20)
stray_cats['breed'].value_counts().head(5).plot.bar(figsize = fig_size)
dogs['outcome_type'].value_counts()
a_cats = cats[cats.outcome_type == 'Adoption']

r_cats = cats[cats.outcome_type == 'Return to Owner']

a_r_cats = cats[(cats.outcome_type == 'Adoption') | (cats.outcome_type == 'Return to Owner')]



# Top 20

a_cats['breed'].value_counts().head(5)
r_cats['breed'].value_counts().head(5)
a_r_cats['breed'].value_counts().head(5)
cats_year_in  = cats.groupby(['intake_year'])

cats_year_out = cats.groupby(['outcome_year'])
cats_year_in['intake_month'].value_counts().sort_index().plot.bar(figsize = fig_size)
cats_year_out['outcome_month'].value_counts().sort_index().plot.bar(figsize = fig_size)