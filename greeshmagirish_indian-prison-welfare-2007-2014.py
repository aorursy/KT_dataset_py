import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid')

sns.set_context("talk", font_scale=0.6)
edu_df = pd.read_csv('/kaggle/input/prison-welfare-data-india-20072014/prison_education_2007_2014.csv')

rehab_df = pd.read_csv('/kaggle/input/prison-welfare-data-india-20072014/prison_rehab_2007_2014.csv')

edu_df = edu_df.drop(['Unnamed: 0'], axis = 1)

rehab_df = rehab_df.drop(['Unnamed: 0'], axis = 1)
elementary = edu_df.groupby(by=['Year', 'State/UT'])['No. of Prisoners Benefitted by Elementary Education'].sum().reset_index().sort_values('No. of Prisoners Benefitted by Elementary Education', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(elementary.Year.unique()):

    plt.subplot(len(elementary.Year.unique()),1,count)



    sns.lineplot(elementary[(elementary['Year'] == year)]['State/UT'],elementary[(elementary['Year'] == year)]['No. of Prisoners Benefitted by Elementary Education'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('No. of Prisoners Benefitted by Elementary Education - 2007-2014', fontsize=15)
adult = edu_df.groupby(by=['Year', 'State/UT'])['No. of Prisoners Benefitted by Adult Education'].sum().reset_index().sort_values('No. of Prisoners Benefitted by Adult Education', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(adult.Year.unique()):

    plt.subplot(len(adult.Year.unique()),1,count)



    sns.lineplot(adult[(adult['Year'] == year)]['State/UT'],adult[(adult['Year'] == year)]['No. of Prisoners Benefitted by Adult Education'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('No. of Prisoners Benefitted by Adult Education - 2007-2014', fontsize=15)
higher = edu_df.groupby(by=['Year', 'State/UT'])['No. of Prisoners Benefitted by Higher Education'].sum().reset_index().sort_values('No. of Prisoners Benefitted by Higher Education', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(higher.Year.unique()):

    plt.subplot(len(higher.Year.unique()),1,count)



    sns.lineplot(higher[(higher['Year'] == year)]['State/UT'],higher[(higher['Year'] == year)]['No. of Prisoners Benefitted by Higher Education'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('No. of Prisoners Benefitted by Higher Education - 2007-2014', fontsize=15)
computer = edu_df.groupby(by=['Year', 'State/UT'])['No. of Prisoners Benefitted by Computer Course'].sum().reset_index().sort_values('No. of Prisoners Benefitted by Computer Course', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(computer.Year.unique()):

    plt.subplot(len(computer.Year.unique()),1,count)



    sns.lineplot(computer[(computer['Year'] == year)]['State/UT'],computer[(computer['Year'] == year)]['No. of Prisoners Benefitted by Computer Course'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('No. of Prisoners Benefitted by Computer Course - 2007-2014', fontsize=15)
df1 =pd.pivot_table(edu_df, index=['State/UT', 'Year'],values=['No. of Prisoners Benefitted by Higher Education'],aggfunc=np.sum).reset_index()

df1 = df1[df1.Year == 2013].sort_values('No. of Prisoners Benefitted by Higher Education', ascending=False).head(20)



fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.barplot(x=df1['State/UT'],y=df1['No. of Prisoners Benefitted by Higher Education'])

plt.xticks(rotation=45)

plt.ylabel('No. of Prisoners')

plt.title('No. of Prisoners Benefitted by Higher Education - 2013', fontsize=15)

release = rehab_df.groupby(by=['Year', 'State/UT'])['No. of Prisoners to whom Financial Assistance Was Provided on Release'].sum().reset_index().sort_values('No. of Prisoners to whom Financial Assistance Was Provided on Release', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(release.Year.unique()):

    plt.subplot(len(release.Year.unique()),1,count)



    sns.lineplot(release[(release['Year'] == year)]['State/UT'],release[(release['Year'] == year)]['No. of Prisoners to whom Financial Assistance Was Provided on Release'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('No. of Prisoners to whom Financial Assistance Was Provided on Release - 2007-2014', fontsize=15)
rehab = rehab_df.groupby(by=['Year', 'State/UT'])['No. of Convicts Rehabilitated'].sum().reset_index().sort_values('No. of Convicts Rehabilitated', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(rehab.Year.unique()):

    plt.subplot(len(rehab.Year.unique()),1,count)



    sns.lineplot(rehab[(rehab['Year'] == year)]['State/UT'],rehab[(rehab['Year'] == year)]['No. of Convicts Rehabilitated'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('No. of Convicts Rehabilitated - 2007-2014', fontsize=15)
legal = rehab_df.groupby(by=['Year', 'State/UT'])['No. of Prisoners to whom Legal Aid Was Provided'].sum().reset_index().sort_values('No. of Prisoners to whom Legal Aid Was Provided', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(legal.Year.unique()):

    plt.subplot(len(legal.Year.unique()),1,count)



    sns.lineplot(legal[(legal['Year'] == year)]['State/UT'],legal[(legal['Year'] == year)]['No. of Prisoners to whom Legal Aid Was Provided'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('No. of Prisoners to whom Legal Aid Was Provided - 2007-2014', fontsize=15)
skill_wage = rehab_df.groupby(by=['Year', 'State/UT'])['Wages Paid per Day to Convicts (in Rs.) - Skilled'].sum().reset_index().sort_values('Wages Paid per Day to Convicts (in Rs.) - Skilled', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(skill_wage.Year.unique()):

    plt.subplot(len(skill_wage.Year.unique()),1,count)



    sns.lineplot(skill_wage[(skill_wage['Year'] == year)]['State/UT'],skill_wage[(skill_wage['Year'] == year)]['Wages Paid per Day to Convicts (in Rs.) - Skilled'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('Wages Paid per Day to Convicts (in Rs.) - Skilled - 2007-2014', fontsize=15)
semiskill_wage = rehab_df.groupby(by=['Year', 'State/UT'])['Wages Paid per Day to Convicts (in Rs.) - Semi Skilled'].sum().reset_index().sort_values('Wages Paid per Day to Convicts (in Rs.) - Semi Skilled', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(semiskill_wage.Year.unique()):

    plt.subplot(len(semiskill_wage.Year.unique()),1,count)



    sns.lineplot(semiskill_wage[(semiskill_wage['Year'] == year)]['State/UT'],semiskill_wage[(semiskill_wage['Year'] == year)]['Wages Paid per Day to Convicts (in Rs.) - Semi Skilled'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('Wages Paid per Day to Convicts (in Rs.) - Semi Skilled - 2007-2014', fontsize=15)
unskilled_wage = rehab_df.groupby(by=['Year', 'State/UT'])['Wages Paid per Day to Convicts (in Rs.) - Unskilled'].sum().reset_index().sort_values('Wages Paid per Day to Convicts (in Rs.) - Unskilled', ascending=False)



plt.figure(figsize=(20,25))

count = 1



for year in sorted(unskilled_wage.Year.unique()):

    plt.subplot(len(unskilled_wage.Year.unique()),1,count)



    sns.lineplot(unskilled_wage[(unskilled_wage['Year'] == year)]['State/UT'],unskilled_wage[(unskilled_wage['Year'] == year)]['Wages Paid per Day to Convicts (in Rs.) - Unskilled'],ci=None)

    plt.subplots_adjust(hspace=1.8)

    plt.xlabel('')

    plt.ylabel('')

    plt.title(year)

    plt.xticks(rotation=90)

    count+=1



plt.suptitle('Wages Paid per Day to Convicts (in Rs.) - Unskilled - 2007-2014', fontsize=15)