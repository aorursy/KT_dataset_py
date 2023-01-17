# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

#Import necessary libraries



import seaborn as sns

import matplotlib.pyplot as plt



# Set plot stlye to ggplot

plt.style.use('ggplot')

# Import the dataset



df=pd.read_csv('../input/degrees-that-pay-back.csv')



print(df.head())

print(df.info())

df.reset_index()



# Rename cols and remove the dollar signs, then convert certain features to numeric ones

df.columns=['major','starter_med','mid_med','percentage', 'mid_10','mid_25','mid_75','mid_90']

           

to_num_cols=['starter_med','mid_med','mid_10','mid_25','mid_75','mid_90']



df[to_num_cols]=df[to_num_cols].replace({'\$': '', ',': ''}, regex=True)



for x in to_num_cols:

    df[x]=pd.to_numeric(df[x]);

print (df.info())



#Start exploring visually!

df.sort_values(by = 'starter_med', ascending = False, inplace=True)



sns.set_style('dark')

fig, ax = plt.subplots()

fig.set_size_inches(22, 12)

majors_distr=sns.barplot(x=df['major'], y=df['starter_med'], data=df)

for item in majors_distr.get_xticklabels():

    item.set_rotation(90)

majors_distr.axes.set_title('Entry salaries by profession',fontsize=18)

majors_distr.set_xlabel("Major degree(Bachelor)",fontsize=14)

majors_distr.set_ylabel("Salary ($)",fontsize=14)

majors_distr.tick_params(labelsize=12)

plt.show();



#Now let's see it in a comparison to middle of the career salaries



sns.set_style('dark')

fig, ax = plt.subplots()

fig.set_size_inches(22, 12)



_=sns.barplot(x=df['major'], y=df['mid_med'], data=df, color='orange')

for item in _.get_xticklabels():

    item.set_rotation(90)

_.axes.set_title('Entry and middle salaries',fontsize=18)

_.tick_params(labelsize=12)



plot2 = sns.barplot(x = 'major', y = 'starter_med', data=df, color = 'green')

plot2.set_xlabel('Major degree(Bachelor)')

plot2.set_ylabel('Salary($)')

handles, labels = ax.get_legend_handles_labels()

plt.legend(handles, labels)

plt.show()



#See who gets the highest salary after college and in the middle of the career.



highest_salary=max(df['starter_med'])

highest_salary_major=df['major'].loc[df.starter_med == highest_salary].values[0]



print('{} is the job type with the highest entry salary, which is {} dollars (The value is a median).'.format(highest_salary_major, int(highest_salary)))



highest_salary_mid=max(df['mid_med'])

highest_salary_major_mid=df['major'].loc[df.mid_med == highest_salary_mid].values[0]



print('{} is the job type with highest middle career salary, which is {} dollars (The value is a median).'.format(highest_salary_major_mid, int(highest_salary_mid)))



#Now, inspect the other table and see which schools have the best prospects.

# First, import the csv and make a few preprocessing steps (rename columns, drop the unnecessary ones, remove dollar signs and commas, convert salaries to numeric)



df1=pd.read_csv('../input/salaries-by-college-type.csv')



df1=df1.drop(['Mid-Career 10th Percentile Salary', 'Mid-Career 25th Percentile Salary','Mid-Career 75th Percentile Salary','Mid-Career 90th Percentile Salary'], axis=1)

df1.reset_index()



df1.columns=['school','school_type','school_start_med','school_mid_med']



to_num_cols_2=['school_start_med','school_mid_med']



df1[to_num_cols_2]=df1[to_num_cols_2].replace({'\$': '', ',': ''}, regex=True)



for x in to_num_cols_2:

    df1[x]=pd.to_numeric(df1[x]);

print (df1.info())



df1_start_top=df1.nlargest(n=20,columns='school_start_med')

df1_mid_top=df1.nlargest(n=20,columns='school_mid_med')



sns.set_style('dark')

fig, ax = plt.subplots()

fig.set_size_inches(14, 6)



school_distr=sns.barplot(x=df1_start_top['school'],y=df1_start_top['school_start_med'],data=df1_start_top)

for item in school_distr.get_xticklabels():

    item.set_rotation(90)

school_distr.axes.set_title('Entry salaries by school',fontsize=18)

school_distr.set_xlabel('School name',fontsize=14)

school_distr.set_ylabel('Salary ($)',fontsize=14)

school_distr.tick_params(labelsize=12)

plt.show();



sns.set_style('dark')

fig, ax = plt.subplots()

fig.set_size_inches(14, 6)



school_distr2=sns.barplot(x=df1_mid_top['school'],y=df1_mid_top['school_mid_med'],data=df1_mid_top)

for item in school_distr2.get_xticklabels():

    item.set_rotation(90)

school_distr2.axes.set_title('Mid-Career salaries by school',fontsize=18)

school_distr2.set_xlabel('School name',fontsize=14)

school_distr2.set_ylabel('Salary ($)',fontsize=14)

school_distr2.tick_params(labelsize=12)

plt.show();



print('It is clearly visible that fresh graduates get the highest salaries after earning technical degrees, however this changes with time and the Ivy League Univerities are taking the lead.')



#Let's look at the top 5 technical and Ivy League univerities in terms of financial outlook.



df1_toptech=df1.sort_values(by=['school_mid_med'],ascending=False)

df1_toptech=df1_toptech.loc[df1_toptech['school_type']=='Engineering']



#There is 8 Ivy League Unis, so let's make them equal.



df1_toptech=df1_toptech.nlargest(n=8, columns='school_mid_med')



df1_topIvy=df1.sort_values(by=['school_mid_med'],ascending=False)

df1_topIvy=df1_topIvy.loc[df1_topIvy['school_type']=='Ivy League']



print(df1_toptech.head())

print(df1_topIvy.head())



df1_top= pd.concat([df1_toptech, df1_topIvy], ignore_index=True)

df1_top=df1_top.sort_values(by=['school_mid_med'], ascending=False)





sns.set_style('dark')

fig, ax = plt.subplots()

fig.set_size_inches(14, 6)



top_distr=sns.barplot(x=df1_top['school'],y=df1_top['school_mid_med'], hue='school_type' , data=df1_top)

for item in top_distr.get_xticklabels():

    item.set_rotation(90)

top_distr.axes.set_title('Mid-Career salaries by top schools and school type',fontsize=18)

top_distr.set_xlabel('School name',fontsize=14)

top_distr.set_ylabel('Salary ($)',fontsize=14)

top_distr.tick_params(labelsize=12)

plt.show();





#Do some stats. First, drop the unnecessary cols.



TopTech=df1_toptech.drop(['school_type'], axis=1)

IvyLeague=df1_topIvy.drop(['school_type'], axis=1)



#Look at the descriptives!

print(TopTech.describe())

print(IvyLeague.describe())



#It is clear that TopTech school salary mean is higher for fresh grads, but the mid-career salary mean is higher for the Ivy League students

# See if the difference is significant or not. T test would be cool, but the distribution has nothing to do with the normal one, so choose an alternative.



import scipy.stats



u1,p1=scipy.stats.mannwhitneyu(TopTech['school_start_med'], IvyLeague['school_start_med'])



u2,p2=scipy.stats.mannwhitneyu(TopTech['school_mid_med'], IvyLeague['school_mid_med'])



print('The Mann-Whitney U value is {} and the significance level is {} (as for starting salary)'.format(u1, p1))



print('The Mann-Whitney U value is {} and the significance level is {} (as for mid-career salary)'.format(u2, p2))



print('We can see that (strictly statistically), TopTech schools really have better salaries in the beginning, and after that (although the Ivy League takes the lead), the latter differences remain statistically insignificant.')