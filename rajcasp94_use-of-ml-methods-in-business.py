# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plotting and visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_questions=pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

print(df_questions)
df_mcq=pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")
print(df_mcq.head())
mcq_q8=df_mcq['Q8']

print(mcq_q8)
mcq_q8=mcq_q8[1:len(mcq_q8)]

print(mcq_q8)
mcq_q8=mcq_q8.dropna()
print(mcq_q8.drop_duplicates())
no_use='No'

not_sure='know'

exploring='exploring'

started='started'

insights='insights'

established='established'



no_use_list=[]

not_sure_list=[]

exploring_list=[]

started_list=[]

insights_list=[]

established_list=[]



for i in mcq_q8:

    if no_use in i:

        no_use_list.append(i)

    if not_sure in i:

        not_sure_list.append(i)

    if exploring in i:

        exploring_list.append(i)

    if started in i:

        started_list.append(i)

    if insights in i:

        insights_list.append(i)

    if established in i:

        established_list.append(i)

print(len(no_use_list))

print(len(not_sure_list))

print(len(exploring_list))

print(len(started_list))

print(len(established_list))

print(len(insights_list))
objects = ('No', 'Not Sure', ' Exploring', 'Started', 'Establised', 'Insights','Total')

y_pos = np.arange(len(objects))

performance = [len(no_use_list),len(not_sure_list),len(exploring_list),len(started_list),len(established_list),len(insights_list),len(mcq_q8)]



plt.bar(y_pos, performance, align='center', alpha=1)

plt.xticks(y_pos, objects)

plt.ylabel('Number of respondents')

plt.title('Number of respondents in each category')



plt.show()
df_mcq['Q8']=df_mcq['Q8'].fillna(0)

df_mcq = df_mcq.loc[df_mcq['Q8'] != 0]

print(df_mcq)
df_mcq=df_mcq[1:len(df_mcq)]

print(df_mcq)
criteria_1=[]

criteria_2=[]

criteria_3=[]

criteria_4=[]

criteria_5=[]

criteria_6=[]

for i in df_mcq['Q8']:

    criteria_1.append(no_use in i)

    criteria_2.append(not_sure in i)

    criteria_3.append(exploring in i)

    criteria_4.append(started in i)

    criteria_5.append(established in i)

    criteria_6.append(insights in i)

#print(criteria)



df_no=df_mcq[criteria_1]

df_not_sure=df_mcq[criteria_2]

df_exp=df_mcq[criteria_3]

df_start=df_mcq[criteria_4]

df_est=df_mcq[criteria_5]

df_insight=df_mcq[criteria_6]
print(len(df_no))

print(len(df_not_sure))

print(len(df_exp))

print(len(df_start))

print(len(df_est))

print(len(df_insight))
print(df_exp)
# dataframe lists

df_list=[df_exp,df_start,df_est,df_insight]

# dataframe for exploring group extracting Q6, Q7 and Q10

list_h1=['Q6','Q7','Q10']

# In order to make sure each group has equal number of unique response for all the questions, we will implement the following piece of code

for i in df_list:

    for j in list_h1:

        print('for',j)

        print(i[j].drop_duplicates())

question='Q6'

question_exp=df_exp.groupby(question).size()

question_start=df_start.groupby(question).size()

question_est=df_est.groupby(question).size()

question_insight=df_insight.groupby(question).size()

question_exp_size=[]

question_start_size=[]

question_est_size=[]

question_insight_size=[]

for i in range(len(question_exp)):

    question_exp_size.append(question_exp[i])

    question_start_size.append(question_start[i])

    question_est_size.append(question_est[i])

    question_insight_size.append(question_insight[i])

    

print(question_exp)    

print(question_start) 

print(question_est)

print(question_insight) 



first_choice=[question_exp_size[0],question_start_size[0],question_est_size[0],question_insight_size[0]]

second_choice=[question_exp_size[1],question_start_size[1],question_est_size[1],question_insight_size[1]]

third_choice=[question_exp_size[2],question_start_size[2],question_est_size[2],question_insight_size[2]]

fourth_choice=[question_exp_size[3],question_start_size[3],question_est_size[3],question_insight_size[3]]

fifth_choice=[question_exp_size[4],question_start_size[4],question_est_size[4],question_insight_size[4]]



ind = np.arange(4) 

width = 0.1

fig, ax = plt.subplots(figsize=(20,5))

rects1 = ax.bar(ind, first_choice, width, color='r')

rects2 = ax.bar(ind+width, second_choice, width, color='b')

rects3 = ax.bar(ind+(2*width), third_choice, width, color='y')

rects4 = ax.bar(ind+(3*width), fourth_choice, width, color='g')

rects5 = ax.bar(ind+(4*width), fifth_choice, width,color='black')



ax.set_ylabel('Count',fontsize=16)

ax.set_title('Count of different company-group size',fontsize=16)

ax.set_xticks(ind + width / 2)

ax.set_xticklabels(('Exploring', 'Started', 'Established', 'Insight'),fontsize=16)



ax.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0]), ('0-49(Scale 1)', '1000-9999(Scale 4)','250-999(Scale 3)','50-249(Scale 2)','>10000(Scale 5)'))



def autolabel(rects):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')

autolabel(rects1)

autolabel(rects2)

autolabel(rects3)

autolabel(rects4)

autolabel(rects5)

    



scale_1='0-49 employees'

scale_2='1000-9,999 employees'

scale_3='250-999 employees'       

scale_4='50-249 employees'        

scale_5='> 10,000 employees' 



# For exploring group

criteria_1=[] # small scale criteria

criteria_2=[] # mid scale criteria

criteria_3=[] # large scale criteria

for i in df_exp['Q6']:

    criteria_1.append(scale_1 in i)

    criteria_2.append(scale_2 in i or scale_3 in i)

    criteria_3.append(scale_4 in i or scale_5 in i)



df_exp_small_scale=df_exp[criteria_1]

df_exp_mid_scale=df_exp[criteria_2]

df_exp_large_scale=df_exp[criteria_3]

# For started group

criteria_1=[] # small scale criteria

criteria_2=[] # mid scale criteria

criteria_3=[] # large scale criteria

for i in df_start['Q6']:

    criteria_1.append(scale_1 in i)

    criteria_2.append(scale_2 in i or scale_3 in i)

    criteria_3.append(scale_4 in i or scale_5 in i)



df_start_small_scale=df_start[criteria_1]

df_start_mid_scale=df_start[criteria_2]

df_start_large_scale=df_start[criteria_3]

# For established group

criteria_1=[] # small scale criteria

criteria_2=[] # mid scale criteria

criteria_3=[] # large scale criteria

for i in df_est['Q6']:

    criteria_1.append(scale_1 in i)

    criteria_2.append(scale_2 in i or scale_3 in i)

    criteria_3.append(scale_4 in i or scale_5 in i)



df_est_small_scale=df_est[criteria_1]

df_est_mid_scale=df_est[criteria_2]

df_est_large_scale=df_est[criteria_3]

# For insight group

criteria_1=[] # small scale criteria

criteria_2=[] # mid scale criteria

criteria_3=[] # large scale criteria

for i in df_insight['Q6']:

    criteria_1.append(scale_1 in i)

    criteria_2.append(scale_2 in i or scale_3 in i)

    criteria_3.append(scale_4 in i or scale_5 in i)



df_insight_small_scale=df_insight[criteria_1]

df_insight_mid_scale=df_insight[criteria_2]

df_insight_large_scale=df_insight[criteria_3]

opt_1='0'

opt_2='1-2'

opt_3='3-4'

opt_4='5-9'

opt_5='10-14'

opt_6='15-19'

opt_7='20+'



# Function to append sizes of new group

def size_append(dataframe,question):

    data_frame_option_size=[]

    temp_small_group=[]

    temp_medium_group=[]

    temp_large_group=[]

    for i in dataframe[question]:

        if(i==opt_1 or i==opt_2 or i==opt_3):

            temp_small_group.append(i)

        if(i==opt_4 or i==opt_5):

            temp_medium_group.append(i)

        if(i==opt_6 or i == opt_7):

            temp_large_group.append(i)

    data_frame_option_size.append(len(temp_small_group))

    data_frame_option_size.append(len(temp_medium_group))

    data_frame_option_size.append(len(temp_large_group))

    return data_frame_option_size

    

           

question_exp_small_scale_size=size_append(df_exp_small_scale,'Q7')

question_exp_mid_scale_size=size_append(df_exp_mid_scale,'Q7')

question_exp_large_scale_size=size_append(df_exp_large_scale,'Q7')



question_start_small_scale_size=size_append(df_start_small_scale,'Q7')

question_start_mid_scale_size=size_append(df_start_mid_scale,'Q7')

question_start_large_scale_size=size_append(df_start_large_scale,'Q7')

        

question_est_small_scale_size=size_append(df_est_small_scale,'Q7')

question_est_mid_scale_size=size_append(df_est_mid_scale,'Q7')

question_est_large_scale_size=size_append(df_est_large_scale,'Q7')



question_insight_small_scale_size=size_append(df_insight_mid_scale,'Q7')

question_insight_mid_scale_size=size_append(df_insight_mid_scale,'Q7')

question_insight_large_scale_size=size_append(df_insight_large_scale,'Q7')



first_choice=[question_exp_small_scale_size[0],question_exp_mid_scale_size[0],question_exp_large_scale_size[0],

             question_start_small_scale_size[0],question_start_mid_scale_size[0],question_start_large_scale_size[0],

             question_est_small_scale_size[0],question_est_mid_scale_size[0],question_est_large_scale_size[0],

             question_insight_small_scale_size[0],question_insight_mid_scale_size[0],question_insight_large_scale_size[0]]



second_choice=[question_exp_small_scale_size[1],question_exp_mid_scale_size[1],question_exp_large_scale_size[1],

             question_start_small_scale_size[1],question_start_mid_scale_size[1],question_start_large_scale_size[1],

             question_est_small_scale_size[1],question_est_mid_scale_size[1],question_est_large_scale_size[1],

             question_insight_small_scale_size[1],question_insight_mid_scale_size[1],question_insight_large_scale_size[1]]



third_choice=[question_exp_small_scale_size[2],question_exp_mid_scale_size[2],question_exp_large_scale_size[2],

             question_start_small_scale_size[2],question_start_mid_scale_size[2],question_start_large_scale_size[2],

             question_est_small_scale_size[2],question_est_mid_scale_size[2],question_est_large_scale_size[2],

             question_insight_small_scale_size[2],question_insight_mid_scale_size[2],question_insight_large_scale_size[2]]



ind = np.arange(12) 

width = 0.2

fig, ax = plt.subplots(figsize=(40,5))

rects1 = ax.bar(ind, first_choice, width)

rects2 = ax.bar(ind+width, second_choice, width)

rects3 = ax.bar(ind+(2*width), third_choice, width)



ax.set_ylabel('Count',fontsize=16)

ax.set_title('Count of People responsible for DS workload across various groups',fontsize=16)

ax.set_xticks(ind + width / 2)

ax.set_xticklabels(('Exploring SS', 'Exploring MS', 'Exploring LS', 'Started SS', 'Started MS', 'Started LS',

                    'Established SS', 'Established MS', 'Established LS','Insight SS', 'Insight MS', 'Insight LS' ),fontsize=16)



ax.legend((rects1[0], rects2[0],rects3[0]), ('Small Group', 'Medium Group', 'Large Group'),fontsize=16)



def autolabel(rects):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')

autolabel(rects1)

autolabel(rects2)

autolabel(rects3)

print(df_mcq.groupby('Q10').size())
sal_1='$0-999 '

sal_2='1,000-1,999'

sal_3='2,000-2,999'

sal_4='3,000-3,999'

sal_5='4,000-4,999'

sal_6='5,000-7,499'

sal_7='7,500-9,999'

sal_8='10,000-14,999'

sal_9='15,000-19,999'

sal_10='20,000-24,999'

sal_11='25,000-29,999'

sal_12='30,000-39,999'

sal_13='40,000-49,999'

sal_14='50,000-59,999'

sal_15='60,000-69,999'

sal_16='70,000-79,999'

sal_17='80,000-89,999'

sal_18='90,000-99,999'

sal_19='100,000-124,999'

sal_20='125,000-149,999'

sal_21='150,000-199,999'

sal_22='200,000-249,999'

sal_23='250,000-299,999'

sal_24='300,000-500,000'

sal_25='> $500,000'
def size_append_new(dataframe,question):

    dataframe_option_size=[]

    very_small_scale_pay=[]

    small_scale_pay=[]

    mid_scale_pay=[]

    large_scale_pay=[]

    very_large_scale_pay=[]

    executive_pay=[]

    for i in dataframe[question]:

        if(i==sal_1 or i==sal_2 or i==sal_3 or i==sal_4 or i==sal_5 or i==sal_6 or i==sal_7 or i==sal_8 or i==sal_9 or i==sal_10):

            very_small_scale_pay.append(i)

        if(i==sal_11 or i==sal_12 or i==sal_13):

            small_scale_pay.append(i)

        if(i==sal_14 or i==sal_15 or i==sal_16 or i==sal_17):

            mid_scale_pay.append(i)

        if(i==sal_18 or i==sal_19 or i==sal_20 or i==sal_21):

            large_scale_pay.append(i)

        if(i==sal_22 or i==sal_23):

            very_large_scale_pay.append(i)

        if(i==sal_24 or i==sal_25):

            executive_pay.append(i)

    dataframe_option_size.append(len(very_small_scale_pay))  

    dataframe_option_size.append(len(small_scale_pay))

    dataframe_option_size.append(len(mid_scale_pay))

    dataframe_option_size.append(len(large_scale_pay))

    dataframe_option_size.append(len(very_large_scale_pay))

    dataframe_option_size.append(len(executive_pay))

    return dataframe_option_size

question_exp_small_scale_pay_size=size_append_new(df_exp_small_scale,'Q10')

question_exp_mid_scale_pay_size=size_append_new(df_exp_mid_scale,'Q10')

question_exp_large_scale_pay_size=size_append_new(df_exp_large_scale,'Q10')



question_start_small_scale_pay_size=size_append_new(df_start_small_scale,'Q10')

question_start_mid_scale_pay_size=size_append_new(df_start_mid_scale,'Q10')

question_start_large_scale_pay_size=size_append_new(df_start_large_scale,'Q10')



question_est_small_scale_pay_size=size_append_new(df_est_small_scale,'Q10')

question_est_mid_scale_pay_size=size_append_new(df_est_mid_scale,'Q10')

question_est_large_scale_pay_size=size_append_new(df_est_large_scale,'Q10')



question_insight_small_scale_pay_size=size_append_new(df_insight_small_scale,'Q10')

question_insight_mid_scale_pay_size=size_append_new(df_insight_mid_scale,'Q10')

question_insight_large_scale_pay_size=size_append_new(df_insight_large_scale,'Q10')



first_choice=[question_exp_small_scale_pay_size[0],question_exp_mid_scale_pay_size[0],question_exp_large_scale_pay_size[0],

             question_start_small_scale_pay_size[0],question_start_mid_scale_pay_size[0],question_start_large_scale_pay_size[0],

             question_est_small_scale_pay_size[0],question_est_mid_scale_pay_size[0],question_est_large_scale_pay_size[0],

             question_insight_small_scale_pay_size[0],question_insight_mid_scale_pay_size[0],question_insight_large_scale_pay_size[0]]



second_choice=[question_exp_small_scale_pay_size[1],question_exp_mid_scale_pay_size[1],question_exp_large_scale_pay_size[1],

             question_start_small_scale_pay_size[1],question_start_mid_scale_pay_size[1],question_start_large_scale_pay_size[1],

             question_est_small_scale_pay_size[1],question_est_mid_scale_pay_size[1],question_est_large_scale_pay_size[1],

             question_insight_small_scale_pay_size[1],question_insight_mid_scale_pay_size[1],question_insight_large_scale_pay_size[1]]



third_choice=[question_exp_small_scale_pay_size[2],question_exp_mid_scale_pay_size[2],question_exp_large_scale_pay_size[2],

             question_start_small_scale_pay_size[2],question_start_mid_scale_pay_size[2],question_start_large_scale_pay_size[2],

             question_est_small_scale_pay_size[2],question_est_mid_scale_pay_size[2],question_est_large_scale_pay_size[2],

             question_insight_small_scale_pay_size[2],question_insight_mid_scale_pay_size[2],question_insight_large_scale_pay_size[2]]



fourth_choice=[question_exp_small_scale_pay_size[3],question_exp_mid_scale_pay_size[3],question_exp_large_scale_pay_size[3],

             question_start_small_scale_pay_size[3],question_start_mid_scale_pay_size[3],question_start_large_scale_pay_size[3],

             question_est_small_scale_pay_size[3],question_est_mid_scale_pay_size[3],question_est_large_scale_pay_size[3],

             question_insight_small_scale_pay_size[3],question_insight_mid_scale_pay_size[3],question_insight_large_scale_pay_size[3]]



fifth_choice=[question_exp_small_scale_pay_size[4],question_exp_mid_scale_pay_size[4],question_exp_large_scale_pay_size[4],

             question_start_small_scale_pay_size[4],question_start_mid_scale_pay_size[4],question_start_large_scale_pay_size[4],

             question_est_small_scale_pay_size[4],question_est_mid_scale_pay_size[4],question_est_large_scale_pay_size[4],

             question_insight_small_scale_pay_size[4],question_insight_mid_scale_pay_size[4],question_insight_large_scale_pay_size[4]]



sixth_choice=[question_exp_small_scale_pay_size[5],question_exp_mid_scale_pay_size[5],question_exp_large_scale_pay_size[5],

             question_start_small_scale_pay_size[5],question_start_mid_scale_pay_size[5],question_start_large_scale_pay_size[5],

             question_est_small_scale_pay_size[5],question_est_mid_scale_pay_size[5],question_est_large_scale_pay_size[5],

             question_insight_small_scale_pay_size[5],question_insight_mid_scale_pay_size[5],question_insight_large_scale_pay_size[5]]





ind = np.arange(12) 

width = 0.1

fig, ax = plt.subplots(figsize=(40,10))

rects1 = ax.bar(ind, first_choice, width)

rects2 = ax.bar(ind+width, second_choice, width)

rects3 = ax.bar(ind+(2*width), third_choice, width)

rects4 = ax.bar(ind+(3*width), fourth_choice, width)

rects5 = ax.bar(ind+(4*width), fifth_choice, width)

rects6 = ax.bar(ind+(5*width), sixth_choice, width)



ax.set_ylabel('Count',fontsize=16)

ax.set_title('Count of Peoples compensation across different groups',fontsize=16)

ax.set_xticks(ind + width / 2)

ax.set_xticklabels(('Exploring SS', 'Exploring MS', 'Exploring LS', 'Started SS', 'Started MS', 'Started LS',

                    'Established SS', 'Established MS', 'Established LS','Insight SS', 'Insight MS', 'Insight LS' ),fontsize=16)



ax.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0]), ('Very Small Scale Pay', 'Small Scale Pay', 'Mid Scale Pay',

                                                                           'Large Scale Pay','Very Large Scale Pay','Executive Pay'),fontsize=16)



def autolabel(rects):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom',fontsize=16)

autolabel(rects1)

autolabel(rects2)

autolabel(rects3)

autolabel(rects4)

autolabel(rects5)

autolabel(rects6)







# For exploring group

criteria_pay_1=[] # mid scale pay criteria

criteria_pay_2=[] # large scale pay criteria

criteria_pay_3=[] # very large scale pay criteria

for i in df_exp['Q10']:

    criteria_pay_1.append(sal_15 in str(i) or sal_16 in str(i) or sal_17 in str(i))

    criteria_pay_2.append(sal_18 in str(i) or sal_19 in str(i) or sal_20 in str(i) or sal_21 in str(i))

    criteria_pay_3.append(sal_22 in str(i) or sal_23 in str(i) or sal_24 in str(i) or sal_25 in str(i))

df_exp_mid_pay=df_exp[criteria_pay_1]

df_exp_large_pay=df_exp[criteria_pay_2]

df_exp_very_large_pay=df_exp[criteria_pay_3]



# For started group

criteria_pay_1=[] # mid scale pay criteria

criteria_pay_2=[] # large scale pay criteria

criteria_pay_3=[] # very large scale pay criteria

for i in df_start['Q10']:

    criteria_pay_1.append(sal_15 in str(i) or sal_16 in str(i) or sal_17 in str(i))

    criteria_pay_2.append(sal_18 in str(i) or sal_19 in str(i) or sal_20 in str(i) or sal_21 in str(i))

    criteria_pay_3.append(sal_22 in str(i) or sal_23 in str(i) or sal_24 in str(i) or sal_25 in str(i))

df_start_mid_pay=df_start[criteria_pay_1]

df_start_large_pay=df_start[criteria_pay_2]

df_start_very_large_pay=df_start[criteria_pay_3]



# For established group

criteria_pay_1=[] # mid scale pay criteria

criteria_pay_2=[] # large scale pay criteria

criteria_pay_3=[] # very large scale pay criteria

for i in df_est['Q10']:

    criteria_pay_1.append(sal_15 in str(i) or sal_16 in str(i) or sal_17 in str(i))

    criteria_pay_2.append(sal_18 in str(i) or sal_19 in str(i) or sal_20 in str(i) or sal_21 in str(i))

    criteria_pay_3.append(sal_22 in str(i) or sal_23 in str(i) or sal_24 in str(i) or sal_25 in str(i))

df_est_mid_pay=df_est[criteria_pay_1]

df_est_large_pay=df_est[criteria_pay_2]

df_est_very_large_pay=df_est[criteria_pay_3]



# For insight group

criteria_pay_1=[] # mid scale pay criteria

criteria_pay_2=[] # large scale pay criteria

criteria_pay_3=[] # very large scale pay criteria

for i in df_insight['Q10']:

    criteria_pay_1.append(sal_15 in str(i) or sal_16 in str(i) or sal_17 in str(i))

    criteria_pay_2.append(sal_18 in str(i) or sal_19 in str(i) or sal_20 in str(i) or sal_21 in str(i))

    criteria_pay_3.append(sal_22 in str(i) or sal_23 in str(i) or sal_24 in str(i) or sal_25 in str(i))

df_insight_mid_pay=df_insight[criteria_pay_1]

df_insight_large_pay=df_insight[criteria_pay_2]

df_insight_very_large_pay=df_insight[criteria_pay_3]
print(df_mcq.groupby('Q5').size())
choice_1='Business Analyst'

choice_2='DBA/Database Engineer'

choice_3='Data Analyst'

choice_4='Data Engineer'

choice_5='Data Scientist'

choice_6='other'

choice_7='Product/Project Manager'

choice_8='Research Scientist'

choice_9='Software Engineer'

choice_10='Statistician'
def size_pay_append(dataframe,question):

    choice_1_list=[]

    choice_2_list=[]

    choice_3_list=[]

    choice_4_list=[]

    choice_5_list=[]

    choice_7_list=[]

    choice_8_list=[]

    choice_9_list=[]

    choice_10_list=[]

    choice_list=[]

    for choice in dataframe[question]:

        if(choice==choice_1):

            choice_1_list.append(choice)

        if(choice==choice_2):

            choice_2_list.append(choice)

        if(choice==choice_3):

            choice_3_list.append(choice)

        if(choice==choice_4):

            choice_4_list.append(choice)

        if(choice==choice_5):

            choice_5_list.append(choice)

        if(choice==choice_7):

            choice_7_list.append(choice)

        if(choice==choice_8):

            choice_8_list.append(choice)

        if(choice==choice_9):

            choice_9_list.append(choice)

        if(choice==choice_10):

            choice_10_list.append(choice)

    choice_list.append(len(choice_1_list))

    choice_list.append(len(choice_2_list))

    choice_list.append(len(choice_3_list))

    choice_list.append(len(choice_4_list))

    choice_list.append(len(choice_5_list))

    choice_list.append(len(choice_7_list))

    choice_list.append(len(choice_8_list))

    choice_list.append(len(choice_9_list))

    choice_list.append(len(choice_10_list))

    return choice_list



question_exp_mid_pay=size_pay_append(df_exp_mid_pay,'Q5')

question_exp_large_pay=size_pay_append(df_exp_large_pay,'Q5')

question_exp_very_large_pay=size_pay_append(df_exp_very_large_pay,'Q5')



question_start_mid_pay=size_pay_append(df_start_mid_pay,'Q5')

question_start_large_pay=size_pay_append(df_start_large_pay,'Q5')

question_start_very_large_pay=size_pay_append(df_start_very_large_pay,'Q5')



question_est_mid_pay=size_pay_append(df_est_mid_pay,'Q5')

question_est_large_pay=size_pay_append(df_est_large_pay,'Q5')

question_est_very_large_pay=size_pay_append(df_est_very_large_pay,'Q5')



question_insight_mid_pay=size_pay_append(df_insight_mid_pay,'Q5')

question_insight_large_pay=size_pay_append(df_insight_large_pay,'Q5')

question_insight_very_large_pay=size_pay_append(df_insight_very_large_pay,'Q5')
labels = 'BA', 'DBA', 'DA', 'DE','DS','PM','RS','SE','St'



fig, ax = plt.subplots(2,2,figsize=(20,20))



ax[0,0].pie(question_exp_mid_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[0,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[0, 0].set_title('Job position distribution in exploring group for mid pay scale',loc='right',fontsize=16)



ax[0,1].pie(question_start_mid_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[0,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[0,1].set_title('Job position distribution in started group for mid pay scale',loc='right',fontsize=16)



ax[1,0].pie(question_est_mid_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[1,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1,0].set_title('Job position distribution in established group for mid pay scale',loc='right',fontsize=16)



ax[1,1].pie(question_insight_mid_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[1,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1,1].set_title('Job position distribution in insight group for mid pay scale',loc='right',fontsize=16)



plt.show()
labels = 'BA', 'DBA', 'DA', 'DE','DS','PM','RS','SE','St'



fig, ax = plt.subplots(2,2,figsize=(20,20))



ax[0,0].pie(question_exp_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[0,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[0, 0].set_title('Job position distribution in exploring group for large pay scale',loc='right',fontsize=16)



ax[0,1].pie(question_start_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[0,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[0,1].set_title('Job position distribution in started group for large pay scale',loc='right',fontsize=16)



ax[1,0].pie(question_est_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[1,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1,0].set_title('Job position distribution in established group for large pay scale',loc='right',fontsize=16)



ax[1,1].pie(question_insight_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[1,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1,1].set_title('Job position distribution in insight group for large pay scale',loc='right',fontsize=16)
labels = 'BA', 'DBA', 'DA', 'DE','DS','PM','RS','SE','St'



fig, ax = plt.subplots(2,2,figsize=(20,20))



ax[0,0].pie(question_exp_very_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[0,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[0, 0].set_title('Job position distribution in exploring group for very large pay scale',loc='right',fontsize=16)



ax[0,1].pie(question_start_very_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[0,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[0,1].set_title('Job position distribution in started group for very large pay scale',loc='right',fontsize=16)



ax[1,0].pie(question_est_very_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[1,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1,0].set_title('Job position distribution in established group for very large pay scale',loc='right',fontsize=16)



ax[1,1].pie(question_insight_very_large_pay, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90,textprops={'fontsize': 11})

ax[1,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1,1].set_title('Job position distribution in insight group for very large pay scale',loc='right',fontsize=16)