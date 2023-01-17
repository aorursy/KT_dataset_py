import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt
df=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

df.head()
df['gender'].value_counts()
#summary of the data

df.describe()
#na values

df.isna().count()
#mean score of maths only male 

filt_math=df['gender']=='male'

male_maths=df.loc[filt_math,'math score'].mean()

male_maths
#mean score in mathc of only female 

filtfmath=df['gender']=='female'

female_maths=df.loc[filtfmath,'math score'].mean()

female_maths
df['parental level of education'].value_counts()

df['test preparation course'].value_counts()
#male scores categorization



filt_completed= df[(df['gender']=='male') & (df['test preparation course']=='completed')]



male_completed=df.loc[filt_completed['math score']].mean()

male_completed
#female scores categorization



filt_fcompleted= df[(df['gender']=='female') & (df['test preparation course']=='completed')]



female_completed=df.loc[filt_fcompleted['math score']].mean()

female_completed
df.isnull().sum()

#Suppose we take the passing marks equals to 40

#let us check how many students are pass in all the subject



pass_score=df[(df['math score']>=40) & (df['reading score']>=40) & (df['writing score']>=40)]



new_df=pd.DataFrame(pass_score)



new_df['All_Clear']=new_df['math score']+new_df['reading score']+new_df['writing score']



new_df.count()



# people got above 40 in all subject are 949 out of 1000

#percentage scored by each students

new_df['percentage']=(new_df['All_Clear'])/3

median=new_df['math score'].median()

median
median_per=new_df['percentage'].median()

median_per
#visualize students score in math and show how many of them are below mediana and above median



plt.style.use('fivethirtyeight')



math_s = new_df['math score']



bins=[40,50,60,70,80,90,100]



plt.hist(math_s,bins = bins,edgecolor='black')



median=67, #median score in maths 

color='#fc4f30'



plt.axvline(median,color=color,label='Med_score',linewidth=2)



plt.legend()

plt.title('Marks Obtain in Maths')

plt.ylabel('Count')

plt.xlabel('Marks')

plt.tight_layout()

plt.show()

            
#Students percentage who have scored above 40 in all the subjects



plt.style.use('fivethirtyeight')



percent = new_df['percentage']



bins=[40,50,60,70,80,90,100]



plt.hist(percent,bins = bins,edgecolor='black')



median_per=67, #median score in maths 



plt.axvline(median_per,color='y',label='Med_per',linewidth=2)



plt.legend()

plt.title('OverAll Percentage')

plt.ylabel('Count')

plt.xlabel('Marks')

plt.tight_layout()

plt.show()
def per_category(percentage):

    if ( percentage >= 80 ):

        return 'A'

    if ( percentage >= 70):

        return 'B'

    if ( percentage >= 60):

        return 'C'

    if ( percentage >= 50):

        return 'D'

    else: 

        return 'E'



new_df['Grades']=new_df.apply(lambda x: per_category(x['percentage']),axis=1)



import seaborn as sns

sns.countplot(x="Grades", data = new_df, order=['A','B','C','D','E'],  palette="muted")

plt.show()




p = sns.countplot(x='parental level of education', data = new_df, hue='Grades', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 



plt.legend(loc='upper left',prop={'size': 8})
