# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #For plotting charts

import warnings 



warnings.filterwarnings("ignore", category=DeprecationWarning) #To ignore warning message

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)





hdi=pd.read_csv("../input/kaggle-2019-survey-add-on-data/HDI.csv")

hdi.describe()



mcr=pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv",low_memory=False)

mcr=mcr.drop([0])



questions=pd.read_csv("../input/kaggle-2019-survey-add-on-data/QuestionDetails.csv")



mcr['Q3'].unique()



mcr['Q3'].replace({'United States of America':'United States',

                   'Russia':'Russian Federation', 

                   'South Korea':'Korea (Republic of)',

                   'United Kingdom of Great Britain and Northern Ireland':'United Kingdom',

                   'Czech Republic':'Czechia',

                   'Hong Kong (S.A.R.)':'Hong Kong, China (SAR)',

                   'Republic of Korea':'Korea (Republic of)',

                   'Iran, Islamic Republic of...':'Iran (Islamic Republic of)'},

        inplace=True)
mcr_hdi=pd.merge(mcr, hdi, how='left', left_on='Q3', right_on='Country')
questions
#Set this to true to display the output of different methods that we would invoke for analysis

display_output_df=False

#Figure size

figsize=(10,5)
#This method helps to plot a chart for the Simple Choice questions

#That inturn helps to analyze the data.  It also groups the data

def SimpleChoicePlot(qn_col,qn_desc=None,inp_df=mcr_hdi,figsize=figsize):

    if qn_desc == None:

        qn_desc=questions[questions['QN_Number']==qn_col]['QN_Short_Desc'].values[0]

    temp_df=inp_df.groupby(['HDI_Category',qn_col])[qn_col].count().rename('Count')

    temp_df=((temp_df/temp_df.groupby(level=0).sum())*100).to_frame().unstack()

    #Flatten the Multi Index created

    temp_df.columns=temp_df.columns.to_flat_index()

    #Make column names without the 'Count' Name. Need to display only various choice values

    temp_df.columns=[item[1] for item in temp_df.columns]

    temp_df.plot(kind='bar',stacked=True,figsize=figsize).legend(bbox_to_anchor=(1,1))

    #plt.legend(loc='center right')

    plt.title(qn_desc)

    plt.show()

    

    #Change the index name as follows:

    temp_df.index.name='Values Given'

    if display_output_df:

        print("Answer values distribution is given below across all HDI Categories:")

        print(temp_df.T)

    return temp_df
#This method helps to plot a chart for the Multi Choice questions

#That inturn helps to analyze the data.  It also groups the data

def MultiChoicePlot(qn_col_abr,figsize=figsize):

    #Extract question column names and descriptions based on question column abbrevation

    qn_name_list=questions[questions['QN_Number'].str.contains(qn_col_abr)]['QN_Number'].values

    qn_desc_list=questions[questions['QN_Number'].str.contains(qn_col_abr)]['QN_Short_Desc'].values

    qn_choices=questions[questions['QN_Number'].str.contains(qn_col_abr)]['ShortenChoices'].values

    cur_qn_df=pd.DataFrame(columns=['HDI_Category','Choice'])    

    for itn in range(0,len(qn_name_list)):

        qn_col=qn_name_list[itn]

        qn_cur_choice=qn_choices[itn]

        temp_df=mcr_hdi[['HDI_Category',qn_col]]

        #Populate shorten description

        temp_df.loc[~temp_df[qn_col].isnull(),qn_col]=qn_cur_choice

        temp_df.columns=['HDI_Category','Choice']

        cur_qn_df=cur_qn_df.append(temp_df)



    #Remove all Nulls (That is whereever people have not entered)

    cur_qn_df=cur_qn_df[~cur_qn_df['Choice'].isnull()]

    temp_df=cur_qn_df.groupby(['HDI_Category','Choice'])['Choice'].count().rename('Count')

    temp_df=((temp_df/temp_df.groupby(level=0).sum())*100).to_frame().unstack()

    #Flatten the Multi Index created

    temp_df.columns=temp_df.columns.to_flat_index()

    #Make column names without the 'Count' Name. Need to display only various choice values

    temp_df.columns=[item[1] for item in temp_df.columns]

    temp_df.plot(kind='bar',stacked=True,figsize=figsize).legend(bbox_to_anchor=(1,1))

    plt.title(qn_desc_list[0])

    plt.show()

    

    #Change the index name as follows:

    temp_df.index.name='Choices Given'

    if display_output_df:

        print("Answer choices distribution is given below across all HDI Categories:")

        print(temp_df.T)

    

    #List down how many choices are given by people. As its multi choice people might have chosen more than one

    qn_summary_cols=questions[(questions['QN_Number'].str.contains(qn_col_abr)) &

                           (questions['ShortenChoices']!='None')]['QN_Number'].values

    

    QSummaryColName=qn_col_abr+'_Summary'

    mcr_hdi[QSummaryColName]=len(qn_summary_cols)-mcr_hdi[qn_summary_cols].isnull().sum(axis=1)

    

    #Combined Question plot

    SimpleChoicePlot(QSummaryColName,qn_desc_list[0]+' - # Of Choices Chosen')

    return temp_df
#This method helps to plot a chart for the Simple Choice questions against another question

#That inturn helps to analyze the data.  It also groups the data

def GroupSimpleChoicePlot(qn_col,qn_grp,qn_desc,grp_lvl=[0,1],inp_df=mcr_hdi):

    #If multiple qn_cols are passed then pass grp_lvl as [0,1,2] etc    

    group_cols=['HDI_Category']

    for gc in qn_col:

        group_cols.append(gc)

    group_cols.append(qn_grp)

       

    temp_df=inp_df.groupby(group_cols)[qn_grp].count().rename('Count')

    temp_df=((temp_df/temp_df.groupby(level=grp_lvl).sum())*100).to_frame().unstack()

    #Flatten the Multi Index created

    temp_df.columns=temp_df.columns.to_flat_index()

    #Make column names without the 'Count' Name. Need to display only various choice values

    temp_df.columns=[item[1] for item in temp_df.columns]

    temp_df.plot(kind='bar',stacked=True,figsize=figsize).legend(bbox_to_anchor=(1,1))

    #plt.legend(loc='center right')

    plt.title(qn_desc)

    plt.show()

    

    #Change the index name as follows:

    temp_df.index.name='Values Given'

    

    if display_output_df:

        print("Answer values distribution is given below across all HDI Categories:")

        print(temp_df)

    return temp_df
#This method helps to plot a chart for the Multi Choice questions against another question

#That inturn helps to analyze the data.  It also groups the data

def GroupMultiChoicePlot(qn_col_abr,qn_grp,qn_addnl_desc,grp_lvl=[0,1],inp_df=mcr_hdi):

    #Extract question column names and descriptions based on question column abbrevation

    qn_name_list=questions[questions['QN_Number'].str.contains(qn_col_abr)]['QN_Number'].values

    qn_desc_list=questions[questions['QN_Number'].str.contains(qn_col_abr)]['QN_Short_Desc'].values

    qn_choices=questions[questions['QN_Number'].str.contains(qn_col_abr)]['ShortenChoices'].values



    all_grp_cols=['HDI_Category','Choice']+qn_grp

    cur_qn_df=pd.DataFrame(columns=all_grp_cols)    

    for itn in range(0,len(qn_name_list)):

        qn_col=qn_name_list[itn]

        qn_cur_choice=qn_choices[itn]

        temp_df=inp_df[['HDI_Category',qn_col]+qn_grp]

        #Populate shorten description

        temp_df.loc[~temp_df[qn_col].isnull(),qn_col]=qn_cur_choice

        temp_df.columns=all_grp_cols

        cur_qn_df=cur_qn_df.append(temp_df)



    #Remove all Nulls (That is whereever people have not entered)

    cur_qn_df=cur_qn_df[~cur_qn_df['Choice'].isnull()]

    temp_df=cur_qn_df.groupby(all_grp_cols)['Choice'].count().rename('Count')

    temp_df_grp=((temp_df/temp_df.groupby(level=grp_lvl).sum())*100).to_frame().unstack()

    #Flatten the Multi Index created

    temp_df_grp.columns=temp_df_grp.columns.to_flat_index()

    #Make column names without the 'Count' Name. Need to display only various choice values

    temp_df_grp.columns=[item[1] for item in temp_df_grp.columns]

    temp_df_grp.plot(kind='bar',stacked=True,figsize=figsize).legend(bbox_to_anchor=(1,1))

    plt.title(qn_desc_list[0] + ' cum ' + qn_addnl_desc)

    plt.show()

    

    #Change the index name as follows:

    temp_df_grp.index.name='Choices Given'

    

    if display_output_df:

        print("Answer choices distribution is given below across all HDI Categories:")

        print(temp_df_grp)

    return temp_df,temp_df_grp
#This method helps to summarize and look at the Grouped output

#Eg:- Get ML Tools usage pattern (None vs ML Tools Usage) by Gender 

#choiceval - MLToolsUsed

def SummarizeResult(temp_grp_df,choiceval,agg_lvl=[0,1]):

    temp_grp_df1=temp_grp_df.reset_index()

    temp_grp_df1['ModifiedChoice']='No'+choiceval

    temp_grp_df1.loc[temp_grp_df1['Choice']!='None','ModifiedChoice']=choiceval

    temp_grp_df1=temp_grp_df1.groupby(['HDI_Category',col_name,'ModifiedChoice'])['Count'].sum().rename('Aggregated')

    temp_grp_df1=((temp_grp_df1/temp_grp_df1.groupby(level=agg_lvl).sum())*100).unstack()

    print(temp_grp_df1)

    return temp_grp_df1
CategorizedCountriesCount=len(mcr_hdi)-mcr_hdi['HDI_Category'].isnull().sum()

((mcr_hdi['HDI_Category'].value_counts())/CategorizedCountriesCount).plot(kind='pie',figsize=figsize,autopct='%1.1f%%',title='% of respondents across all HDI categories').legend(bbox_to_anchor=(1,1))
temp=mcr_hdi[['HDI_Category','Country']].drop_duplicates().groupby(['HDI_Category'])['Country'].count().sort_values(ascending=False)

temp.plot(kind='barh',figsize=figsize).legend(bbox_to_anchor=(1,1))

temp


temp_df=SimpleChoicePlot('Q1')

#Lets take only 18-21 age range people and check their roles. 

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q1'].isin(['18-21'])]

temp_df=GroupSimpleChoicePlot(['Q1'],'Q5','18-21 Age cum Current Role  Vs HDI',inp_df=mcr_hdi_filtered)
#Let us take only 50+ age range people and check their roles

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q1'].isin(['50-54','55-59','60-69','70+'])]

temp_df=GroupSimpleChoicePlot(['Q5'],'Q1','50+ Age cum Current Role  Vs HDI',inp_df=mcr_hdi_filtered)
#Take only Male and Female records alone

mcr_hdi_m_f_alone=mcr_hdi[mcr_hdi['Q2'].isin(['Male','Female'])]

mcr_hdi_m_f_alone['Q2'].value_counts()/len(mcr_hdi_m_f_alone)

temp_df=SimpleChoicePlot('Q2',inp_df=mcr_hdi_m_f_alone)
temp_df=SimpleChoicePlot('Q3')

temp_df=GroupSimpleChoicePlot(['Country'],'Q2','Gender vs Country',inp_df=mcr_hdi_m_f_alone)

temp_df[temp_df['Female']>=16.59].sort_values(by='Female')
temp_df[temp_df['Female']>=16.59].groupby(level=0)['Female'].count()
temp_df=SimpleChoicePlot('Q4')
mcr_hdi.Q4.unique()

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q4'].isin(['Doctoral degree'])]

temp_df=GroupSimpleChoicePlot(['Q4'],'Q5','Doctoral degree cum Current Role  Vs HDI',inp_df=mcr_hdi_filtered)



temp_df=GroupSimpleChoicePlot(['Q4'],'Q10','Doctoral degree cum Salary Vs HDI',inp_df=mcr_hdi_filtered)
mcr_hdi_filtered=mcr_hdi_filtered[mcr_hdi_filtered['Q10']=='$0-999']

temp_df=GroupSimpleChoicePlot(['Q4'],'Q5','Doctoral degree with 0-999 USD salary cum Role Vs HDI',inp_df=mcr_hdi_filtered)

temp_df=GroupSimpleChoicePlot(['Q4'],'Q6','Doctoral degree with 0-999 USD salary cum Org Size Vs HDI',inp_df=mcr_hdi_filtered)
#Lets us look at their roles and salary of the Master’s degree holders.

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q4'].isin(['Master’s degree'])]

temp_df=GroupSimpleChoicePlot(['Q4'],'Q5','Master’s degree cum Current Role  Vs HDI',inp_df=mcr_hdi_filtered)

temp_df=GroupSimpleChoicePlot(['Q4'],'Q10','Master\'s degree with 0-999 USD salary cum Role Vs HDI',inp_df=mcr_hdi_filtered)
mcr_hdi_filtered=mcr_hdi_filtered[mcr_hdi_filtered['Q10']=='$0-999']

temp_df=GroupSimpleChoicePlot(['Q4'],'Q5','Master’s degree with 0-999 USD salary cum Role Vs HDI',inp_df=mcr_hdi_filtered)

temp_df=GroupSimpleChoicePlot(['Q4'],'Q6','Master\'s  degree with 0-999 USD salary cum Org Size Vs HDI',inp_df=mcr_hdi_filtered)



#Let us compare education of Female vs Male

temp_df=GroupSimpleChoicePlot(['Q2'],'Q4','Gender cum Education Vs HDI',inp_df=mcr_hdi_m_f_alone)

temp_df=SimpleChoicePlot('Q5')
temp_df=SimpleChoicePlot('Q6')
temp_df=SimpleChoicePlot('Q7')

temp_df=SimpleChoicePlot('Q8')
temp_df=SimpleChoicePlot('Q10')

#Let us compare salary of Female vs Male

temp_df=GroupSimpleChoicePlot(['Q2'],'Q10','Gender cum Salary Vs HDI',inp_df=mcr_hdi_m_f_alone)

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()
temp_df=SimpleChoicePlot('Q11')

temp_df=SimpleChoicePlot('Q14')
mcr_hdi.Q14.unique()

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q14'].isin(['Basic statistical software (Microsoft Excel, Google Sheets, etc.)'])]



temp_df=GroupSimpleChoicePlot(['Q14'],'Q5','Basic statistical software  cum Current Role Vs HDI',inp_df=mcr_hdi_filtered)

#Filter only statisticians

mcr_hdi.Q5.unique()

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q5'].isin(['Statistician'])]

temp_df=GroupSimpleChoicePlot(['Q5'],'Q14','Basic statistical software  cum Statistician Vs HDI',inp_df=mcr_hdi_filtered)
temp_df=SimpleChoicePlot('Q15')



#Lets check age of people who has written code <=2 years

mcr_hdi.Q15.unique()

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q15'].isin(['1-2 years','< 1 years'])]

mcr_hdi_filtered['Q15']='<=2 years'

temp_df=GroupSimpleChoicePlot(['Q15'],'Q1','Age cum Code Writting years for analysis Vs HDI',inp_df=mcr_hdi_filtered)



#Lets check role of people who have never written code for data analysis

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q15'].isin(['I have never written code'])]

temp_df=GroupSimpleChoicePlot(['Q15'],'Q5','Role cum Code Writting years for analysis Vs HDI',inp_df=mcr_hdi_filtered)



#Lets look at the tools used by them

mcr_hdi_filtered=mcr_hdi_filtered[mcr_hdi_filtered['Q5'].isin(['Student'])]

temp_df=GroupSimpleChoicePlot(['Q15'],'Q14','Tools cum Code Writting years for analysis by students Vs HDI',inp_df=mcr_hdi_filtered)
temp_df=SimpleChoicePlot('Q19')



temp_df=MultiChoicePlot('Q18_Part')



#Python and SQL vs their roles

col_name='Q18_Part_1' #Python

title='Pythong Lang cum Current Role Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q5',title,inp_df=mcr_hdi_filtered)

temp_df.T



col_name='Q18_Part_3' #SQL

title='SQL cum Current Role Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q5',title,inp_df=mcr_hdi_filtered)

temp_df.T



#Python and SQL vs their roles

col_name='Q18_Part_1' #Python

title='Pythong Lang cum Salary Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q10',title,inp_df=mcr_hdi_filtered)

temp_df.T

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()





col_name='Q18_Part_3' #SQL

title='SQL cum Salary Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q10',title,inp_df=mcr_hdi_filtered)

temp_df.T

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()
temp_df=SimpleChoicePlot('Q22')



#Lets check role of the who are using TPU <=5 times

mcr_hdi_filtered=mcr_hdi[mcr_hdi['Q22'].isin(['Once','2-5 times'])]

mcr_hdi_filtered['Q22']='<=5 times'

temp_df=GroupSimpleChoicePlot(['Q22'],'Q5','TPU <=5 times cum Role Vs HDI',inp_df=mcr_hdi_filtered)



#Lets check how many years they have used ML methods vs who are using TPU <=5 times

temp_df=GroupSimpleChoicePlot(['Q22'],'Q23','TPU <=5 times cum ML Methods Usage Years Vs HDI')#,inp_df=mcr_hdi_filtered)



temp_df=MultiChoicePlot('Q21_Part')

#Lets check the Salary, ML Methods, ML Algorithms used with CPUs

#Lets check the Salary, ML Methods and ML Algorithms used with GPUs



#CPU vs their salaries

col_name='Q21_Part_1' #CPU

title='CPU cum Salary Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q10',title,inp_df=mcr_hdi_filtered)

temp_df.T

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()



#GPU vs their salaries

col_name='Q21_Part_2' #GPU

title='GPU cum Salary Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q10',title,inp_df=mcr_hdi_filtered)

temp_df.T

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()





#Check usage across HDI Categories for CPU

df_1=mcr_hdi[~mcr_hdi['Q21_Part_1'].isnull()]['HDI_Category'].value_counts()

print("CPU Usage:\n",df_1/df_1.sum())



#Check usage across HDI Categories for GPU

df_1=mcr_hdi[~mcr_hdi['Q21_Part_2'].isnull()]['HDI_Category'].value_counts()

print("GPU Usage:\n",df_1/df_1.sum())





##Check against ML Algorithms, ML Frameworks

#Check against ML Algorithms used by people for the above

col_name='Q21_Part_2' #GPU

title='GPU Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q24_Part',[col_name],title,grp_lvl=[0,2])#,inp_df=mcr_hdi_filtered)

temp_df.T



#Check against ML Algorithms used by people for the above

col_name='Q21_Part_1' #CPU

title='CPU Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q24_Part',[col_name],title,grp_lvl=[0,2])#,inp_df=mcr_hdi_filtered)

temp_df.T



#Check against ML Frameworks used by people for the above

col_name='Q21_Part_2' #GPU

title='GPU Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q28_Part',[col_name],title,grp_lvl=[0,2])#,inp_df=mcr_hdi_filtered)

temp_df.T



#Check against ML Algorithms used by people for the above

col_name='Q21_Part_1' #CPU

title='CPU Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q28_Part',[col_name],title,grp_lvl=[0,2])#,inp_df=mcr_hdi_filtered)

temp_df.T
temp_df=MultiChoicePlot('Q9_Part')
temp_df=MultiChoicePlot('Q12_Part')



#Take only Favourite Media Source -Kaggle and check their roles

mcr_hdi.Q12_Part_4.unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi['Q12_Part_4'].isnull()]

temp_df=GroupSimpleChoicePlot(['Q12_Part_4'],'Q5','Favourite Media Source -Kaggle cum Current Role Vs HDI',inp_df=mcr_hdi_filtered)

if display_output_df:

    temp_df.T



    

temp_df=MultiChoicePlot('Q13_Part')



#Coursera and Kaggle vs their roles

col_name='Q13_Part_2' #Coursera

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q5','Favourite Media Source -Coursera cum Current Role Vs HDI',inp_df=mcr_hdi_filtered)

temp_df.T





col_name='Q13_Part_6' #Kaggle

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q5','Favourite Media Source -Kaggle cum Current Role Vs HDI',inp_df=mcr_hdi_filtered)

temp_df.T



#Lets check their salaries.

#Coursera and Kaggle vs their salaries

col_name='Q13_Part_2' #Coursera

title='Coursera course cum Salary Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q10',title,inp_df=mcr_hdi_filtered)

temp_df.T

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()





col_name='Q13_Part_6' #Kaggle

title='Kaggle course cum Salary Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q10',title,inp_df=mcr_hdi_filtered)

temp_df.T

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()



#Notebooks used

temp_df=MultiChoicePlot('Q17_Part')



#Kaggle Notebook vs their roles

col_name='Q17_Part_1' #Kaggle Notebook

title='Kaggle Notebook cum Current Role Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q5',title,inp_df=mcr_hdi_filtered)

temp_df.T





#Kaggle Notebook vs their salaries

col_name='Q17_Part_1' #Kaggle Notebook

title='Kaggle Notebook cum Salary Vs HDI'

mcr_hdi[col_name].unique()

mcr_hdi_filtered=mcr_hdi[~mcr_hdi[col_name].isnull()]

temp_df=GroupSimpleChoicePlot([col_name],'Q10',title,inp_df=mcr_hdi_filtered)

temp_df.T

temp_df1=temp_df.T.copy().reset_index()

temp_df1['SalRange']='<100K'

temp_df1.loc[temp_df1['index'].isin(['> $500,000','100,000-124,999',

             '125,000-149,999','150,000-199,999','200,000-249,999',

             '250,000-299,999','300,000-500,000']),'SalRange']='>=100K'



temp_df1.groupby('SalRange').sum()
#Automated MLs

temp_df=MultiChoicePlot('Q25_Part')



#Cloud computing platforms

temp_df=MultiChoicePlot('Q29_Part')



#Cloud computing products

temp_df=MultiChoicePlot('Q30_Part')

##ML Products

temp_df=MultiChoicePlot('Q32_Part')



##ML Tools

temp_df=MultiChoicePlot('Q33_Part')



##RDB

temp_df=MultiChoicePlot('Q34_Part')



##Big data / Analytics Products

temp_df=MultiChoicePlot('Q31_Part')
#Q24_Part_2	Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests

#Check against Gender

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q24_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q25_Part_2	Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q25_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q26_Part_2	Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image segmentation methods (U-Net, Mask R-CNN, etc)

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q26_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q27_Part_2	Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Encoder-decorder models (seq2seq, vanilla transformers)

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q27_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q28_Part_2	Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow 

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q28_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q29_Part_2	Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Amazon Web Services (AWS) 

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q29_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q30_Part_2	Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Google Compute Engine (GCE)

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q30_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q31_Part_2	Which specific big data / analytics products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS Redshift

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q31_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q32_Part_2	Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Cloudera

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q32_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q33_Part_2	Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H20 Driverless AI  

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q33_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T



#Q34_Part_2	Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL

col_name='Q2' #Gender

title='Gender Vs HDI'

temp_grp_df,temp_df=GroupMultiChoicePlot('Q34_Part',[col_name],title,grp_lvl=[0,2],inp_df=mcr_hdi_m_f_alone)

temp_df.T

from IPython.display import HTML, display

import tabulate

table = [["Activity / Factor","Female","Male"],

         ["ML Algorithms","Linear Logistic Regression, Decision Trees or RF","CNN and GB Machines"],

         ["ML Tools","Auto Model Selection","Auto Data Augmentation"],

         ["Computer Vision Methods","Image Segment Methods","General Purpose Image"],

         ["Natural Language Processing","Encoder Decoder Models (Except in MHD) & Word Embeddings (in MHD)","Transformer Language Models (Except in LHD) & Word Embeddings (in LHD)"],

         ["Machine Learning Frameworks","Scikit-learn","Keras and PyTorch"],

         ["Cloud Computing Platforms","IBMCloud","AmazonWebServices"],

         ["Cloud Computing Products","AzureVirtualMachines (Except in MHD) & GoogleCloudFunctions (in MHD)","AWS_EC_Cloud"],

         ["Big Data / Analytics Products","MS Analysis Service","AWSAthena, AWSElasticMapReduce, AWSKinesis and AWSRedshift"],

         ["Machine Learning Products","Azure_ML_Studio and RapidMiner","GoogleCloudVision and AmazonSageMaker"],

         ["Automated Machine Learning Tools","MLbox (Except in LHD) and H20DriverlessAI (in LHD)","GoogleAutoML"],

         ["Relational Database Products","MicrosoftAccess","PostgresSQL (Except in LHD) & MicrosoftSQLServer (in LHD)"]]

display(HTML(tabulate.tabulate(table, tablefmt='html')))