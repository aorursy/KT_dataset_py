import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import matplotlib.pyplot as plt

from plotly.graph_objs import *

import colorlover as cl

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



import gc

gc.enable()





import warnings

warnings.filterwarnings("ignore")

%matplotlib inline



#DATASET VIEW

path="/kaggle/input/kaggle-survey-2019/"



data_files=list(os.listdir(path))

df_files=pd.DataFrame(data_files,columns=['File_Name'])

df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path+x).st_size/(1024*1024),2))



with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    display(df_files.sort_values('File_Name'))
%%time

#READING DATASET

print('READING MULTIPLE CHOICE RESPONSE...')

df_mchresp=pd.read_csv(path+'multiple_choice_responses.csv',low_memory=False)



print('READING OTHER TEXT RESPONSE...')

df_othtxtresp=pd.read_csv(path+'other_text_responses.csv')



print('READING QUESTINGS ONLY...')

df_quesonly=pd.read_csv(path+'questions_only.csv')



print('READING SURVEY SCHEMA...')

df_surshma=pd.read_csv(path+'survey_schema.csv')
#All FUNCTIONS



#FUNCTION FOR PROVIDING FEATURE SUMMARY

def feature_summary(df_fa):

    print('DataFrame shape')

    print('rows:',df_fa.shape[0])

    print('cols:',df_fa.shape[1])

    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']

    df=pd.DataFrame(index=df_fa.columns,columns=col_list)

    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])

    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])

    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])

    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])

    for i,col in enumerate(df_fa.columns):

        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):

            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))

            df.at[col,'Mean']=df_fa[col].mean()

            df.at[col,'Std']=df_fa[col].std()

            df.at[col,'Skewness']=df_fa[col].skew()

        elif 'datetime64[ns]' in str(df_fa[col].dtype):

            df.at[col,'Max/Min']=str(df_fa[col].max())+'/'+str(df_fa[col].min())

        df.at[col,'Sample_values']=list(df_fa[col].unique())

    display(df_fa.head())       

    return(df.fillna('-'))
#FEATURE SUMMARY SURVEY SCHEMA

pd.set_option('display.max_colwidth', -1)

display(feature_summary(df_surshma))



#SCHEMA TABLE TRANSPOSING

df_survey_schema=pd.DataFrame()

df_survey_schema['Ques_no']=df_surshma.columns[1:-1]

df_survey_schema['Sort_index']=[int(x[1:]) for x in df_survey_schema['Ques_no']]

df_survey_schema['Ques_text']=list(df_surshma.iloc[0,1:-1])

for i in range(1,10):

    df_survey_schema[df_surshma.iloc[i,0]]=list(df_surshma.iloc[i,1:-1])



print('Survey Schema shape:',df_survey_schema.shape)



df_survey_schema.sort_values('Sort_index')



#FEATURE SUMMARY MULTIPLE CHOICE RESPONSE

df_mchresp.replace(np.nan,'Not Applicable',inplace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    display(feature_summary(df_mchresp))

    

#FEATURE SUMMARY OTHER TEXT RESPONSES    

display(feature_summary(df_othtxtresp))



#FEATURE SUMMARY QUESTIONS ONLY

display(feature_summary(df_quesonly))





#ALL MULTIPLE CHOICE QUESTIONS

df_mul_ch_resp=pd.DataFrame()

df_mul_ch_resp['Ques_text']=df_mchresp.iloc[0,1:].values

df_mul_ch_resp['Ques_no']=df_mchresp.columns[1:]

df_mul_ch_resp['Parent']=df_mul_ch_resp.Ques_no.apply(lambda x:x.split('_')[0] if x.split('_')[0]!=x else x)

# df_mul_ch_resp['Child']=df_mul_ch_resp.Ques_no.apply(lambda x:x.split('_')[2] if x.split('_')[2]!=x else np.nan)

# df_mul_ch_resp['Duration']=df_mchresp['Time from Start to Finish (seconds)'][1:].values

df_mul_ch_resp['Choices']='choice'

df_mul_ch_resp['Count_per_choice']='choice'



for i,col in enumerate(df_mchresp.columns[1:]):

    cols=[col,'Time from Start to Finish (seconds)']

    df_mul_ch_resp.at[i,'Choices']=list(df_mchresp[col][1:].unique())

    df_mul_ch_resp.at[i,'Count_per_choice']=list(df_mchresp[cols][1:].groupby(col).count().reset_index()['Time from Start to Finish (seconds)'].values)



    

df_mul_ch_resp['No_of_choices']=df_mul_ch_resp.Choices.apply(lambda x:len(x))

df_mul_ch_resp['Perc_of_count']=df_mul_ch_resp.Count_per_choice.apply(lambda x:list((np.array(x)*100/np.array(x).sum()).round(2)))



with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    display(df_mul_ch_resp)

    



#CREATING DATAFRAME FOR SUB QUESTION COUNT

df_sub_ques=df_mul_ch_resp[['Ques_no','Parent']].groupby('Parent').count().reset_index()

df_sub_ques.columns=['Ques_no','Sub_ques_count']

display(df_sub_ques)



#JOINING SURVEY SCHEMA WITH SUB QUESTION COUNT

df_survey_schema_f=pd.merge(df_survey_schema,df_sub_ques,how='left',on='Ques_no')



df_survey_schema_f.sort_values('Sort_index',inplace=True)

df_survey_schema_f.reset_index(inplace=True)

df_survey_schema_f.drop(['index','Sort_index'],axis=1,inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    display(df_mul_ch_resp[['Ques_text','Ques_no','Parent','No_of_choices','Choices','Count_per_choice']])
#UNDERSTANDING SURVEY SCHEMA

cols=['Ques_no','Ques_text','Sub_ques_count','# of Respondents:']

df_survey_schema_f[cols]
df_survey_schema_f['# of Respondents:']=df_survey_schema_f['# of Respondents:'].astype(int)

plt.figure(figsize=(8,10))

sns.barplot(x=df_survey_schema_f['# of Respondents:'],y=df_survey_schema_f['Ques_no'])

plt.xlabel('RESPONSE COUNT',color='blue')

plt.ylabel('QUESTION',color='blue')

plt.title('QUESTIONS WITH RESPONDENT COUNT',color='blue')

plt.show()
df_survey_schema_f['Sub_ques_count']=df_survey_schema_f['Sub_ques_count'].astype(int)

plt.figure(figsize=(8,10))

sns.barplot(x=df_survey_schema_f['Sub_ques_count'],y=df_survey_schema_f['Ques_no'])

plt.xlabel('SUB QUESTION COUNT',color='blue')

plt.ylabel('QUESTION',color='blue')

plt.title('QUESTIONS WITH SUB QUESTION COUNT',color='blue')

plt.show()
for i in range(0,df_mul_ch_resp.shape[0]):

    if ('TEXT' not in df_mul_ch_resp.loc[i,'Ques_no']):

        if (df_mul_ch_resp.loc[i,'No_of_choices']<15):

            plt.figure(figsize=(6, 4))

        else:

            plt.figure(figsize=(6, 10))

        sns.barplot(x=df_mul_ch_resp.loc[i,'Count_per_choice'],y=df_mul_ch_resp.loc[i,'Choices'])

        plt.xlabel('RESPONSE COUNT',color='blue')

        plt.ylabel('CHOICES',color='blue')

        plt.title(df_mul_ch_resp.loc[i,'Ques_no']+':'+df_mul_ch_resp.loc[i,'Ques_text'],color='blue')

        plt.show()