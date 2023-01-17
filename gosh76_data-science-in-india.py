import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#multi_choice = pd.read_csv('C:\\Users\\goldie.sahni\\Downloads\\KaggleSurvey\\multiple_choice_responses.csv')
multi_choice = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")
multi_choice.shape
listcols = []



for x in multi_choice.columns:

    

    if '_OTHER_TEXT' in x:

        

        listcols.append(x)
multi_choice.drop(listcols,axis=1,inplace=True)
#Qs = multi_choice.loc[0,:]
multi_choice.drop([0],inplace=True)
listnum = ['Q9','Q12','Q13','Q16','Q17','Q18','Q20','Q21','Q24','Q25','Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33','Q34']
for n in listnum:

    

    print(n)

    

    listcols1 = []

    

    na_string = 'No_Value'    

       

    for x in multi_choice.columns:

        

        if n in x:

            

            listcols1.append(x)

            

    df1 = multi_choice.loc[:,listcols1]

    

    df1[n + '_whole'] = ''

    

    for c in listcols1:

        

        new = df1[c].copy()

        

        df1[n + '_whole']= df1[n + '_whole'].str.cat(new, sep =",", na_rep = na_string)

        

    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x[1: ])

    

    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace(',No_Value',''))

    

    #df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace('No_Value,',''))

    

    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace(',None',''))

    

    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace('No_Value,',''))

    

    multi_choice[n + '_whole'] = df1[n + '_whole'].copy()



    print(df1.loc[13,n + '_whole'])   

           

part_cols = []



for x in multi_choice.columns:

    

    if '_Part_' in x:

        

        part_cols.append(x)
multi_choice.drop(part_cols,axis=1,inplace=True)
%matplotlib inline
ax = (multi_choice['Q3'].value_counts()[0:5]*100/multi_choice.shape[0]).plot(kind='bar',figsize=(20,10))

ax.set_xlabel("country",fontsize=20)

ax.set_ylabel("%",fontsize = 20)

ax.tick_params(labelsize=20)
ax1 = (multi_choice.loc[multi_choice['Q3']=='India','Q2'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q2'].shape[0]).plot(kind='bar',figsize=(20,10))

ax1.set_xlabel("gender",fontsize=20)

ax1.set_ylabel("%",fontsize = 20)

ax1.tick_params(labelsize=20)
ax2 = (multi_choice.loc[multi_choice['Q3']=='India','Q1'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q1'].shape[0]).plot(kind='bar',figsize=(20,10))

ax2.set_xlabel("age_group",fontsize=20)

ax2.set_ylabel("%",fontsize = 20)

ax2.tick_params(labelsize=20)
ax3 = (multi_choice.loc[multi_choice['Q3']=='India','Q4'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q4'].shape[0]).plot(kind='bar',figsize=(20,10))

ax3.set_xlabel("degree",fontsize=20)

ax3.set_ylabel("%",fontsize = 20)

ax3.tick_params(labelsize=20)
ax4 = (multi_choice.loc[multi_choice['Q3']=='India','Q5'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q5'].shape[0]).plot(kind='bar',figsize=(20,10))

ax4.set_xlabel("occupation",fontsize=20)

ax4.set_ylabel("%",fontsize = 20)

ax4.tick_params(labelsize=20)
ax5 = (multi_choice.loc[multi_choice['Q3']=='India','Q6'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q6'].shape[0]).plot(kind='bar',figsize=(20,10))

ax5.set_xlabel("no._of_employees",fontsize=20)

ax5.set_ylabel("%",fontsize = 20)

ax5.tick_params(labelsize=20)
ax6 = (multi_choice.loc[multi_choice['Q3']=='India','Q7'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q7'].shape[0]).plot(kind='bar',figsize=(20,10))

ax6.set_xlabel("people_sharing_workload",fontsize=20)

ax6.set_ylabel("%",fontsize = 20)

ax6.tick_params(labelsize=20)
ax7 = (multi_choice.loc[multi_choice['Q3']=='India','Q8'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q8'].shape[0]).plot(kind='bar',figsize=(20,10))

ax7.set_xlabel("state_of_ML_in_company",fontsize=20)

ax7.set_ylabel("%",fontsize = 20)

ax7.tick_params(labelsize=20)
ax8 = (multi_choice.loc[multi_choice['Q3']=='India','Q9_whole'].value_counts()[0:3]*100/multi_choice.loc[multi_choice['Q3']=='India','Q9_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax8.set_xlabel("gender",fontsize=20)

ax8.set_ylabel("%",fontsize = 20)

ax8.tick_params(labelsize=20)
ax9 = (multi_choice.loc[multi_choice['Q3']=='India','Q10'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q10'].shape[0]).plot(kind='bar',figsize=(20,10))

ax9.set_xlabel("annual_salary",fontsize=20)

ax9.set_ylabel("%",fontsize = 20)

ax9.tick_params(labelsize=20)
ax10 = (multi_choice.loc[multi_choice['Q3']=='India','Q11'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q11'].shape[0]).plot(kind='bar',figsize=(20,10))

ax10.set_xlabel("investment",fontsize=20)

ax10.set_ylabel("%",fontsize = 20)

ax10.tick_params(labelsize=20)
ax11 = (multi_choice.loc[multi_choice['Q3']=='India','Q12_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q12_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax11.set_xlabel("forum",fontsize=20)

ax11.set_ylabel("%",fontsize = 20)

ax11.tick_params(labelsize=20)
ax12 = (multi_choice.loc[multi_choice['Q3']=='India','Q13_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q13_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax12.set_xlabel("course_platform",fontsize=20)

ax12.set_ylabel("%",fontsize = 20)

ax12.tick_params(labelsize=20)
ax13 = (multi_choice.loc[multi_choice['Q3']=='India','Q14'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q14'].shape[0]).plot(kind='bar',figsize=(20,10))

ax13.set_xlabel("development_environment",fontsize=20)

ax13.set_ylabel("%",fontsize = 20)

ax13.tick_params(labelsize=20)
ax14 = (multi_choice.loc[multi_choice['Q3']=='India','Q15'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q15'].shape[0]).plot(kind='bar',figsize=(20,10))

ax14.set_xlabel("experience",fontsize=20)

ax14.set_ylabel("%",fontsize = 20)

ax14.tick_params(labelsize=20)
ax15 = (multi_choice.loc[multi_choice['Q3']=='India','Q16_whole'].value_counts()[0:4]*100/multi_choice.loc[multi_choice['Q3']=='India','Q16_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax15.set_xlabel("environment",fontsize=20)

ax15.set_ylabel("%",fontsize = 20)

ax15.tick_params(labelsize=20)
ax16 = (multi_choice.loc[multi_choice['Q3']=='India','Q17_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q17_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax16.set_xlabel("notebook_type",fontsize=20)

ax16.set_ylabel("%",fontsize = 20)

ax16.tick_params(labelsize=20)
ax17 = (multi_choice.loc[multi_choice['Q3']=='India','Q18_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q18_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax17.set_xlabel("tools",fontsize=20)

ax17.set_ylabel("%",fontsize = 20)

ax17.tick_params(labelsize=20)
ax18 = (multi_choice.loc[multi_choice['Q3']=='India','Q19'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q19'].shape[0]).plot(kind='bar',figsize=(20,10))

ax18.set_xlabel("computer_language",fontsize=20)

ax18.set_ylabel("%",fontsize = 20)

ax18.tick_params(labelsize=20)
ax19 = (multi_choice.loc[multi_choice['Q3']=='India','Q20_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q20_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax19.set_xlabel("visualization_tools",fontsize=20)

ax19.set_ylabel("%",fontsize = 20)

ax19.tick_params(labelsize=20)
ax20 = (multi_choice.loc[multi_choice['Q3']=='India','Q21_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q21_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax20.set_xlabel("processing_unit_type",fontsize=20)

ax20.set_ylabel("%",fontsize = 20)

ax20.tick_params(labelsize=20)
ax21 = (multi_choice.loc[multi_choice['Q3']=='India','Q22'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q22'].shape[0]).plot(kind='bar',figsize=(20,10))

ax21.set_xlabel("TPU_use_frequency",fontsize=20)

ax21.set_ylabel("%",fontsize = 20)

ax21.tick_params(labelsize=20)
ax22 = (multi_choice.loc[multi_choice['Q3']=='India','Q23'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q23'].shape[0]).plot(kind='bar',figsize=(20,10))

ax22.set_xlabel("machine_learning_use",fontsize=20)

ax22.set_ylabel("%",fontsize = 20)

ax22.tick_params(labelsize=20)
ax23 = (multi_choice.loc[multi_choice['Q3']=='India','Q24_whole'].value_counts()[0:3]*100/multi_choice.loc[multi_choice['Q3']=='India','Q24_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax23.set_xlabel("machine_learning_algorithms",fontsize=20)

ax23.set_ylabel("%",fontsize = 20)

ax23.tick_params(labelsize=20)
ax24 = (multi_choice.loc[multi_choice['Q3']=='India','Q25_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q25_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax24.set_xlabel("automated_tool_selection",fontsize=20)

ax24.set_ylabel("%",fontsize = 20)

ax24.tick_params(labelsize=20)
ax25 = (multi_choice.loc[multi_choice['Q3']=='India','Q26_whole'].value_counts()[0:2]*100/multi_choice.loc[multi_choice['Q3']=='India','Q26_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax25.set_xlabel("image_processing_tools",fontsize=20)

ax25.set_ylabel("%",fontsize = 20)

ax25.tick_params(labelsize=20)
ax26 = (multi_choice.loc[multi_choice['Q3']=='India','Q27_whole'].value_counts()[0:2]*100/multi_choice.loc[multi_choice['Q3']=='India','Q27_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax26.set_xlabel("nlp_tools",fontsize=20)

ax26.set_ylabel("%",fontsize = 20)

ax26.tick_params(labelsize=20)
ax27 = (multi_choice.loc[multi_choice['Q3']=='India','Q28_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q28_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax27.set_xlabel("machine_learning_frameworks",fontsize=20)

ax27.set_ylabel("%",fontsize = 20)

ax27.tick_params(labelsize=20)
ax28 = (multi_choice.loc[multi_choice['Q3']=='India','Q29_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q29_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax28.set_xlabel("cloud_computing_platform",fontsize=20)

ax28.set_ylabel("%",fontsize = 20)

ax28.tick_params(labelsize=20)
ax29 = (multi_choice.loc[multi_choice['Q3']=='India','Q30_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q30_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax29.set_xlabel("cloud_computing_products",fontsize=20)

ax29.set_ylabel("%",fontsize = 20)

ax29.tick_params(labelsize=20)
ax30 = (multi_choice.loc[multi_choice['Q3']=='India','Q31_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q31_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax30.set_xlabel("data_analysis_products",fontsize=20)

ax30.set_ylabel("%",fontsize = 20)

ax30.tick_params(labelsize=20)
ax31 = (multi_choice.loc[multi_choice['Q3']=='India','Q32_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q32_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax31.set_xlabel("machine_learning_products",fontsize=20)

ax31.set_ylabel("%",fontsize = 20)

ax31.tick_params(labelsize=20)
ax32 = (multi_choice.loc[multi_choice['Q3']=='India','Q33_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q33_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax32.set_xlabel("automated_platforms",fontsize=20)

ax32.set_ylabel("%",fontsize = 20)

ax32.tick_params(labelsize=20)
ax33 = (multi_choice.loc[multi_choice['Q3']=='India','Q34_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q34_whole'].shape[0]).plot(kind='bar',figsize=(20,10))

ax33.set_xlabel("relational_database_products",fontsize=20)

ax33.set_ylabel("%",fontsize = 20)

ax33.tick_params(labelsize=20)