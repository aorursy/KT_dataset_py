import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

%matplotlib inline

multiple = pd.read_csv('../input/multipleChoiceResponses.csv', dtype=np.object)
mulcQ = multiple.iloc[0,:]
mulcA = multiple.iloc[1:,:]

fast = mulcA[round(mulcA.iloc[:,0].astype(int) / 60) <= 3].index
mulcA = mulcA.drop(fast, axis=0)
rol = mulcA.Q5
rol.value_counts(normalize=True).plot(kind='bar')
rem = mulcA.Q9
rem.value_counts().plot(kind='bar')
#
mulcA.replace({'Q9':{'0-10,000':1,'10-20,000':2,'20-30,000':3,'30-40,000':4,'40-50,000':5,'50-60,000':6,
                       '60-70,000':7,'70-80,000':8,'80-90,000':9,'90-100,000':10,'100-125,000':11,'125-150,000':12,
                       '150-200,000':13,'200-250,000':14,'250-300,000':15,'300-400,000':16,'400-500,000':17,
                                 '500,000+':18}},inplace = True)
mulcA.replace({'Q3':{'United Kingdom of Great Britain and Northern Ireland':'Great B.','Iran, Islamic Republic of...':'Iran',
                       'United States of America':'USA','Hong Kong (S.A.R.)':'Hong Kong','Republic of Korea':'R. Korea',
                      'Czech Republic':'Czech R.'}},inplace = True)

q16 = multiple.filter(regex="(Q{t}$|Q{t}_)".format(t = 16))[1:]
q16_col = {'Q16_Part_1':'Python','Q16_Part_2':'R','Q16_Part_3':'SQL','Q16_Part_4': 'Bash','Q16_Part_5':'Java',
           'Q16_Part_6':'Javascript','Q16_Part_7':'VBA','Q16_Part_8':'C/C++','Q16_Part_9':'MATLAB',
           'Q16_Part_10':'Scala','Q16_Part_11':'Julia','Q16_Part_12':'Go','Q16_Part_13':'C#/.NET','Q16_Part_14':'PHP',
           'Q16_Part_15':'Ruby','Q16_Part_16':'SAS/STATA'}

q16_lim= q16.rename(columns=q16_col).fillna(0).replace('[^\\d]',1, regex=True)
q16_lim.pop('Q16_Part_17')
q16_lim.pop('Q16_Part_18')
q16_lim.pop('Q16_OTHER_TEXT')
lab = list(q16_lim.iloc[:0])
for i in lab:
    mulcA[i]= q16_lim['{}'.format(i)]
#
com_sci = mulcA[(mulcA.Q5 == 'Computer science (software engineering, etc.)') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
eng_nco = mulcA[(mulcA.Q5 == 'Engineering (non-computer focused)') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
mat_sta = mulcA[(mulcA.Q5 == 'Mathematics or statistics') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
biu_dis = mulcA[(mulcA.Q5 == 'A business discipline (accounting, economics, finance, etc.)') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
phy_ast = mulcA[(mulcA.Q5 == 'Physics or astronomy') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
inf_tec = mulcA[(mulcA.Q5 == 'Information technology, networking, or system administration') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
med_sci = mulcA[(mulcA.Q5 == 'Medical or life sciences (biology, chemistry, medicine, etc.)') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
soc_sci = mulcA[(mulcA.Q5 == 'Social sciences (anthropology, psychology, sociology, etc.)') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
hum_tie = mulcA[(mulcA.Q5 == 'Humanities (history, literature, philosophy, etc.)') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
env_sci = mulcA[(mulcA.Q5 == 'Environmental science or geology') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]
fin_art = mulcA[(mulcA.Q5 == 'Fine arts or performing arts') & (mulcA.Q9 != 'I do not wish to disclose my approximate yearly compensation')]


rem_prom = [com_sci.Q9.mean(),eng_nco.Q9.mean(),mat_sta.Q9.mean(),biu_dis.Q9.mean(),phy_ast.Q9.mean(),inf_tec.Q9.mean(),med_sci.Q9.mean(),
            soc_sci.Q9.mean(),hum_tie.Q9.mean(),env_sci.Q9.mean(),fin_art.Q9.mean()]
plt.figure(figsize=(20,10))
plt.bar(np.arange(11),rem_prom,color=['dodgerblue','c','tomato','silver','midnightblue','tan'])
plt.xticks(np.arange(11), ('Com. Science', 'Engieneering', 'Mathematics', 'Economics', 'Physics','Inf. Tecnologist','Medics','Social sciences','Humanities',
                            'Env. Sciense','Arts'))
plt.yticks(np.arange(10),('$10000','$20000','$30000','$40000','$50000','$60000','$70000','$80000','$90000','$100000'))

age_18_21 = mulcA.Q5[(mulcA.Q2 == '18-21')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_22_24 = mulcA.Q5[(mulcA.Q2 == '22-24')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_25_29 = mulcA.Q5[(mulcA.Q2 == '25-29')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_30_34 = mulcA.Q5[(mulcA.Q2 == '30-34')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_35_39 = mulcA.Q5[(mulcA.Q2 == '35-39')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_40_44 = mulcA.Q5[(mulcA.Q2 == '40-44')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_45_49 = mulcA.Q5[(mulcA.Q2 == '45-49')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_50_54 = mulcA.Q5[(mulcA.Q2 == '50-54')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_55_59 = mulcA.Q5[(mulcA.Q2 == '55-59')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_60_69 = mulcA.Q5[(mulcA.Q2 == '60-69')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_70_79 = mulcA.Q5[(mulcA.Q2 == '70-79')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]
age_80m = mulcA.Q5[(mulcA.Q2 == '80+')&(mulcA.Q5 != 'Other')&(mulcA.Q5 != 'I never declared a major')]

df_arol = pd.DataFrame([age_18_21.value_counts(normalize=True),age_22_24.value_counts(normalize=True),age_25_29.value_counts(normalize=True),age_30_34.value_counts(normalize=True),
                        age_35_39.value_counts(normalize=True),age_40_44.value_counts(normalize=True),age_45_49.value_counts(normalize=True),age_50_54.value_counts(normalize=True)
                        ,age_55_59.value_counts(normalize=True),age_60_69.value_counts(normalize=True),age_70_79.value_counts(normalize=True),age_80m.value_counts(normalize=True)],
                       index=['age18-21','age22-24','age25-29','age30-34','age35-39','age40-44','age45-49','age50-54','age55-59'
                              ,'age60-69','age70-79','age80+']).T

axes = df_arol.plot.barh(rot=0, subplots=True,figsize=(15,35))

plt.figure(figsize=(15,3))
names = ['Comp. Science','Engeneering','Mathematics','Inf. Technology','Business','Physics','Medical','Soc. Science']
df_coun = pd.DataFrame([com_sci.Q3.value_counts(),eng_nco.Q3.value_counts(),mat_sta.Q3.value_counts(),inf_tec.Q3.value_counts()
                        ,biu_dis.Q3.value_counts(),phy_ast.Q3.value_counts(),med_sci.Q3.value_counts(),soc_sci.Q3.value_counts()]
                       ,index=[names])
df_coun= df_coun.drop(['Other'],axis=1).T
fig, axes = plt.subplots(4, 2,  sharey=True,figsize=(15,20))
axes[0, 0].barh(df_coun.index[:10],list(map(int,df_coun['Comp. Science'].values[:10])),color='cadetblue')
axes[0, 0].set_title('Comp. Science')
axes[0, 1].barh(df_coun.index[:10],list(map(int,df_coun['Engeneering'].values[:10])),color='slategray')
axes[0, 1].set_title('Engeneering')
axes[1, 0].barh(df_coun.index[:10],list(map(int,df_coun['Mathematics'].values[:10])),color='seagreen')
axes[1, 0].set_title('Mathematics')
axes[1, 1].barh(df_coun.index[:10],list(map(int,df_coun['Inf. Technology'].values[:10])),color='palevioletred')
axes[1, 1].set_title('Inf. Technology')
axes[2, 0].barh(df_coun.index[:10],list(map(int,df_coun['Business'].values[:10])),color='darkblue')
axes[2, 0].set_title('Business')
axes[2, 1].barh(df_coun.index[:10],list(map(int,df_coun['Physics'].values[:10])),color='khaki')
axes[2, 1].set_title('Physics')
axes[3, 0].barh(df_coun.index[:10],list(map(int,df_coun['Medical'].values[:10])),color='k')
axes[3, 0].set_title('Medical')
axes[3, 1].barh(df_coun.index[:10],list(map(int,df_coun['Soc. Science'].values[:10])),color='firebrick')
axes[3, 1].set_title('Soc. Science')
lengs = pd.DataFrame([mulcA['Python'].sum(),mulcA['R'].sum(),mulcA['SQL'].sum(),mulcA['Bash'].sum(),mulcA['Java'].sum(),
                           mulcA['Javascript'].sum(),mulcA['VBA'].sum(),mulcA['C/C++'].sum(),mulcA['MATLAB'].sum(),
                           mulcA['Scala'].sum(),mulcA['Julia'].sum(),mulcA['Go'].sum(),mulcA['C#/.NET'].sum(),
                           mulcA['PHP'].sum(),mulcA['Ruby'].sum(),mulcA['SAS/STATA'].sum()],index=lab)
lengs.plot(kind='bar',color='brown')
com_sci_lengs=[mulcA[i][mulcA.Q5=='Computer science (software engineering, etc.)'].sum() for i in lab ]
eng_nco_lengs=[mulcA[i][mulcA.Q5=='Engineering (non-computer focused)'].sum() for i in lab ]
mat_sta_lengs=[mulcA[i][mulcA.Q5=='Mathematics or statistics'].sum() for i in lab ]
biu_dis_lengs=[mulcA[i][mulcA.Q5=='A business discipline (accounting, economics, finance, etc.)'].sum() for i in lab ]
phy_ast_lengs=[mulcA[i][mulcA.Q5=='Physics or astronomy'].sum() for i in lab ]
inf_tec_lengs=[mulcA[i][mulcA.Q5=='Information technology, networking, or system administration'].sum() for i in lab ]
med_sci_lengs=[mulcA[i][mulcA.Q5=='Medical or life sciences (biology, chemistry, medicine, etc.)'].sum() for i in lab ]
soc_sci_lengs=[mulcA[i][mulcA.Q5=='Social sciences (anthropology, psychology, sociology, etc.)'].sum() for i in lab ]

fig, axes = plt.subplots(4, 2,  sharey=True,figsize=(15,20))
axes[0, 0].barh(lab,com_sci_lengs,color='c')
axes[0, 0].set_title('Comp. Science')
axes[0, 1].barh(lab,eng_nco_lengs,color='r')
axes[0, 1].set_title('Engeneering')
axes[1, 0].barh(lab,mat_sta_lengs,color='k')
axes[1, 0].set_title('Mathematics')
axes[1, 1].barh(lab,inf_tec_lengs)
axes[1, 1].set_title('Inf. Technology')
axes[2, 0].barh(lab,biu_dis_lengs,color = 'tomato')
axes[2, 0].set_title('Business')
axes[2, 1].barh(lab,phy_ast_lengs,color='b')
axes[2, 1].set_title('Physics')
axes[3, 0].barh(lab,med_sci_lengs,color='y')
axes[3, 0].set_title('Medical')
axes[3, 1].barh(lab,soc_sci_lengs,color='g')
axes[3, 1].set_title('Soc. Science')
fem_oc = mulcA.Q5[mulcA.Q1 == "Female"]
mal_oc = mulcA.Q5[mulcA.Q1 == "Male"]
df_gen = pd.DataFrame([fem_oc.value_counts(normalize=True),mal_oc.value_counts(normalize=True)],index=['Female','Male']).T
axes = df_gen.plot.barh(rot=0, subplots=True,figsize=(15,15))    