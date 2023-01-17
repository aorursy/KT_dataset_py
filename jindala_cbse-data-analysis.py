import numpy as np
import pandas as pd
import matplotlib as mlt 
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


cbse = pd.read_csv('../input/schools_detailed.csv')
cbse.insert(0, 'SchoolId', range(0, len(cbse)))
SchoolDetails = cbse[['name','aff_no','state','district','region','address','pincode','ph_no','off_ph_no','res_ph_no','fax_no',\
'email','website','year_found','date_opened','status','aff_type','aff_start','aff_end','soc_name',
'n_category','n_medium','n_school_type']]


plt.subplots(figsize=(10,6))
ax=SchoolDetails['state'].value_counts().plot.bar(width=0.8,align = 'center',label ='Guido')       
plt.ylabel("Number of school",fontsize=15,color='navy')    
plt.title("Number of school per state",fontsize=30,color='navy')
plt.xlabel("States--->",fontsize=15,color='navy')
plt.show()


plt.subplots(figsize=(10,6))
ax=SchoolDetails['n_medium'].value_counts().plot.bar(width=0.8,align = 'center',label ='Guido')       
plt.ylabel("Number of school",fontsize=15,color='navy')    
plt.xlabel("Medium--->",fontsize=15,color='navy')
plt.show()



plt.subplots(figsize=(10,6))
ax=SchoolDetails['status'].value_counts().plot.bar(width=0.8,align = 'center',label ='Guido')       
plt.ylabel("Number of school",fontsize=15,color='navy')    
plt.xlabel("Status--->",fontsize=15,color='navy')
plt.show()



plt.subplots(figsize=(10,6))
st = SchoolDetails.groupby(['state','status'],as_index=False).size().reset_index(name ='Count').sort_values(by ='Count', ascending = False)
sns.barplot(x='state',y='Count',data=st,hue='status',palette='viridis')
plt.ylabel("Number of school",fontsize=15,color='navy')    
plt.title("Number of school per state",fontsize=30,color='navy')
plt.xlabel("States--->",fontsize=15,color='navy')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()


plt.subplots(figsize=(10,6))
st = SchoolDetails.groupby(['state','n_school_type'],as_index=False).size().reset_index(name ='Count').sort_values(by ='Count', ascending = False)
sns.barplot(x='state',y='Count',data=st,hue='n_school_type',palette='viridis')
plt.ylabel("Number of school",fontsize=15,color='navy')    
plt.title("Number of school per state",fontsize=30,color='navy')
plt.xlabel("States--->",fontsize=15,color='navy')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

statesArea = {'MAHARASHTRA':307713,'WEST BENGAL':88752,\
              'TAMILNADU':130058,'ANDHRA PRADESH':160205,\
              'KARNATAKA':191791,'KERALA':38863,'MADHYA PRADESH':308350,\
              'GUJARAT':196024,'CHATTISGARH':135191,'ODISHA':155707,\
              'RAJASTHAN':342239,'UTTAR PRADESH':243290,'ASSAM':78438,\
              'HARYANA':44212,'DELHI':1484,'JHARKHAND':79714,\
              'PUNJAB':50362,'BIHAR':94163,'TRIPURA':10486,'PUDUCHERRY':562,\
              'HIMACHAL PRADESH':55673,'UTTARAKHAND':53483,'GOA':3702,\
              'JAMMU & KASHMIR':222236,'SIKKIM':7096,'ANDAMAN & NICOBAR':8249,\
              'ARUNACHAL PRADESH':83743,'MEGHALAYA':22429,\
              'CHANDIGARH':114,'MIZORAM':21081,'DADAR & NAGAR HAVELI':491,\
              'MANIPUR':22327,'NAGALAND':16579,'DAMAN & DIU':112,\
              'LAKSHADWEEP':32,'TELANGANA' :112077}

st = SchoolDetails.groupby(['state'],as_index=False).size().reset_index(name ='Count').sort_values(by ='Count', ascending = False)
for state in statesArea.keys():
    st.loc[st['state']==state,'Area'] = statesArea[state]
st['School_per_squareKm'] = st['Count']/st['Area']
sortedStates = st.sort_values('School_per_squareKm',ascending=False)[:10]
sns.barplot(x='state',y='School_per_squareKm',data=sortedStates,palette='viridis')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()

statesPop = {'MAHARASHTRA':112372972,'WEST BENGAL':91347736,\
              'TAMILNADU':72138958,'ANDHRA PRADESH':49386799,\
              'KARNATAKA':61130704,'KERALA':33387677,'MADHYA PRADESH':72597565,\
              'GUJARAT':60383628,'CHATTISGARH':135191,'ODISHA':41947358,\
              'RAJASTHAN':68621012,'UTTAR PRADESH':207281477,'ASSAM':31169272,\
              'HARYANA':25540196,'DELHI':18980000,'JHARKHAND':32966238,\
              'PUNJAB':27704236,'BIHAR':103804637,'TRIPURA':3671032,'PUDUCHERRY':1244464,\
              'HIMACHAL PRADESH':6864602,'UTTARAKHAND':10116752,'GOA':1457723,\
              'JAMMU & KASHMIR':12548926,'SIKKIM':607688,'ANDAMAN & NICOBAR':379944,\
              'ARUNACHAL PRADESH':1382611,'MEGHALAYA':2964007,\
              'CHANDIGARH':1055450,'MIZORAM':1091014,'DADAR & NAGAR HAVELI':342853,\
              'MANIPUR':2721756,'NAGALAND':1980602,'DAMAN & DIU':242911,\
              'LAKSHADWEEP':64429,'TELANGANA' :35286757}

for state in statesPop.keys():
    st.loc[st['state']==state,'Population'] = statesPop[state]
st['School_per_TenThousand'] = st['Count']*10000/st['Population']    

sortedStates = st.sort_values('School_per_TenThousand',ascending=False)[:10]
sns.barplot(x='state',y='School_per_TenThousand',data=sortedStates,palette='viridis')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()


PrincipalDetails =cbse[['SchoolId','princi_name',	'sex',	'princi_qual',	\
                        'princi_exp_adm'	,'princi_exp_teach','n_school_type']]

IndpSchPri= PrincipalDetails[PrincipalDetails['n_school_type']  ==  'Independent'].sort_values(by ='princi_exp_adm',ascending =False)[:20]
GovtSchPri= PrincipalDetails[PrincipalDetails['n_school_type']  ==  'Govt'].sort_values(by ='princi_exp_adm',ascending =False)[:20]
KVSpSchPri= PrincipalDetails[PrincipalDetails['n_school_type']  ==  'KVS'].sort_values(by ='princi_exp_adm',ascending =False)[:20]
 
print(IndpSchPri[['princi_name','princi_qual','princi_exp_adm'	,'princi_exp_teach']])
print(GovtSchPri[['princi_name','princi_qual','princi_exp_adm'	,'princi_exp_teach']])
print(KVSpSchPri[['princi_name','princi_qual','princi_exp_adm'	,'princi_exp_teach']])

def sexinfo( r):
    if (r ==1):
        return 'Male'
    elif (r == 2):
        return 'Female'
    else:
        return 'Not Defined'

PrincipalDetails['sex'] =PrincipalDetails['sex'].apply(sexinfo)
plt.subplots(figsize=(10,6))
st = PrincipalDetails.groupby(['sex','n_school_type'],as_index=False).size().reset_index(name ='Count').sort_values(by ='Count', ascending = False)
sns.barplot(x='n_school_type',y='Count',data=st,hue='sex',palette='viridis')
plt.ylabel("Number of principals",fontsize=15,color='navy')    
plt.title("Number of Principals per SchoolType",fontsize=30,color='navy')
plt.xlabel("School Type--->",fontsize=15,color='navy')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()


plt.subplots(figsize=(10,6))
st = PrincipalDetails.groupby(['sex'],as_index=False).size().reset_index(name ='Count').sort_values(by ='Count', ascending = False)
plt.pie((st['Count']).values,labels =st['sex'].values,shadow = True,startangle =90,autopct ='%10.2f%%',)
plt.show()


NearestFac=cbse[['SchoolId','l_nearest_railway','l_nearest_railway_dist','l_nearest_police',\
                 'l_nearest_police_dist','l_nearest_bank','l_nearest_bank_dist','n_school_type']]

df = NearestFac.groupby('n_school_type').mean().reset_index()
df.columns =['School Type','SchoolId','Railway','Police','Bank']
df.plot(x="School Type", y=["Railway", "Police", "Bank"], kind="bar")
plt.ylabel("KMs--->",fontsize=15,color='navy')    
plt.xlabel("School Type--->",fontsize=15,color='navy')
plt.title("Facility",fontsize=30,color='navy')
plt.xticks(rotation=45,ha='right')


students = cbse[['SchoolId','e_nursery_students',
                    'e_i_v_students','e_vi_viii_students',
                    'e_ix_x_students','e_xi_xii_students',
                    'e_i_students','e_ii_students',
                    'e_iii_students','e_iv_students',
                    'e_v_students','e_vi_students',
                    'e_vii_students','e_viii_students',
                    'e_ix_students','e_x_students',
                    'e_xi_students','e_xii_students','n_school_type']]


studentsdf = students[['SchoolId','e_nursery_students',
                    'e_i_v_students','e_vi_viii_students',
                    'e_ix_x_students','e_xi_xii_students','n_school_type']]
studentsdf = studentsdf.groupby('n_school_type').count().reset_index()
studentsdf.columns =['School Type','SchoolId','Nursery','i_v_students','vi_viii_students','ix_x_students','xi_xii_students']


studentsdf.plot(x="School Type", y=['Nursery','i_v_students','vi_viii_students','ix_x_students','xi_xii_students'], kind="bar")
plt.ylabel("NoOfStudents",fontsize=15,color='navy')    
plt.xlabel("School Type--->",fontsize=15,color='navy')
plt.xticks(rotation=45,ha='right')
plt.show()

Fac= cbse[['SchoolId',
        'p_area_meter','p_area_acre',
'p_area_builtup_meter','p_num_sites',
'p_area_playground','p_urinal_type',
'p_boys_urinal','p_girls_urinal',
'p_potable_water','p_health_cert',
'f_total_books','f_periodicals',
'f_dailies','f_reference_books',
'f_magazine','f_swimming_pool',
'f_indoor_games','f_dance_rooms',
'f_gym','f_music_rooms',
'f_hostel','f_health_checkup','n_school_type' ]]


Facdf = Fac.groupby('n_school_type').count().reset_index()
print(Facdf.head(2))

Facdf.columns =['School Type','SchoolId','p_area_meter','p_area_acre'\
                ,'p_area_builtup_meter','p_num_sites','p_area_playground'\
                ,'p_urinal_type','Boys_urinal','Girls_urinal'\
                ,'Potable_Water','p_health_cert','Total_Books'\
                ,'f_periodicals','f_dailies','f_reference_books'\
                ,'f_magazine','f_swimming_pool','f_indoor_games'\
                ,'f_dance_rooms','f_gym','f_music_rooms'\
                ,'f_hostel','Health_checkup'
                ]


Facdf.plot(x="School Type", y=['Boys_urinal','Girls_urinal','Potable_Water','Health_checkup',\
                               'Total_Books'], kind="bar")
plt.ylabel("Count",fontsize=15,color='navy')    
plt.xlabel("School Type--->",fontsize=15,color='navy')
plt.xticks(rotation=45,ha='right')
plt.show()