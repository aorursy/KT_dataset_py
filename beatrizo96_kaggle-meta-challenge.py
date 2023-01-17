import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from collections import Counter
from numpy import linalg as la
#from __future__ import print_function
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
%matplotlib inline
MCR = pd.read_csv('../input/multipleChoiceResponses.csv')
#the answers of the participants
MCR_A = MCR.iloc[1:,:]
#the questions
MCR_Q = MCR.iloc[0,:]
#cleaning the answers (the answers written by the interviewed are not taken into account)
personal_data = MCR_A.iloc[:,1:13].copy()
personal_data.drop(['Q1_OTHER_TEXT','Q6_OTHER_TEXT','Q7_OTHER_TEXT'], axis=1, inplace=True)
personal_data = personal_data.drop(list(personal_data[personal_data.Q9 == 'I do not wish to disclose my approximate yearly compensation'].index), axis=0)
personal_data = personal_data[~personal_data.Q9.isnull()].copy()
personal_data.head(3)
#It's created a column with an average of the possible intervals of incomes that were given to the interviewed 
salary = personal_data.Q9.str.replace(',', '').str.replace('500000\+', '500-500000').str.split('-')
personal_data['Q9_1'] = salary.apply(lambda x: (int(x[0])*1000 + int(x[1]))/2)
# Column with boolean values that says if the interviewed is in the top 20% of highest incomes.
personal_data['Q9_2'] = personal_data.Q9_1 > personal_data.Q9_1.quantile(0.8)
#series with all the personal data for those who recieve more than 90.000 dollars a year.
Top_salary=personal_data.loc[list(personal_data.Q9_2[personal_data.Q9_2==True].index)]
#with this function it's found the frequency for every possible answers of a specific question.
def freq_salary(x_salary):
    freq_x=np.array(x_salary.value_counts(dropna=False, sort=True))
    freq_x=freq_x[1:]
    freq_x=freq_x*100/np.amax(freq_x) 
    ind_x=np.arange(len(freq_x))
    return(freq_x,ind_x)
#Here are created series for men and women who have an income higher than 90.000 dollars a year. 
male_top_salary=Top_salary.loc[list(personal_data[personal_data.Q1== 'Male'].index)]
female_top_salary=Top_salary.loc[list(personal_data[personal_data.Q1== 'Female'].index)]
#The frequencies for the different intervals of incomes are calculated, both for men and women 
freq_male,ind_male=freq_salary(male_top_salary.Q9_1)
freq_female,ind_female=freq_salary(female_top_salary.Q9_1)
#a list of string the intervals of incomes higher than 90.000 dollars is created
L=('90-100.000','100-125.000','125-150.000','150-200.000','200-250.000',
   ',250-300.000','300-400.000','400-500.000','500.000+')
#the result are given in a bar plot.
plt.figure(figsize=(15,10))
w=0.3
plt.xticks(ind_male,L,rotation='vertical')
plt.bar(ind_male-w/2,freq_male,width=w,label='Men')
plt.bar(ind_female+w/2,freq_female,width=w,label='Women')
plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Incomes for Men and Women(Top 20% of all incomes)')
plt.legend()
L=('90-100.000','100-125.000','125-150.000','150-200.000','200-250.000',
   ',250-300.000','300-400.000','400-500.000','500.000+')
business_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'A business discipline (accounting, economics, finance, etc.)'].index)]
enviroment_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5 == 'Environmental science or geology '].index)]
physics_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Physics or astronomy'].index)]
engineering_not_data_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5 == 'Engineering (non-computer focused)'].index)]
math_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Mathematics or statistics'].index)]
medical_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Medical or life sciences (biology, chemistry, medicine, etc.)'].index)]
information_tech_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Information technology, networking, or system administration'].index)]
computar_science_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Computer science (software engineering, etc.)'].index)]
arts_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Fine arts or performing arts'].index)]
humanities_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Humanities (history, literature, philosophy, etc.)'].index)]
other_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Other'].index)]
not_declared_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'I never declared a major'].index)]
social_sciences_top_salary=Top_salary.loc[list(personal_data[personal_data.Q5== 'Social sciences (anthropology, psychology, sociology, etc.)'].index)]

freq_business,ind_business=freq_salary(business_top_salary.Q9_1)
freq_physics, ind_physics=freq_salary(physics_top_salary.Q9_1)
freq_engineering_not_data,ind_engineering_not_data=freq_salary(engineering_not_data_top_salary.Q9_1)
freq_math,ind_math=freq_salary(math_top_salary.Q9_1)
freq_medical,ind_medical=freq_salary(medical_top_salary.Q9_1)
freq_information,ind_information=freq_salary(information_tech_top_salary.Q9_1)
freq_computar_science,ind_computar_science=freq_salary(computar_science_top_salary.Q9_1)
freq_arts,ind_arts=freq_salary(arts_top_salary.Q9_1)
freq_other,ind_other=freq_salary(other_top_salary.Q9_1)
freq_not_declared,ind_not_declared=freq_salary(not_declared_top_salary.Q9_1)
freq_social_sciences,ind_social_sciences=freq_salary(social_sciences_top_salary.Q9_1)
w=0.135
plt.figure(figsize=(15,10))
plt.xticks(ind_computar_science,L,rotation='vertical')
plt.bar(ind_engineering_not_data-w*5/2,freq_engineering_not_data,width=w,label='Engineering (non-computer focused)')
plt.bar(ind_math-w*4/2,freq_math,width=w,label='Mathematics or statistics')
plt.bar(ind_medical-w*3/2,freq_medical,width=w,label='Medical or life sciences (biology, chemistry, medicine, etc.)')
plt.bar(ind_information-w*2/2,freq_information,width=w,label='Information technology, networking, or system administration')
plt.bar(ind_computar_science-w*1/2,freq_computar_science,width=w,label='Computer science (software engineering, etc.)')
plt.bar(ind_arts+w*1/2,freq_arts,width=w,label='Fine arts or performing arts')
plt.bar(ind_business+w*2/2,freq_business,width=w,label='A business discipline (accounting, economics, finance, etc.)')
plt.bar(ind_physics+w*3/2,freq_physics,width=w,label='Physics or astronomy')
plt.bar(ind_other+w*4/2,freq_other,width=w,label='Other')
plt.bar(ind_not_declared+w*5/2,freq_not_declared,width=w,label='I never declared a major')
plt.bar(ind_social_sciences+w*6/2,freq_social_sciences,width=w,label='Social sciences (anthropology, psychology, sociology, etc.)')
plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Income and Major (Top 20% of all incomes)')
plt.legend()
L=('0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000','50-60,000', '60-70,000', '70-80,000','80-90,000',
   '90-100,000','100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000','300-400,000', '400-500,000',
   '500,000+')
a_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '18-21'].index)]
b_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '22-24'].index)]
c_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '25-29'].index)]
d_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '30-34'].index)]
e_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '35-39'].index)]
f_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '40-44'].index)]
g_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '45-49'].index)]
h_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '50-54'].index)]
i_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '55-59'].index)]
j_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '60-69'].index)]
k_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '70-79'].index)]
l_salary=personal_data.loc[list(personal_data[personal_data.Q2 == '80+'].index)]

freq_a,ind_a =freq_salary(a_salary.Q9_1)
freq_b,ind_b =freq_salary(b_salary.Q9_1)
freq_c,ind_c =freq_salary(c_salary.Q9_1)
freq_d,ind_d =freq_salary(d_salary.Q9_1)
freq_e,ind_e =freq_salary(e_salary.Q9_1)
freq_f,ind_f =freq_salary(f_salary.Q9_1)
freq_g,ind_g =freq_salary(g_salary.Q9_1)
freq_h,ind_h =freq_salary(h_salary.Q9_1)
freq_i,ind_i =freq_salary(i_salary.Q9_1)
freq_j,ind_j =freq_salary(j_salary.Q9_1)
freq_k,ind_k =freq_salary(k_salary.Q9_1)
freq_l,ind_l =freq_salary(l_salary.Q9_1)
w=0.1
plt.figure(figsize=(20,10))
plt.xticks(ind_b,L,rotation='vertical')
plt.bar(ind_a-w*6/2,freq_a,width=w,label='18-21')
plt.bar(ind_b-w*5/2,freq_b,width=w,label='22-24')
plt.bar(ind_c-w*4/2,freq_c,width=w,label='25-29')
plt.bar(ind_d-w*3/2,freq_d,width=w,label='30-34')
plt.bar(ind_e-w*2/2,freq_e,width=w,label='35-39')
plt.bar(ind_f-w*1/2,freq_f,width=w,label='40-44')
plt.bar(ind_g+w*1/2,freq_g,width=w,label='45-49')
plt.bar(ind_h+w*2/2,freq_h,width=w,label='50-54')
plt.bar(ind_i+w*3/2,freq_i,width=w,label='55-59')
plt.bar(ind_j+w*4/2,freq_j,width=w,label='60-69')
plt.bar(ind_k+w*5/2,freq_k,width=w,label='70-79')
plt.bar(ind_l+w*6/2,freq_l,width=w,label='80+')

plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Income and Age')
plt.legend()

energy_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Energy/Mining'].index)]
goverment_salary=Top_salary.loc[list(personal_data[personal_data.Q7 == 'Government/Public Service'].index)]
shipping_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Shipping/Transportation'].index)]
other_salary=Top_salary.loc[list(personal_data[personal_data.Q7 == 'Other'].index)]
academics_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Academics/Education'].index)]
online_serv_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Online Service/Internet-based Services'].index)]
computers_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Computers/Technology'].index)]
sports_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Hospitality/Entertainment/Sports'].index)]
marketing_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Marketing/CRM'].index)]
insurance_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Insurance/Risk Assessment'].index)]
online_buss_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Online Business/Internet-based Sales'].index)]
broadcasting_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Broadcasting/Communications'].index)]
sales_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Retail/Sales'].index)]
accounting_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Accounting/Finance'].index)]
military_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Military/Security/Defense'].index)]
service_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Non-profit/Service'].index)]
student_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'I am a student'].index)]
fabrication_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Manufacturing/Fabrication'].index)]
medical_salary=Top_salary.loc[list(personal_data[personal_data.Q7== 'Medical/Pharmaceutical'].index)]

freq_energy,ind_energy =freq_salary(energy_salary.Q9_1)
freq_goverment,ind_goverment =freq_salary(goverment_salary.Q9_1)
freq_shipping,ind_shipping =freq_salary(shipping_salary.Q9_1)
freq_other,ind_other=freq_salary(other_salary.Q9_1)
freq_academics,ind_academics =freq_salary(academics_salary.Q9_1)
freq_online_serv,ind_online_serv =freq_salary(online_serv_salary.Q9_1)
freq_computers,ind_computers =freq_salary(computers_salary.Q9_1)
freq_sports,ind_sports=freq_salary(sports_salary.Q9_1)
freq_marketing,ind_marketing =freq_salary(marketing_salary.Q9_1)
freq_insurance,ind_insurance =freq_salary(insurance_salary.Q9_1)
freq_online_buss,ind_online_buss =freq_salary(online_buss_salary.Q9_1)
freq_broadcasting,ind_broadcasting =freq_salary(broadcasting_salary.Q9_1)
freq_sales,ind_sales =freq_salary(sales_salary.Q9_1)
freq_accounting,ind_accounting =freq_salary(accounting_salary.Q9_1)
freq_military,ind_military=freq_salary(military_salary.Q9_1)
freq_service,ind_service =freq_salary(service_salary.Q9_1)
freq_student,ind_student =freq_salary(student_salary.Q9_1)
freq_fabrication,ind_fabrication =freq_salary(fabrication_salary.Q9_1)
freq_medical,ind_medical =freq_salary(medical_salary.Q9_1)

L=('90-100.000','100-125.000','125-150.000','150-200.000','200-250.000',
   ',250-300.000','300-400.000','400-500.000','500.000+')
w=0.15
plt.figure(figsize=(20,10))
plt.xticks(ind_computers,L,rotation='vertical')
plt.bar(ind_goverment-w*5/2,freq_goverment,width=w,label='Government/Public Service')
plt.bar(ind_shipping-w*4/2,freq_shipping,width=w,label='Shipping/Transportation')
plt.bar(ind_online_serv-w*3/2,freq_online_serv,width=w,label='Other')
plt.bar(ind_other-w*2/2,freq_other,width=w,label='Academics/Education')
plt.bar(ind_academics-w*1/2,freq_academics,width=w,label='Online Service/Internet-based Services')
plt.bar(ind_computers+w*1/2,freq_computers,width=w,label='Computers/Technology')
plt.bar(ind_sports+w*2/2,freq_sports,width=w,label='Hospitality/Entertainment/Sports')
plt.bar(ind_marketing+w*3/2,freq_marketing,width=w,label='Marketing/CRM')
plt.bar(ind_insurance+w*4/2,freq_insurance,width=w,label='Insurance/Risk Assessment')
plt.bar(ind_online_buss+w*5/2,freq_online_buss,width=w,label='Online Business/Internet-based Sales')
plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Income and Industries(Top 20% of all incomes)')
plt.legend()
L=('90-100.000','100-125.000','125-150.000','150-200.000','200-250.000',
   ',250-300.000','300-400.000','400-500.000','500.000+')
w=0.17
plt.figure(figsize=(20,10))
plt.xticks(ind_b,L,rotation='vertical')
plt.bar(ind_sales-w*4/2,freq_sales,width=w,label='Retail/Sales')
plt.bar(ind_accounting-w*3/2,freq_accounting,width=w,label='Accounting/Finance')
plt.bar(ind_military-w*2/2,freq_military,width=w,label='Military/Security/Defense')
plt.bar(ind_service-w*1/2,freq_service,width=w,label='Non-profit/Service')
plt.bar(ind_energy+w*1/2,freq_energy,width=w,label='Energy/Mining')
plt.bar(ind_broadcasting+w*2/2,freq_broadcasting,width=w,label='Broadcasting/Communications')
plt.bar(ind_student+w*3/2,freq_student,width=w,label='I am a student')
plt.bar(ind_fabrication+w*4/2,freq_fabrication,width=w,label='Manufacturing/Fabrication')
plt.bar(ind_medical+w*5/2,freq_medical,width=w,label='Medical/Pharmaceutical')
plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Income and Industries(Top 20% of all incomes)')
plt.legend()
prof_salary=personal_data.loc[list(personal_data[personal_data.Q4== 'Professional degree'].index)]
mast_salary=personal_data.loc[list(personal_data[personal_data.Q4 == 'Master’s degree'].index)]
bach_salary=personal_data.loc[list(personal_data[personal_data.Q4== 'Bachelor’s degree'].index)]
nofor_salary=personal_data.loc[list(personal_data[personal_data.Q4 == 'No formal education past high school'].index)]
doc_salary=personal_data.loc[list(personal_data[personal_data.Q4== 'Doctoral degree'].index)]
somecol_salary=personal_data.loc[list(personal_data[personal_data.Q4== 'Some college/university study without earning a bachelor’s degree'].index)]

freq_prof,ind_prof =freq_salary(prof_salary.Q9_1)
freq_mast,ind_mast =freq_salary(mast_salary.Q9_1)
freq_bach,ind_bach =freq_salary(bach_salary.Q9_1)
freq_nofor,ind_nofor=freq_salary(nofor_salary.Q9_1)
freq_doc,ind_doc =freq_salary(doc_salary.Q9_1)
freq_somecol,ind_somecol =freq_salary(somecol_salary.Q9_1)

L=('0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000','50-60,000', '60-70,000', '70-80,000','80-90,000',
   '90-100,000','100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000','300-400,000', '400-500,000',
   '500,000+')
w=0.2
plt.figure(figsize=(20,10))
plt.xticks(ind_b,L,rotation='vertical')

plt.bar(ind_doc-w*3/2,freq_doc,width=w,label='Doctoral degree')
plt.bar(ind_prof-w*2/2,freq_prof,width=w,label='Professional degree')
plt.bar(ind_mast-w*1/2,freq_mast,width=w,label='Master’s degree')
plt.bar(ind_bach+w*1/2,freq_bach,width=w,label='Bachelor’s degree')
plt.bar(ind_nofor+w*2/2,freq_nofor,width=w,label='No formal education past high school')
plt.bar(ind_somecol+w*3/2,freq_somecol,width=w,label='Some college/university study without earning a bachelor’s degree')


plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Income and Level of Education')
plt.legend()
L=('90-100,000','100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000','300-400,000', '400-500,000',
   '500,000+')
a_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '15-20'].index)]
b_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '0-1'].index)]
c_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '5-10'].index)]
d_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '10-15'].index)]
e_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '30 +'].index)]
f_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '2-3'].index)]
g_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '20-25'].index)]
h_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '3-4'].index)]
i_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '1-2'].index)]
j_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '4-5'].index)]
k_salary=Top_salary.loc[list(personal_data[personal_data.Q8 == '25-30'].index)]

freq_a,ind_a =freq_salary(a_salary.Q9_1)
freq_b,ind_b =freq_salary(b_salary.Q9_1)
freq_c,ind_c =freq_salary(c_salary.Q9_1)
freq_d,ind_d =freq_salary(d_salary.Q9_1)
freq_e,ind_e =freq_salary(e_salary.Q9_1)
freq_f,ind_f =freq_salary(f_salary.Q9_1)
freq_g,ind_g =freq_salary(g_salary.Q9_1)
freq_h,ind_h =freq_salary(h_salary.Q9_1)
freq_i,ind_i =freq_salary(i_salary.Q9_1)
freq_j,ind_j =freq_salary(j_salary.Q9_1)
freq_k,ind_k =freq_salary(k_salary.Q9_1)
w=0.2
plt.figure(figsize=(20,10))
plt.xticks(ind_e,L,rotation='vertical')

plt.bar(ind_b-w*3/2,freq_b,width=w,label='0-1')
plt.bar(ind_i-w*2/2,freq_i,width=w,label='1-2')
plt.bar(ind_f-w*1/2,freq_f,width=w,label='2-3')
plt.bar(ind_h+w*1/2,freq_h,width=w,label='3-4')
plt.bar(ind_j+w*2/2,freq_j,width=w,label='4-5')
plt.bar(ind_c+w*3/2,freq_c,width=w,label='5-10')


plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Income and Years of Experience(Top 20% of all incomes)')
plt.legend()
w=0.2
plt.figure(figsize=(20,10))
plt.xticks(ind_e,L,rotation='vertical')
plt.bar(ind_d-w*3/2,freq_d,width=w,label='10-15')
plt.bar(ind_a-w*2/2,freq_a,width=w,label='15-20')
plt.bar(ind_g-w*1/2,freq_g,width=w,label='20-25')
plt.bar(ind_k+w*1/2,freq_k,width=w,label='25-30')
plt.bar(ind_e+w*2/2,freq_e,width=w,label='30 +')
plt.xlabel('Income [Dollars]')
plt.ylabel('Normalized Frequencies(%)')
plt.title('Income and Years of Experience(Top 20% of all incomes)')
plt.legend()
#Assigning dummies for the different answers of the personal data
dummiesQ1 = pd.get_dummies(personal_data['Q1'], prefix='Q1')
dummiesQ2 = pd.get_dummies(personal_data['Q2'], prefix='Q2')
dummiesQ4 = pd.get_dummies(personal_data['Q4'], prefix='Q4')
dummiesQ5 = pd.get_dummies(personal_data['Q5'], prefix='Q5')
dummiesQ6 = pd.get_dummies(personal_data['Q6'], prefix='Q6')
dummiesQ8 = pd.get_dummies(personal_data['Q8'], prefix='Q8')
dummiesQ9 = pd.get_dummies(personal_data['Q9'], prefix='Q9')
#All the dummies together
data = pd.concat([dummiesQ1,dummiesQ2,dummiesQ4,dummiesQ5,dummiesQ8,dummiesQ9], axis=1)
#All the possible answers for all the personal data questions
x_features = data.columns[:]
#Function to calculate projection of vectors
def proy(a,b):
    return np.dot(a,b)/np.sqrt(np.dot(a,a))
cov_matrix = np.array(data[x_features].cov())
val, vec = la.eig(cov_matrix)
vec = vec[:,val.argsort()[::-1]]
val = val[val.argsort()[::-1]]
vec1 = vec[:,0]
vec2 = -vec[:,1]
#New coordinates of the data
new_c = []
labels = []
for index, rows in data.iterrows():
    fila = rows[x_features].values
    new_c.append([proy(vec1,fila),proy(vec2,fila)])
new_c=np.array(new_c)
plt.scatter(new_c[:,0],new_c[:,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

#Kmeans for 11 clusters
cluster11 = KMeans(n_clusters=11, random_state=10)
cluster_labels = cluster11.fit_predict(data[x_features])
data['clusters'] = cluster_labels
data[data.clusters==8].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data[data.clusters==6].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
range_n_clusters = np.arange(94,99)
X = data[x_features]
score = []
for n_clusters in range_n_clusters:
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    score.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
plt.plot(range_n_clusters,score)
plt.xlabel('K')
plt.ylabel('silhouette score')
plt.grid()
cluster97 = KMeans(n_clusters=97, random_state=10)
cluster_labels = cluster97.fit_predict(data[x_features])
data['clusters'] = cluster_labels
data[data.clusters==12].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data[data.clusters==33].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data[data.clusters==67].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data[data.clusters==88].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
Q5=MCR_A.iloc[:,6].copy()
Q6=MCR_A.iloc[:,7].copy()
Q9=MCR_A.iloc[:,12].copy()
Q9_1=personal_data.iloc[:,9].copy()
Q9_2=personal_data.iloc[:,10].copy()
#Selecting the answers for the computational knowledge questions
Q10=MCR_A.iloc[:,13].copy()
Q23=MCR_A.iloc[:,126].copy()
Q24=MCR_A.iloc[:,127].copy()
Q25=MCR_A.iloc[:,128].copy()
Q26=MCR_A.iloc[:,129].copy()
Q32=MCR_A.iloc[:,263].copy()
Q37=MCR_A.iloc[:,305].copy()
Q11=MCR_A.iloc[:,14:21].copy()
Q36=MCR_A.iloc[:,291:304].copy()
Q38=MCR_A.iloc[:,307:329].copy()
Q44=MCR_A.iloc[:,343:349].copy()
Q45=MCR_A.iloc[:,349:355].copy()
Q47=MCR_A.iloc[:,356:372].copy()
#Here a create dummies for all the possible answers for each of the questions a named before
dummiesQ5 = pd.get_dummies(MCR_A['Q5'], prefix='Q5')
dummiesQ6 = pd.get_dummies(MCR_A['Q6'], prefix='Q6')
dummiesQ9 = pd.get_dummies(MCR_A['Q9'], prefix='Q9')
dummiesQ9_1 = pd.get_dummies(personal_data['Q9_1'], prefix='Q9_1')
dummiesQ9_2 = pd.get_dummies(personal_data['Q9_2'], prefix='Q9_2')
dummiesQ10 = pd.get_dummies(MCR_A['Q10'], prefix='Q10')
dummiesQ23 = pd.get_dummies(MCR_A['Q23'], prefix='Q23')
dummiesQ24 = pd.get_dummies(MCR_A['Q24'], prefix='Q24')
dummiesQ25 = pd.get_dummies(MCR_A['Q25'], prefix='Q25')
dummiesQ26 = pd.get_dummies(MCR_A['Q26'], prefix='Q26')
dummiesQ32 = pd.get_dummies(MCR_A['Q32'], prefix='Q32')
dummiesQ37 = pd.get_dummies(MCR_A['Q37'], prefix='Q37')
dummiesQ40 = pd.get_dummies(MCR_A['Q40'], prefix='Q40')
dummiesQ11_1 = pd.get_dummies(MCR_A['Q11_Part_1'], prefix='Q11')
dummiesQ11_2 = pd.get_dummies(MCR_A['Q11_Part_2'], prefix='Q11')
dummiesQ11_3 = pd.get_dummies(MCR_A['Q11_Part_3'], prefix='Q11')
dummiesQ11_4 = pd.get_dummies(MCR_A['Q11_Part_4'], prefix='Q11')
dummiesQ11_5 = pd.get_dummies(MCR_A['Q11_Part_5'], prefix='Q11')
dummiesQ11_6 = pd.get_dummies(MCR_A['Q11_Part_6'], prefix='Q11')
dummiesQ11_7 = pd.get_dummies(MCR_A['Q11_Part_7'], prefix='Q11')
dummiesQ36_1 = pd.get_dummies(MCR_A['Q36_Part_1'], prefix='Q36')
dummiesQ36_2 = pd.get_dummies(MCR_A['Q36_Part_2'], prefix='Q36')
dummiesQ36_3 = pd.get_dummies(MCR_A['Q36_Part_3'], prefix='Q36')
dummiesQ36_4 = pd.get_dummies(MCR_A['Q36_Part_4'], prefix='Q36')
dummiesQ36_5 = pd.get_dummies(MCR_A['Q36_Part_5'], prefix='Q36')
dummiesQ36_6 = pd.get_dummies(MCR_A['Q36_Part_6'], prefix='Q36')
dummiesQ36_7 = pd.get_dummies(MCR_A['Q36_Part_7'], prefix='Q36')
dummiesQ36_8 = pd.get_dummies(MCR_A['Q36_Part_8'], prefix='Q36')
dummiesQ36_9 = pd.get_dummies(MCR_A['Q36_Part_9'], prefix='Q36')
dummiesQ36_10 = pd.get_dummies(MCR_A['Q36_Part_10'], prefix='Q36')
dummiesQ36_11 = pd.get_dummies(MCR_A['Q36_Part_11'], prefix='Q36')
dummiesQ36_12 = pd.get_dummies(MCR_A['Q36_Part_12'], prefix='Q36')
dummiesQ36_13 = pd.get_dummies(MCR_A['Q36_Part_13'], prefix='Q36')
dummiesQ44_1 = pd.get_dummies(MCR_A['Q44_Part_1'], prefix='Q44')
dummiesQ44_2 = pd.get_dummies(MCR_A['Q44_Part_2'], prefix='Q44')
dummiesQ44_3 = pd.get_dummies(MCR_A['Q44_Part_3'], prefix='Q44')
dummiesQ44_4 = pd.get_dummies(MCR_A['Q44_Part_4'], prefix='Q44')
dummiesQ44_5 = pd.get_dummies(MCR_A['Q44_Part_5'], prefix='Q44')
dummiesQ44_6 = pd.get_dummies(MCR_A['Q44_Part_6'], prefix='Q44')
dummiesQ45_1 = pd.get_dummies(MCR_A['Q45_Part_1'], prefix='Q45')
dummiesQ45_2 = pd.get_dummies(MCR_A['Q45_Part_2'], prefix='Q45')
dummiesQ45_3 = pd.get_dummies(MCR_A['Q45_Part_3'], prefix='Q45')
dummiesQ45_4 = pd.get_dummies(MCR_A['Q45_Part_4'], prefix='Q45')
dummiesQ45_5 = pd.get_dummies(MCR_A['Q45_Part_5'], prefix='Q45')
dummiesQ45_6 = pd.get_dummies(MCR_A['Q45_Part_6'], prefix='Q45')
dummiesQ47_1 = pd.get_dummies(MCR_A['Q47_Part_1'], prefix='Q47')
dummiesQ47_2 = pd.get_dummies(MCR_A['Q47_Part_2'], prefix='Q47')
dummiesQ47_3 = pd.get_dummies(MCR_A['Q47_Part_3'], prefix='Q47')
dummiesQ47_4 = pd.get_dummies(MCR_A['Q47_Part_4'], prefix='Q47')
dummiesQ47_5 = pd.get_dummies(MCR_A['Q47_Part_5'], prefix='Q47')
dummiesQ47_6 = pd.get_dummies(MCR_A['Q47_Part_6'], prefix='Q47')
dummiesQ47_7 = pd.get_dummies(MCR_A['Q47_Part_7'], prefix='Q47')
dummiesQ47_8 = pd.get_dummies(MCR_A['Q47_Part_8'], prefix='Q47')
dummiesQ47_9 = pd.get_dummies(MCR_A['Q47_Part_9'], prefix='Q47')
dummiesQ47_10 = pd.get_dummies(MCR_A['Q47_Part_10'], prefix='Q47')
dummiesQ47_11 = pd.get_dummies(MCR_A['Q47_Part_11'], prefix='Q47')
dummiesQ47_12 = pd.get_dummies(MCR_A['Q47_Part_12'], prefix='Q47')
dummiesQ47_13 = pd.get_dummies(MCR_A['Q47_Part_13'], prefix='Q47')
dummiesQ47_14 = pd.get_dummies(MCR_A['Q47_Part_14'], prefix='Q47')
dummiesQ47_15 = pd.get_dummies(MCR_A['Q47_Part_15'], prefix='Q47')
dummiesQ47_16 = pd.get_dummies(MCR_A['Q47_Part_16'], prefix='Q47')
dummiesQ47=pd.concat([dummiesQ47_1,dummiesQ47_2,dummiesQ47_3,dummiesQ47_4,dummiesQ47_5,dummiesQ47_6,dummiesQ47_7,dummiesQ47_8,dummiesQ47_9,dummiesQ47_10,dummiesQ47_11,dummiesQ47_12,dummiesQ47_13], axis=1)
dummiesQ45=pd.concat([dummiesQ45_1,dummiesQ45_2,dummiesQ45_3,dummiesQ45_4,dummiesQ45_5,dummiesQ45_6], axis=1)
dummiesQ44=pd.concat([dummiesQ44_1,dummiesQ44_2,dummiesQ44_3,dummiesQ44_4,dummiesQ44_5,dummiesQ44_6], axis=1)
dummiesQ36=pd.concat([dummiesQ36_1,dummiesQ36_2,dummiesQ36_3,dummiesQ36_4,dummiesQ36_5,dummiesQ36_6,dummiesQ36_7,dummiesQ36_8,dummiesQ36_9,dummiesQ36_10,dummiesQ36_11,dummiesQ36_12,dummiesQ36_13], axis=1)
dummiesQ11=pd.concat([dummiesQ11_1,dummiesQ11_2,dummiesQ11_3,dummiesQ11_4,dummiesQ11_5,dummiesQ11_6,dummiesQ11_7], axis=1)
#Here I gruped the dummies in the subcategories
data1 = pd.concat([dummiesQ5,dummiesQ6,dummiesQ9_1,dummiesQ10,dummiesQ11], axis=1)
data2 = pd.concat([dummiesQ5,dummiesQ6,dummiesQ9_1,dummiesQ23,dummiesQ24,dummiesQ25,dummiesQ26], axis=1)
data3 = pd.concat([dummiesQ5,dummiesQ6,dummiesQ36,dummiesQ37], axis=1)
data4 = pd.concat([dummiesQ5,dummiesQ6,dummiesQ44,dummiesQ45,dummiesQ47], axis=1)
data1 = data1.dropna()
data2 = data2.dropna()
data3 = data3.dropna()
data4 = data4.dropna()
x_features1 = data1.columns[:]
x_features2 = data2.columns[:]
x_features3 = data3.columns[:]
x_features4 = data4.columns[:]
cov_matrix = np.array(data1[x_features1].cov())
val, vec = la.eig(cov_matrix)
vec = vec[:,val.argsort()[::-1]]
val = val[val.argsort()[::-1]]
vec1 = vec[:,0]
vec2 = -vec[:,1]
new_c = []
labels = []
for index, rows in data1.iterrows():
    fila = rows[x_features1].values
    new_c.append([proy(vec1,fila),proy(vec2,fila)])
new_c=np.array(new_c)
plt.plot(new_c[:,0],new_c[:,1],'.',)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
range_n_clusters = np.arange(88,91)
X = data1[x_features1]
score = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    score.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
plt.plot(range_n_clusters,score)
plt.xlabel('K')
plt.ylabel('silhouette score')
plt.xlim(88,90)
plt.ylim(0.076,0.08)
plt.grid()
cluster89 = KMeans(n_clusters=89, random_state=10)
cluster_labels = cluster89.fit_predict(data1[x_features1])
data1['clusters'] = cluster_labels
data1[data1.clusters==20].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data1[data1.clusters==33].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data1[data1.clusters==67].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data1[data1.clusters==88].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
cov_matrix = np.array(data2[x_features2].cov())
val, vec = la.eig(cov_matrix)
vec = vec[:,val.argsort()[::-1]]
val = val[val.argsort()[::-1]]
vec1 = vec[:,0]
vec2 = -vec[:,1]
new_c = []
labels = []
for index, rows in data2.iterrows():
    fila = rows[x_features2].values
    new_c.append([proy(vec1,fila),proy(vec2,fila)])
new_c=np.array(new_c)
plt.plot(new_c[:,0],new_c[:,1],'.',)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
range_n_clusters = np.arange(72,78)
X = data2[x_features2]
score = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    score.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
plt.plot(range_n_clusters,score)
plt.xlabel('K')
plt.ylabel('silhouette score')
plt.grid()
plt.xlim(72,77)
plt.ylim(0.05,0.065)
cluster75 = KMeans(n_clusters=75, random_state=10)
cluster_labels = cluster75.fit_predict(data2[x_features2])
data2['clusters'] = cluster_labels
data2[data2.clusters==20].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data2[data2.clusters==33].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data2[data2.clusters==67].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data2[data2.clusters==88].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
cov_matrix = np.array(data3[x_features3].cov())
val, vec = la.eig(cov_matrix)
vec = vec[:,val.argsort()[::-1]]
val = val[val.argsort()[::-1]]
vec1 = vec[:,0]
vec2 = -vec[:,1]
new_c = []
labels = []
for index, rows in data3.iterrows():
    fila = rows[x_features3].values
    new_c.append([proy(vec1,fila),proy(vec2,fila)])
new_c=np.array(new_c)
plt.plot(new_c[:,0],new_c[:,1],'.',)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
range_n_clusters = np.arange(95,100)
X = data3[x_features3]
score = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    score.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)  
plt.plot(range_n_clusters,score)
plt.xlabel('K')
plt.ylabel('silhouette score')
plt.grid()
plt.xlim(96,99)
cluster98 = KMeans(n_clusters=98, random_state=10)
cluster_labels = cluster98.fit_predict(data3[x_features3])
data3['clusters'] = cluster_labels
data3[data3.clusters==20].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data3[data3.clusters==33].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data3[data3.clusters==67].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data3[data3.clusters==88].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
cov_matrix = np.array(data4[x_features4].cov())
val, vec = la.eig(cov_matrix)
vec = vec[:,val.argsort()[::-1]]
val = val[val.argsort()[::-1]]
vec1 = vec[:,0]
vec2 = -vec[:,1]
new_c = []
labels = []
for index, rows in data4.iterrows():
    fila = rows[x_features4].values
    new_c.append([proy(vec1,fila),proy(vec2,fila)])
new_c=np.array(new_c)
plt.plot(new_c[:,0],new_c[:,1],'.',)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
range_n_clusters = np.arange(96,105)
X = data4[x_features4]
score = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    score.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
plt.plot(range_n_clusters,score)
plt.xlabel('K')
plt.ylabel('silhouette score')
plt.grid()
cluster102 = KMeans(n_clusters=102, random_state=10)
cluster_labels = cluster102.fit_predict(data4[x_features4])
data4['clusters'] = cluster_labels
data4[data4.clusters==20].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data4[data4.clusters==33].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data4[data4.clusters==67].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data4[data4.clusters==88].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data4[data4.clusters==25].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data4[data4.clusters==47].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data4[data4.clusters==72].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
data4[data4.clusters==83].sum()[0:-2].plot.bar(fig = plt.figure(figsize = (20,5)))
