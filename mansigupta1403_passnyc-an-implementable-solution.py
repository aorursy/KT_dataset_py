print('Initial steps: Ingesting, pre processing data')
#importing libraries

import matplotlib.pyplot as plt
import matplotlib
import plotly.plotly as py
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
%matplotlib inline
from math import sin, cos, sqrt, atan2, radians
#Loading needed databases

School_exp = pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")
D5 = pd.read_csv("../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv")
School_safety = pd.read_csv("../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv")
SHSAT = pd.read_csv("../input/2017-2018-shsat-admissions-test-offers-by-schools/2017-2018 SHSAT Admissions Test Offers By Sending School.csv")
Center_list = pd.read_csv("../input/passnyc-resource-centers/passnyc-resource-centers.csv")
School_demo = pd.read_csv("../input/ny-school-demographics-and-accountability-snapshot/2006-2012-school-demographics-and-accountability-snapshot.csv")
#Preprocessing

D5.set_index('Year of SHST',inplace=True, drop=True) #Changing index 
D5 = D5.drop([2013, 2014, 2015]) ##Dropping rows that will not be used

School_safety.set_index('School Year',inplace=True, drop=True) #Changing index 
School_safety = School_safety.drop(['2013-14','2014-15'])  ##Dropping columns that will not be used

School_safety = School_safety.dropna(subset = ['DBN'])

School_exp = pd.merge(School_exp, School_safety, how='left', left_on='Location Code', right_on='DBN')
School_exp = pd.merge(School_exp, SHSAT, how='left', left_on='DBN', right_on='School DBN')


School_exp.set_index('DBN',inplace=True, drop=True)
D5.set_index('DBN',inplace=True, drop=True)
School_exp = School_exp.join(D5)

School_exp = School_exp.drop(columns=['Adjusted Grade','New?','Other Location Code in LCGMS','Location Code_x','SED Code','Grade Low','Grade 3 ELA - All Students Tested','Grade 3 ELA 4s - All Students','Grade 3 ELA 4s - American Indian or Alaska Native','Grade 3 ELA 4s - Black or African American','Grade 3 ELA 4s - Hispanic or Latino','Grade 3 ELA 4s - Asian or Pacific Islander','Grade 3 ELA 4s - White','Grade 3 ELA 4s - Multiracial','Grade 3 ELA 4s - Limited English Proficient','Grade 3 ELA 4s - Economically Disadvantaged','Grade 3 Math - All Students tested','Grade 3 Math 4s - All Students','Grade 3 Math 4s - American Indian or Alaska Native','Grade 3 Math 4s - Black or African American','Grade 3 Math 4s - Hispanic or Latino','Grade 3 Math 4s - Asian or Pacific Islander','Grade 3 Math 4s - White','Grade 3 Math 4s - Multiracial','Grade 3 Math 4s - Limited English Proficient','Grade 3 Math 4s - Economically Disadvantaged','Grade 4 ELA - All Students Tested','Grade 4 ELA 4s - All Students','Grade 4 ELA 4s - American Indian or Alaska Native','Grade 4 ELA 4s - Black or African American','Grade 4 ELA 4s - Hispanic or Latino','Grade 4 ELA 4s - Asian or Pacific Islander','Grade 4 ELA 4s - White','Grade 4 ELA 4s - Multiracial','Grade 4 ELA 4s - Limited English Proficient','Grade 4 ELA 4s - Economically Disadvantaged','Grade 4 Math - All Students Tested','Grade 4 Math 4s - All Students','Grade 4 Math 4s - American Indian or Alaska Native','Grade 4 Math 4s - Black or African American','Grade 4 Math 4s - Hispanic or Latino','Grade 4 Math 4s - Asian or Pacific Islander','Grade 4 Math 4s - White','Grade 4 Math 4s - Multiracial','Grade 4 Math 4s - Limited English Proficient','Grade 4 Math 4s - Economically Disadvantaged','Grade 5 ELA - All Students Tested','Grade 5 ELA 4s - All Students','Grade 5 ELA 4s - American Indian or Alaska Native','Grade 5 ELA 4s - Black or African American','Grade 5 ELA 4s - Hispanic or Latino','Grade 5 ELA 4s - Asian or Pacific Islander','Grade 5 ELA 4s - White','Grade 5 ELA 4s - Multiracial','Grade 5 ELA 4s - Limited English Proficient','Grade 5 ELA 4s - Economically Disadvantaged','Grade 5 Math - All Students Tested','Grade 5 Math 4s - All Students','Grade 5 Math 4s - American Indian or Alaska Native','Grade 5 Math 4s - Black or African American','Grade 5 Math 4s - Hispanic or Latino','Grade 5 Math 4s - Asian or Pacific Islander','Grade 5 Math 4s - White','Grade 5 Math 4s - Multiracial','Grade 5 Math 4s - Limited English Proficient','Grade 5 Math 4s - Economically Disadvantaged','Grade 6 ELA - All Students Tested','Grade 6 ELA 4s - All Students','Grade 6 ELA 4s - American Indian or Alaska Native','Grade 6 ELA 4s - Black or African American','Grade 6 ELA 4s - Hispanic or Latino','Grade 6 ELA 4s - Asian or Pacific Islander','Grade 6 ELA 4s - White','Grade 6 ELA 4s - Multiracial','Grade 6 ELA 4s - Limited English Proficient','Grade 6 ELA 4s - Economically Disadvantaged','Grade 6 Math - All Students Tested','Grade 6 Math 4s - All Students','Grade 6 Math 4s - American Indian or Alaska Native','Grade 6 Math 4s - Black or African American','Grade 6 Math 4s - Hispanic or Latino','Grade 6 Math 4s - Asian or Pacific Islander','Grade 6 Math 4s - White','Grade 6 Math 4s - Multiracial','Grade 6 Math 4s - Limited English Proficient','Grade 6 Math 4s - Economically Disadvantaged','Grade 7 ELA 4s - American Indian or Alaska Native','Grade 7 ELA 4s - Asian or Pacific Islander','Grade 7 ELA 4s - Multiracial','Grade 7 ELA 4s - Economically Disadvantaged','Grade 7 Math 4s - American Indian or Alaska Native','Grade 7 Math 4s - Asian or Pacific Islander','Grade 7 Math 4s - Multiracial','Grade 8 ELA 4s - American Indian or Alaska Native','Grade 8 ELA 4s - Asian or Pacific Islander','Grade 8 Math 4s - American Indian or Alaska Native','Grade 8 Math 4s - Asian or Pacific Islander','Grade 8 Math 4s - Multiracial','Building Code','Location Name','Location Code_y','Address','Borough_x','Geographical District Code','Register','Building Name','# Schools','Schools in Building','Postcode','Latitude_y','Longitude_y','BBL','NTA','School DBN','Borough_y','School Category','School Name_y'])
                                      
School_exp = School_exp[School_exp['Grade High'] != '0K']
School_exp['Grade High'] = School_exp['Grade High'].astype(np.object).astype(int)

School_demo.set_index('schoolyear', inplace=True, drop=True)
School_demo = School_demo.drop([20052006, 20062007, 20072008, 20082009, 20092010, 20102011])
School_demo = School_demo.fillna(0)
School_demo.set_index('DBN',inplace=True, drop=True)
School_demo.grade6 = School_demo[['grade6']].convert_objects(convert_numeric=True).fillna(0)
School_demo.grade7 = School_demo[['grade7']].convert_objects(convert_numeric=True).fillna(0)
School_demo.grade8 = School_demo[['grade8']].convert_objects(convert_numeric=True).fillna(0)
School_demo['Reach'] = School_demo['grade7'] + School_demo['grade8'] + School_demo['grade6']
School_demo['grade7'] = School_demo['grade7'].astype(np.object).astype(float)

School_exp = pd.merge(School_exp, School_demo, how='left', left_on='DBN', right_on='DBN')

for col in School_exp.columns.values:
    if col.startswith("Percent") or col.endswith("%") or col.endswith("Rate"):
        School_exp[col] = School_exp[col].astype(np.object).str.replace('%', '').astype(float)

School_exp.replace(np.NaN,0, inplace=True)

School_exp['grade7'] = School_exp['grade7'].astype(np.object).astype(float)

#Add conditional subset flags

School_exp['Minority'] = np.where(School_exp['Percent Black / Hispanic']>60, 'Minority', 'no')

School_exp['7to9'] = np.where(School_exp['Grade High']>6, 'SHST', 'no')
print('Creating our clusters')
#Creating our recommendation clusters

School_exp['Cluster 1'] = np.where((School_exp['Number of students who registered for the SHSAT']<35) & (School_exp['7to9'] == 'SHST'), 'Awareness', 'no')
School_exp['Cluster 2'] = np.where((School_exp['Cluster 1'] == 'Awareness') & (School_exp['Average ELA Proficiency'] > 2.3) & (School_exp['Average Math Proficiency'] > 2.3), 'Mentoring', 'no')
School_exp['Cluster 3'] = np.where((School_exp['Cluster 1'] == 'Awareness') & (School_exp['Supportive Environment %'] < 85), 'Afterschool programs', 'no')
School_exp['Cluster 4'] = np.where((School_exp['7to9'] == 'SHST') & (School_exp['Percent of eight grade students who received offer'] < 10) & (School_exp['Rigorous Instruction %'] < 85), 'Test prep', 'no')

#Nearest resource center locator

place = []
dist = []
for i in range(1282):
    dist.append(99999)
R = 3959.0

for i in range(1282):
    lat1 = radians(School_exp['Latitude_x'][i])
    lon1 = radians(School_exp['Longitude_x'][i])
    center_name = ''
    for j in range(82):
        lat2 = radians(Center_list['Lat'][j])
        lon2 = radians(Center_list['Long'][j])
        
        if Center_list['Test Prep'][j] == 1 and School_exp['Cluster 4'][i] == 'Test prep':
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c
            if dist[i] > distance:
                center_name = Center_list['Resource Center Name'][j]
                dist[i] = round(distance,1)
                
        if Center_list['After School Program'][j] == 1 and School_exp['Cluster 3'][i] == 'Afterschool programs':
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c
            if dist[i] > distance:
                center_name = Center_list['Resource Center Name'][j]
                dist[i] = round(distance,1)

    place.append(center_name)
School_awareness = School_exp.loc[School_exp['Cluster 1'] == 'Awareness']
School_awareness = School_awareness[['School Name_x', 'Reach']]
School_awareness['Awareness_rank']=School_awareness['Reach'].rank(ascending=0)

School_awareness = School_awareness.sort_values(by=['Awareness_rank'], ascending=[True]) #sort based on order of priority
School_awareness.to_csv('1_Increase_awareness.csv')
print(School_awareness.head(n=10))

print('Please note: The entire ranked list will be available as an output of this notebook')
Inc_participation = School_exp.loc[School_exp['Cluster 2'] == 'Mentoring']
Inc_participation = Inc_participation[['School Name_x', 'Average ELA Proficiency', 'Average Math Proficiency']]
Inc_participation['Proficiency'] = Inc_participation['Average ELA Proficiency'] + Inc_participation['Average Math Proficiency']
Inc_participation['Mentoring_rank']=Inc_participation['Proficiency'].rank(ascending=0)
Inc_participation = Inc_participation.sort_values(by = ['Mentoring_rank'], ascending=[True]) #sort based on order of priority
Inc_participation.to_csv('2_Offer_mentoring.csv')
print(Inc_participation.head(n=10))

print('Please note: The entire ranked list will be available as an output of this notebook')
Afterschool_prog = School_exp.loc[School_exp['Cluster 3'] == 'Afterschool programs']
Afterschool_prog = Afterschool_prog[['School Name_x', 'Supportive Environment %']]

Afterschool_prog['Afterschool_rank']=Afterschool_prog['Supportive Environment %'].rank(ascending=0)
Afterschool_prog = Afterschool_prog.sort_values(by = ['Afterschool_rank'], ascending=[True]) #sort based on order of priority

Afterschool_prog.to_csv('3_Afterschool_programs.csv')
print(Afterschool_prog.head(n=10))

print('Please note: The entire ranked list will be available as an output of this notebook')
Test_prep = School_exp.loc[School_exp['Cluster 4'] == 'Test prep']
Test_prep = Test_prep[['School Name_x', 'Economic Need Index', 'Rigorous Instruction %', 'Percent of eight grade students who received offer']]
Test_prep['Prep_rank']=Test_prep['Economic Need Index'].rank(ascending=0)
Test_prep = Test_prep.sort_values(by = ['Prep_rank'], ascending=[True]) #sort based on order of priority

Test_prep.to_csv('4_Test_prep.csv')
print(Test_prep.head(n=10))

print('Please note: The entire ranked list will be available as an output of this notebook')
#Making the csv for distance between all the Partner Resource Centers and the schools

rows = 1283
columns = 83
Center_school = [[0 for x in range(columns)] for y in range(rows)] 
for i in range(1,1283):
    lat1 = radians(School_exp['Latitude_x'][i-1])
    lon1 = radians(School_exp['Longitude_x'][i-1])
    for j in range(1,83):
        lat2 = radians(Center_list['Lat'][j-1])
        lon2 = radians(Center_list['Long'][j-1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        Center_school[i][j] = round(distance,1)

Columns_csv = [0 for x in range(83)]

for i in range(1,1282):
    Center_school[i][0] = School_exp['School Name_x'][i]
for j in range(1,82):
    Center_school[0][j] = Center_list['Resource Center Name'][j]

Resource_school = pd.DataFrame(Center_school)
Resource_school.to_csv('5_Resource center vs Schools Distance.csv')
#Making csv for Cluster 3 : Increasing Awareness

Cluster_3 = School_exp[['School Name_x', 'Cluster 3']]
Cluster_3['Nearest Resource Center'] = place
Cluster_3['Distance to the Nearest Resource Center'] = dist
cols = Cluster_3.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, '')) else x)
Cluster_3.columns = cols
Cluster_3 = Cluster_3.drop(Cluster_3[Cluster_3.Cluster_3 == 'no'].index)
Cluster_3 = Cluster_3.drop(columns=['Cluster_3'])
cols = Cluster_3.columns
cols = cols.map(lambda x: x.replace('_', ' ') if isinstance(x, (str, '')) else x)
Cluster_3.columns = cols
Cluster_3.to_csv('6_Partners for After School Programs.csv')
print(Cluster_3.head(n=10))

print('Please note: The entire list will be available as an output of this notebook')
#Making csv for Cluster 4 : Test Prep

Cluster_4 = School_exp[['School Name_x', 'Cluster 4']]
Cluster_4['Nearest Resource Center'] = place
Cluster_4['Distance to the Nearest Resource Center'] = dist
cols = Cluster_4.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, '')) else x)
Cluster_4.columns = cols
Cluster_4 = Cluster_4.drop(Cluster_4[Cluster_4.Cluster_4 == 'no'].index)
Cluster_4 = Cluster_4.drop(columns=['Cluster_4'])
cols = Cluster_4.columns
cols = cols.map(lambda x: x.replace('_', ' ') if isinstance(x, (str, '')) else x)
Cluster_4.columns = cols
Cluster_4.to_csv('7_Partners for Test Prep.csv')
print(Cluster_4.head(n=10))

print('Please note: The entire list will be available as an output of this notebook')