import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

path_to_files = "../input"

import folium

from folium import plugins
#data in excel

universities = pd.read_excel(os.path.join(path_to_files,'university_names.xlsx'))

uni_shape = np.zeros(shape = (40,4))

universities_df = pd.DataFrame(uni_shape)



#grouping students by university (as opposed to major of study)

for i,v in enumerate(universities['Higher Education Institution'].value_counts().keys()):

    universities_df.iloc[i,0] = v



    

    

universities.fillna(value = 0, inplace = True)

#filling in NaN values to mean 0 students

current_students = universities.iloc[:,:-2]

new_series = []

current_students.info()

grouped_students = current_students.groupby(by ='Higher Education Institution')['Enrolled _ UnderGraduate','Enrolled_Post Graduate'].apply(sum)

grouped_students.head()



grouped_students['Total_students'] = grouped_students['Enrolled _ UnderGraduate'] +grouped_students['Enrolled_Post Graduate']

    

university_locations = pd.read_excel(os.path.join(path_to_files,'university_locations.xlsx'), header = None)



#inserting data into the map

for i in university_locations.iloc[:,0]:

    if i in grouped_students.index:

        grouped_students.loc[i,'lat']= float(university_locations.loc[university_locations[0] == i][1])

        grouped_students.loc[i,'long']= float(university_locations.loc[university_locations[0] == i][2])





grouped_students.to_excel('students_grouped_by_university.xlsx')



#%%

schools = pd.read_excel(os.path.join(path_to_files,'og_school_data.xlsx'), header = 0)

#%%

schools['Grades'].value_counts()

kids_only = ['FS1-Y6','FS1-Y9','KG1-KG2','KG1-G5','KG1-G10','FS2-Y6','KG1-G11','FS2-Y11','FS2-Y10','KG1-G10','FS1-Y7','Pre-Primary - Gr10','KG1-G9','FS1 - Year6','Pre Primary-Y6','Pre Primary-G9','KG1-G6','FS1-Y8','KG1-G8','KG1-G7','Y1-Y8','Pre Primary-G3','Pre Primary - G5','Pre-Primary-Gr9','FS2-Y8','Pre Primary-G7','KG1-G4','KG1 - G4','G1 - G8','Pre Primary-Y8','Pre Primary-G6','FS1 - Year 8','Pre Primary-KG2','pre-Primary - Gr5','FS1 - Year8','KG1 - KG2','FS1 - Year 9','Pre-Primary-G8','KG2-G9','Pre-Primary- Gr8','KG2- G6','FS2 - Year6']



not_primary_schools = schools[schools['Grades'].apply(lambda x: x not in kids_only)]





grouped_students.info()

#make a point for every 100 students

import folium

from folium import plugins





dubai =folium.Map([25.276381,55.368653],zoom_start = 10, zoom_control = True)

list_of_randoms = (np.random.rand(2,20))

for index, row in grouped_students.iterrows():

    if int(row['Total_students']//100) > 1:

        for i in range(int(row['Total_students']//100)+1):

            folium.Marker([row['lat']+(0.001*np.random.choice(list_of_randoms[:,0])), row['long']+(0.001*np.random.choice(list_of_randoms[:,1]))],

                        popup=index,

                        icon=folium.Icon(color='red', prefix = 'University')

                       ).add_to(dubai)

    else:

        folium.Marker([row['lat'], row['long']],

                        popup=index,

                        icon=folium.Icon(color='red', prefix = 'University')

                       ).add_to(dubai)

            
# for non primary students, create a point for every 100 students, offset by a small amount to be seen more clearly

for index, row in not_primary_schools.iterrows():

    if int(row['2018/19 Enrollments']//100) > 1:

        for i in range(int(row['2018/19 Enrollments']//500)+1):

            folium.Marker([row['Latitude']+(0.0001*np.random.choice(list_of_randoms[:,0])), row['Longitude']+(0.0001*np.random.choice(list_of_randoms[:,1]))]).add_to(dubai)

    else: 

        dubai.add_child(folium.Marker([row['Latitude'], row['Longitude']]))

            

dubai