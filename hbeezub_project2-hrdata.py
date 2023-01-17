import csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv('../input/HR_comma_sep.csv')
data.shape
data.head(20)
data.tail()
type(data)
data.isnull().any()
data.describe()
df_hr1=data.drop(['number_project', 'Work_accident','average_montly_hours','Work_accident','sales'], axis=1)
quit = df_hr1['left'].value_counts()

print(quit)
fig = plt.figure(figsize=(10,6))

ax = fig.add_axes((0,0,.5,1))

ax.set_title('Distribution of employees that leave or stay ie: Turnover')

labels = ['stayed', 'quit']

sizes = [11428, 3571]

colors = ['gold', 'darkgoldenrod']

plt.pie(sizes, labels=labels,colors = colors, autopct = '%1.0f%%',startangle=90, radius=1.5)

plt.axis("equal") #make Pie appear round & not oval

plt.show()
df_hr1.hist('satisfaction_level')

plt.xlabel('Satisfaction Level')

plt.ylabel('Number of Employees')

plt.title('Distribution of Satisfaction Levels')

plt.show()
not_employed = df_hr1[df_hr1['left']==1]

still_employed = df_hr1[df_hr1['left']==0]

n_ext_satis= not_employed[not_employed.satisfaction_level > .91]

n_very_satis = not_employed[(not_employed.satisfaction_level > .81) & (not_employed.satisfaction_level <= .91)]

n_satis = not_employed[(not_employed.satisfaction_level > .71) & (not_employed.satisfaction_level <= .81)]

n_dis = not_employed[(not_employed.satisfaction_level > .61) & (not_employed.satisfaction_level <= .71)]

n_very_dis = not_employed[not_employed.satisfaction_level <= .61]

count_n_ext_satis=n_ext_satis.count()

count_n_very_satis=n_very_satis.count()

count_n_satis=n_satis.count()

count_n_dis=n_dis.count()

count_n_very_dis=n_very_dis.count()

print(count_n_ext_satis, count_n_very_satis, count_n_satis,count_n_dis, count_n_very_dis)

objects = ('Extremely Satisfied', 'Very Satisfied',' Satisfied' ,'Dissatisfied','Very Dissatisfied')

y_pos = np.arange(len(objects))

performance = [20,478,431,36,2606]

 

plt.bar(y_pos, performance, align='center', alpha=0.5,  color='g')

plt.xticks(y_pos, objects, rotation='70')

plt.ylabel('number of employees')

plt.title('Satisfaction levels of Employees that left')

 

plt.show()



s_ext_satis= still_employed[still_employed.satisfaction_level > .91]

s_very_satis = still_employed[(still_employed.satisfaction_level > .81) & (still_employed.satisfaction_level <= .91)]

s_satis = still_employed[(still_employed.satisfaction_level > .71) & (still_employed.satisfaction_level <= .81)]

s_dis = still_employed[(still_employed.satisfaction_level > .61) & (still_employed.satisfaction_level <= .71)]

s_very_dis = still_employed[still_employed.satisfaction_level <= .61]
count_s_ext_satis=s_ext_satis.count()

count_s_very_satis=s_very_satis.count()

count_s_satis=s_satis.count()

count_s_dis=s_dis.count()

count_s_very_dis=s_very_dis.count()

print(count_s_ext_satis, count_s_very_satis, count_s_satis,count_s_dis, count_s_very_dis)
objects = ('Extremely Satisfied', 'Very Satisfied',' Satisfied' ,'Dissatisfied','Very Dissatisfied')

y_pos = np.arange(len(objects))

performance = [1540,1744,1914,1899,4331]

 

plt.bar(y_pos, performance, align='center', alpha=0.5,  color='b')

plt.xticks(y_pos, objects, rotation='70')

plt.ylabel('number of employees')

plt.title('Satisfaction levels of Employees Still at Company')

 

plt.show()
objects = ('Extremely Satisfied', 'Very Satisfied',' Satisfied' ,'Dissatisfied','Very Dissatisfied')

y_pos = np.arange(len(objects))

s_performance = [1540,1744,1914,1899,4331]

n_performance = [20,478,431,36,2606]



fig, ax = plt.subplots()

index = np.arange(5)

bar_width = 0.35

 

rects1 = plt.bar(index, s_performance, 

                 color='b',

                 label='still employed')

 

rects2 = plt.bar(index+ bar_width, n_performance,

                 color='g',

                 label='left company')

 

plt.bar(y_pos, s_performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects, rotation='70')

plt.ylabel('number of employees')

plt.xlabel('Person')

plt.title('Satisfaction levels of Employees Still at Company')

plt.legend()



plt.show()

df_hr1.plot(kind='scatter',x='satisfaction_level',y='last_evaluation', label='Job Satisfaction vs Performance Appraisal', color='orchid', figsize=(20, 10)) 

plt.show()