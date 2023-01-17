import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = "../input/openpowerlifting.csv"
data = pd.read_csv(path)
data = data[1:147149:]
data.shape
data.head(5)
data_new = data[['Name','Sex']]
data_new = data_new.drop_duplicates()
data.columns
data = data.drop(labels = ['Squat4Kg','Bench4Kg','Deadlift4Kg','Wilks'], axis = 1)
data.shape
data.tail(5)
print(data.isnull().sum())
data.fillna(method ='ffill',inplace = True)
data.isnull().sum()
data.tail(5)

gender_size = data_new.Sex.value_counts().sort_index().tolist()
gender_names = ['Female','Male']
col = ['#c973d0','#4a73ab']
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(gender_size, radius=3.3, labels=gender_names, colors = ['#e54370','#0093b7']) 
plt.setp( mypie, width=0.9, edgecolor='white')

def squat_calculate(x):
    if(x < 10.0):
        return "05-10"
    if(x >= 10.0 and x < 20.0):
        return "10-20"
    if(x >= 20.0 and x < 30.0):
        return "20-30"
    if(x >= 30.0 and x < 40.0):
        return "30-40"
    if(x >= 40.0 and x < 50.0):
        return "40-50"
    if(x >= 50.0 and x < 60.0):
        return "50-60"
    if(x >= 60.0 and x < 70.0):
        return "60-70"
    if(x >= 70.0 and x < 80.0):
        return "70-80"
    if(x >= 80.0 and x < 90.0):
        return "80-90"
    else:
        return "90-100"
    


data['Agecategory'] = pd.DataFrame(data.Age.apply(lambda x : squat_calculate(x)))

data.head(20)
data_male = pd.DataFrame(data[data['Sex'] == 'M'])
data_female = pd.DataFrame(data[data['Sex'] == 'F'])
lifting_capacity_m = pd.DataFrame(data_male.groupby('Agecategory')[['BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()
lifting_capacity_f = pd.DataFrame(data_female.groupby('Agecategory')[['BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()

#plt.figure(figsize = (20,10))
lifting_capacity_m.plot(kind = 'bar', color = ['#63cdd7','#0093b7','#005f89'], figsize = (15,10), x = 'Agecategory', rot = 30)

lifting_capacity_f.plot(kind = 'bar', color = ['#f9dff0','#f0acc3','#e54370'], figsize = (15,10), x = 'Agecategory', rot = 30)

import seaborn as sns
plt.figure(figsize = (20,15))

plt.subplot(1,3,1)

plt.ylim(0,600)
sns.violinplot(data = data, x = 'Sex', y = 'BestSquatKg',hue = 'Sex', scale = 'count',dodge = True, palette = ['#e54370','#0093b7'])
plt.style.use("fast")
plt.title('Squat Capacity by Gender')
plt.xlabel('Gender')
plt.ylabel('Squat Lifting Capacity')


plt.subplot(1,3,2)
plt.ylim(0,500)
plt.style.use("fast")
sns.violinplot(data = data, x = 'Sex', y = 'BestBenchKg',hue = 'Sex',scale = 'count',dodge = True, palette = ['#e54370','#0093b7'])
plt.xlabel('Gender')
plt.ylabel('Bench Lifting Capacity')
plt.title('Bench Capacity by Gender')


plt.subplot(1,3,3)
plt.ylim(0,500)
plt.style.use("fast")
sns.violinplot(data = data, x = 'Sex', y = 'BestDeadliftKg',hue = 'Sex',scale = 'count',dodge = True, palette = ['#e54370','#0093b7'])
plt.xlabel('Gender')
plt.ylabel('Deadlift Lifting Capacity')
plt.title('Deadlift Lifting Capacity by Gender')


plt.show()
data_male = pd.DataFrame(data[data['Sex'] == 'M'])
data_female = pd.DataFrame(data[data['Sex'] == 'F'])
bodyw_lcm = pd.DataFrame(data_male.groupby('Agecategory')[['BodyweightKg','BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()
bodyw_lcf = pd.DataFrame(data_female.groupby('Agecategory')[['BodyweightKg','BestSquatKg','BestBenchKg','BestDeadliftKg']].mean()).reset_index()


bodyw_lcm 
bodyw_lcf
bodyw_lcm['Total'] = bodyw_lcm['BestSquatKg'] + bodyw_lcm['BestBenchKg']+bodyw_lcm['BestDeadliftKg']
bodyw_lcf['Total'] = bodyw_lcf['BestSquatKg'] + bodyw_lcf['BestBenchKg']+bodyw_lcf['BestDeadliftKg']

bodyw_lcm['wRatio'] = bodyw_lcm['Total']/bodyw_lcm['BodyweightKg']
bodyw_lcf['wRatio'] = bodyw_lcf['Total']/bodyw_lcf['BodyweightKg']

bodyw_lcm
bodyw_lcf
plt.figure(figsize = (20,10))
plt.plot(bodyw_lcm.Agecategory,bodyw_lcm.wRatio, color = '#0093b7')
plt.plot(bodyw_lcf.Agecategory,bodyw_lcf.wRatio, color = '#e54370')
#plt.plot(bodyw_lcf.Agecategory, y = bodyW_lcf.wRatio, kind = 'line')