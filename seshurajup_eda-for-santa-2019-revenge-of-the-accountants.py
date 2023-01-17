import seaborn as sns

import matplotlib.pyplot as plt

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



fpath = '/kaggle/input/santa-2019-revenge-of-the-accountants/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')



fpath = '/kaggle/input/santa-2019-revenge-of-the-accountants/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')
family_size_dict = data[['n_people']].to_dict()['n_people']

m,_ = data.shape 



cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].T.to_dict()



N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



# from 100 to 1

days = list(range(N_DAYS,0,-1))
data.head()
plt.figure(figsize=(16,9))

sns.countplot(data['n_people'])
plt.figure(figsize=(16,9))

sns.distplot(data['choice_0'],bins=100,kde=False)

plt.title('Preferred Choice of each Family')
plt.figure(figsize=(16,9))

sns.distplot(data['choice_9'],bins=100,kde=False)

plt.title('Least Preferred Choice for each Family')
def weekday(days):

    weekday = []

    for day in days:

        if day%7==2:

            weekday.append('Monday')

        elif day%7==1:

            weekday.append('Tuesday')

        elif day%7==0:

            weekday.append('Wednesday')

        elif day%7==6:

            weekday.append('Thursday')

        elif day%7==5:

            weekday.append('Friday')

        elif day%7==4:

            weekday.append('Saturday')

        else:

            weekday.append('Sunday')

    return weekday

weekday_list = [weekday(data[cols].values[i][:].tolist()) for i in range(5000)]

weekday_list = pd.DataFrame(weekday_list,columns = ['weekday_'+cols[i] for i in range(10)])

weekday_list
plt.figure(figsize=(16,9))

sns.countplot(weekday_list['weekday_choice_0'],order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.figure(figsize=(16,9))

sns.countplot(weekday_list['weekday_choice_9'],order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
#lets set a difficulty score per choice, 0 is weekdays, 1 is weekends, 2 is christmas eve

def difficulty(days):

    dif = []

    for day in days:

        wd = weekday([day])

        if day == 1 or day == 2 or day == 3:

            dif.append(1)

        elif wd[0]=='Monday' or wd[0]=='Tuesday' or wd[0]=='Wednesday' or wd[0]=='Thursday':

            dif.append(0)

        else:

            dif.append(1)

    return dif



dif_list = [difficulty(data[cols].values[i][:].tolist()) for i in range(5000)]

dif_list = pd.DataFrame(dif_list,columns = ['difficulty_'+cols[i] for i in range(10)])

dif_sum = pd.DataFrame(np.sum(dif_list,axis=1),columns=['dif_sum'])

#data = pd.concat([data,dif_sum],axis=1)

plt.figure(figsize=(16,9))

sns.countplot(dif_sum['dif_sum'])

plt.title("Number of difficult days per family")

print("The mean number of difficult days per family is: {}".format(np.mean(dif_sum['dif_sum'])))

print("Standard deviation of difficult days is: {}".format(np.std(dif_sum['dif_sum'])))
family_size = list(range(2,9))

cost_per_family_size = np.zeros((11,len(family_size)))

def choice_cost(n,choice):

    if choice == 0:

        return 0

    elif choice == 1:

        return 50

    elif choice == 2:

        return 50 + 9 * n

    elif choice == 3:

        return 100 + 9 * n

    elif choice == 4:

        return 200 + 9 * n

    elif choice == 5:

        return 200 + 18 * n

    elif choice == 6:

        return 300 + 18 * n

    elif choice == 7:

        return 300 + 36 * n

    elif choice == 8:

        return 400 + 36 * n

    elif choice == 9:

        return 500 + 36 * n + 199 * n

    else:

        return 500 + 36 * n + 398 * n

j = 0

for n in family_size:  

    for i in range(11):

        cost_per_family_size[i,j]=choice_cost(n,i)

    j+=1



cost_per_family_size = pd.DataFrame(cost_per_family_size.T,index = [str(x)+'_people' for x in range(2,9)],columns = ['choice_'+str(x) for x in range(11)])

cost_per_family_size['n_people']=list(range(2,9))

cost_per_family_size
plt.figure(figsize=(16,9))

for i in range(11):

    sns.lineplot(data=cost_per_family_size,x='n_people',y='choice_'+str(i))
plt.figure(figsize=(16,9))

for i in range(2,9):

    sns.lineplot(data=cost_per_family_size,x='n_people',y='choice_'+str(i))
sorted_families = []

daily_occupancy={i:0 for i in range(1,101)}

answer = np.zeros((m,3))

for j in range(4):

    for i in range(m):

        if difficulty(data.loc[i][['choice_'+str(j)]])==[0] and i not in sorted_families and daily_occupancy[data.loc[i][['choice_'+str(j)]][0]]+data.loc[i][['n_people']][0]<300:

            sorted_families.append(i)

            answer[i,0]=i

            answer[i,1]=data.loc[i][['choice_'+str(j)]]

            answer[i,2]=j

            daily_occupancy[data.loc[i][['choice_'+str(j)]][0]]+=data.loc[i][['n_people']][0]
def daily_plot(answer):

    def get_daily_occupancy(answer):

        daily_occupancy={i:0 for i in range(0,101)}

        for i in range(m):

            daily_occupancy[int(answer[i,1])]+=data.loc[i]['n_people']

        return daily_occupancy

    daily_occupancy = get_daily_occupancy(answer)

    plt.figure(figsize=(12,7))

    fig = sns.lineplot(x=list(range(1,101)),y=[daily_occupancy[i] for i in range(1,101)])

    ax = plt.axes()

    x = np.linspace(0, 100, 100)

    y= np.linspace(125,125,100)

    ax.plot(x, y,color='green');

    x = np.linspace(0, 100, 100)

    y= np.linspace(300,300,100)

    ax.plot(x, y,color='red')

daily_plot(answer)
for j in range(4,8):

    for i in range(m):

        if difficulty(data.loc[i][['choice_'+str(j)]])==[0] and i not in sorted_families and daily_occupancy[data.loc[i][['choice_'+str(j)]][0]]<125:

            sorted_families.append(i)

            answer[i,0]=i

            answer[i,1]=data.loc[i][['choice_'+str(j)]]

            answer[i,2]=j

            daily_occupancy[data.loc[i][['choice_'+str(j)]][0]]+=data.loc[i][['n_people']][0]

daily_plot(answer)
for j in range(4):

    for i in range(m):

        if (difficulty(data.loc[i][['choice_'+str(j)]])==[1] or difficulty(data.loc[i][['choice_'+str(j)]])==[2]) and i not in sorted_families and daily_occupancy[data.loc[i][['choice_'+str(j)]][0]]+data.loc[i][['n_people']][0]<300:

            sorted_families.append(i)

            answer[i,0]=i

            answer[i,1]=data.loc[i][['choice_'+str(j)]]

            answer[i,2]=j

            daily_occupancy[data.loc[i][['choice_'+str(j)]][0]]+=data.loc[i][['n_people']][0]

daily_plot(answer)
del sorted_families

for k,v in daily_occupancy.items():

    if v<125:

        for j in range(4):

            for i in range(m):

                if daily_occupancy[data.loc[i][['choice_'+str(int(answer[i,2]))]][0]]-data.loc[i][['n_people']][0]>=125 and data.loc[i][['choice_'+str(j)]][0]==k and daily_occupancy[k]<=125:

                    answer[i,0]=i

                    answer[i,1]=data.loc[i][['choice_'+str(j)]]

                    answer[i,2]=j

                    daily_occupancy[data.loc[i][['choice_'+str(j)]][0]]+=data.loc[i][['n_people']][0]

daily_plot(answer)