import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset1 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

dataset1[dataset1.columns[0:32]].isnull().sum()/len(dataset1)*100

dataset2 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

dataset2.isnull().sum()/len(dataset2)*100

covid = dataset2.copy()

df= pd.DataFrame(covid['symptom_onset'])

df.fillna(0)

df['month']=0

df['day'] = 0

for x in range(0,len(df)):



    if type(df['symptom_onset'][x]) == float :

        df['month'][x]=0

        df['day'][x]=0

        

    else:

        slash= (df['symptom_onset'][x]).find('/')

        second_slash= df['symptom_onset'][x][slash+1:].find('/')

        df['month'][x]= int(df['symptom_onset'][x][0:slash].strip()) 

        df['day'][x]=int(df['symptom_onset'][x][slash+1:slash+second_slash+1].strip())

        

df

    
dataset2= pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
mask = ((dataset2['symptom_onset_date'].notnull() | dataset2['confirmed_date'].notnull()) & (dataset2['released_date'].notnull() | dataset2['deceased_date'].notnull()) & (dataset2['birth_year'].notnull() | dataset2['age'].notnull()))

dataset2.loc[mask, 'age'].notnull().sum()
column_names=['birth_year','age','symptom_onset_date','confirmed_date','released_date','deceased_date','sex','state']

newdf = dataset2.loc[mask,column_names].copy()

newdf.head()
date_columns = ['symptom_onset_date','confirmed_date','released_date','deceased_date']

for column_name in date_columns:

    newdf[column_name] = pd.to_datetime(newdf[column_name], format="%Y-%m-%d")

print()

print(newdf.dtypes)
newdf['is_male'] = pd.get_dummies(newdf.sex)['male']

newdf['is_female'] = pd.get_dummies(newdf.sex)['female']

print(newdf[['is_male','is_female']].head(10))

print(newdf.dtypes)

print(newdf.isnull().sum()/len(newdf)*100)

newdf['recovery_time']=0

for i in range(0,1035):

    if newdf['state'].iloc[i]=='deceased':

        newdf['recovery_time'].iloc[i]=newdf.iloc[i]['deceased_date'] - newdf.iloc[i]['confirmed_date']

    elif newdf['state'].iloc[i]=='released':

        newdf['recovery_time'].iloc[i]= newdf.iloc[i]['released_date']- newdf.iloc[i]['confirmed_date']

        

newdf
print(newdf.isnull().sum()/len(newdf)*100)
dayss= newdf['recovery_time'].astype('timedelta64[D]')

newdf['timelength']= dayss / np.timedelta64(1,'D')
alive = newdf[newdf['state']!='deceased']

alive.state.unique()

x=alive.age.unique()

x.sort()

y=[]

avgtime=0

for age in x:

    t_avg= alive[alive['age']==age]['timelength'].mean()

    y.append(t_avg)



import matplotlib.pyplot as plt

%matplotlib inline



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(x,y)

plt.show()

import seaborn as sns

density= sns.kdeplot(alive['timelength'],bw=5)



plt.title ('recovery time density plot')

plt.xlabel('Recovery time')

plt.ylabel('density')



plt.show()
alive['acc_age']=2020- alive['birth_year']

alive['agegroup']= 0

for i in range(len(alive)):

    s= alive['age'].iloc[i].find('s')

    alive['agegroup'].iloc[i]= int((alive['age'].iloc[i])[:s])

    
Y= alive['agegroup']

plt.scatter(alive['acc_age'], alive['timelength'], c=Y.values.ravel())



plt.title('scatter plot for age against recovery time')

plt.xlabel('age')

plt.ylabel('recovery time')



plt.show()



plt.scatter(alive['agegroup'], alive['timelength'])



plt.title('scatter plot for agegroup against recovery time')

plt.xlabel('age group')

plt.ylabel('recovery time')



plt.show()
plt.violinplot([alive['timelength']], showextrema=True, showmedians=True)



plt.title('Recovery time')

plt.ylabel('time(days)')



plt.show()