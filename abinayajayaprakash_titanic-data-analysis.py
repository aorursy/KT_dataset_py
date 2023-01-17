# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv
titanic_data=[]
with open('../input/titanic/train.csv', 'rt') as f:
  reader = csv.DictReader(f)
  titanic_data = list(reader)

print(titanic_data[0])
#Checking for duplicate entries

print(len(titanic_data))

unique_passenger_records= set()
for passenger_record in titanic_data:
  unique_passenger_records.add(passenger_record['PassengerId'])
print(len(unique_passenger_records))

#Removing unnecessary columns

for passenger_record in titanic_data:
  passenger_record.pop('Cabin',None)
  passenger_record.pop('Ticket',None)
  passenger_record.pop('Fare',None)

print(titanic_data[1])
import pandas as pd
%pylab inline
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('../input/titanic/train.csv')

print(titanic_df.head())
#Removing unnecessary columns

titanic_df_new = titanic_df.drop(['PassengerId','Ticket','Cabin','Fare'], axis=1)
print(titanic_df_new.head())
#Using the isnull function helps us spot missing values in all the respective columns in the dataframe.
#Finding the number of rows with missing data in each column

titanic_df_new.isnull().sum()
#Finding the total number of rows which contain the value '1' in the column 'Survived' to find how many survived altogether

survived = titanic_df_new['Survived']== True
non_survived = titanic_df_new['Survived']== False

print('Number of passengers who survived: {}'.format(survived.sum()))
print('Number of passengers who did not survive: {}'.format(non_survived.sum()))

percent_survived =(survived.sum()/891) *100
print('Percentage of passengers who survived: {} %'.format(percent_survived))
names=['Survivors','Non-Survivors']
values=[survived.sum(),non_survived.sum()]
plt.bar(names,values)
plt.title('Number of survivors')
#Checking the number of rows that meet two conditions:
   #contain the value '1' in the 'Survived' column
   #have either 'female' or 'male' in the 'Sex' column

female_survived = titanic_df_new[survived]['Sex']=='female'
male_survived = titanic_df_new[survived]['Sex']=='male'

print('Number of female passengers who survived: {}'.format(female_survived.sum()))
print('Number of male passengers who survived: {}'.format(male_survived.sum()))

percent_female_survived = (female_survived.sum()/survived.sum()) *100
print('{}% of the passengers who survived were female.'.format(percent_female_survived))
names=['Female Survivors',' Male Survivors']
values=[female_survived.sum(),male_survived.sum()]
plt.bar(names,values)
plt.title('Number of female survivors')
#The 'argmin' or 'argmax' function returns the row number with the minimum/maximum age value
#The 'iloc' function then locates the row, identifies the column given and returns the entry in it along the respective row number

youngest_survivor = titanic_df_new[survived]['Age'].argmin()
youngest_survivor_name = titanic_df_new[survived].iloc[youngest_survivor]['Name']
youngest_survivor_age= titanic_df_new[survived].iloc[youngest_survivor]['Age']
print('The youngest survivor {} was {} years old.'.format(youngest_survivor_name,youngest_survivor_age))

oldest_survivor = titanic_df_new[survived]['Age'].argmax()
oldest_survivor_name = titanic_df_new[survived].iloc[oldest_survivor]['Name']
oldest_survivor_age= titanic_df_new[survived].iloc[oldest_survivor]['Age']
print('The oldest survivor {} was {} years old.'.format(oldest_survivor_name,oldest_survivor_age))
youngest_non_survivor = titanic_df_new[non_survived]['Age'].argmin()
youngest_non_survivor_name = titanic_df_new[non_survived].iloc[youngest_non_survivor]['Name']
youngest_non_survivor_age= titanic_df_new[non_survived].iloc[youngest_non_survivor]['Age']
print('The youngest non-survivor {} was {} years old.'.format(youngest_non_survivor_name,youngest_non_survivor_age))

oldest_non_survivor = titanic_df_new[non_survived]['Age'].argmax()
oldest_non_survivor_name = titanic_df_new[non_survived].iloc[oldest_non_survivor]['Name']
oldest_non_survivor_age= titanic_df_new[non_survived].iloc[oldest_non_survivor]['Age']
print('The oldest non-survivor {} was {} years old.'.format(oldest_non_survivor_name,oldest_non_survivor_age))
#The sum function is used to find the total number of passengers who survived and satisfy the condition on the age value
#Then the corresponding values are subtracted to find how many fall in each interval 

below_12 = titanic_df_new[survived]['Age']<=12
below_19 = titanic_df_new[survived]['Age']<=19
below_59 = titanic_df_new[survived]['Age']<=59
below_100 = titanic_df_new[survived]['Age']<=150

children_survived = below_12.sum()
teenagers_survived = below_19.sum() - below_12.sum()
adults_survived = below_59.sum() - (below_19.sum())
elderly_survived = below_100.sum() - (below_59.sum())

print('Number of children who survived: {}'.format(children_survived))
print('Number of teenagers who survived: {}'.format(teenagers_survived))
print('Number of adults who survived: {}'.format(adults_survived))
print('Number of elderly who survived: {}'.format(elderly_survived))
below_12_all = titanic_df_new['Age']<=12
below_19_all= titanic_df_new['Age']<=19
below_59_all= titanic_df_new['Age']<=59
below_100_all = titanic_df_new['Age']<=100

children_total = below_12_all.sum()
teenagers_total = below_19_all.sum() - below_12_all.sum()
adults_total = below_59_all.sum() - (below_19_all.sum())
elderly_total = below_100_all.sum() - (below_59_all.sum())

children_sr = (children_survived/children_total)*100
teenager_sr = (teenagers_survived/teenagers_total)*100
adults_sr = (adults_survived /adults_total)*100
elderly_sr = (elderly_survived/elderly_total)*100

names=['Children',' Teenagers','Adults','Elderly']
values=[children_sr,teenager_sr,adults_sr,elderly_sr]
plt.bar(names,values)
plt.title('Survival rates according to age groups')
#This function groups all records by which 'Pclass' the belong to and then the 'size' argument finds the number of records it contains

def survival(Pclass):
  return titanic_df_new.groupby(['Pclass','Survived']).size()[Pclass,1].astype('float')

print('Out of all passengers who survived, {} belonged to the upper class , {} belonged to the middle class and {} belonged to the lower class.'.format(survival(1),survival(2),survival(3)))
names=['Upper class',' Middle class','Lower class']
values=[survival(1),survival(2),survival(3)]
plt.bar(names,values)
plt.title('Number of survivors in each socio-economic class')
#The above function was modified to include the 'Sex' column with the condition for it to be 'female' to see how many females survived in each 'Pclass'

def survival_female(Pclass):
  return titanic_df_new.groupby(['Pclass','Survived','Sex']).size()[Pclass,1,'female'].astype('float')

print( '{} % of Upper class citizens who survived were females'.format((survival_female(1)/survival(1)) *100))

print('Similarly {} % in the Middle class and {} % in the Lower class'.format(((survival_female(2)/survival(2)) *100), ((survival_female(3)/survival(3)) *100)))
#The 'total' function helps find how many passengers were in each of the socio-economic classes.Both the survival and total functions were used
#to find the percentage of survivors in each Pclass

def total(Pclass):
  return titanic_df_new.groupby(['Pclass']).size()[Pclass].astype('float')

Pclass1_sr = (survival(1)/total(1))*100
Pclass2_sr = (survival(2)/total(2))*100
Pclass3_sr = (survival(3)/total(3))*100

print('{} % of upper class citizens survived'.format(Pclass1_sr))
print('{} % of middle class citizens survived'.format(Pclass2_sr))
print('{} % of lower class citizens survived'.format(Pclass3_sr))
#The sum function is used to find the total number of passengers who survived according to their port of embarkation 

Port_C_s= titanic_df_new[survived]['Embarked']=='C'
Port_S_s= titanic_df_new[survived]['Embarked']=='S'
Port_Q_s= titanic_df_new[survived]['Embarked']=='Q'

print('Out of all passengers who survived, {} were to be embarked at Cherbourg , {} were to be embarked at Southhampton and {} were to be embarked at Queenstown.'.format(Port_C_s.sum(),Port_S_s.sum(),Port_Q_s.sum()))
names=['Port Southhampton',' Port Cherbourg','Port Queenstown']
values=[Port_S_s.sum(),Port_C_s.sum(),Port_Q_s.sum()]
plt.bar(names,values)
plt.title('Number of survivors according to their port of embarkation')
#The 'count' function is used to get the sum of non-null values that satisy all conditions specified on respective columns

with_relatives=titanic_df_new[(titanic_df_new.Parch!=0) & (titanic_df_new.SibSp != 0)].count()

with_relatives_survived =titanic_df_new[(titanic_df_new.Survived) & (titanic_df_new.Parch!=0) & (titanic_df_new.SibSp != 0)].count()

print('{} % of passengers with relatives aboard survived'.format((62/142)*100))
without_relatives=titanic_df_new[(titanic_df_new.Parch==0) & (titanic_df_new.SibSp == 0)].count()

without_relatives_survived =titanic_df_new[(titanic_df_new.Survived) & (titanic_df_new.Parch==0) & (titanic_df_new.SibSp == 0)].count()

print('{} % of passengers without relatives aboard survived'.format((163/537)*100))
#The 'count' function is used to get the sum of non-null values that satisy all conditions specified on respective columns

children_with_nannies = titanic_df_new[(titanic_df_new.Parch==0)  & (titanic_df_new.SibSp==0) & (titanic_df_new.Age <= 18)].count()
print(children_with_nannies)
children_with_nannies_survived = titanic_df_new[(titanic_df_new.Survived) & (titanic_df_new.Parch==0)  & (titanic_df_new.SibSp==0) & (titanic_df_new.Age <= 18)].count()
print(children_with_nannies_survived)