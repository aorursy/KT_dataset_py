# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



data = pd.read_csv('../input/No-show-Issue-300k.csv', sep=';')



data.Age = data.Age.apply(lambda x: x if (x>0) else 0)

data['GenderNumb'] = data.Gender.apply(lambda x: 0 if(x=='F') else 1)

data['StatusNumb'] = data.Status.apply(lambda x: 0 if(x=='No-Show') else 1)

data['WeekdayNumb'] = data.DayOfTheWeek.apply(lambda x: weekdays.index(x))

data.describe()
no_show = data.loc[data.Status == 'No-Show']

show = data.loc[data.Status == 'Show-Up']



len(no_show)/(len(no_show)+len(show))
import matplotlib.pyplot as plt

columns = data.columns

age_range = range(120)

age_show = np.zeros(120)

age_no_show = age_show.copy()



no_show_age_count = no_show.groupby('Age').Age.count()

show_age_count = show.groupby('Age').Age.count()

#print(no_show_age_count)

for index, count in zip(no_show_age_count.index, no_show_age_count.values):

    #print(index)

    age_no_show[index] = count



for index, count in zip(show_age_count.index, show_age_count.values):

    age_show[index] = count



percentage = age_show.copy()

for index, value in enumerate(age_no_show):

    x = value/(percentage[index]+value)

    percentage[index] = 0.3 if np.isnan(x) else x



below_or_above = percentage -0.3



plt.bar(age_range, age_no_show, bottom=age_show, color='red', label='No-Show')

plt.bar(age_range, age_show, label='Show')

plt.legend()

plt.title('total Amount of Show-Up per Age')

plt.xlabel('Age')

plt.show()

#plt.show()

plt.bar(age_range, below_or_above)

plt.title('No-Show Trend')

plt.xlabel('Age')

plt.show()
f = data.StatusNumb[data.GenderNumb==0]

m = data.StatusNumb[data.GenderNumb==1]

percent_m = 1-sum(m)/len(m)

percent_f = 1-sum(f)/len(f)



print('Percent of male No-Show: %f'%(percent_m))

print('Percent of female No-Show: %f'%(percent_f))
  

columns = ['Smokes', 'Alcoolism', 'Scholarship', 'Handcap']

#print(data.columns)

for column in columns:

    k = data.StatusNumb[data[column]==0]

    l = data.StatusNumb[data[column]==1]

    percent_m = 1-sum(k)/len(k)

    percent_f = 1-sum(l)/len(l)



    print('Percent of non '+column+' No-Show: %f'%(percent_m))

    print('Percent of  '+column+' No-Show: %f'%(percent_f))
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



for index, weekday in enumerate(weekdays):

    k = data.StatusNumb[data.DayOfTheWeek==weekday]

    percent_m = 1-sum(k)/len(k)

    plt.bar(index, percent_m, color='blue')

    

plt.xticks(range(len(weekdays)),weekdays, rotation=50)

plt.title('Percent of No-Show per DayOfWeek')

plt.show()

from sklearn import ensemble, cross_validation, datasets

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



#print(data.Age)

target = data.StatusNumb.values

#print(target)

train_data = data[['Age', 'Smokes', 'Alcoolism', 'Handcap', 'WeekdayNumb']].values

#print(train_data)

#plt.bar(np.arange(7),np.bincount(train_data[:,4]))

randomForest = ensemble.RandomForestClassifier()

score = cross_val_score(randomForest, train_data, target, cv=5)

print(score)