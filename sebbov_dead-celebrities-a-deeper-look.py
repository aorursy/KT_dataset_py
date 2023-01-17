# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('fivethirtyeight')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/celebrity_deaths_4.csv", encoding = "ISO-8859-1")

data['celeb_name'] = data.name

data.fame_score = data.fame_score.fillna(0)

data['wiki_length'] = data.fame_score
data.head(10)
plt.semilogy(data.age,data.wiki_length,'o',alpha = 0.1)

plt.ylim([1*10**3,3*10**6])

plt.xlim([5,120])

x = data.age

y = np.array(data.wiki_length)

fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit) 

plt.plot(x,fit_fn(x))

plt.title('Do older celebrites have longer Wikipedia articles?')

plt.xlabel('Age at death')

plt.ylabel('Length of Wikipedia article [words]',fontsize = 13)

plt.show()
avg_len_year = []

std_len_year = []

for year in np.arange(2006,2017):

    avg_len_year.append(np.mean(data.wiki_length[data.death_year == year]))

    std_len_year.append(np.std(data.wiki_length[data.death_year == year]))

plt.figure(figsize = (7,7))

plt.errorbar(np.arange(2006,2017),avg_len_year,yerr = std_len_year,marker = 'o', markersize = 12)

plt.xlim([2005,2017])

plt.ylabel('Number of words in Wikipedia articles')

plt.title('Wikipedia article length vs Year')

#plt.errorbar(x, y, xerr=0.2, yerr=0.4)
import seaborn as sns

sns.distplot(data.wiki_length, bins = 400,kde= False,norm_hist=False)

plt.xlim([0,35000])
data.famous_for = data.famous_for.fillna(0)

words = ['singer','actor','actress','musician','writer','author','TV','movie','television','comedian','film']

for i,row in enumerate(data.iterrows()):

    data.loc[i,'flag'] = 0

    for word in words:

        if row[1].famous_for is not 0:

            if word in row[1].famous_for:

           # print row[1].celeb_name, row[1].famous_for

                data.loc[i,'flag'] = 1

                break
real_celeb = data[data.flag == 1]

real_celeb.to_csv('real_celeb.csv')

#super_stars = real_celeb[real_celeb.wiki_length>25000]

dead_super = []

yearly_len_avg = []

for year in np.arange(2005,2017):

    dead_super.append(len(real_celeb[real_celeb.death_year == year]))

    yearly_len_avg.append(np.mean(real_celeb.wiki_length))

    

plt.xlim([2005,2017])

plt.bar(np.arange(2005,2017),dead_super)#,'o')

plt.title('Dead real celebrities per year')
plt.title('Article Length vs Age')

plt.ylabel('Article length in Wikipedia [Words]')

plt.xlabel('Age at death')

plt.semilogy(real_celeb.age,real_celeb.wiki_length,'o',alpha = 0.3,markersize = 6)

#plt.ylim([8*10**3,3*10**5])

plt.xlim([5,120])

plt.ylim([2*10**3,2*10**5])

x = real_celeb.age

y = real_celeb.wiki_length

fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit) 

plt.plot(x,fit_fn(x))

plt.show()
super_stars = real_celeb[real_celeb.wiki_length>25000]

dead_super = []

yearly_len_avg = []

for year in np.arange(2005,2017):

    dead_super.append(len(super_stars[super_stars.death_year == year]))

    yearly_len_avg.append(np.mean(super_stars.wiki_length))

    

plt.xlim([2005,2017])

plt.bar(np.arange(2005,2017),dead_super)#,'o')

plt.title('Dead Super Stars per year')
young_super = super_stars[super_stars.age<70]



dead_super = []

yearly_len_avg = []

for year in np.arange(2006,2017):

    dead_super.append(len(young_super[young_super.death_year == year]))

    yearly_len_avg.append(np.mean(young_super.wiki_length))

    

plt.bar(np.arange(2006,2017),dead_super)

plt.xlim([2005,2017])

plt.ylim([0,max(dead_super)+1])

plt.title('Super Stars who died under 70')
young_super = super_stars[super_stars.age<60]



dead_super = []

yearly_len_avg = []

for year in np.arange(2006,2017):

    dead_super.append(len(young_super[young_super.death_year == year]))

    yearly_len_avg.append(np.mean(young_super.wiki_length))

    

plt.bar(np.arange(2006,2017),dead_super)

plt.xlim([2005,2017])

plt.ylim([0,max(dead_super)+1])

plt.title('Super Stars who died under 60')
import seaborn as sns

x = data.age

#sns.distplot(data.age, bins = 50,hist = False,kde = True)#,label = 'All Celebs')

sns.kdeplot(x,shade=True);

#sns.distplot(real_celeb.age, bins = 50,hist = False,kde= True, label = 'Real Celebs')

#sns.kdeplot(x, shade=True);

#sns.distplot(super_stars.age, bins = 50,hist = False,kde= True,label = 'Super Stars')

plt.xlim([0,120])

plt.title('Death Age Distribution')

#plt.legend(['all celebs','only "real" celebs','only super stars'])

plt.show()
avg_age = []

for year in np.arange(2006,2017):

    avg_age.append(np.mean(data.age[data.death_year == year]))





plt.bar(np.arange(2006,2017),avg_age)

plt.ylim([72,78])

plt.xlim([2006,2017])

plt.ylabel('Average Age')

plt.title('Average Age of death - all famous people')
avg_age = []

for year in np.arange(2006,2017):

    avg_age.append(np.mean(real_celeb.age[real_celeb.death_year == year]))





plt.bar(np.arange(2006,2017),avg_age)

plt.ylim([72,78])

plt.xlim([2006,2017])

plt.ylabel('Average Age')

plt.title('Average Age of death - real celebs')