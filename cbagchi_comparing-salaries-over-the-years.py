# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
obama_salaries=pd.read_csv("../input/obama_staff_salaries.csv")

trump_salaries=pd.read_csv("../input/white_house_2017_salaries.csv")

print(obama_salaries.head())

print(trump_salaries.head())
#Remove dollar sign from every tuple

def remove_dollar_sign(x):

    salary=list()

    for i in x.index:

        k=x[i]

        k=k.replace("$","")

        salary.append(k)

    return(pd.DataFrame(salary)) 

def remove_dollar_comma(x):

    salary=list()

    for i in x.index:

        k=x[i]

        k = k.replace(",", "")

        k=k.replace("$","")

        salary.append(k)

    return(pd.DataFrame(salary))

#Extracting salary of each year separately

indexes=["2009","2010","2011","2012","2013","2014","2015","2016","2017"]

avg=pd.Series()

_2009_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2009]).astype(float)

avg=avg.append(_2009_salaries.mean())

_2010_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2010]).astype(float)

avg=avg.append(_2010_salaries.mean())

_2011_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2011]).astype(float)

avg=avg.append(_2011_salaries.mean())

_2012_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2012]).astype(float)

avg=avg.append(_2012_salaries.mean())

_2013_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2013]).astype(float)

avg=avg.append(_2013_salaries.mean())

_2014_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2014]).astype(float)

avg=avg.append(_2014_salaries.mean())

_2015_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2015]).astype(float)

avg=avg.append(_2015_salaries.mean())

_2016_salaries=remove_dollar_sign(obama_salaries["salary"][obama_salaries["year"]==2016]).astype(float)

avg=avg.append(_2016_salaries.mean())

_2017_salaries=remove_dollar_comma(trump_salaries["SALARY"]).astype(float)

avg=avg.append(_2017_salaries.mean())

avg=np.array(avg)



frames=[_2009_salaries,_2010_salaries,_2011_salaries,_2012_salaries,_2013_salaries,_2014_salaries,_2015_salaries,_2016_salaries,_2017_salaries]

salaries=pd.DataFrame(pd.concat(frames,axis=1))

salaries.columns=indexes

salaries.head()
#Creating barplot

N=9

ind=np.arange(N)

plt.bar(ind,avg,width=0.35,color="lightblue")

plt.xticks(ind,indexes)

#ax.set_xticklabels()

plt.xlabel("Year")

plt.ylabel("Average Salary")

plt.show()
salaries["2009"]=salaries["2009"].fillna(salaries["2009"].median())

salaries["2010"]=salaries["2010"].fillna(salaries["2010"].median())

salaries["2011"]=salaries["2011"].fillna(salaries["2011"].median())

salaries["2012"]=salaries["2012"].fillna(salaries["2012"].median())

salaries["2013"]=salaries["2013"].fillna(salaries["2013"].median())

salaries["2014"]=salaries["2014"].fillna(salaries["2014"].median())

salaries["2015"]=salaries["2015"].fillna(salaries["2015"].median())

salaries["2016"]=salaries["2016"].fillna(salaries["2016"].median())

salaries["2017"]=salaries["2017"].fillna(salaries["2017"].median())

salaries_array=np.array(salaries)

#fig,ax=plt.subplots(1,10)

#width=1

#ax.boxplot(salaries["2009"])

#ax.boxplot(salaries["2010"])

bp=plt.boxplot(salaries_array,patch_artist=True)

plt.xticks(ind+1,indexes)

for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:

        plt.setp(bp[element], color="darkgrey")



for patch in bp['boxes']:

        patch.set(facecolor="cyan")

plt.show()