# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)
plt.rcParams['figure.figsize'] = (17.0, 6.0)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', 1000)
plt.rcParams['figure.figsize'] = (17.0, 6.0)
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.labelweight"] = 'bold'
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
data = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv")
g = sns.countplot(data.FinanciallySupporting,hue=data.AttendedBootcamp)
g.set_xticklabels(["No","Yes"])
g.legend(["No","Yes"],title="Attended Bootcamp")
g.set_title("Are the bootcampers financially supporting?")
plt.show()
non_boot_inc = data[data.AttendedBootcamp == 0]['Income'].dropna()
boot_inc = data[data.AttendedBootcamp == 1]['Income'].dropna()
avg_income = np.average(data['Income'].dropna())

print ("\nAverage income of programmers : "+str(avg_income))
print ("Average income of bootcampers : "+str(np.average(boot_inc)))
print ("Average income of non-bootcampers : "+str(np.average(non_boot_inc)))


print ("\n"+str((len(data[(data.AttendedBootcamp == 0) & 
         (data.Income >= avg_income)])/float(len(non_boot_inc)))*100)+"% of non-bootcampers are above the average income")
    
print (str((len(data[(data.AttendedBootcamp == 1) & 
         (data.Income >= avg_income)])/float(len(boot_inc)))*100)+"% of bootcampers are above the average income")
    
print ("\nExpected Earning of programmers attending bootcamp: "+
       str(np.average(data[data.AttendedBootcamp == 1]['ExpectedEarning'].dropna())))


print ("\nExpected Earning of programmers not attending bootcamp: "+
       str(np.average(data[data.AttendedBootcamp == 0]['ExpectedEarning'].dropna())))
labels = [ "{0} - {1}".format(i, i + 5) for i in range(10, 90, 5) ]
data['group'] = pd.cut(data.Age, range(10, 92, 5), right=False, labels=labels)
age_att = pd.pivot_table(data=data,index="group",columns=["AttendedBootcamp"],aggfunc={"AttendedBootcamp":np.size})
age_att.columns = age_att.columns.droplevel()
age_att.columns = ['0','1']
labels = [ "{0} - {1}".format(i, i + 5) for i in range(10, 90, 5) ]
data['group'] = pd.cut(data.Age, range(10, 92, 5), right=False, labels=labels)
colors = matplotlib.cm.Paired(np.linspace(0, 1, 60))
age_att['1'].plot.bar(stacked=True,color = colors)
plt.title("Age distribution of programmers who attend boot camps")
plt.xlabel("Age Group")
g = plt.ylabel("Count")
age_att['0'].plot.bar(stacked=True,color = colors)
plt.title("Age distribution of programmers who haven't attended any boot camps")
plt.xlabel("Age Group")
g = plt.ylabel("Count")
g = sns.countplot(data[data.AttendedBootcamp == 1]['BootcampFullJobAfter'])
g = g.set_xticklabels(["No","Yes"])
g = plt.title("Scored full-time job after Bootcamp?")
boot_inc_dist = data[data.AttendedBootcamp == 1][["BootcampPostSalary","Income"]].dropna()
print ("\nAverage increment after attending bootcamp : "+str(np.sum(boot_inc_dist.BootcampPostSalary - boot_inc_dist.Income)/np.sum(boot_inc_dist.Income)*100)+"%")
bootcamp_sal = data[data.BootcampName != "Free Code Camp is not a bootcamp - please scroll up and change answer to \"no\""][['BootcampName','BootcampPostSalary','Income']].dropna()
bootcamp_sal = bootcamp_sal.groupby('BootcampName').sum()[['BootcampPostSalary','Income']]
bootcamp_sal = pd.DataFrame(((bootcamp_sal['BootcampPostSalary']-bootcamp_sal['Income'])/bootcamp_sal['Income'])*100)
bootcamp_sal.columns = ['Percentage_change_in_salary']
g = bootcamp_sal[bootcamp_sal.Percentage_change_in_salary > 0].sort('Percentage_change_in_salary',ascending=False).plot.bar()
plt.legend(frameon=False)
plt.ylabel("Percentage change in salary")
plt.title("Percentage change in salary after attending particular bootcamp")
g = g.set_xticklabels(g.get_xticklabels(),rotation=90)