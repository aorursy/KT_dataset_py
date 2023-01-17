# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from numpy import * # linear algebra
from pandas import *# data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pyplot import *
from plotly import *
from scipy.stats import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importing data
school=read_csv('../input/2016 School Explorer.csv')
#To view all the column names
set_option('display.max_seq_items',None)
school.columns
school.isnull().sum()
school.drop(['Adjusted Grade','New?','Other Location Code in LCGMS'],axis=1,inplace=True)
school.dropna(inplace=True)
school.isnull().sum()
#Distribution of Community Schools in each City
school_city=crosstab(school['City'], school['Community School?']).sort_values('No',ascending=True)
school_city

school_city.plot(kind='barh', stacked=True,figsize=(10,10),width=0.8)

title('Distribution of Community Schools in each City ')
xlabel('Name of the City')
ylabel('Frequency of Community Schools and Non-Community Schools')
legend(title='Community School',loc="center right")
show()
sns.set(rc={'figure.figsize':(7,5)})
sns.boxplot(x="Community School?", y="Economic Need Index", data=school,dodge=False,palette="seismic",width=0.3)
show()
sns.boxplot(x="Community School?", y="Average ELA Proficiency", data=school,dodge=False,palette="husl",width=0.3)
show()
sns.set(rc={'figure.figsize':(7,5)})
sns.boxplot(x="Community School?", y="Average Math Proficiency", data=school,dodge=False,palette="hot",width=0.3)
show()
grp=crosstab(school['Supportive Environment Rating'],school["Community School?"])
grp = grp.div(grp.sum(1), axis=0)
colors = ["#2E86C1","#D68910"]
grp.plot(kind='bar', stacked=True,figsize=(15,5),width=0.3,color=colors)
index=arange(9)
xticks(rotation=0)
title('Comparison of Supportive Environment Rating between Community schools and Non-Community schools')
# xticks(index,names)

legend(loc="upper right")
show()
grp=crosstab(school['Rigorous Instruction Rating'],school["Community School?"])
grp = grp.div(grp.sum(1), axis=0)
colors = ["#2E86C1","#D68910"]
grp.plot(kind='bar', stacked=True,figsize=(15,5),width=0.3,color=colors)
index=arange(9)
xticks(rotation=0)
title('Comparison of Rigorous Instruction Rating between Community schools and Non-Community schools')
# xticks(index,names)

legend(loc="upper right")
show()
grp=crosstab(school['Collaborative Teachers Rating'],school["Community School?"])
grp = grp.div(grp.sum(1), axis=0)
colors = ["#2E86C1","#D68910"]
grp.plot(kind='bar', stacked=True,figsize=(15,5),width=0.3,color=colors)
index=arange(9)
xticks(rotation=0)
title('Comparison of Collaborative Teachers Rating between Community schools and Non-Community schools')
# xticks(index,names)

legend(loc="upper right")
show()
school_community=school.copy()
school_community=school_community[school_community['Community School?']=='Yes']
sns.set(rc={'figure.figsize':(12,5)})
sns.boxplot(x="City", y="Economic Need Index", data=school_community,palette="seismic",width=0.3)
show()
school_community_eco_need=school_community.loc[(school_community['City']=='NEW YORK') | (school_community['City']=='BRONX') | (school_community['City']=='BROOKLYN'),:]
len(school_community_eco_need)
school_community_eco_need['Student Achievement Rating']
school_community_eco_need.columns
achieve_community=DataFrame(melt(school_community_eco_need,id_vars=['School Name'], value_vars=['Student Achievement Rating']))
achieve_community
del achieve_community['variable']
achieve_community
sns.set(rc={'figure.figsize':(13,8)})
school['Percent of Students Chronically Absent']=school['Percent of Students Chronically Absent'].astype('str')      
school['Percent of Students Chronically Absent'] = school['Percent of Students Chronically Absent'].str.rstrip('%').astype('float') *10
 
sns.boxplot(x="Supportive Environment Rating", y="Percent of Students Chronically Absent", data=school,dodge=False,\
            order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"],palette="Oranges",width=0.4)
title('Association between Students Chronically Missing School and Supportive Environment Rating')
show()

sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Rigorous Instruction Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,\
            palette="Blues",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Rigorous Instruction Rating')
show()

sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Collaborative Teachers Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,\
            palette="Greens",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Collaborative Teachers Rating')
show()


sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Strong Family-Community Ties Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,\
            palette="Purples",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Strong Family-Community Ties Rating')
show()

sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Trust Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,\
            palette="Reds",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Trust Rating')
show()