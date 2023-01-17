# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read CSV, print

data = pd.read_csv("../input/train.csv")

data.head(5)
data_cor = data.corr()

sns.heatmap(data_cor, annot=True, fmt=".2f")
pandas_profiling.ProfileReport(data)
# get count of passengers by sex

fvsmc = data.groupby(['Sex']).size()



# create a dataframe for the count and sex

fvsmcarr = []

fvsmcarr.append({'Sex': 'female', 'Count': fvsmc.female})

fvsmcarr.append({'Sex': 'male', 'Count': fvsmc.male})

fvsmcdf = pd.DataFrame(fvsmcarr)

fvsmcdf = fvsmcdf[['Sex','Count']]

fvsmcdf
# pie chart for count and sex of passengers

fvsmfig, fvsmax = plt.subplots ()

fvsmax.pie(fvsmcdf['Count'],explode=[0.1,0.1], labels=(fvsmcdf['Sex']), autopct='%1.1f%%',startangle=90)

fvsmax.axis('equal')

fvsmpie = plt.title("Count of People on Board based on Sex")

plt.show()
# Count how many survived per gender

svss = data.groupby(['Sex', 'Survived']).size()

psvss = []

psvss.append({'Sex' : 'female', 'Survived' : 'false', 'Count' : svss.female[0],'Percentage' : svss.female[0]/svss.sum()})

psvss.append({'Sex' : 'female', 'Survived' : 'true', 'Count' : svss.female[1],'Percentage' : svss.female[1]/svss.sum()})

psvss.append({'Sex' : 'male', 'Survived' : 'false', 'Count' : svss.male[0],'Percentage' : svss.male[0]/svss.sum()})

psvss.append({'Sex' : 'male', 'Survived' : 'true', 'Count' : svss.male[1],'Percentage' : svss.male[1]/svss.sum()})

psvssdf = pd.DataFrame(psvss)

psvssdf = psvssdf[['Sex', 'Survived', 'Count', 'Percentage']]

psvssdf
# Barchart of Survival and Sex

sns.barplot(x="Sex", y="Count", hue="Survived", data=psvssdf)
# Check survival of male and female passengers

sfvsma = []

sfvsmat = svss.female[1] + svss.male[1]

sfvsma.append({'Sex' : 'female', 'Survived' : 'true', 'Count' : svss.female[1],'Percentage' : svss.female[1]/sfvsmat})

sfvsma.append({'Sex' : 'male', 'Survived' : 'true', 'Count' : svss.male[1],'Percentage' : svss.male[1]/sfvsmat})



sfvsmadf = pd.DataFrame(sfvsma)

sfvsmadf = sfvsmadf[['Sex', 'Survived', 'Count', 'Percentage']]

sfvsmadf

# pie chart for survival of male and female passengers

sfvsmfig, sfvsmax = plt.subplots ()

sfvsmax.pie(sfvsmadf['Count'],explode=[0.1,0.1], labels=(sfvsmadf['Sex']), autopct='%1.1f%%',startangle=90)

sfvsmax.axis('equal')

sfvsmpie = plt.title("Count of Male and Female Survivors")

plt.show()
# graph to show age distribution of survival and deaths

graph = sns.FacetGrid(data, hue="Survived",aspect=3)

graph.map(sns.kdeplot,'Age',shade= True)

graph.set(xlim=(0, data['Age'].max()))

graph.add_legend()

plt.title("Survival Based on Age")
# get count by Pclass

pcc = data.groupby(['Pclass']).size()

pccarr = []

pccarr.append({'Pclass': 1, 'Count': pcc[1]})

pccarr.append({'Pclass': 2, 'Count': pcc[2]})

pccarr.append({'Pclass': 3, 'Count': pcc[3]})



pccdf = pd.DataFrame(pccarr)

pccdf = pccdf[['Pclass', 'Count']]

pccdf
# pie chart for count and sex of passengers

pccfig, pccax = plt.subplots ()

pccax.pie(pccdf['Count'],explode=[0.1,0.1,0.1], labels=(pccdf['Pclass']), autopct='%1.1f%%',startangle=90)

pccax.axis('equal')

pccpie = plt.title("Survival based on Social Class")

plt.show()
# get count of survival by Pclass

svspc = data.groupby(['Pclass','Survived']).size()

svspcarr = []

svspcarr.append({'Pclass' : 1, 'Survived' : 'false', 'Count' : svspc[1][0], 'Percentage':svspc[1][0]/svspc.sum()})

svspcarr.append({'Pclass' : 1, 'Survived' : 'true', 'Count' : svspc[1][1], 'Percentage':svspc[1][1]/svspc.sum()})

svspcarr.append({'Pclass' : 2, 'Survived' : 'false', 'Count' : svspc[2][0], 'Percentage':svspc[2][0]/svspc.sum()})

svspcarr.append({'Pclass' : 2, 'Survived' : 'true', 'Count' : svspc[2][1], 'Percentage':svspc[2][1]/svspc.sum()})

svspcarr.append({'Pclass' : 3, 'Survived' : 'false', 'Count' : svspc[3][0], 'Percentage':svspc[3][0]/svspc.sum()})

svspcarr.append({'Pclass' : 3, 'Survived' : 'true', 'Count' : svspc[3][1], 'Percentage':svspc[3][1]/svspc.sum()})



svspcdf = pd.DataFrame(svspcarr)

svspcdf = svspcdf[['Pclass', 'Survived', 'Count', 'Percentage']]

svspcdf
# get survived count per Pclass

svspcsarr = []

svspcsarr.append(svspcarr[1])

svspcsarr.append(svspcarr[3])

svspcsarr.append(svspcarr[5])



svspcsdf = pd.DataFrame(svspcsarr)

svspcsdf = svspcsdf[['Pclass', 'Survived', 'Count']]



# pie chart for count and sex of passengers

svspcfig, svspcax = plt.subplots ()

svspcax.pie(svspcsdf['Count'],explode=[0.1,0.1,0.1], labels=(svspcsdf['Pclass']), autopct='%1.1f%%',startangle=90)

svspcax.axis('equal')

svspcpie = plt.title("Survival based on Social Class")

plt.show()
# a histogram to show the distribution of fares

plt.hist(data.Fare,bins=range(0, 600, 50))
# graph to show distribution of survivors based on fare

graph = sns.FacetGrid(data, hue="Survived",aspect=3)

graph.map(sns.kdeplot,'Fare',shade= True)

graph.set(xlim=(0, data['Fare'].max()))

graph.add_legend()

plt.title("Survival Based on Fare")
# boxplot of Pclass and fare

sns.boxplot(x=data.Pclass, data=data, y=data.Fare)
# Check for correlation between survival and number of siblings and spouse on Titanic

data['Survived'].corr(data['SibSp'])
# Scatterplot of survival and number of siblings and spouse on board

sns.stripplot(x="Survived", y="SibSp", data=data);
# Check for correlation between survival and number of parents and children on Titanic

data['Survived'].corr(data['Parch'])


# Scatterplot of survival and number of siblings and spouse on board

sns.stripplot(x="Survived", y="Parch", data=data);
# Count survivors and deaths per port of embarkation

evss = data.groupby(["Embarked","Survived"]).size()



# Create Dataframe for Ports of Embarkation

evss_arr = []

evss_arr.append({'Embarked' : 'Southampton', 'Survived': 'True', 'Count' : evss.S[1]})

evss_arr.append({'Embarked' : 'Southampton', 'Survived': 'False', 'Count' : evss.S[0]})

evss_arr.append({'Embarked' : 'Cherbourg', 'Survived': 'True', 'Count' : evss.C[1]})

evss_arr.append({'Embarked' : 'Cherbourg', 'Survived': 'False', 'Count' : evss.C[0]})

evss_arr.append({'Embarked' : 'Queenstown', 'Survived': 'True', 'Count' : evss.Q[1]})

evss_arr.append({'Embarked' : 'Queenstown', 'Survived': 'False', 'Count' : evss.Q[0]})

evss_df = pd.DataFrame(evss_arr)

evss_df = evss_df[['Embarked', 'Survived', 'Count']]

evss_df
# Pie Chart for Distribution of Port of Embarkation



fig1, ax = plt.subplots()

ax.pie(data.Embarked.value_counts(), explode=[0.1, 0.1, 0.1],  labels=('S', 'C', 'Q'), autopct='%1.1f%%',

       startangle=90)

ax.axis('equal')



embarked = plt.title("Distribution of Port of Embarkation")

plt.show()
# Barchart of Survival of Port of Embarkation



ax = sns.countplot(x="Embarked", hue="Survived", data=data)