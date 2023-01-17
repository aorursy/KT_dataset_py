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

data
data.corr()
pandas_profiling.ProfileReport(data)
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



ax = sns.countplot(x="Survived", hue="Sex", data=data)
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