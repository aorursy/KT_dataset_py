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
# import data

traindf = pd.read_csv("../input/train.csv")

traindf.head(5) # print to check if loaded
# count how many survived per gender

svss = traindf.groupby(['Sex', 'Survived']).size()

svss
# organize data into a dataframe

psvss = []

psvss.append({'Sex' : 'female', 'Survived' : 'false', 'Count' : svss.female[0],'Percentage' : svss.female[0]/svss.sum()})

psvss.append({'Sex' : 'female', 'Survived' : 'true', 'Count' : svss.female[1],'Percentage' : svss.female[1]/svss.sum()})

psvss.append({'Sex' : 'male', 'Survived' : 'false', 'Count' : svss.male[0],'Percentage' : svss.male[0]/svss.sum()})

psvss.append({'Sex' : 'male', 'Survived' : 'true', 'Count' : svss.male[1],'Percentage' : svss.male[1]/svss.sum()})

psvssdf = pd.DataFrame(psvss)

psvssdf = psvssdf[['Sex', 'Survived', 'Count', 'Percentage']]

psvssdf
# graph count of people who died/survived depending on sex

import seaborn as sns

sns.barplot(x="Sex", y="Count", hue="Survived", data=psvssdf)

fvsmc = traindf.groupby(['Sex']).size()

fvsmc
fvsmcarr = []

fvsmcarr.append({'Sex': 'female', 'Count': fvsmc.female})

fvsmcarr.append({'Sex': 'male', 'Count': fvsmc.male})

fvsmcdf = pd.DataFrame(fvsmcarr)

fvsmcdf = fvsmcdf[['Sex','Count']]

fvsmcdf
import matplotlib.pyplot as plt

fvsmfig, fvsmax = plt.subplots ()

fvsmax.pie(fvsmcdf['Count'],explode=[0.1,0.1], labels=(fvsmcdf['Sex']), autopct='%1.1f%%',startangle=90)

fvsmax.axis('equal')

fvsmpie = plt.title("Count of People on Board based on Sex")

plt.show()