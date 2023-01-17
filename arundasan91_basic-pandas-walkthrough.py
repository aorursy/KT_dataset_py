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
# Load the dataset.

data = pd.read_csv('../input/astronauts.csv')
# View first 5 rows

data.head()
# Information regarding the classes

data.info()
features = list(data.columns)
print("Unique values in each features:\n")

unique_features = {}

for feature in features:

    unique_features[str(feature)]=data[feature].unique().shape[0]

    print('{} : {}'.format(feature, data[feature].unique().shape[0]))
# How many Astronauts are there from Texas ?

print("There are", data[data['Birth Place'].str.contains(r'TX')]['Name'].count(), \

      "Astronauts from TX") 

#print("Names: \n", data[data['Birth Place'].str.contains(r'TX')]['Name'].values)

data[data['Birth Place'].str.contains(r'TX')][['Name','Birth Place']]
#Details of Astronauts from Texas

data[data['Birth Place'].str.contains(r'TX')]
# How many astronauts lost their lives during flight / Death Mission.

print("{} astronauts lost their lives (Death Missions).".format(data[data['Death Mission'].notnull()]['Name'].count()))
# Names of all astronauts who left us during flight / Death Mission.

list(data[data['Death Mission'].notnull()]['Name'])