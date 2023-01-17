# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv',index_col='AnimalID').fillna('-1 value')

# **Note: all these functions can be implemented using "map" functions that could be easily 
# implemented in PySpark. 
# Wishing to see the implementation of map functios in pandas too, since I am not 
# experienced with them!**

print(train.head())


# **I am assuming that the Name column will not provide information**
# 
# **I still not understanding the value of OutcomeSubtype!!! I will remove it**
train.drop('Name', axis=1, inplace=True)
train.drop('OutcomeSubtype', axis=1, inplace=True)
# **the "sexuponOutcome" is a combination of the real sex of the animal 
# and the state of the sex such Neutered,Spayed or Intact.
# I will split these two attributes in two columns of the dataframe**
# splitting sex type and the animal state and removing the original columns
# splitting sex type and the animal state and removing the original columns
train["sex"]  = [x[1] if len(x)==2 else "Unknown" for x in [ x.split(" ") for x in train['SexuponOutcome'].values] ]
train["sex_state"] = [x[0] if len(x)==2 else "Unknown" for x in [ x.split(" ") for x in train['SexuponOutcome'].values] ]
train.drop('SexuponOutcome', axis=1, inplace=True)
# **The Color appears like a composite column. 
# I will split this column in Primary Color and Secondary Color**
# color can be mostly splitted between primary color and secondary color 
train["primary color"]   = [x[1] if len(x)==2 else x[0].split(" ")[0] for x in [ x.split("/") for x in train['Color'].values] ]
train["secondary color"] = [x[0] if len(x)==2 else x[0].split(" ")[0] for x in [ x.split("/") for x in train['Color'].values] ]
train.drop('Color', axis=1, inplace=True)
# **The Breed attribute also appears like a composite column. 
# Since the animal can be a mix of several breeds, I add a further column is the animal
# is a mix.
# I will further split the Breed column in a firtBreedAttribute and a secondaryBreedAttribute.
# Finally add a 3rd column if the breed has more than 3 attributed**
# 
# *Note: I am not happy with this split since I see I am lost some important words about the real 
# breed of the animal. Help is appreciated!*
# check if the animal is a mix 
train["isMix"]     = [x[-1] == "Mix" for x in [ x.split(" ") for x in train['Breed'].values] ]
train["firtBreedAttribute"] = [x[0]  if len(x)>1 else x[0] for x in [ x.split(" ") for x in train['Breed'].values] ]
train["secondaryBreedAttribute"] = [x[1]  if len(x)>2 else x[0] for x in [ x.split(" ") for x in train['Breed'].values] ]
train["hasMoreBreedAttributes"] = [len(x)>3 if x[-1] == "Mix" else len(x)>2 for x in [ x.split(" ") for x in train['Breed'].values] ]

train.drop('Breed', axis=1, inplace=True)
# **The age upon outcome is mixed between years, months and weeks. 
# I will normalized this column considering only the age in weeks**
# normalized age in weeks 
numOfWeeksPerYear = 52
numOfWeeksPerMonth = 4
# int(x[0])*numOfWeeksPerYear if x[1].startswith('year') elseif x[0]
def getNormalizedWeeks(x):
    if x[1].startswith('year'):
        nw = int(x[0])*52
    elif x[1].startswith('month'):
        nw = int(x[0])*4
    else:
        nw = int(x[0])
    return nw

train["normalizedAgeuponOutcome"] = [ getNormalizedWeeks(x) for x in [ x.split(" ") for x in train['AgeuponOutcome'].values]]
train.drop('AgeuponOutcome', axis=1, inplace=True)
# **As last, I assume here that the information about the month when the animal
# has been outcome could bring some information. Instead of the full date, I will
# create a column that will bring only the information of the month.**
train["OutcomeMonth"] = [x[0].split("-")[1] for x in [ x.split(" ") for x in train['DateTime'].values] ]
train.drop('DateTime', axis=1, inplace=True)
# ### A final data format looks like this
print(train.head(20))

