# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Upload data - only digimon list

df = pd.read_csv("../input/DigiDB_digimonlist.csv")
#Examine top five data entries to get an idea bout the data

df.head()
#Get a summary of the numerical data, transpose switches rows and columns

df.describe().transpose()
#Check data completeness and size

df.isnull().values.any(), df.shape
df.columns
#Graph the Stage variable 

sns.countplot(df['Stage']).set_title("Distribution of Digimon based on Stage")
sns.pairplot(df,hue="Type")
sns.lmplot(x="Lv50 SP", y="Lv50 Int", col="Type", hue="Type", data=df)
sns.lmplot(x="Lv50 SP", y="Lv50 Int", col="Attribute", hue="Attribute", data=df)