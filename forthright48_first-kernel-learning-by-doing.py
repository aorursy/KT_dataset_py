# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from IPython.display import display
csv = pd.read_csv("../input/train.csv")



csv.head()
# Lets start with processing gender.

# Woman and kids should have higher chance to survive

# Is there really such correlation?



gender = csv[["Sex","Survived"]]



total = len(gender)

male = gender.query("Sex == 'male'")

female = gender.query("Sex == 'female'")



totalMale = len(male)

totalFemale = len(female)

maleAlive = len( male.query("Survived == 1") )

femaleAlive = len ( female.query("Survived == 1") )

maleDead = totalMale - maleAlive

femaleDead = totalFemale - femaleAlive



# Total Population

plt.figure(figsize=(3,3))

plt.title("Ratio of Gender Survival")

labels = ["Male Alive", "Male Dead", "Female Alive", "Female Dead"]

fracs = [maleAlive,maleDead,femaleAlive,femaleDead]

plt.pie(fracs,labels=labels,autopct='%.2f%%')



# Male population Ratio

plt.figure(figsize=(3,3))

plt.title("Percentage of Male Survived")

labels = ["Alive", "Dead"]

fracs = [maleAlive,maleDead]

plt.pie(fracs,labels=labels,autopct='%.2f%%')

         

# Female population Ratio

plt.figure(figsize=(3,3))

plt.title("Percentage of Female Survived")

labels = ["Alive", "Dead"]

fracs = [femaleAlive,femaleDead]

plt.pie(fracs,labels=labels,autopct='%.2f%%')



plt.show()



print ( "Female have stronger chances to survive")