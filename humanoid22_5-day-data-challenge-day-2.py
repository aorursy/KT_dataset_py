# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

#import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

nutrition = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

nutrition.describe()

print(nutrition.columns)

calories = nutrition["Calories"]

#plt.hist(calories)

#plt.title("Calories")



sns.distplot(calories, kde = False).set_title("Calories") 



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.