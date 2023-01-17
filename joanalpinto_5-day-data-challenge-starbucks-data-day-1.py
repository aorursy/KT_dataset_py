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



# Read and describe data

df_drinks = pd.read_csv("../input/starbucks-menu-nutrition-drinks.csv")

df_drinks.describe()

df_food = pd.read_csv("../input/starbucks-menu-nutrition-food.csv", encoding='latin-1')

df_food.describe()

df_drinks_expanded = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

df_drinks_expanded.describe()
