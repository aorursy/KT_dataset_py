import numpy as np

import pandas as pd

from scipy.stats import chisquare,chi2_contingency
# Chi-Square Test to find the Goodness of Fit for Categorical Variables



car = pd.read_csv('../input/car-evaluation/car.data.txt',encoding = 'utf-8',header = None, usecols=[0,1,2,3,4,5])

car.head()
car.describe()
car = car.rename(columns={0: 'buying', 1: 'maint', 2: 'doors', 3: 'persons', 4: 'lug_boot', 5: 'safety'})
car.head()
car.info()
#Goodness of fit test for a single Categorical variable



# Let	pi denote	the	proportion	in	the	ith category

# H0	:	All	pi s	are	the	same	

# Ha	:	At	least	one	pi differs	from	the	others	



chisquare(car["doors"].value_counts())



#The p-value > 0.05 hence we conclude that all proportions are the same
chisquare(car["lug_boot"].value_counts())



#The p-value > 0.05 hence we conclude that al proportions are equal
# Goodness of Fit Test between 2 categorical variables



# H0: The two categorical variables are independent

# Ha: The two categorical variables are dependent



# Creating contingency table

cont = pd.crosstab(car["doors"],

                   car["lug_boot"])
cont
chi2_contingency(cont)



#The p-value > 0.05 hence we conclude that the 2 categorical variables are Independent