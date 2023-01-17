import numpy as np

import pandas as pd
#Let's suppose we have a data_set of population of different countries.

#There is a country name column in that data set with followig entries.



country_name=["India","USA","China","Lebanon","india","InDia","South Africa","china"]



#check unique values in country_name.



print(np.unique(country_name))
#In above column we can see that there is a clear inconsistency in country name.

#As we know India,india,InDia all represent same country and same thing in case of China and china, but it's obvious that any programming lnguage will

#treat them differently.

#We can deal with this problem by converting them into lowercase or uppercase letters



country_name_lower=[name.lower() for name in country_name]

country_name_upper = [name.upper() for name in country_name]

print("Countries Name")

print(country_name_lower)

print(country_name_upper)

print("Unique Countries Name")

print(np.unique(country_name_lower))

print(np.unique(country_name_upper))
#we are taking same country_name list.

country_name=["India","USA","China","Lebanon","SouthAfrica","Austria","Australia","Pakistan","South Africa","South'Africa"]

print(np.unique(country_name))
#In above country_name both SouthAfrica,South Africa and South'Afrcia all are same for us but not for programming language.

#We can solve this by replacing one of them with other.

#Like we can replace "South Africa" with "SouthAfrica" or vice versa.



import string



country_name_lower = [name.lower() for name in country_name]

country_name_lower_right=[]

for name in country_name_lower:

    name_char=[ch for ch in name if ch not in string.punctuation+" "] #space is not in punctuation. run in code `" " in string.punctuation`

    country_name_lower_right.append("".join(name_char))

    

print("Country Name Before")

print(country_name_lower)

print("Country Name After")

print(country_name_lower_right)

print("Unique Country Name Before")

print(len(np.unique(country_name_lower)))

print("Unique Country Name After")

print(len(np.unique(country_name_lower_right)))
#let's take example of height



height_of_children_before = [1.2,1.7,1,-1.2,2.4,1.6,-1.8]



#this height_of_children contains negative values which is thoroughly wrong because height of anyone can't be negative it's always greater than zero.

#let's make it correct



height_of_children_after = np.abs(height_of_children_before)



print(height_of_children_after)