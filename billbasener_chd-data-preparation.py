import numpy as np 
import pandas as pd
chd_data = pd.read_csv("../input/coronary-heart-disease/CHDdata.csv")
chd_data.describe()
#drop "famhist" column
chd_data = chd_data.drop(columns= "famhist")
#check that it worked
chd_data.head()
for key in chd_data.keys()[0:8]:
    print("Standardizing "+key+".")
    chd_data[key] = chd_data[key] - np.mean(chd_data[key])
    chd_data[key] = chd_data[key] / np.std(chd_data[key])
# Check that it worked
chd_data.describe()
chd_positive = chd_data[chd_data.chd == 1]
chd_negative = chd_data[chd_data.chd == 0]
# Check that it worked
chd_positive.describe()
# Check that it worked
chd_negative.describe()
# Lets check the mean of each class to get a first look at the seperation
print("Mean for CHD Positive:")
print(np.array([chd_positive.mean()[0:8]]))
print("Mean for CHD Negative:")
print(np.array([chd_negative.mean()[0:8]]))