import scipy.stats #for a bit of statistiscs
import numpy as np # for numerical python
import pandas as pd # for handling dataframes
import seaborn as sns # for better plots
import matplotlib.pyplot as plt # for plots
# importing the data
alone = pd.read_csv("../input/foreveralone.csv", sep= ",", encoding = "ISO-8859-1")
alone.info()
# dropping jobtitle
alone.drop("job_title", axis = 1, inplace = True)
alone["what_help_from_others"].unique()
# getting the most common form of help people want
alone[alone["what_help_from_others"] != "I don't want help"]["what_help_from_others"].str. \
    get_dummies(",").sum().sort_values(ascending = False)[:10]
alone["improve_yourself_how"].str.get_dummies(",").sum().sort_values(ascending = False)[:10]
%matplotlib inline
sns.set()
plt.style.use("ggplot") # yea, I still love ggplot2
sns.countplot(x = "virgin", data = alone, hue = "depressed")
scipy.stats.chisquare(f_obs = [52, 65, 105, 247], f_exp = [39, 78, 118, 234]) 
# expected values were calculated by hand
sns.countplot(x = "virgin", data = alone, hue = "bodyweight")
plt.title("Bodyweight and virginity")
plt.xlabel("Are you a virgin?")
alone.pivot_table("age", index = "virgin", columns = "bodyweight", aggfunc = "count")
# chisquare test on virginity and bodyweight
scipy.stats.chisquare(f_obs = [77, 7, 25, 8, 192, 18, 88, 54], 
	f_exp = [67.10, 6.23, 28.19, 15.46, 201.89, 18.76, 84.81, 46.53])