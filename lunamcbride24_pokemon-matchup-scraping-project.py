# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt #Plotting graphs
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pokemon = pd.read_csv("../input/pokemon-type-matchup-data/PokeTypeMatchupData.csv") #Open the type matchup data
pokemon.head() #Take a peek at the data
print(pokemon.isnull().any()) #Check for any null values
#Type matchup values to make floats
types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
         "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
         "Dragon", "Dark", "Steel", "Fairy"]

pokemon.replace(to_replace = "[#, *]", value = "", regex = True, inplace = True) #Remove the multiplier * and the # in the pokedex number
pokemon[types] = pokemon[types].astype(float) #Set all the matchup values to floats
pokemon["Number"] = pokemon["Number"].astype(int) #Set the pokedex number to an int
pokemon.head() #Take a peek at the dataset
fig, axes = plt.subplots(nrows = 3, ncols = 6, figsize = (40, 20)) #Set the figures

#For each type
for i in range(0, len(types)):
    #If else to segment the data to fit the subplots, as it has 3 rows for 18 types
    if i<6:
        a = 0
    elif i<12:
        a = 1
    else:
        a = 2
    pokemon.hist(column = types[i], edgecolor = "white", ax = axes[a][i - a * 6]) #Take a look at commonality of values

plt.plot() #Plot the data
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
sums = pd.DataFrame(pokemon[types].sum(), columns = ["Sum"]) #Make a dataframe of the sums of the multipliers for each type
sums["Type"] = types #Add the types to the sums
sums = sums.sort_values(by = "Sum", ascending  = False) #Sort the sums of the multipliers
sums.plot.bar("Type", "Sum", edgecolor = "white", ax = axes) #Plot the sums
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
means = pd.DataFrame(pokemon[types].mean(), columns = ["Mean"]) #Make a dataframe of the means of the multipliers for each type
means["Type"] = types #Add the types to the means
means = means.sort_values(by = "Mean", ascending  = False) #Sort the means of the multipliers
means.plot.bar("Type", "Mean", edgecolor = "white", ax = axes) #Plot the means
#CategorizeMatchup: Removes x4 and x0.25 values to simply gage super effective and not very effective
#Input: The damage multiplier
#Output: The damage multiplier, just with x4 -> x2 and x0.25 -> x0.5
def categorizeMatchup(mult):
    if mult > 1: #If the value is more than 1
        return 2 #Return the super effective value 2
    if mult < 1 and mult > 0: #If the value is between 0 and 1
        return 0.5 #Return the not very effective value
    return mult #Return the original value if it is normal effect (1) or not effective (0)

#For loop to fix the matchups for each type column
for typ in types:
    pokemon[typ] = pokemon[typ].apply(categorizeMatchup) #Fix the effectiveness values for the type
pokemon.head() #Take a peek at the database (Remember: grass on Bulbasaur was 0.25 before)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
sums = pd.DataFrame(pokemon[types].sum(), columns = ["Sum"]) #Make a dataframe of the sums of the multipliers for each type
sums["Type"] = types #Add the types to the sums
sums = sums.sort_values(by = "Sum", ascending  = False) #Sort the sums of the multipliers
sums.plot.bar("Type", "Sum", edgecolor = "white", ax = axes) #Plot the sums
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
means = pd.DataFrame(pokemon[types].mean(), columns = ["Mean"]) #Make a dataframe of the means of the multipliers for each type
means["Type"] = types #Add the types to the means
means = means.sort_values(by = "Mean", ascending  = False) #Sort the means of the multipliers
means.plot.bar("Type", "Mean", edgecolor = "white", ax = axes) #Plot the means
#SuperEffective: changes every multiplier except x2 into 0, thus not counting them in the analysis
#Input: the multiplier
#Output: 1 for super effective, 0 for not super effective
def superEffective(mult):
    if mult == 2: #If the type is super effective
        return 1 #Return 1 for isSuperEffective
    return 0 #Return a 0 for NotSuperEffective

superEff = pokemon[types].copy() #Take a copy of the types to change for super effective only

#For each type column, adjust the super effectiveness
for typ in types:
    superEff[typ] = superEff[typ].apply(superEffective) #Adjust the column following the superEffective method
superEff.head() #Take a peek at the data
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
sums = pd.DataFrame(superEff.sum(), columns = ["Sum"]) #Make a dataframe of the sums of the multipliers for each type
sums["Type"] = types #Add the types to the sums
sums = sums.sort_values(by = "Sum", ascending  = False) #Sort the sums of the multipliers
sums.plot.bar("Type", "Sum", edgecolor = "white", ax = axes) #Plot the sums
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
means = pd.DataFrame(superEff.mean(), columns = ["Mean"]) #Make a dataframe of the means of the multipliers for each type
means["Type"] = types #Add the types to the means
means = means.sort_values(by = "Mean", ascending  = False) #Sort the means of the multipliers
means.plot.bar("Type", "Mean", edgecolor = "white", ax = axes) #Plot the means
#NotEffective: changes every multiplier except those less than 1 into 0, thus not counting them in the analysis
#Input: the multiplier
#Output: 1 for not very effective/not effective, 0 for effective
def notEffective(mult):
    if mult < 1: #If the type is less than effective
        return 1 #Return 1 for isNotVery/NotEffective
    return 0 #Return a 0 for Effective

notEff = pokemon[types].copy() #Take a copy of the types to change for super effective only

#For each type column, adjust the super effectiveness
for typ in types:
    notEff[typ] = notEff[typ].apply(notEffective) #Adjust the column following the superEffective method
notEff.head() #Take a peek at the data
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
sums = pd.DataFrame(notEff.sum(), columns = ["Sum"]) #Make a dataframe of the sums of the multipliers for each type
sums["Type"] = types #Add the types to the sums
sums = sums.sort_values(by = "Sum", ascending  = False) #Sort the sums of the multipliers
sums.plot.bar("Type", "Sum", edgecolor = "white", ax = axes) #Plot the sums
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6)) #Set the figures
means = pd.DataFrame(notEff.mean(), columns = ["Mean"]) #Make a dataframe of the means of the multipliers for each type
means["Type"] = types #Add the types to the means
means = means.sort_values(by = "Mean", ascending  = False) #Sort the means of the multipliers
means.plot.bar("Type", "Mean", edgecolor = "white", ax = axes) #Plot the means