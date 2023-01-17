# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
forest_data = pd.read_csv("../input/forest-area-of-land-area/forest_area.csv")
forest_data.head(20)
forest_data.info()
happiness_data = pd.read_csv("../input/world-happiness/2015.csv")
happiness_data.info()
happiness_data.head(5)
new_forest_data = pd.concat([forest_data.iloc[:,:1],forest_data.iloc[:,-1]],axis=1) # getting country name and 2015 forest rate.
#merging happiness data and forest data, and then rename 2015 column to ForestRatio and Drop CountryName column which is dublicate because of merging.
result = happiness_data.merge(new_forest_data, left_on='Country',right_on='CountryName') 
result.rename(columns={'2015': 'ForestRatio'}, inplace=True)
result.drop(["CountryName"], axis= 1, inplace=True)
result.head(20)
main_data = result[pd.notnull(result.ForestRatio) & result.ForestRatio > 0] #ForestRatio should be not null and greater than 0
main_data.columns = [each.replace(" ","").replace("(","_").replace(")","") for each in main_data.columns]
main_data.corr()
plt.scatter(x = main_data.HappinessScore, y = main_data.ForestRatio)
plt.xlabel("Happiness Score")
plt.ylabel("Forest Ratio")
plt.title("Happiness Score vs Forest Ratio")
#ok lets start to change our perspective
new_data = main_data.groupby(["Region"]).mean()
new_data.corr()
new_data
plt.scatter(x = new_data.ForestRatio, y = new_data.Freedom)
plt.xlabel("Forest Ratio")
plt.ylabel("Freedom")
plt.title("Forest Ratio vs Freedom")
happiness_data.columns = [ each.replace(" ","").replace("(","_").replace(")","") for each in happiness_data.columns ]
happiness_data.columns
happiness_data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(happiness_data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
happiness_data.Family.plot(color = 'r',label = 'Family',linewidth=1, alpha = 0.9,grid = True)
happiness_data.Economy_GDPperCapita.plot(color = 'orange',label = 'Economy_GDPperCapita',linewidth=1, alpha = 0.9,grid = True)
happiness_data.Health_LifeExpectancy.plot(color = 'yellow',label = 'Health (Life Expectancy)',linewidth=1, alpha = 1.0,grid = True)
happiness_data.Freedom.plot(color = 'g',label = 'Freedom',linewidth=1, alpha = 0.9,grid = True)
happiness_data.Trust_GovernmentCorruption.plot(color = 'black',label = 'Trust (Government Corruption)',linewidth=1, alpha = 0.9,grid = True)
happiness_data.Generosity.plot(color = 'b',label = 'Generosity',linewidth=1, alpha = 0.9,grid = True)
happiness_data.DystopiaResidual.plot(color = 'gray',label = 'Dystopia Residual',linewidth=1, alpha = 0.9,grid = True)

plt.legend() 
plt.xlabel('Happiness Rank')
plt.ylabel('Score')
plt.title('Happiness Factors Line Plot Graph')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
plt.show()
fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize=(30, 15))
happiness_data.plot(kind = "scatter", x= "Family", y = "HappinessScore", ax = axes[0][0])
happiness_data.plot(kind = "scatter", x= "Economy_GDPperCapita", y = "HappinessScore", ax = axes[0][1])
happiness_data.plot(kind = "scatter", x= "Health_LifeExpectancy", y = "HappinessScore", ax = axes[0][2])
happiness_data.plot(kind = "scatter", x= "Freedom", y = "HappinessScore", ax = axes[0][3])
happiness_data.plot(kind = "scatter", x= "Trust_GovernmentCorruption", y = "HappinessScore", ax = axes[1][0])
happiness_data.plot(kind = "scatter", x= "Generosity", y = "HappinessScore", ax = axes[1][1])
happiness_data.plot(kind = "scatter", x= "DystopiaResidual", y = "HappinessScore", ax = axes[1][2])
plt.show()
fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize=(30, 15))
happiness_data.Family.plot(kind = 'hist',bins = 50, ax = axes[0][0])
happiness_data.Economy_GDPperCapita.plot(kind = "hist", ax = axes[0][1])
happiness_data.Health_LifeExpectancy.plot(kind = "hist", ax = axes[0][2])
happiness_data.Freedom.plot(kind = "hist", ax = axes[0][3])
happiness_data.Trust_GovernmentCorruption.plot(kind = "hist", ax = axes[1][0])
happiness_data.Generosity.plot(kind = "hist", ax = axes[1][1])
happiness_data.DystopiaResidual.plot(kind = "hist",  ax = axes[1][2])
plt.show()
happiness_data.boxplot(column='HappinessScore',by = 'Region', figsize=(30, 15))
happiness_data["HappinessDegree"] = ['Happy' if each > 6 else 'Normal' if each > 5 else 'Unhappy' for each in happiness_data.HappinessScore ]
pivot_data = happiness_data.pivot_table( index=['Region'], columns = "HappinessDegree", values = "HappinessRank",aggfunc='count')
pivot_data