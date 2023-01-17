# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import data
pokemon = pd.read_csv("../input/pokemon.csv", sep = ",", encoding = "ISO-8859-1")
# basic information about the data
pokemon.info()
# renaming the columns
pokemon.rename(index = str, columns = {"#": "Id"}, inplace = True)
# Id of the missing name
pokemon['Id'][pokemon["Name"].isnull()]
pokemon.loc[pokemon["Id"].isin(list(range(62, 64))), :] 
# the pokemon is primeape, mankey's evolved form
# treating the missing value in the Name field.
pokemon.loc[pokemon.Name.isnull(), "Name"] = "Primeape"
# filtering out the gen 6 and 7 pokemon
analysis_set = pokemon[pokemon["Generation"] < 6]
analysis_set = analysis_set.loc[analysis_set["Legendary"] == False, :]
# changing the type fairy to normal
analysis_set.loc[analysis_set["Type 1"] == "Fairy", "Type 1"] = "Normal"
analysis_set.loc[analysis_set["Type 2"] == "Fairy", "Type 2"] = np.nan
# filtering out the mega evolutions
analysis_set = analysis_set[~analysis_set.Name.str.contains("Mega")]
# since we'll be reusing this groupby, let's store it.
type1_grp = analysis_set.groupby("Type 1")
# most common and least common types
type1_grp["Id"].count().sort_values(ascending = False)
type1_grp["Attack"].agg(["min", "mean", "max"]).sort_values(by = ["min", "mean", "max"], ascending = False)
# Special Attack
type1_grp["Sp. Atk"].agg(["min", "mean", "max"]).sort_values(["mean", "max"], ascending = False)
# defense
type1_grp["Defense"].agg(["min", "mean", "max"]).sort_values(["mean", "max"], ascending = False)
# special defense
type1_grp["Sp. Def"].agg(["min", "mean", "max"]).sort_values(["mean", "max"], ascending = False)
# shuckle
analysis_set.loc[pokemon["Name"] == "Shuckle", :]
# fastest type
type1_grp["Speed"].agg(["min", "mean", "max"]).sort_values(["mean", "max"], ascending = False)
# the fastest pokemon
analysis_set.loc[analysis_set["Speed"] == 160, :]
# most common secondary type
analysis_set.groupby("Type 2")["Id"].count().sort_values(ascending = False) # Flying it is...
# most common typing
analysis_set.groupby(["Type 1", "Type 2"])["Id"].count().sort_values(ascending = False)
# the ground-electric pokemon
analysis_set[(analysis_set["Type 1"] == "Ground") & (analysis_set["Type 2"] == "Electric")]