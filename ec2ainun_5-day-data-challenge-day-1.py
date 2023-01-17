import pandas as pd
#read and print entire csv digimon list

pd.read_csv("../input/DigiDB_digimonlist.csv")
#assign digimon list data to variable dataset and describe it

dataset = pd.read_csv("../input/DigiDB_digimonlist.csv")

dataset.describe()
dataset.describe().transpose()