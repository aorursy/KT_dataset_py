import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pokedata = pd.read_csv("../input/pokemon-all-cleaned/pokemon_cleaned.csv")
pokedata.head()
pokedata[10:15]
pokedata[pokedata['Name']=='Charizard']
pokedata[pokedata['Speed']>150]
pokedata[(pokedata['Type 1']=='Electric') & (pokedata['Total']>500) & (pokedata['Legendary']==False)]
pokedata['Total'].idxmax()
pokedata['Total'].idxmin()
pokedata.loc[[50]]
pokedata.loc[10:15]
pokedata.loc[[pokedata['Total'].idxmax()]]
pokedata.loc[[pokedata['Total'].idxmin()]]