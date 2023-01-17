# DataFrame Manipulation Package
import pandas as pd

# Altair is a declarative statistical visualization library
# for Python, based on Vega and Vega-Lite.
# Data in Altair is built around the Pandas Dataframe!
import altair as alt
import numpy as np
file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter03/bank-full.csv'
bankData = pd.read_csv(file_url, sep=";")
bankData.head()
filter_mask = bankData['y'] == 'yes'
bankSub1 = bankData[filter_mask].groupby('age')['y'].agg(agegrp='count').reset_index()
# Visualising the relationship using altair
alt.Chart(bankSub1).mark_line().encode(x='age', y='agegrp')
# Getting another perspective
ageTot = bankData.groupby('age')['y'].agg(ageTot='count').reset_index()
ageTot.head()
# Getting all the details in one place
ageProp = bankData.groupby(['age','y']).agg(np.sum).reset_index()
#ageProp = bankData.groupby(['age','y'])['age'] ageCat='count'
ageProp.head()
# Merging both the data frames
ageComb = pd.merge(ageProp, ageTot,left_on = ['age'], right_on = ['age'])
ageComb['catProp'] = (ageComb.ageCat/ageComb.ageTot)*100
ageComb.head()
# Visualising the relationship using altair
alt.Chart(ageComb).mark_line().encode(x='age', y='catProp').facet(column='y')