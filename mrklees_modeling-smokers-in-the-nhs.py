import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import statsmodels.api as sm

# I've added Daft to the stack so that I can visualize my graphical models beautifully

import daft

from daft import PGM



flatdata = pd.read_csv('../input/nhs-tobacco-flattened/flattened.csv').drop("Unnamed: 0", axis=1)
flatdata.head(5)
flatdata.describe()
# This is was final version of my causal graph.  While initially taken a priori, it's validity is (somewhat) confirmed by the data. 

pgm = PGM(shape=[8, 5], origin=[0, 0])



pgm.add_node(daft.Node("Deaths", r"Deaths", 4, 0.5, aspect=1.75))

pgm.add_node(daft.Node("Admissions", r"Admissions", 2.5, 1.5, aspect=2.5))

pgm.add_node(daft.Node("Smokers", r"Smokers", 4, 2.5, aspect=2.25))

pgm.add_node(daft.Node("Sex", r"Sex", 7, 3.5, aspect=2))

pgm.add_node(daft.Node("Age", r"Age", 5.5, 3.5, aspect=2))

pgm.add_node(daft.Node("Drug Cost", r"Drug Cost", 5.75, 4.5, aspect=2.25))

pgm.add_node(daft.Node("In Treatment", r"In Treatment", 4, 3.5, aspect=2.75))

pgm.add_node(daft.Node("Year", r"Year", 1, 4.5, aspect=2.25)) # There isn't a great place for Year to go

pgm.add_node(daft.Node("Cost", r"Cost", 2.25, 3.5, aspect=2.25))

pgm.add_node(daft.Node("Income", r"Income", 0.75, 3.5, aspect=2.25))

pgm.add_node(daft.Node("Spent", r"Spend", 1.5, 2.5, aspect=2.25))



pgm.add_edge("Year", "In Treatment")

pgm.add_edge("Year", "Admissions")

pgm.add_edge("Year", "Deaths")

pgm.add_edge("Year", "Income")

pgm.add_edge("Year", "Cost")

pgm.add_edge("Year", "Drug Cost")

pgm.add_edge("Year", "Smokers")

pgm.add_edge("Age", "Smokers")

pgm.add_edge("Sex", "Smokers")

pgm.add_edge("Cost", "Smokers")

pgm.add_edge("Cost", "Spent")

pgm.add_edge("Drug Cost", "Smokers")

pgm.add_edge("Drug Cost", "In Treatment")

pgm.add_edge("In Treatment", "Smokers")

pgm.add_edge("Income", "Spent")

pgm.add_edge("Income", "Smokers")

pgm.add_edge("Smokers", "Admissions")

pgm.add_edge("Admissions", "Deaths")

pgm.add_edge("Smokers", "Deaths")



pgm.render()
model = sm.OLS(endog=flatdata["Smokers"], exog=sm.add_constant(flatdata[['Drug Cost', 'Year']])).fit()

model.summary()
model = sm.OLS(endog=flatdata["Smokers"], exog=sm.add_constant(flatdata[['Year']])).fit()

model.summary()
model = sm.OLS(endog=flatdata["Deaths"], exog=sm.add_constant(flatdata[['Smokers', 'Year']])).fit()

model.summary()