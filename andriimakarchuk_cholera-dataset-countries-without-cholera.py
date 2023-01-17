import numpy as np

import pandas as pd
data = pd.read_csv("../input/cholera-dataset/data.csv").dropna()

data.head()
data = data[["Country", "Year", "Number of reported cases of cholera"]]

data.head()
countries = list( data["Country"].unique() )

print(countries)
data = data.drop( labels=data[ data["Year"]<2010 ].index )

data = data.drop( labels=data[ data["Number of reported cases of cholera"]==0 ].index )

wantedCountries = list( set(countries)-set(data["Country"].unique()) )



print("Wanted countries:\n"+str(wantedCountries))