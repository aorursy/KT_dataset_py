# The script lists the products in US with most sugar content

import numpy as np
import pandas as pd
import matplotlib.pyplot as matplt

# Any results you write to the current directory are saved as output.

wff = pd.read_csv('../input/FoodFacts.csv')
wff.countries = wff.countries.str.lower()
#print( wff.shape )

df1 = wff[wff.countries == 'united states']
df2 = wff[wff.countries == 'en:us']
df3 = wff[wff.countries == 'us']
us_wff = pd.concat([df1, df2, df3])
#print( us_wff.shape )

us_wff = us_wff[['product_name','sugars_100g']]
#print( us_wff[:5] )

us_wff = us_wff.sort_values(by=['sugars_100g'], ascending=[False])
print( us_wff[:25] )

