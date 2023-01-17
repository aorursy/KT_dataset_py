import pandas as pd
import numpy as np 
import statistics as s
a="../input/comptab_2018-01-29 16_00_comma_separated.csv"
df=pd.read_csv(a)
print(df)
df.describe()
df.plot.scatter('Importer reported quantity','Exporter reported quantity')
df.box.plot('Importer reported quantity','Exporter reported quantity')
df.plot.bar('Importer reported quantity','Exporter reported quantity')
b=df["Year"]
print("median:-",s.median(b))
print("mean:-",s.mean(b))
print("mode:-",s.mode(b))

