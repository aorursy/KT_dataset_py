import numpy as np

import pandas as pd



from subprocess import check_output

check_output(["ls", "../input"]).decode("utf8")



%matplotlib inline

pd.set_option('display.max_rows', 20)



df = pd.read_csv("../input/winemag-data-130k-v2.csv")

#df["country"].value_counts().head(10)
#ARVUSTUSTE HULK RIIKIDE KAUPA#

df["country"].value_counts().head(10)
# VEINI HINNA JA KVALITEEDI SUHE

df.plot.scatter("points", "price", alpha = 0.2);
#TOP 20 KÕRGEIMA VEINIKVALITEEDIGA RIIKI"

df2=pd.DataFrame(df.groupby("country")["points"].mean())

df2.sort_values("points", ascending = 0)



df2["max"]=pd.DataFrame(df.groupby("country")["points"].max())

df2["min"]=pd.DataFrame(df.groupby("country")["points"].min())

df2["review count"]=df["country"].value_counts()

df2.sort_values("points", ascending = 0).head(20)
# ARVUSTUSTE KVALITEEDIPUNKTIDE KOGUSED

df.points.plot.hist(bins = 20, grid = 0, rwidth = 0.8); 
# TOP 10 KÕIGE KALLIMAT VEINI

(df[["country", "points", "title", "price", "province", "winery", "description"]]

 .sort_values("price", ascending = 0)).head(10)

from IPython.display import Image

from IPython.core.display import HTML 

Image(url = "http://avis-vin.lefigaro.fr/var/img/480/119905-650x330-8-chateaux-de-l-appellation-medoc-etaient-presents-dont-le-chateau-les-ormes-sorbet.jpg")