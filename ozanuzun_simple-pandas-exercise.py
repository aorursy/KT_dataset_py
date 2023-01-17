import pandas as pd

import numpy as np
poke_df=pd.read_csv("../input/pokemon/Pokemon.csv").drop("#",axis=1)
poke_df.head()
poke_df[poke_df["Type 1"]=="Fire"].head()
poke_df[(poke_df["Type 1"]=="Fire") & (poke_df["Type 2"]=="Flying")]
poke_df[(poke_df["Type 1"]=="Fire") & (poke_df["Type 2"].notnull())]
poke_df[poke_df["Defense"]<poke_df["Attack"]]
poke_df[poke_df["Attack"]<poke_df["Defense"]]
poke_df[poke_df["Attack"]==poke_df["Defense"]]
poke_df2=poke_df.set_index(["Type 1","Type 2"])
poke_df2.head(15)
poke_df2.loc["Fire","Flying"]
poke_df3=poke_df.copy()

poke_df3.insert(3,"Type 3","NaN")



for i in poke_df3.index:

    if poke_df3["Attack"][i]<poke_df3["Defense"][i]:

        poke_df3["Type 3"][i]="Defense"

    elif poke_df3["Defense"][i]<poke_df3["Attack"][i]:

        poke_df3["Type 3"][i]="Attack"

poke_df3=poke_df3.set_index(["Type 1","Type 2","Type 3"])

poke_df3.head(15)
poke_df3.loc["Fire","Flying","Attack"]
poke_df3.loc["Rock","Fairy","Defense"]
poke_df3.loc["Rock","Fairy","Attack"]