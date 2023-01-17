import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.graphics.mosaicplot import mosaic



pokemons = pd.read_csv("../input/Pokemon.csv")

print(pokemons)
pokemons["Type 1"].value_counts().plot(

    kind="bar",

    title="Pokemons by type"

)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))



_ = mosaic(pokemons, ["Type 1", "Type 2"], ax=ax)