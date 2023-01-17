import pandas as pd
deep_sea_coral = pd.read_csv("../input/deep_sea_corals.csv", header=0, skiprows=[1])
coral_by_frequency = (

    deep_sea_coral

    .groupby("ScientificName")

    .size()

    .sort_values(ascending=False)

)



coral_by_frequency.head(10)