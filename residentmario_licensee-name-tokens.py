import pandas as pd

licensees = pd.read_csv("../input/federal-firearm-licensees.csv", index_col=0)[1:]

licensees.head(3)
import seaborn as sns
import itertools



name_tokens = list(

    itertools.chain(

        *licensees['License Name'].str.split(" ").map(lambda n: [ni.strip() for ni in n]).values.tolist()

    )

)
pd.Series(name_tokens).value_counts().head(20).plot.bar(figsize=(12, 6), fontsize=20)

sns.despine()
bus_tokens = list(

    itertools.chain(

        *licensees['Business Name'].str.split(" ").dropna().map(lambda n: 

                                                       [ni.strip() for ni in n]

                                                      ).values.tolist()

    )

)

pd.Series(bus_tokens).value_counts().head(20).plot.bar(figsize=(12, 6), fontsize=20)

sns.despine()