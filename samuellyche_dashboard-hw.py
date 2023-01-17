%config IPCompleter.greedy=True



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/daily-inmates-in-custody.csv")
data.head()
data.columns
plt.hist(

    data.loc[~data['AGE'].isna(),'AGE']

    , bins=list(range(16,96,5))

)

plt.xlabel("Age")

plt.ylabel("Count")

plt.show()
gang_affiliated = data.SRG_FLG.map(dict(N=0,Y=1))

plt.hist(gang_affiliated)

plt.xlabel("Gang Affiliated")

plt.ylabel("Count")

plt.show()
plt.hist(

    [

        data.loc[data.SRG_FLG == 'Y', 'AGE']

        , data.loc[data.SRG_FLG == 'N', 'AGE']

    ]

    , bins=list(range(16,96,5))

)

plt.legend(['Ganster', 'Non-Gangster'])

plt.xlabel("Age")

plt.ylabel("Count")

plt.show()