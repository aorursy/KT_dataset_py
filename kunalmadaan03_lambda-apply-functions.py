import pandas as pd

import numpy as np
ms = pd.read_csv("../input/alc-consumption-and-higher-education/student-mat.csv")

ps =pd.read_csv("../input/alc-consumption-and-higher-education/student-por.csv")

df = pd.concat([ms,ps]).reset_index(drop=True)
df.info()
df = df.loc[:,:"guardian"]

df.head()
def Capital():

    return lambda x: x.capitalize()
df.Mjob = df.Mjob.apply(Capital())

df.Fjob = df.Fjob.apply(Capital())

df.head()
df.tail(1)
# Already done in Question 6
def majority(x):

    if x == 1:

        return True

    else:

        return False

df["legal_drinker"] = [majority(1) if x>=18 else majority(0) for x in df["age"]]
df.head()
temp = df.apply(lambda x: x*10 if x.name in ['Medu', 'Fedu'] else x)
temp