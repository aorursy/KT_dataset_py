import pandas as pd

import seaborn as sb
gifts = pd.read_csv("../input/gifts.csv")

gifts['toy'] = gifts['GiftId'].apply(lambda x: x.split("_")[0])

gifts



toys = gifts.groupby(['toy']).agg(['count'])

toys
gifts = pd.read_csv("../input/gifts.csv")

gifts['toy'] = gifts['GiftId'].apply(lambda x: x.split("_")[0])

gifts



toys = gifts.groupby(['toy']).agg(['count'])

toys
q = toys.sum().values[0]

idxs = [w % 1000 for w in range(q)]

idxs