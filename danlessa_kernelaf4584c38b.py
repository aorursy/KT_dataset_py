import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_feather("../input/sao_paulo-traffic_jams.feather")
data.loc[:, ['segment']] = data.segment.str.lower()

paulista_data = data.where(data.segment.str.contains('paulista')).dropna()

subset = paulista_data.loc[:, ["timestamp", "jam_size"]]
subset.groupby(subset.timestamp.dt.month).mean().plot()

plt.title("Mean traffic jam by month")

plt.ylabel("Jam size in meters")

plt.show()
subset.groupby(subset.timestamp.dt.hour).mean().plot()

plt.title("Mean traffic jam by hour")

plt.ylabel("Jam size in meters")

plt.show()
subset.jam_size.hist(log=True)

plt.title("Traffic jam log-distribution")

plt.show()