import pandas as pd
%matplotlib inline
data = pd.read_csv("../input/bus-breakdown-and-delays.csv")
data.head()
notify_parents = data.Has_Contractor_Notified_Parents.value_counts().plot.bar()
borough_affected = data.Boro.value_counts().plot.bar()