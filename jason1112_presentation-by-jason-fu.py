import pandas as pd
crime = pd.read_csv("../input/reported.csv", index_col=0)
crime.head(65)
crime['drunk.driving'].head(65).plot.bar()
crime['drunk.driving'].sort_index().plot.line()