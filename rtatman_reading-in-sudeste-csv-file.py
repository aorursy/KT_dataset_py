# libraries we'll use
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# read in the first 10,000 rows
data = pd.read_csv("../input/sudeste.csv", nrows = 10000)
# look at the first few lines
data.head()