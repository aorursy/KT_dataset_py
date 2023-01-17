import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
data.head()
data.info()
data.describe()
by_month = data.drop(['year', 'state', 'date'], axis = 1)
by_month.head()
by_month = by_month.groupby(by_month['month']).aggregate({'number': 'sum'})
by_month
by_month.plot.line()
by_year = data.drop(['month', 'state', 'date'], axis = 1)
by_year
by_year = by_year.groupby(by_year['year']).aggregate({'number': 'sum'})
by_year
by_year.plot.line()