import pandas as pd

data = pd.read_csv('/kaggle/input/migration-data-worldbank-1960-2018/migration_population.csv')

data.head()
from IPython.display import IFrame

IFrame(src='https://www.dashboardom.com/migration-population', width='100%', height=700)