import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
my_filepath = "../input/avocado-prices/avocado.csv"
df = pd.read_csv(my_filepath)
df.head()
sales_SanFrancisco = df[df.region == 'SanFrancisco']
print('completed')
years = sales_SanFrancisco.groupby('year').agg(
    **{'Всего продано': pd.NamedAgg(column='Total Volume', aggfunc='sum'),
    }
)
print('completed')
years.head()
plt.title("Продажи авокадо в Сан-Франциско")
years_to_compare = years.loc[[2015, 2016, 2017, 2018]]
sns.barplot(x=years_to_compare.index, y=years_to_compare['Всего продано'])
plt.show()