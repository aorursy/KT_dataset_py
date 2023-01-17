import pandas as pd

data = pd.read_csv('../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')

data.describe()
import seaborn as sns

sns.distplot(data['Open'], kde=False)

sns.distplot(data['High'], kde=False)
from scipy.stats import ttest_ind

print(ttest_ind(data['Open'], data['High'], equal_var=False))