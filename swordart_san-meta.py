%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def parse_date(date):
    tem = date.split()
    datetime, time = tem[0], tem[1]
    tem = datetime.split('-')
    year, month, day = tem[0], tem[1], tem[2]
    tem = time.split(':')
    hour = tem[0]
    return hour
data = pd.read_csv('../input/train.csv', header=0)

data['Chour'] = data['Dates'].apply(parse_date)
data = pd.crosstab(data.Category, data.Chour)

heat_map = sns.heatmap(data, square=True, linewidths=1, label='tiny')
plt.show()

