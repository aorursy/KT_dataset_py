import pandas as pd # to clean data
data = pd.read_csv('../input/DigiDB_digimonlist.csv')

print(data.describe())
import seaborn as sns # to plot histogram

sns.distplot(data['Lv 50 HP'], kde=True).set_title('My first seaborn plot')