import seaborn as sns
import pandas as pd
data = pd.read_csv('../input/DigiDB_digimonlist.csv')
data.head()
import seaborn as sns

sns.distplot(data['Lv 50 HP'], kde = False,bins=30).set_title('My First Seaborn Plot')