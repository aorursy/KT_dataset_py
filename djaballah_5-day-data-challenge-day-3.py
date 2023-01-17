import pandas as pd
import scipy.stats 
import matplotlib.pyplot as plt
cereal_df = pd.read_csv('../input/cereal.csv')
cereal_df.head()
cereal_df.describe()
hot = cereal_df[cereal_df['type'] == 'H']
cold = cereal_df= cereal_df[cereal_df['type'] == 'C']
hot.describe()
cold.describe()
plt.hist(hot['sugars'])
plt.title('Historam of the sugar quantity in hot cereals')
plt.hist(cold['sugars'])
plt.title('Historam of the sugar quantity in cold cereals')
scipy.stats.ttest_ind(hot['sugars'], cold['sugars'], equal_var= False)
