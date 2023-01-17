# Load modules
import pandas as pd # Data manipulation
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns
%matplotlib inline

# Read in the data
recipes = pd.read_csv('../input/epi_r.csv')
recipes.info()

recipes.head()
cal=recipes["calories"][(recipes["calories"]>0) & (recipes["calories"]<3000)]
#sns.distplot(recipes["rating"]>0)
#plt.savefig('rating_distribution.png')
sns.distplot(cal)
plt.savefig('calories_distribution.png')
