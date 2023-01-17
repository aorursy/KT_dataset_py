import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Limiting floats output to 3 decimal points:
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 

# Adjusting the displays of the dataset (for some reason, I had to save the data in a variable before pd would allow me to change the display options. Let me know if there is a way around this.):
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

print('Dependencies installed!')
school_explorer = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
registration_testers = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')
school_explorer.head()
registration_testers.head()
print(school_explorer.shape)
school_explorer.describe()
print(registration_testers.shape)
registration_testers.describe()
school_explorer.columns.tolist()
registration_testers.columns.tolist()
total = school_explorer.isnull().sum().sort_values(ascending=False)
percent = ((school_explorer.isnull().sum() / school_explorer.isnull().count()) * 100).sort_values(ascending=False)
missing_values = pd.DataFrame({'Total ': total, 'Missing ratio ': percent})
missing_values.head(25)
total = registration_testers.isnull().sum().sort_values(ascending=False)
percent = ((registration_testers.isnull().sum() / registration_testers.isnull().count()) * 100).sort_values(ascending=False)
missing_values = pd.DataFrame({'Total ': total, 'Missing ratio ': percent})
missing_values.head(10)
total = pd.DataFrame(school_explorer['City'].value_counts().reset_index())
total.columns = ['city', 'total']

plt.figure(figsize=(20, 10))

barplot = sns.barplot(x=total['total'], y=total['city'])
barplot.set(xlabel='', ylabel='')
plt.title('Total of schools in each city:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=10)
plt.show()
# Census downloaded from: https://www.census.gov/quickfacts/fact/table/kingscountybrooklynboroughnewyork/IPE120216
brooklyn_census = pd.read_csv('../input/brooklyn-census/QuickFacts Jun-27-2018 (1).csv') 

# Dropping the second and last column since it does not have any values.
brooklyn_census = brooklyn_census.drop('Fact Note', axis=1)
brooklyn_census = brooklyn_census.drop('Value Note for Kings County (Brooklyn Borough), New York', axis=1)

# Renaming the 'Kings County (Brooklyn Borough), New York' for easier use:
brooklyn_census.columns = ['fact', 'total']
brooklyn_census.head(60)
# Saving only Brooklyn schools in a variable:
brooklyn = pd.DataFrame(school_explorer[school_explorer['City'] == 'BROOKLYN'])
print(brooklyn.shape)
brooklyn.head()







