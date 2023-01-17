
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

DOWNLOAD_ROOT = "../input/kc_house_data.csv"

def create_dataframe(data_path):
    df = pd.read_csv(data_path)
    return df
housing = create_dataframe(DOWNLOAD_ROOT)
housing.head()
housing.info()

print(housing.isnull().any())

with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(housing[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[])
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=10, figsize=(20,15))
plt.show()
import seaborn as sns
corr = housing.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
housing.plot(kind="scatter", x="long", y="lat",alpha=0.1)

