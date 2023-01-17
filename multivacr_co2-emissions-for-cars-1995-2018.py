from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import utils, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('display.max_columns', 50)
dataframes = []
filename = '../input/Fuel Consumption Ratings'
indexes = ['MODEL YEAR', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS','TRANSMISSION','FUEL TYPE','FUEL CONSUMPTION CITY(L/100 km)','HWY(L/100 km)','COMB(L/100 km)','COMB(mpg)','CO2 EMISSIONS(g/km)']

for i in range(20):
    to_add = pd.read_csv(filename+str(i)+'.csv', names=indexes)
    dataframes.append(to_add)
df = pd.concat(dataframes)
df.T
print(df.shape)
#checking if there is NaN values
df.isna().sum()
#checking for innecesary data; i.e. just 1 value for feature.
df.nunique()
df.drop_duplicates(keep='first', inplace=True)
df.drop(['MODEL YEAR'], axis=1, inplace=True)
df.T
num_vars = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
num_vars
cat_vars = [c for c in df if not c in num_vars]
cat_vars
cat_dict = {}

for n, col in df.items():
    if n in cat_vars:
        df[n] = df[n].astype('category')
        cat_dict[n] = {i+1:e for i,e in enumerate(df[n].cat.categories)}
cat_dict
for n,col in df.items():
    if n in cat_vars:
        df[n] = df[n].cat.codes + 1
df.T
df.dtypes
#splitting independent variables from the label
x = df.drop(['CO2 EMISSIONS(g/km)'], axis=1)
y = df['CO2 EMISSIONS(g/km)']

#splitting the data for testing and training:
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1507)
m = RandomForestRegressor(1000, n_jobs=-1, oob_score=True)
m.fit(x_train, y_train)
def score():
    print(f'Scores:')
    print(f'Train      = {m.score(x_train, y_train):.4}')
    print(f'Validation = {m.score(x_val, y_val):.4}')
    if hasattr(m, 'oob_score_'): print(f'OOB        = {m.oob_score_:.4}')
score()
imp = pd.DataFrame({'cols':x_train.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
imp.style.bar(color='lightblue')
