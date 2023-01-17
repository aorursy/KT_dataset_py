"""Importing libraries """



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from scipy import stats

sns.set(style="ticks", color_codes=True)



from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import make_scorer 

from sklearn.model_selection import RandomizedSearchCV 



import eli5

from eli5.sklearn import PermutationImportance



import warnings 

warnings.simplefilter('ignore')

"""Loading the data """



df = pd.read_csv("../input/ehpm-2017/ehpm-2017.csv", na_values=' ')
df.head()
df.shape
selected_variables = [ 'r104', 'r106', 'r107', 'r215a', 'r32601a', 

                       'r32603a', 'r32605a',  'r32606a',  'r32609a', 

                       'r32610a',  'r32611a', 'r32612a',  'r32616a',  

                       'r32617a',  'r32619a', 'r442',  'r7041a', 

                       'r7041b',  'r706e',  'r424'

                     ]



df = df[selected_variables]
df["r104"] = pd.Categorical(df.r104)

df["r107"] = pd.Categorical(df.r107)

df["r215a"] = pd.Categorical(df.r215a)

df["r32601a"] = pd.Categorical(df.r32601a)

df["r32603a"] = pd.Categorical(df.r32603a)

df["r32605a"] = pd.Categorical(df.r32605a)

df["r32606a"] = pd.Categorical(df.r32606a)

df["r32609a"] = pd.Categorical(df.r32609a)

df["r32610a"] = pd.Categorical(df.r32610a)

df["r32611a"] = pd.Categorical(df.r32611a)

df["r32612a"] = pd.Categorical(df.r32612a)

df["r32616a"] = pd.Categorical(df.r32616a)

df["r32617a"] = pd.Categorical(df.r32617a)

df["r32619a"] = pd.Categorical(df.r32619a)

df["r442"] = pd.Categorical(df.r442)
''' Percentage of missing values in the variable salary '''



df["r424"].isnull().sum() / df.shape[0]
df = df.dropna(subset=['r424'])
df = df[ (df["r106"] >=18)  & (df["r424"] > 0) ]
df.isnull().sum() / df.shape[0]
df = df.drop(['r442'], axis=1)
vars_to_combine = ['r7041a', 'r7041b', 'r706e']

df['remittance'] = df.loc[:, vars_to_combine].sum(axis=1)



#if we sum three na the value must be na but with the 

# code above the result is 0 that is why we need to run the

# following code.



df.loc[df[vars_to_combine].isnull().sum(axis=1) == 3, ['remittance']] = np.nan
df["remittance"].isnull().sum()/df.shape[0]
df = df.drop(['r7041a', 'r7041b', 'r706e'], axis=1)
df_temp = df.select_dtypes(include=['category'])
for col in df_temp.columns:     

    print(df_temp[col].value_counts())

    
del df_temp

df = df.drop(['r32616a'], axis=1)

condition = (df['r215a'] == 2) | (df['r215a'] == 3) | (df['r215a'] == 4)  | (df['r215a'] == 5) | (df['r215a'] == 8) 

df = df.loc[condition,  :]

df['r215a'].cat.remove_unused_categories(inplace=True)
g = sns.FacetGrid(df, col = "r104", row="r215a")

g = g.map(sns.distplot, "r424", hist=True, rug=False).set(xscale='log')
selected_variables = ["r424", "remittance", "r106"]



df_temp = df[selected_variables]



g = sns.pairplot(df_temp).set(xscale='log', yscale='log')
df_temp.corr(method='pearson')
del df_temp
g = sns.FacetGrid(df, col="r104")

g.map(plt.hist, "r424", density=True)
male = df.loc[df["r104"] == 1, "r424"].to_numpy()

female = df.loc[df["r104"] == 2, "r424"].to_numpy()

stats.ks_2samp(male, female)
g = sns.FacetGrid(df, col="r215a")

g.map(plt.hist, "r424",  bins=5, density=True)
summaries_by_education = df.groupby('r215a').describe()

summaries_by_education["r424"]
first_group = df.loc[df["r215a"] == 2, "r424"].to_numpy()

second_group = df.loc[df["r215a"] == 3, "r424"].to_numpy()

third_group = df.loc[df["r215a"] == 4, "r424"].to_numpy()

fourth_group = df.loc[df["r215a"] == 5, "r424"].to_numpy()

fifth_group = df.loc[df["r215a"] == 8, "r424"].to_numpy()



stats.kruskal(first_group, second_group, third_group, fourth_group, fifth_group)
X = df[['r104', 'r106', 'r107', 'r215a', 'r32601a', 'r32603a', 'r32605a',

       'r32606a', 'r32609a', 'r32610a', 'r32611a', 'r32612a', 'r32617a',

       'r32619a', 'remittance']]

y = df['r424']



# In order to deal with missing value we transform remittance into a binary indicator: 1 if the person receives 

# remittance, 0 otherwise. 



X['remittance'] = X['remittance'].map(lambda x : 0 if np.isnan(x)  else 1)

X['remittance'] = pd.Categorical(X.remittance)
def create_dummy_df(df, cat_cols, dummy_na):

    '''

    INPUT:

    df - pandas dataframe with categorical variables you want to dummy

    cat_cols - list of strings that are associated with names of the categorical columns

    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

    

    OUTPUT:

    df - a new dataframe that has the following characteristics:

            1. contains all columns that were not specified as categorical

            2. removes all the original columns in cat_cols

            3. dummy columns for each of the categorical columns in cat_cols

            4. if dummy_na is True - it also contains dummy columns for the NaN values

            5. Use a prefix of the column name with an underscore (_) for separating 

    '''

    for col in  cat_cols:

        try:

            # for each cat add dummy var, drop original column

            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=dummy_na)], axis=1)

        except:

            continue

    return df
cat_cols = X.select_dtypes(include=['category']).columns.to_list()



X = create_dummy_df(X, cat_cols, False)
#Split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .10, random_state=42) 
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

rf = RandomForestRegressor(random_state=1234)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=10, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train);
rf_random.best_params_
model = RandomForestRegressor(random_state=1234, n_estimators  = 600,

                              min_samples_split = 10,

                              min_samples_leaf = 4,

                              max_features = 'sqrt',

                              max_depth = 40,

                              bootstrap = True)

model.fit(X, y);

perm = PermutationImportance(model, random_state=0).fit(X, y)



colnames = X.columns



eli5.show_weights(perm, feature_names = colnames.tolist())