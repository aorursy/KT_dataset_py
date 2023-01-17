# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from scipy import stats

import warnings



from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score





import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno



pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
fish_data = pd.read_csv("../input/fish-market/Fish.csv")

fish_data.sort_values("Species",ascending=False,inplace=True)

fish_data.reset_index(drop=True,inplace=True)

fish_data.head()
fish_data.info()
fish_data.describe()
fish_data.describe(include=['O'])
weight_by_species = fish_data[["Weight","Species"]].groupby(by="Species", as_index=False).mean().sort_values("Weight", ascending=False)

weight_by_species
by_species = fish_data.drop("Weight", axis=1).groupby(by="Species", as_index=False).mean().sort_values("Length1", ascending=False)

by_species
msno.matrix(fish_data, figsize=(12,5));
fig, axs = plt.subplots()

fig.set_size_inches(6,5)

sns.boxplot(data=fish_data, y="Weight")

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(8,6)

sns.boxplot(data=fish_data, x="Species", y="Weight")

plt.show()
outliers = np.abs(fish_data["Weight"]-fish_data["Weight"].mean())<=(3*fish_data["Weight"].std())

outliers.value_counts()
outlier_df = outliers.to_frame()

fish_data.iloc[outlier_df[outlier_df["Weight"]==False].index,:]
fish_data_wout_outliers = fish_data[outliers]

print("Shape of before outliers: {}".format(fish_data.shape))

print("Shape of after outliers: {}".format(fish_data_wout_outliers.shape))
fig, axs = plt.subplots()

fig.set_size_inches(6,5)

sns.boxplot(data=fish_data_wout_outliers, y="Weight")

plt.show()
corrMat = fish_data_wout_outliers[fish_data_wout_outliers.drop("Species",axis=1).columns].corr()

plt.figure(figsize=(10,10))

sns.heatmap(corrMat, annot=True)

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

fig.set_size_inches(14,8)

sns.regplot(x="Length3", y="Weight", data = fish_data_wout_outliers, ax=ax1);

sns.regplot(x="Height", y="Weight", data = fish_data_wout_outliers, ax=ax2, color='r');

sns.regplot(x="Width", y="Weight", data = fish_data_wout_outliers, ax=ax3, color='y');
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

fig.set_size_inches(14,8)

sns.regplot(x="Length3", y="Weight", data = fish_data_wout_outliers, ax=ax1,order=3);

sns.regplot(x="Height", y="Weight", data = fish_data_wout_outliers, ax=ax2, color='r');

sns.regplot(x="Width", y="Weight", data = fish_data_wout_outliers, ax=ax3, color='y',order=2);
df_shuffled = shuffle(fish_data_wout_outliers)

df_shuffled
categ = df_shuffled["Species"]

df_dummy = pd.get_dummies(categ, drop_first=True)

df_dummy.reset_index(drop=True,inplace=True)

df_dummy
target = df_shuffled["Weight"]

target.reset_index(drop=True, inplace=True)

target
numerical = df_shuffled.drop(["Species","Weight"], axis=1)

scaler = StandardScaler()

num_norm = scaler.fit_transform(numerical)

num_norm

df_num = pd.DataFrame(num_norm,columns=numerical.columns)

df_num.describe()
merged_df = pd.concat([df_num,df_dummy,target],axis=1)

merged_df
def get_reg_model_metrics(model,actual,predicted):

    

    reg_metrics = {

                        "MSE": mean_squared_error(actual,predicted),

                        "RMSE": pow(mean_squared_error(actual,predicted),0.5),

                        "R\u00b2 Score" : r2_score(actual,predicted)

                  }

    

    df_reg_metrics = pd.DataFrame.from_dict(reg_metrics, orient='index')

    df_reg_metrics.columns = [model]

    

    return df_reg_metrics
def fit_model(model, data):

    metrics = pd.DataFrame()

    X = data.drop(["Weight"],axis=1)

    y = data["Weight"]

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    

    i = 0

    

    for name, reg in model.items():

        print("Fitting model: " + name)

        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)

        

        if i == 0:

            metrics = get_reg_model_metrics(name, y_test, y_pred)

            print(metrics)

            

        elif i != 0:

            new_metric = get_reg_model_metrics(name, y_test, y_pred)

            print(new_metric)

            metrics = pd.concat([metrics, new_metric],axis=1)



        i= i+1



        print("====================")

        print("\n")

        

    return metrics.sort_values(by="R\u00b2 Score", axis=1)
model1 = {

    "Linear Reg": LinearRegression(),

    "Ridge": Ridge(),

    "Lasso": Lasso(),

    "Random Forest": RandomForestRegressor(n_estimators=100),

    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100),

    "AdaBoost Regressor": AdaBoostRegressor(DecisionTreeRegressor(max_depth=2),random_state=42)

}
fit_model(model1, merged_df)