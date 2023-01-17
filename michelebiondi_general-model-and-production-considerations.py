import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import pandas_profiling



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error
data_folder = '/kaggle/input/used-car-dataset-ford-and-mercedes/'

dataset_names = ['bmw', 'merc', 'hyundi', 'ford', 'vauxhall', 'vw', 'audi','skoda', 'toyota']
df = pd.DataFrame()

for dataset_name in dataset_names:

    dataset = pd.read_csv(data_folder+dataset_name + '.csv')

    if(dataset_name == 'hyundi'):

        dataset.rename(columns={"tax(£)": "tax"}, inplace=True)

    dataset['manufacturer'] = dataset_name

    df = pd.concat([df, dataset], ignore_index=True)
df.info()
df.describe()
plt.figure(figsize=(12,7))



plt.title('Price distribution')



sns.distplot(df['price'])
plt.figure(figsize=(14,11))

sns.boxplot(x="manufacturer", y="price", data=df)
costly = df[df.price > 100000]

costly.describe()
list(costly['manufacturer'].unique())
cheap = df[df.price < 1000]

cheap.describe()
list(cheap['manufacturer'].unique())
cheap[cheap['manufacturer'] == 'merc']
plt.figure(figsize=(7,6))

sns.countplot(x="manufacturer", data=df)
profile = df.profile_report(title="Pandas Profiling Report")
profile.to_widgets()
duplicates = df[df.duplicated(keep=False)] # Just for visualization

duplicates.head()
df.drop_duplicates(ignore_index=True, inplace=True)
df.describe()
categorical_cols = list(df.select_dtypes('object').columns)

categorical_cols
threshold = 10 # If less than 10 unique values we suppose it is low cardinality



low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() < threshold]



high_cardinality_cols = list(set(categorical_cols)-set(low_cardinality_cols))



print(low_cardinality_cols)

print(high_cardinality_cols)
general_df = df.drop(['model', 'manufacturer'],axis=1)

general_df
general_df_encoded = pd.get_dummies(general_df, drop_first=True)

general_df_encoded
y = general_df_encoded['price']

X = general_df_encoded.drop('price', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4242)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression().fit(X_train, y_train)
preds = linreg.predict(X_test)
baseline_rmse = mean_squared_error(y_test, preds, squared=False) # RMSE

baseline_rmse
baseline_mae = mean_absolute_error(y_test, preds)

baseline_mae
def score_result(y_test, preds):

    print("----------------")

    rmse = mean_squared_error(y_test, preds, squared=False)

    mae = mean_absolute_error(y_test, preds)

    print("RMSE: ", rmse)

    print("MAE: ", mae)

    print("\nImprovement from baseline:")

    print("RMSE Improvement:",  baseline_rmse - rmse)

    print("MAE Improvement:", baseline_mae - mae)

    print("----------------")
from sklearn.ensemble import RandomForestRegressor
rf_basic = RandomForestRegressor(random_state=4242)

rf_basic.fit(X_train, y_train)



rf_basic_preds = rf_basic.predict(X_test)
score_result(rf_basic_preds, y_test)