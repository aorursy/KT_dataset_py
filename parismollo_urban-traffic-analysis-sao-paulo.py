import numpy as np



import pandas as pd

from pandas.plotting import scatter_matrix



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR





from sklearn.model_selection import cross_val_score



from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
HOURS = {7:00, 7:30, 8:00, 8:30, 9:00, 9:30, 10:00, 10:30, 11:00, 11:30, 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:00, 15:30, 16:00, 16:30, 17:00, 17:30, 18:00, 18:30, 19:00, 19:30, 20:00} # reference from dataset folder

DATA_PATH = "../input/spurbantraffic/urban_traffic_sp.csv"

TARGET_VAR = "Slowness in traffic (%)"

MONDAY  = 26

TUESDAY = 53

WEDNESDAY = 80

THURSDAY = 107

FRIDAY = 134

DAYS_TO_CODE = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}
def load_data(path:str=DATA_PATH, sep:str=";"):

    try:

        df = pd.read_csv(path, sep=sep)

        print("Data loaded with success")

        return df

    except FileNotFoundError:

        print("Check your data directory! Nothing there yet...")

        return False
df = load_data()

df.head()
df.isnull().sum()
print(f"Dataframe shape: {df.shape}\n")

df.info()
df.describe()
df.hist(bins=50, figsize=(20, 15))

plt.show()
def transform_target(df, target_var=TARGET_VAR, to=float):

    df[target_var] = df[target_var].str.replace(',', '.').astype(to)

transform_target(df)
# New attribute

def transform_days(df, create_column=False, to_numerical=False):

    #check is day column exists if not create

    #if numerical transformation, go from day to number

    #else go from number to day names

    if create_column:

        df['Day'] = '0'



    position=-1

    if to_numerical is False:

        for idx in df.index:

            if idx <= MONDAY:

                df.iloc[idx, position] = 'Monday'

            elif idx <= TUESDAY:

                df.iloc[idx, position] = 'Tuesday'

            elif idx <= WEDNESDAY:

                df.iloc[idx, position] = 'Wednesday'

            elif idx <= THURSDAY:

                df.iloc[idx, position] = 'Thursday'

            elif idx <= FRIDAY:

                df.iloc[idx, position] = 'Friday'

    else:

        df_values = df["Day"].unique()

        for key, value in DAYS_TO_CODE.items():

            assert key in df_values, "First transform your data into weekday by setting to_numerical=False, then apply the numerical transformation"

            df.loc[(df.Day == key), 'Day'] = value

        df['Day'] = df['Day'].astype(int)

        

transform_days(df, create_column=True)
# Create code to hour dict

def set_hours_dict(df, hours:dict =HOURS)-> dict:

    hours_arr = []



    for hour, minute in hours.items():

      s1 = str(hour) + ':' + '00'

      s2 = str(hour) + ':' + str(minute)

      if hour != 20:

        hours_arr.append(s1)

        hours_arr.append(s2)

      else:

        hours_arr.append(s1)



    code_to_hour = {}

    for code, hour in zip(df['Hour (Coded)'], hours_arr):

      code_to_hour[code] = hour



    return code_to_hour



code_to_hour = set_hours_dict(df)



def code_hour(code):

  return code_to_hour[code]
transform_days(df, to_numerical=True)

plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), cmap='Blues', annot=True)

plt.show()
corr_matrix = df.corr()

corr_matrix['Slowness in traffic (%)'].sort_values(ascending=False)
attributes = ["Slowness in traffic (%)", "Hour (Coded)"]

scatter_matrix(df[attributes], figsize=(12, 8))
def slowness_over_time(df, coded_hours=False):

    fig = plt.figure(figsize=(20, 12))

    ax = fig.add_axes([0, 0, 1, 1])



    colors = {'Monday': 'r', 'Tuesday': 'b', 'Wednesday': 'g', 'Thursday': 'yellow', 'Friday':'black'}

    transform_days(df)

    for e in df['Day'].unique():

        subset = df[df['Day'] == e]

        ax.plot(subset['Hour (Coded)'], subset['Slowness in traffic (%)'],color=colors[e])



    ax.set_title('Slowness in traffic VS. Hour of the day', fontsize=25, pad=15)

    ax.set_xlabel('Hour of the day', fontsize=15)

    ax.set_ylabel('Slowness in traffic (%)', fontsize=15)

    

    if coded_hours is False:

        ax.set_xticks(range(1, 28))

        ax.set_xticklabels(map(code_hour, subset['Hour (Coded)'].unique()))



    ax.legend(colors, fontsize=20)



    plt.show()

slowness_over_time(df)

transform_days(df, to_numerical=True)
num_cols = df.nunique()[df.nunique() > 2].keys() # Discriminate non-categorical data

num_cols = num_cols.drop('Day')



l = num_cols.values

number_of_columns=len(num_cols.values)

number_of_rows = len(l)-1/number_of_columns

plt.figure(figsize=(number_of_columns,5*number_of_rows))



for i in range(0,len(l)):

    plt.subplot(number_of_rows + 1,number_of_columns,i+1)

    sns.set_style('whitegrid')

    sns.boxplot(df[l[i]],color='green',orient='v')

    plt.tight_layout()
df.groupby('Day')['Slowness in traffic (%)'].mean()
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Hour (Coded)"])



# Verifying stratified distribution

print("Train set class proportions:\n")

print(train_set["Hour (Coded)"].value_counts() / len(train_set))

print("\nFull set:")

print(df["Hour (Coded)"].value_counts() / len(df))
X_train = train_set.drop("Slowness in traffic (%)", axis=1)

y_train = train_set["Slowness in traffic (%)"].copy()
x_num_cols = X_train.nunique()[X_train.nunique() > 2].keys()

x_num_cols = x_num_cols.drop('Day')

numerical_data = list(x_num_cols)
num_pipeline = Pipeline([('std_scaler', StandardScaler())])

full_pipeline = ColumnTransformer([("num", num_pipeline, numerical_data)], remainder='passthrough')
X_train_prepared = full_pipeline.fit_transform(X_train)
tree_reg = DecisionTreeRegressor(random_state=42)

scores = cross_val_score(tree_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
svm_reg = SVR(kernel="linear")

svr_scores = cross_val_score(svm_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)

svr_rmse_scores = np.sqrt(-svr_scores)
lin_reg = LinearRegression()

lin_scores = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_scores = cross_val_score(forest_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)
data = {

    "Model":["Linear Reg", "Decision Tree", "SVR", "Random Forest"],

    "Mean Score": [lin_rmse_scores.mean(), tree_rmse_scores.mean(), svr_rmse_scores.mean(), forest_rmse_scores.mean()],

    "Standard Deviation": [lin_rmse_scores.std(), tree_rmse_scores.std(), svr_rmse_scores.std(), forest_rmse_scores.std()]

}

scores_df = pd.DataFrame(data)

scores_df
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8, 10, 12, 14, 16, 18]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [10, 12, 14, 16, 18]},

]



forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(X_train_prepared, y_train)
best_grid_model = grid_search.best_estimator_

print("Best model paramateres:", grid_search.best_params_)
feature_importances = grid_search.best_estimator_.feature_importances_

sorted(zip(feature_importances, list(X_train)), reverse=True)
final_model = grid_search.best_estimator_



X_test = test_set.drop("Slowness in traffic (%)", axis=1)

y_test = test_set["Slowness in traffic (%)"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse
# Here are a few examples...

predictions = final_predictions[:10]

actual_results = y_test[:10]



for p, a in zip(predictions, actual_results):

    print("Predicted: {:.2f} - Expected: {}".format(p, a))