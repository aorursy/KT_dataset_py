import pandas as pd

import numpy as np



from scipy import stats



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold



from xgboost import XGBRegressor, plot_importance



from matplotlib import pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



df = pd.read_csv('../input/ted-talks/ted_main.csv')

df.info()
# There are only 6 rows with missing data for speaker_occupation and I decide to drop those as it will not affect the dataset in a measurable way.

df.dropna(axis=0, subset=['speaker_occupation'], inplace=True)

df.info()
df.describe()
sns.pairplot(df[['comments', 'duration', 'languages', 'num_speaker']],  size=2)

plt.show()
fig, (ax1, ax2) = plt.subplots(1,2)

sns.boxplot(y= df.views, ax=ax1)

sns.boxplot(y=df.comments, ax=ax2)

plt.subplots_adjust(wspace=0.50)

plt.show()
# Compute z-score for each value in a colun relative to the column mean and std and use the resulting value to filter outliers from each specified column in the dataframe.



df = df[(np.abs(stats.zscore(df[['comments', 'views']])) < 3).all(axis=1)]
df.info()
df.describe()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df['film_date'] = pd.to_datetime(df['film_date'],unit='s')

df['film_month'] = df['film_date'].apply(lambda x: month_order[int(x.month) - 1])

df['film_day'] = df['film_date'].apply(lambda x: day_order[int(x.day) % 7 - 1])
fig, ax =plt.subplots(1,2, figsize=(20,10))

sns.countplot(df['film_month'], ax=ax[0])

sns.countplot(df['film_day'], ax=ax[1])

fig.show()
def cal_pos_rating_ratio(ratings):

    counter = {'Funny':0, 'Beautiful':0, 'Ingenious':0, 'Courageous':0, 'Longwinded':0, 'Confusing':0, 'Informative':0, 'Fascinating':0, 'Unconvincing':0, 'Persuasive':0, 'Jaw-dropping':0, 'OK':0, 'Obnoxious':0, 'Inspiring':0}

    neg_descriptors = {"Confusing", "Unconvincing", "Longwinded", "Obnoxious", "OK"}

    for rating_list in ratings:

        counter[rating_list['name']] += rating_list['count']

    neg_desc_count = sum([counter[desc] for desc in neg_descriptors])

    total_desc_count = sum(list(counter.values()))

    pos_desc_count = total_desc_count - neg_desc_count

    popular_data_no_tedex_pct_positive = 100 * (pos_desc_count / total_desc_count)

    return popular_data_no_tedex_pct_positive
df['eval_ratings'] = df['ratings'].apply(lambda x: eval(x))
df['pos_rating_ratio'] = df.eval_ratings.apply(cal_pos_rating_ratio)
def rating_count(ratings):

    counter = {'Funny':0, 'Beautiful':0, 'Ingenious':0, 'Courageous':0, 'Longwinded':0, 'Confusing':0, 'Informative':0, 'Fascinating':0, 'Unconvincing':0, 'Persuasive':0, 'Jaw-dropping':0, 'OK':0, 'Obnoxious':0, 'Inspiring':0}

    for rating_list in ratings:

        counter[rating_list['name']] += rating_list['count']

    return  sum(list(counter.values()))

df['raiting_count'] = df.eval_ratings.apply(rating_count)
y = df.views

X = df.drop(['views', 'film_date', 'published_date', 'eval_ratings'], axis=1) # drop the columns that are not needed for predictions

X.info()


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

# Select categorical columns with relatively low cardinality

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 13 and 

                        X_train_full[cname].dtype == "object"] # 13 is used to fit the month encoding



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'uint64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

# One-hot encode the data

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train.info()
def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    preds_val = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds_val)

    return(mae)





maes_dtr = []

# compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 25, 50, 100, 500, 5000]:

    mae_dtr = get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid)

    maes_dtr.append(mae_dtr)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae_dtr))



def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    preds_val = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds_val)

    return(mae)





maes_rfr = []

# compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:

    mae_rf = get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid)

    maes_rfr.append(mae_rf)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae_rf))


kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb1 = XGBRegressor(random_state=0, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)],)

parameters = {

              'objective':['reg:linear'],

              'learning_rate': [.1, .05, .07], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [50, 500, 1000]}



xgb_grid = GridSearchCV(xgb1,

                        parameters,

                        cv=kf,

                        n_jobs = 5,

                        scoring='neg_mean_absolute_error',

                        verbose=True)



xgb_grid.fit(X_train,

         y_train)



print(xgb_grid.best_score_)

print(xgb_grid.best_params_)
plot_importance(xgb_grid.best_estimator_)
predictions_xgb = xgb_grid.predict(X_valid) 

# Calculate MAE

mae_xgb = mean_absolute_error(predictions_xgb, y_valid)



print("Mean Absolute Error:" , mae_xgb)
y_mae = [min(maes_dtr), min(maes_rfr), mae_xgb]

x_mae = ['Decision Tree MAE', 'Random Forest MAE', 'XGB MAE']
sns.barplot(x=x_mae, y=y_mae)

plt.ylabel("Mean Absolute Error")