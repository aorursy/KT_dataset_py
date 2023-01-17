import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_validate



from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor

from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold, GroupShuffleSplit

from sklearn.multioutput import MultiOutputRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.svm import SVR, LinearSVR



from xgboost import XGBRegressor

from tqdm import tqdm
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_df = pd.read_csv("/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv")
full_profiles = data_df['profile_id'].value_counts().index.values

test_set_profiles = [31, 32, 41, 42, 51, 52, 61, 62, 71, 73]

train_set_profiles = [id for id in full_profiles if id not in test_set_profiles]
train_df = data_df.loc[data_df['profile_id'].isin(train_set_profiles)].copy()

test_df = data_df.loc[data_df['profile_id'].isin(test_set_profiles)].copy()



len(train_df['profile_id'].value_counts()), len(test_df['profile_id'].value_counts())
train_df.shape, test_df.shape
train_df.head(3)
test_df.head(3)
test_profiles = test_df.loc[:, 'profile_id'].copy()

test_labels = test_df.loc[:, ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']].copy()

test_df.drop(['profile_id', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding'], axis=1, inplace=True)
test_df.head(3)
# add names to the index of each df

train_df.index.name = 'timestep'

test_df.index.name = 'timestep'
train_df.loc[:, 'ambient'].plot(figsize=(16,4))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
train_df.loc[:, 'coolant'].plot(figsize=(16,4))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
train_df.describe()
train_df['profile_id'].value_counts()[:5]
EXAMPLE_ID = 10



example_profile = train_df.loc[train_df['profile_id'] == EXAMPLE_ID].copy()

example_profile['ambient'].plot(figsize=(16,7))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
# choose a range of spans to experiment with

spans = [600, 1200, 2400, 4800]



for span in spans:

    new_col = f"ambient_exp_{span}"

    example_exp_ma = example_profile.ewm(span=span, adjust=False).mean()

    example_profile[new_col] = example_exp_ma['ambient'].copy()

    

ambient_cols = [x for x in example_profile.columns.values if x.startswith('ambient')]
example_profile[ambient_cols].plot(figsize=(16,7))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
# choose a range of spans to experiment with

spans = [600, 1200, 2400]



chosen_col = 'coolant'



for span in spans:

    new_col = f"{chosen_col}_exp_{span}"

    example_exp_ma = example_profile.ewm(span=span, adjust=False).mean()

    example_profile[new_col] = example_exp_ma[chosen_col].copy()

    

chosen_cols = [x for x in example_profile.columns.values if x.startswith(chosen_col)]
example_profile[chosen_cols].plot(figsize=(16,8))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Outflow Coolant Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
output_vars = ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']



example_profile[output_vars].plot(figsize=(16,6))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
exp_ma = example_profile[output_vars].ewm(span=500).mean()
exp_ma.plot(figsize=(16,6))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Temperature (Exp Weighted Average)", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
input_cols = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q']



example_profile[input_cols].plot(figsize=(18,8))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
example_profile[output_vars].plot(figsize=(18,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()



example_profile[input_cols].plot(figsize=(18,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
train_df_correlations = train_df.corr()
plt.figure(figsize=(12,10))

sns.heatmap(train_df_correlations, annot=True)

plt.show()
training_sample = train_df.sample(100000)
plt.figure(figsize=(12,7))

sns.scatterplot(x='coolant', y='stator_yoke', data=training_sample, alpha=0.05)

plt.grid()

plt.show()
plt.figure(figsize=(12,7))

ax = sns.regplot(x="i_d", y="pm", data=training_sample, scatter_kws={'alpha':0.01})

plt.grid()

plt.show()
plt.figure(figsize=(12,7))

ax = sns.regplot(x="i_q", y="pm", data=training_sample, scatter_kws={'alpha':0.02})

plt.grid()

plt.show()
pd.plotting.scatter_matrix(training_sample[output_vars], figsize=(16, 10), alpha=0.02)

plt.show()
y = train_df[output_vars]

X = train_df.drop(columns=output_vars + ['profile_id'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
lin_reg = LinearRegression()



lr_scores = cross_val_score(lin_reg, X_train, y_train, 

                         scoring='neg_mean_squared_error', cv=10)



lr_rmse_scores = np.sqrt(-lr_scores)

print(f"Linear Regression RMSE: {lr_rmse_scores.mean():.5f} +/- {lr_rmse_scores.std():.5f}")
lin_reg.fit(X_train, y_train)

val_preds = lin_reg.predict(X_test)

print(f"Linear Regression RMSE on Validation Split: {np.sqrt(mean_squared_error(val_preds, y_test)):.4f}")
profile_4_idx = train_df[train_df['profile_id'] == 4].copy().index

profile_4_preds = lin_reg.predict(X.iloc[profile_4_idx])

profile_4_original = train_df.loc[profile_4_idx, output_vars]

profile_4_preds.shape, profile_4_original.shape



new_cols = [f"{x}_pred" for x in profile_4_original.columns]

profile_4_preds = pd.DataFrame(profile_4_preds)

profile_4_preds.columns = new_cols
profile_4_original[output_vars].plot(figsize=(16,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Output Temperature", weight="bold")

plt.title("Original Output Temperatures - Profile 4", weight='bold')

plt.tight_layout()

plt.grid()

plt.show()
profile_4_preds.plot(figsize=(16,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Output Temperature", weight="bold")

plt.title("Predicted Output Temperatures - Profile 4", weight='bold')

plt.tight_layout()

plt.grid()

plt.show()
for column in profile_4_original.columns:

    mse = mean_squared_error(profile_4_original[column], profile_4_preds[column + '_pred'])

    rmse = np.sqrt(mse)

    print(f"RMSE for {column}: {rmse:.4f}")
def norm(row, col1, col2):

    return np.linalg.norm([row[col1], row[col2]])
def find_new_features(dataframe):

    """ Calculate new electrical features for our dataset """

    extra_df = dataframe.copy()

    

    extra_df['voltage_s'] = extra_df.apply(norm, args=('u_d', 'u_q'), axis=1)

    extra_df['current_s'] = extra_df.apply(norm, args=('i_d', 'i_q'), axis=1)

    extra_df['apparent_power'] = 1.5 * extra_df['voltage_s'] * extra_df['current_s']

    extra_df['effective_power'] = ((extra_df['u_d'] * extra_df['u_q']) +

                                (extra_df['i_d'] * extra_df['i_q']))

    extra_df['speed_current'] = extra_df['current_s'] * extra_df['motor_speed']

    extra_df['speed_power'] = extra_df['current_s'] * extra_df['apparent_power']

    

    return extra_df
train_df = find_new_features(train_df)

train_df.head()
example_profile = train_df.loc[train_df['profile_id'] == EXAMPLE_ID].copy()
example_profile[output_vars].plot(figsize=(18,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()



example_profile[['voltage_s', 'current_s', 'apparent_power']].plot(figsize=(18,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
example_profile[output_vars + ['apparent_power']].plot(figsize=(18,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
example_profile['apparent_power'] = ((example_profile['apparent_power'] - 

                                     example_profile['apparent_power'].mean()) /

                                     example_profile['apparent_power'].std())



example_profile[output_vars + ['apparent_power']].plot(figsize=(18,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
example_profile[output_vars + ['effective_power']].plot(figsize=(18,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Ambient Temperature", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
plot_var = 'apparent_power'

alpha = 0.05



plt.figure(figsize=(14,9))



plt.subplot(221)

ax = sns.regplot(x=plot_var, y="pm", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()



plt.subplot(222)

ax = sns.regplot(x=plot_var, y="stator_yoke", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()



plt.subplot(223)

ax = sns.regplot(x=plot_var, y="stator_tooth", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()



plt.subplot(224)

ax = sns.regplot(x=plot_var, y="stator_winding", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()

plt.tight_layout()

plt.show()
# choose a range of spans to experiment with

spans = [600, 1200, 2400, 4800]



chosen_col = 'apparent_power'



for span in spans:

    new_col = f"{chosen_col}_exp_{span}"

    example_exp_ma = example_profile.ewm(span=span, adjust=False).mean()

    example_profile[new_col] = example_exp_ma[chosen_col].copy()

    

chosen_cols = [x for x in example_profile.columns.values if x.startswith(chosen_col)]
example_profile[chosen_cols].plot(figsize=(16,5))

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Apparent Power (Standardised)", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
cols = output_vars + chosen_cols

color_dict = {}



for column in cols:

    if column == 'apparent_power':

        color_dict[column] = 'black'

    elif column.startswith('apparent_power_exp'):

        color_dict[column] = 'tab:blue'

    else:

        color_dict[column] = 'tab:red'
example_profile[output_vars + chosen_cols].plot(figsize=(16,7), 

                                                color=[color_dict.get(x, 'black') for x in cols])

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Standardised Feature Values", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
# translate apparent power variables by 1.5

example_profile[chosen_cols[1:]] = example_profile[chosen_cols[1:]] - 1.5
example_profile[output_vars + chosen_cols[1:]].plot(figsize=(16,7), 

                                                color=[color_dict.get(x, '#333333') for x in cols])

plt.xlabel("Timestep (0.5s per step)", weight="bold")

plt.ylabel("Standardised Feature Values", weight="bold")

plt.tight_layout()

plt.grid()

plt.show()
plot_var = 'apparent_power_exp_2400'

alpha = 0.05
plt.figure(figsize=(14,9))



plt.subplot(221)

ax = sns.regplot(x=plot_var, y="pm", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()



plt.subplot(222)

ax = sns.regplot(x=plot_var, y="stator_yoke", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()



plt.subplot(223)

ax = sns.regplot(x=plot_var, y="stator_tooth", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()



plt.subplot(224)

ax = sns.regplot(x=plot_var, y="stator_winding", data=example_profile, scatter_kws={'alpha':alpha})

plt.grid()

plt.tight_layout()

plt.show()
corr_matrix = example_profile.corr()

corr_matrix['stator_yoke'].sort_values(ascending=False)
class PMSMDataProcessor(BaseEstimator, TransformerMixin):

    """ PMSM data processor and loader """

    

    def __init__(self, add_exp_terms=True, spans=[600, 1200, 2400, 4800],

                 output_vars=['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']):

        self.add_exp_terms = add_exp_terms

        self.spans = [600, 1200, 2400, 4800]

        self.output_vars = output_vars

        self.training_cols = []

        

        

    def fit(self, X, y=None):      

        return self

                

    

    def transform(self, X):

        """ Create new features from our given data """

        

        extra_df = self._find_new_features(X)

        

        # update training cols

        self._update_train_cols(extra_df)

        

        # if selected add additional exponential terms

        if self.add_exp_terms:

            extra_df = self._find_exp_terms(extra_df).copy()

            

            # update training cols again to reflect new additions

            self._update_train_cols(extra_df)

            

        return extra_df

        

        

    def _find_norm(self, row, col1, col2):

        return np.linalg.norm([row[col1], row[col2]])

    

    

    def _update_train_cols(self, X):

        """ Compile list of base columns used for training """

        self.training_cols = [x for x in X if x != 'profile_id' 

                              and x not in self.output_vars]

    

    

    def _find_new_features(self, dataframe):

        """ Calculate new electrical features for our dataset """

        

        extra_cols = dataframe.copy()

        

        extra_cols['voltage_s'] = extra_cols.apply(self._find_norm, 

                                                   args=('u_d', 'u_q'), axis=1)

        extra_cols['current_s'] = extra_cols.apply(self._find_norm, 

                                                   args=('i_d', 'i_q'), axis=1)

        

        extra_cols['apparent_power'] = 1.5 * extra_cols['voltage_s'] * extra_cols['current_s']

        extra_cols['effective_power'] = ((extra_cols['u_d'] * extra_cols['u_q']) +

                                         (extra_cols['i_d'] * extra_cols['i_q']))

        

        extra_cols['speed_current'] = extra_cols['current_s'] * extra_cols['motor_speed']

        extra_cols['speed_power'] = extra_cols['current_s'] * extra_cols['apparent_power']

    

        return extra_cols

    

    

    def _find_exp_terms(self, dataframe):

        """ Add exponential terms to all features, using the class spans """

        

        extended_df = dataframe.copy()

        

        # add new columns to each feature for each span

        for span in self.spans:

            exp_ma_df = dataframe.ewm(span=span, adjust=False).mean()

            for column in self.training_cols:

                new_col = f"{column}_exp_{span}"

                extended_df[new_col] = exp_ma_df[column].copy()

                

        return extended_df
data_preprocessor = PMSMDataProcessor()



train_df_extra = data_preprocessor.fit_transform(train_df)

test_df_extra = data_preprocessor.transform(test_df)



y = train_df_extra[data_preprocessor.output_vars].copy()

X = train_df_extra.drop(columns=data_preprocessor.output_vars)



X.shape, y.shape, test_df_extra.shape
# select profile id's as groups to split our folds on

groups = X.loc[:, 'profile_id'].copy()

X.drop('profile_id', axis=1, inplace=True)
std_scaler = StandardScaler()

X_std = pd.DataFrame()

X_std[X.columns] = pd.DataFrame(std_scaler.fit_transform(X))

X_std['profile_id'] = groups.values

test_std = std_scaler.transform(test_df_extra)



X_std.head(3)
grouped_kfold = GroupShuffleSplit(n_splits=6)

grouped_kfold.get_n_splits(X_std, y, groups)
for train_index, test_index in grouped_kfold.split(X_std, y, groups):

    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)

    train_inputs, val_inputs = X_std.iloc[train_index], X_std.iloc[test_index]

    train_outputs, val_outputs = y.iloc[train_index], y.iloc[test_index]

    

    profiles_train = train_inputs['profile_id'].value_counts().index.values

    profiles_test = val_inputs['profile_id'].value_counts().index.values

    

    print(f"Profile IDs in Training set: \n{profiles_train}\n")

    print(f"Profile IDs in Test set: \n{profiles_test}\n")
lr_rmse_scores = []

mean_rmse_scores = []



lr_reg = LinearRegression()



# times to repeat the cross validation

repeats = 10



for i in tqdm(range(repeats)):

    lr_scores = cross_val_score(lr_reg, X_std.drop('profile_id', axis=1), y, groups=groups,

                                scoring='neg_mean_squared_error', cv=grouped_kfold)

    rmse_scores = np.sqrt(-lr_scores)

    lr_rmse_scores.append(rmse_scores)

    mean_rmse_scores.append(rmse_scores.mean())



lr_rmse_scores = np.array(lr_rmse_scores)

mean_rmse_scores = np.array(mean_rmse_scores)



print(f"Linear Regression OLS RMSE: {mean_rmse_scores.mean():.5f} +/- {mean_rmse_scores.std():.5f}")
def multi_model_grouped_cross_validation(clf_tuple_list, X, y, groups, K_folds=6, random_seed=0, subsample=False,

                                 score_type='neg_mean_squared_error', subsample_prop=0.1):

    """ Find grouped cross validation scores, and print and return results """

    

    model_names, model_scores = [], []

    

    # if selected, only use a small portion of data for speed

    if subsample:

        X, _, y, _ = train_test_split(X, y, training_size=subsample_prop, random_seed=random_seed)

    

    for name, model in clf_tuple_list:

        grouped_kfold = GroupShuffleSplit(n_splits=K_folds)

        cross_val_results = cross_val_score(model, X, y, groups=groups, 

                                            cv=grouped_kfold, scoring=score_type, n_jobs=-1)

        model_names.append(name)

        rmse_scores = np.sqrt(-cross_val_results)

        model_scores.append(rmse_scores)

        print("{0:<40} {1:.5f} +/- {2:.5f}".format(name, rmse_scores.mean(), rmse_scores.std()))

        

    return model_names, model_scores





def boxplot_comparison(model_names, model_scores, figsize=(14, 10), score_type="RMSE",

                       title="K-Folds Cross-Validation Comparisons"):

    """ Boxplot comparison of a range of models using Seaborn and matplotlib """

    

    fig = plt.figure(figsize=figsize)

    fig.suptitle(title, fontsize=18)

    ax = fig.add_subplot(111)

    sns.boxplot(x=model_names, y=model_scores)

    ax.set_xticklabels(model_names)

    ax.set_xlabel("Model", fontsize=16) 

    ax.set_ylabel("Model Score ({})".format(score_type), fontsize=16)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)

    plt.show()

    return
X_trg_sub, _, y_trg_sub, _ = train_test_split(X_std, y, stratify=groups, test_size=0.99, random_state=12)



# extract our training profiles from the training data prior to fitment

trg_profiles = X_trg_sub.loc[:, 'profile_id'].values

X_trg_sub = X_trg_sub.drop('profile_id', axis=1)



X_trg_sub.shape, y_trg_sub.shape
# list of classifiers to compare

clf_list = [("Linear Regressor", LinearRegression()),

            ("Ridge Regressor", Ridge(alpha=250)),

            ("LASSO Regressor", Lasso(alpha=0.1)),

            ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.1)),

            ("Support Vector Regressor", MultiOutputRegressor(LinearSVR())),

            ("Extra Trees Regressor", ExtraTreesRegressor(n_estimators=10, bootstrap=True, max_depth=60, 

                                                          n_jobs=-1, min_samples_leaf=3, min_samples_split=5)),

            ("Random Forest Regressor", RandomForestRegressor(n_estimators=10, bootstrap=True, max_depth=60, 

                                                              n_jobs=-1, min_samples_leaf=3, min_samples_split=5)),

            ("Multi-layer Perception", MLPRegressor())]



# obtain names and scores for each cross-validation, and print / plot results

print("K-Folds Cross-Validation Results on training set: \n{}".format('-'*60))

%time model_names, model_scores = multi_model_grouped_cross_validation(clf_list, X_trg_sub, y_trg_sub, trg_profiles)

boxplot_comparison(model_names, model_scores)
alpha_values = [0, 0.01, 0.1, 0.5, 1.0, 1.5, 3, 10, 20, 50, 100, 

                200, 250, 500, 750, 1000, 2000, 3000, 4000, 5000]



ridge_rmse_scores = []

mean_ridge_rmse_scores = []



for alpha in tqdm(alpha_values):

    ridge_reg = Ridge(alpha=alpha, fit_intercept=True)

    ridge_scores = cross_val_score(ridge_reg, X, y, groups=groups,

                                   scoring='neg_mean_squared_error', cv=grouped_kfold)

    rmse_scores = np.sqrt(-ridge_scores)

    ridge_rmse_scores.append(rmse_scores)

    mean_ridge_rmse_scores.append(rmse_scores.mean())



ridge_rmse_scores = np.array(ridge_rmse_scores)

mean_ridge_rmse_scores = np.array(mean_ridge_rmse_scores)



print(f"Ridge RMSE: {mean_ridge_rmse_scores.mean():.5f} +/- {mean_ridge_rmse_scores.std():.5f}")



# calculate standard deviation to annotate on our plot

ridge_rmse_scores_std = ridge_rmse_scores.std(axis=1)
# plot mean and standard deviation of cross-val scores

plt.figure(figsize=(18,6))

sns.lineplot(x=alpha_values, y=mean_ridge_rmse_scores)

plt.fill_between(alpha_values, mean_ridge_rmse_scores - ridge_rmse_scores_std, 

                 mean_ridge_rmse_scores + ridge_rmse_scores_std, 

                 color='tab:blue', alpha=0.2)

plt.semilogx(alpha_values, mean_ridge_rmse_scores + ridge_rmse_scores_std, 'b--')

plt.semilogx(alpha_values, mean_ridge_rmse_scores - ridge_rmse_scores_std, 'b--')

plt.ylabel("Cross-Validation RMSE (Average)", weight='bold', size=14)

plt.xlabel("Ridge Alpha Hyper-Parameter", weight='bold', size=14)

plt.show()
# range of alpha values between 0 and 1

alpha_values = np.logspace(-4, 1, 20)



lasso_rmse_scores = []

mean_lasso_rmse_scores = []



for alpha in tqdm(alpha_values):

    lasso_reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000, tol=0.5)

    lasso_scores = cross_val_score(lasso_reg, X, y, groups=groups,

                                   scoring='neg_mean_squared_error', cv=grouped_kfold)

    rmse_scores = np.sqrt(-lasso_scores)

    lasso_rmse_scores.append(rmse_scores)

    mean_lasso_rmse_scores.append(rmse_scores.mean())



lasso_rmse_scores = np.array(lasso_rmse_scores)

mean_lasso_rmse_scores = np.array(mean_lasso_rmse_scores)



print(f"LASSO RMSE: {mean_lasso_rmse_scores.mean():.5f} +/- {mean_lasso_rmse_scores.std():.5f}")
# calculate standard deviation to annotate on our plot

lasso_rmse_scores_std = lasso_rmse_scores.std(axis=1)



# plot mean and standard deviation of cross-val scores

plt.figure(figsize=(18,6))

sns.lineplot(x=alpha_values, y=mean_lasso_rmse_scores)

plt.fill_between(alpha_values, mean_lasso_rmse_scores - lasso_rmse_scores_std, 

                 mean_lasso_rmse_scores + lasso_rmse_scores_std, 

                 color='tab:blue', alpha=0.2)

plt.semilogx(alpha_values, mean_lasso_rmse_scores + lasso_rmse_scores_std, 'b--')

plt.semilogx(alpha_values, mean_lasso_rmse_scores - lasso_rmse_scores_std, 'b--')

plt.ylabel("Cross-Validation RMSE (Average)", weight='bold', size=14)

plt.xlabel("LASSO Alpha Hyper-Parameter", weight='bold', size=14)

plt.show()
# range of alpha values and l1 ratios for exploring

alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

l1_ratios = np.arange(0.0, 1.1, 0.1)



results_tuples = []

en_rmse_scores = []

mean_en_rmse_scores = []



for alpha in tqdm(alpha_values):

    for l1 in l1_ratios:

        

        en_reg = ElasticNet(alpha=alpha, l1_ratio=l1, tol=0.5,

                            fit_intercept=True, max_iter=1000)

        en_scores = cross_val_score(en_reg, X, y, groups=groups,

                                    scoring='neg_mean_squared_error', cv=grouped_kfold)

        rmse_scores = np.sqrt(-en_scores)

        en_rmse_scores.append(rmse_scores)

        mean_en_rmse_scores.append(rmse_scores.mean())

        

        results_tuples.append((alpha, l1, rmse_scores.mean(), rmse_scores.std()))



en_rmse_scores = np.array(en_rmse_scores)

mean_en_rmse_scores = np.array(mean_en_rmse_scores)



print(f"Elastic Net RMSE: {mean_en_rmse_scores.mean():.5f} +/- {mean_en_rmse_scores.std():.5f}")
# form dataframe from the results

elasnet_results = pd.DataFrame(results_tuples)

elasnet_results.columns = ['alpha', 'l1_ratio', 'mean', 'std']



# plot results according to hyper-parameters and mean RMSE

plt.figure(figsize=(16,6))

sns.scatterplot(x='alpha', y='l1_ratio', hue='mean', 

                data=elasnet_results, s=150)

plt.xlabel('ElasticNet Alpha Parameter', weight='bold', size=14)

plt.ylabel('ElasticNet l1-ratio', weight='bold', size=14)

plt.legend(loc='best')

plt.xlim(-0.5, 13.0)

plt.show()
lr_bag_reg = BaggingRegressor(LinearRegression(), n_estimators=5, bootstrap=True)



lr_bag_rmse_scores = []

mean_lr_bag_rmse_scores = []



# times to repeat the cross validation

repeats = 2



for i in tqdm(range(repeats)):

    lr_bag_scores = cross_val_score(lr_bag_reg, X, y, groups=groups,

                                scoring='neg_mean_squared_error', cv=grouped_kfold)

    rmse_scores = np.sqrt(-lr_bag_scores)

    lr_bag_rmse_scores.append(rmse_scores)

    mean_lr_bag_rmse_scores.append(rmse_scores.mean())



lr_bag_rmse_scores = np.array(lr_bag_rmse_scores)

mean_lr_bag_rmse_scores = np.array(mean_lr_bag_rmse_scores)



print(f"Lin Reg Bagging RMSE: {mean_lr_bag_rmse_scores.mean():.5f} +/- {mean_lr_bag_rmse_scores.std():.5f}")
X_trg_sub, _, y_trg_sub, _ = train_test_split(X_std, y, stratify=groups, test_size=0.99, random_state=12)

trg_profiles = X_trg_sub.loc[:, 'profile_id'].values



print(f"Number of profile IDs within sample: {X_trg_sub['profile_id'].value_counts().shape[0]}\n")



# drop profile ID before training

X_trg_sub = X_trg_sub.drop('profile_id', axis=1)



print(f"Shapes of data: X = {X_trg_sub.shape}, y = {y_trg_sub.shape}")



sample_group_kfold = GroupShuffleSplit(n_splits=6)
et_param_grid = {'n_estimators' : [10], 'max_depth' : [5, 10, 20, 60], 

                 'min_samples_split' : [2, 5, 10, 20], 'min_samples_leaf' : [1, 3, 5, 10], 

                 'n_jobs': [-1], 'bootstrap' : [True]}



et_reg = ExtraTreesRegressor()

et_grid_search = GridSearchCV(et_reg, et_param_grid, cv=sample_group_kfold, scoring='neg_mean_squared_error')
# fit to our grid search with grouped kfold

%time et_grid_search.fit(X_trg_sub, y_trg_sub, groups=trg_profiles)
# display best parameters found during grid search

best_et_params = et_grid_search.best_params_

best_et_params
# change best params n_estimators to a higher a more appropriate value

best_et_params['n_estimators'] = 600

best_et_params
et_reg = ExtraTreesRegressor(**best_et_params)



et_scores = cross_val_score(et_reg, X_trg_sub, y_trg_sub, groups=trg_profiles, 

                            scoring='neg_mean_squared_error', cv=sample_group_kfold)



et_rmse_scores = np.sqrt(-et_scores)

print(f"Extra Trees Reg RMSE: {et_rmse_scores.mean():.5f} +/- {et_rmse_scores.std():.5f}")
gb_reg = MultiOutputRegressor(GradientBoostingRegressor())



gb_scores = cross_val_score(gb_reg, X_trg_sub, y_trg_sub, groups=trg_profiles, 

                            scoring='neg_mean_squared_error', cv=sample_group_kfold)



gb_rmse_scores = np.sqrt(-gb_scores)

print(f"Gradient Boosting Reg RMSE: {gb_rmse_scores.mean():.5f} +/- {gb_rmse_scores.std():.5f}")
regressors = [('Bagged Linear Reg', BaggingRegressor(LinearRegression(), n_estimators=20, bootstrap=True)), 

              ('Extra Trees Reg', ExtraTreesRegressor(**best_et_params))]
ensemble_reg = MultiOutputRegressor(VotingRegressor(estimators=regressors, n_jobs=-1))



ensemble_scores = cross_val_score(ensemble_reg, X_trg_sub, y_trg_sub, groups=trg_profiles, 

                            scoring='neg_mean_squared_error', cv=sample_group_kfold)



ensemble_rmse_scores = np.sqrt(-ensemble_scores)

print(f"Ensemble Reg RMSE: {ensemble_rmse_scores.mean():.5f} +/- {ensemble_rmse_scores.std():.5f}")
data_preprocessor = PMSMDataProcessor()



train_df_extra = data_preprocessor.fit_transform(train_df)

test_df_extra = data_preprocessor.transform(test_df)



y = train_df_extra[data_preprocessor.output_vars].copy()

X = train_df_extra.drop(columns=data_preprocessor.output_vars)



# select profile id's as groups to split our folds on

groups = X.loc[:, 'profile_id'].copy()

X.drop('profile_id', axis=1, inplace=True)



X.shape, y.shape, test_df_extra.shape
std_scaler = StandardScaler()

X_std = pd.DataFrame()

X_std[X.columns] = pd.DataFrame(std_scaler.fit_transform(X))

test_std = std_scaler.transform(test_df_extra)
et_opt_params = {'n_estimators' : 600, 'bootstrap': True, 'max_depth': 20, 

                 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_jobs': -1}
# train final extra trees classifier

et_reg = ExtraTreesRegressor(**et_opt_params)

%time et_reg.fit(X_std, y)
# train final extra trees classifier

bag_lr_reg = BaggingRegressor(LinearRegression(), n_estimators=20, bootstrap=True)

%time bag_lr_reg.fit(X_std, y)
et_test_preds = et_reg.predict(test_std)

print(f"Final Extra Trees RMSE on Test Set: {np.sqrt(mean_squared_error(et_test_preds, test_labels)):.5f}")
lr_test_preds = bag_lr_reg.predict(test_std)

print(f"Final Linear Regression RMSE on Test Set: {np.sqrt(mean_squared_error(lr_test_preds, test_labels)):.5f}")
ensemble_preds = (et_test_preds + lr_test_preds) / 2.0

print(f"Final Ensemble RMSE on Test Set: {np.sqrt(mean_squared_error(ensemble_preds, test_labels)):.5f}")
# create an ensemble and compare final performance - we should expect a marginal improvement

regressors = [('Bagged Linear Reg', BaggingRegressor(LinearRegression(), n_estimators=20, bootstrap=True)), 

              ('Extra Trees Reg', ExtraTreesRegressor(**et_opt_params))]



# craete our final ensemble model

ensemble_reg = MultiOutputRegressor(VotingRegressor(estimators=regressors))
#print("Training final ensemble model on entire training set.\n")

#ensemble_reg.fit(X_std, y)
#test_preds = ensemble_reg.predict(test_std)

#print(f"Final Ensemble RMSE on Test Set: {np.sqrt(mean_squared_error(test_preds, test_labels)):.5f}")