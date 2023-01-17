import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import statsmodels.api as sm

from statsmodels.graphics.gofplots import ProbPlot

from scipy import stats
train = pd.read_csv("https://raw.githubusercontent.com/anilak1978/data621/master/moneyball-training-data.csv")

test = pd.read_csv("https://raw.githubusercontent.com/anilak1978/data621/master/moneyball-evaluation-data.csv")
train.head()
train.describe()
train = train.drop(["INDEX"], axis=1)
#create a function to find the missing value percentages

def missing_data(data):

    percentage_missing = data.isnull().sum()*100/len(data)

    data_types= data.dtypes

    missing_values = pd.DataFrame({"data_types": data_types,

                                  "pct_missing": percentage_missing})

    return missing_values



missing_data(train)
# look at correlation and plot heatmap

correlation = train.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation, annot=True)
correlation["TARGET_WINS"].sort_values(ascending=False)
# seperating batting, base running, pitching and fielding

batting = train[['TEAM_BATTING_H', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B',

                 'TEAM_BATTING_HR', 'TEAM_BATTING_BB', 'TEAM_BATTING_SO','TEAM_BATTING_HBP']]



baserun = train[['TEAM_BASERUN_SB', 'TEAM_BASERUN_CS']]

pitching = train[['TEAM_PITCHING_H', 'TEAM_PITCHING_HR', 'TEAM_PITCHING_BB','TEAM_PITCHING_SO']]

fielding = train[['TEAM_FIELDING_E', 'TEAM_FIELDING_DP']]
# looking at batting

plt.figure(figsize=(15,15))

sns.pairplot(batting, kind="reg")
# looking at baserun

plt.figure(figsize=(15,15))

sns.pairplot(baserun, kind="reg")
# looking at pitching

plt.figure(figsize=(15,15))

sns.pairplot(pitching, kind="reg")
# looking at fielding

plt.figure(figsize=(15,15))

sns.pairplot(fielding, kind="reg")
# find skewness for each column

for column in train.columns.values.tolist():

    print(column)

    print(train[column].skew())

    print()
# distribution for each variable

for column in train:

    fig=px.histogram(train, x=train[column])

    fig.show()
# distribution for each variable with boxplot

for column in train:

    fig=px.box(train, y=train[column])

    fig.show()
for column in train:

    plt.figure(figsize=(10,10))

    sns.residplot(x=train[column], y=train["TARGET_WINS"], data=train, lowess=True)
# creating a simple model with one explanatory variable

simple_model = sm.OLS(train["TARGET_WINS"], train["TEAM_BATTING_H"]).fit()

simple_model.summary()
# fitted values (need a constant term for intercept)

model_fitted_y = simple_model.fittedvalues



# model residuals

model_residuals = simple_model.resid



# normalized residuals

model_norm_residuals = simple_model.get_influence().resid_studentized_internal



# absolute squared normalized residuals

model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))



# absolute residuals

model_abs_resid = np.abs(model_residuals)



# leverage, from statsmodels internals

model_leverage = simple_model.get_influence().hat_matrix_diag



# cook's distance, from statsmodels internals

model_cooks = simple_model.get_influence().cooks_distance[0]
plot_lm_1 = plt.figure(1)

plot_lm_1.set_figheight(8)

plot_lm_1.set_figwidth(12)



plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'TARGET_WINS', data=train,

                                  lowess=True,

                                  scatter_kws={'alpha': 0.5},

                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_1.axes[0].set_title('Residuals vs Fitted')

plot_lm_1.axes[0].set_xlabel('Fitted values')

plot_lm_1.axes[0].set_ylabel('Residuals')
# plot Standardized Residuals VS Theoretical Quantiles.

QQ = ProbPlot(model_norm_residuals)

plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)



plot_lm_2.set_figheight(8)

plot_lm_2.set_figwidth(12)



plot_lm_2.axes[0].set_title('Normal Q-Q')

plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')

plot_lm_2.axes[0].set_ylabel('Standardized Residuals');



# annotations

abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)

abs_norm_resid_top_3 = abs_norm_resid[:3]



for r, i in enumerate(abs_norm_resid_top_3):

    plot_lm_2.axes[0].annotate(i, 

                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],

                                   model_norm_residuals[i]));
plot_lm_3 = plt.figure(3)

plot_lm_3.set_figheight(8)

plot_lm_3.set_figwidth(12)



plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)

sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 

            scatter=False, 

            ci=False, 

            lowess=True,

            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_3.axes[0].set_title('Scale-Location')

plot_lm_3.axes[0].set_xlabel('Fitted values')

plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');



# annotations

abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)

abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]



for i in abs_norm_resid_top_3:

    plot_lm_3.axes[0].annotate(i, 

                               xy=(model_fitted_y[i], 

                                   model_norm_residuals_abs_sqrt[i]));
plot_lm_4 = plt.figure(4)

plot_lm_4.set_figheight(8)

plot_lm_4.set_figwidth(12)



plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)

sns.regplot(model_leverage, model_norm_residuals, 

            scatter=False, 

            ci=False, 

            lowess=True,

            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_4.axes[0].set_xlim(0, 0.20)

plot_lm_4.axes[0].set_ylim(-3, 5)

plot_lm_4.axes[0].set_title('Residuals vs Leverage')

plot_lm_4.axes[0].set_xlabel('Leverage')

plot_lm_4.axes[0].set_ylabel('Standardized Residuals')



# annotations

leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]



for i in leverage_top_3:

    plot_lm_4.axes[0].annotate(i, 

                               xy=(model_leverage[i], 

                                   model_norm_residuals[i]))

    

# shenanigans for cook's distance contours

def graph(formula, x_range, label=None):

    x = x_range

    y = formula(x)

    plt.plot(x, y, label=label, lw=1, ls='--', color='red')



p = len(simple_model.params) # number of model parameters



graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 

      np.linspace(0.001, 0.200, 50), 

      'Cook\'s distance') # 0.5 line



graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 

      np.linspace(0.001, 0.200, 50)) # 1 line



plt.legend(loc='upper right');
# remove the variable that has more than 80% missing value

train_2 = train.drop(["TEAM_BATTING_HBP"], axis=1)

train_2.info()
# replace the missing values with mean of each variable

avg_1 = train_2["TEAM_BATTING_SO"].mean(axis=0)

avg_2 = train_2["TEAM_BASERUN_SB"].mean(axis=0)

avg_3 = train_2["TEAM_BASERUN_CS"].mean(axis=0)

avg_4 = train_2["TEAM_FIELDING_DP"].mean(axis=0)

avg_5 = train_2["TEAM_PITCHING_SO"].mean(axis=0)

train_2["TEAM_BATTING_SO"].replace(np.nan, avg_1, inplace=True)

train_2["TEAM_BASERUN_SB"].replace(np.nan, avg_2, inplace=True)

train_2["TEAM_BASERUN_CS"].replace(np.nan, avg_3, inplace=True)

train_2["TEAM_FIELDING_DP"].replace(np.nan, avg_4, inplace=True)

train_2["TEAM_PITCHING_SO"].replace(np.nan, avg_5, inplace=True)
missing_data(train_2)
# find and remove outliers

z= np.abs(stats.zscore(train_2))

filtered=(z<3).all(axis=1)

train_3=train_2[filtered]

train_3.head()
# look at skewness and confirm removal of the outliers

plt.figure(figsize=(10,10))

sns.pairplot(train_3, kind="reg")
# feature selection (select all)

X = train_3[['TEAM_BATTING_H', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B',

       'TEAM_BATTING_HR', 'TEAM_BATTING_BB', 'TEAM_BATTING_SO',

       'TEAM_BASERUN_SB', 'TEAM_BASERUN_CS', 'TEAM_PITCHING_H',

       'TEAM_PITCHING_HR', 'TEAM_PITCHING_BB', 'TEAM_PITCHING_SO',

       'TEAM_FIELDING_E', 'TEAM_FIELDING_DP']]

y = train_3.TARGET_WINS



def stepwise_selection(X, y, 

                       initial_list=[], 

                       threshold_in=0.01, 

                       threshold_out = 0.05, 

                       verbose=True):

    """ Perform a forward-backward feature selection 

    based on p-value from statsmodels.api.OLS

    Arguments:

        X - pandas.DataFrame with candidate features

        y - list-like with the target

        initial_list - list of features to start with (column names of X)

        threshold_in - include a feature if its p-value < threshold_in

        threshold_out - exclude a feature if its p-value > threshold_out

        verbose - whether to print the sequence of inclusions and exclusions

    Returns: list of selected features 

    Always set threshold_in < threshold_out to avoid infinite looping.

    See https://en.wikipedia.org/wiki/Stepwise_regression for the details

    """

    included = list(initial_list)

    while True:

        changed=False

        # forward step

        excluded = list(set(X.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))



        # backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() # null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included



result = stepwise_selection(X, y)



print('resulting features:')

print(result)
# using all features

model_1 = sm.OLS(y, X).fit()

model_1.summary()
# Lets try another model this time with the significant variables based on p-values

X2 = train_3[["TEAM_FIELDING_E", "TEAM_BASERUN_SB", "TEAM_BATTING_3B",

             "TEAM_FIELDING_DP", "TEAM_FIELDING_DP", "TEAM_PITCHING_SO",

             "TEAM_BATTING_SO", "TEAM_BATTING_2B"]]
model_2 = sm.OLS(y, X2).fit()

model_2.summary()
# let's use all offensive categories

X3 = train_3[["TEAM_BATTING_H",

              "TEAM_BATTING_BB",

              "TEAM_BATTING_HR",

              "TEAM_BATTING_2B",

              "TEAM_BATTING_SO",

              "TEAM_BASERUN_CS",

              "TEAM_BATTING_3B"]]

model_3 = sm.OLS(y, X3).fit()

model_3.summary()
# all defensive catagories

X4 = train_3[["TEAM_PITCHING_H",

              "TEAM_PITCHING_BB",

              "TEAM_PITCHING_HR",

              "TEAM_PITCHING_SO",

              "TEAM_FIELDING_E"]]

model_4 = sm.OLS(y, X4).fit()

model_4.summary()
# use only the 5 significant variables we identified from model3

X5 = train_3[["TEAM_PITCHING_H",

              "TEAM_PITCHING_BB",

              "TEAM_PITCHING_HR",

              "TEAM_PITCHING_SO",

              "TEAM_BATTING_3B",

              "TEAM_BASERUN_SB"]]

model_5 = sm.OLS(y, X5).fit()

model_5.summary()
# Prediction

test.head()
test_2 = test.drop(["INDEX"], axis=1)
X_test = test_2[["TEAM_BATTING_H",

              "TEAM_BATTING_BB",

              "TEAM_BATTING_HR",

              "TEAM_BATTING_2B",

              "TEAM_BATTING_SO",

              "TEAM_BASERUN_CS",

              "TEAM_BATTING_3B"]]
y_pred = model_3.predict(X_test)
y_pred[0:5]