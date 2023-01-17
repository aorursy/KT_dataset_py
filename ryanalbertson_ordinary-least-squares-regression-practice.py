import pandas as pd

import numpy as np

from pandas import plotting

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from sklearn import metrics

import statsmodels.api as sm

import statsmodels.stats.diagnostic as smd

from sklearn.model_selection import train_test_split

from scipy.stats import gaussian_kde

from statsmodels.graphics.gofplots import ProbPlot

from scipy import stats
# Load CSV file data into a dataframe.

df = pd.read_csv("/kaggle/input/pga-tour-20102018-data/2019_data.csv")

df.head()
# Transpose the statistic variables such that there is 1 golfer per row, each with columns for

# every statistic variable.

df = df.set_index(['Player Name', 'Variable', 'Date'])['Value'].unstack('Variable').reset_index()



# Typecast the Date column to datetime objects so they can be quantified.

df['Date'] = pd.to_datetime(df['Date'])



# Select data from 8/25/19, which was the end of the 2018-2019 PGA season.

df = df[(df['Date'] == '2019-08-25')]



# Typecast data points to numeric data types, except Player Name and Date columns.

df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0)



df.head()
# Check all of the statistic variables available in the dataset.

for col in df.columns[2:]:

    print(col)
# Visualize the distribution and colinearity for a sample of random variables.



sample_df = df.iloc[:, 2:].sample(n=8, axis=1)



fig1 = pd.plotting.scatter_matrix(sample_df, figsize=(24, 24))



for x in range(len(sample_df.columns)):

    for y in range(len(sample_df.columns)):

        ax = fig1[x, y]

        ax.xaxis.label.set_rotation(45)

        ax.yaxis.label.set_rotation(45)

        ax.yaxis.labelpad = 100
selected_var = 'Greens in Regulation Percentage - (%)'
# Only keep the average, percentage, and points variables, otherwise there will be many 1.0 co-efficients cluttering the analysis.

for col in df.columns:

    if 'AVG' not in col and '%' not in col and 'POINTS' not in col:

        del df[col]
# Create a sorted Pierson correlation matrix to understand variable pair relationships.



matrix_df = df.corr().abs()

unstack_matrix = matrix_df.unstack()

sorted_matrix = unstack_matrix.sort_values(kind='quicksort', ascending=False, na_position='first').dropna()

  

print('ALL CORRELATIONS ARE BETWEEN \"{}\" AND AN ARBITRARY VARIABLE'.format(selected_var))

print('='*95+'\n')

    

count = 0

for pair, val in sorted_matrix.items():

    if pair[1] == selected_var and count < 10:

        print('{:68} PIERSON CO-EFF.'.format(pair[0] + ' ,'))

        print('{:68} {}'.format(pair[1], val))

        print('-'*88)

        count += 1
# Select the highly correlated pairs that contain the selected variable.



pairs = []



for pair, val in sorted_matrix.items():

    var1, var2 = pair

    if var2 == selected_var:

        pairs.append([var1, var2, val])
# Test the significance of the correlations with the selected variable using p-values of the co-effs.



lin_regress_dict = {}



for pair in pairs:

    var1_list = df[pair[0]].values.tolist()

    var2_list = df[pair[1]].values.tolist()

    (slope, intercept, r_value, p_value, std_err) = linregress(var1_list, var2_list)

    

    key_name = "{}, {}".format(pair[0], pair[1])

    lin_regress_dict[key_name] = ((slope, intercept, r_value, p_value, std_err))

    

# Keep the most significantly correlated pairs

for key, val in list(lin_regress_dict.items()):

    if val[3] > 0.05:  # p-value > 0.05

        del lin_regress_dict[key]
# Sort the correlated pairs by p-value.

sorted_pvalues = sorted(lin_regress_dict.items(), key=lambda x: x[1][3])



# Print the most significantly correlated variables to the selected variable.

print('VARIABLE CORRELATIONS TO \"{}\", SORTED BY P-VALUE\n'.format(selected_var))

print('{:58} {:13} {}'.format('NAME', 'R-VALUE', 'P-VALUE'))

print('='*81)



for pair in sorted_pvalues[:100]:

    var1, var2 = pair[0].split(', ')

    #split1, split2 = name1.split(' - ', 1)

    slope, intercept, r_value, p_value, std_err = pair[1]

    

    print('{:52}   |   {:6.4f}   |   {:4}'.format(var1, r_value.round(4), p_value))

    print('-'*81)  
independent_var = 'Driving Distance - (AVG.)'

dependent_var = 'Greens in Regulation Percentage - (%)'



x = pd.DataFrame(df[independent_var])

y = pd.DataFrame(df[dependent_var])
# Split data for later validation.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# Concatenate the variables into one dataframe for easier reference.

train_df = pd.concat([x_train, y_train], axis=1)
# Generate an Ordinary Least Squares regression model.



X_train = sm.add_constant(x_train)



ols_model = sm.OLS(y_train, X_train).fit()



print(ols_model.summary())
# Cross validate with the test set of data.



X_test = sm.add_constant(x_test)



ols_model_test = sm.OLS(y_test, X_test).fit()



print(ols_model_test.summary())
# Set variables for info from the model, to use for analysis.



# Model fitted values.

ols_model_fitted_y = ols_model.fittedvalues



# Model residuals.

ols_model_residuals = ols_model.resid



# Normalized residuals.

ols_model_norm_residuals = ols_model.get_influence().resid_studentized_internal



# Absolute squared normalized residuals.

ols_model_norm_residuals_abs_sqrt = np.sqrt(np.abs(ols_model_norm_residuals))



# Leverage.

ols_model_leverage = ols_model.get_influence().hat_matrix_diag
# Regression plot.



plt.style.use('seaborn')



fig2, ax1 = plt.subplots(1, 1, figsize=(12, 8))



ax1.scatter(x_train, y_train, alpha=0.5)

ax1.plot(x_train, ols_model_fitted_y, color='red', linewidth=2)



ax1.set_title('Regression')

ax1.set_xlabel(independent_var)

ax1.set_ylabel(dependent_var)

ax1.text(200, 10,'y = 0.2259x + 0.0053', fontsize=20)



fig3, ax1 = plt.subplots(1, 1, figsize=(12, 8))



ax1.scatter(x_train, y_train, alpha=0.8)

ax1.plot(x_train, ols_model_fitted_y, color='red', linewidth=2)



ax1.set_title('Regression - (Magnified View)')

ax1.set_xlabel(independent_var)

ax1.set_ylabel(dependent_var)

ax1.set_xbound(min([x for x in x_train[independent_var] if x!=0])-5, \

               max(x_train[independent_var])+5)

ax1.set_ybound(min([y for y in y_train[dependent_var] if y!=0])-5, \

               max(y_train[dependent_var])+5)
# Residuals density plot



mean = np.mean(ols_model_residuals)

std = np.std(ols_model_residuals)



kde = gaussian_kde(ols_model_residuals)

covf = kde.covariance_factor()

bw = covf * std

     

fig4, ax1 = plt.subplots(1, 1, figsize=(12, 8))



sns.distplot(ols_model_residuals, kde_kws={'bw': bw})



ax1.set_title('Residual Density')

ax1.text(2.5, 1.25, "mean = {:.4f}\nstd = {:.4f}".format(mean, std), fontsize=18)
# Normal Q-Q plot.



fig5, ax1 = plt.subplots(figsize=(12, 8))



QQ = ProbPlot(ols_model_norm_residuals)



QQ.qqplot(line='45', alpha=0.3, lw=1, color='#4c72b0', ax=ax1)



ax1.set_title('Normal Q-Q')

ax1.set_xlabel('Theoretical Quantiles')

ax1.set_ylabel('Standardized Residuals')
# Residuals vs Fitted plot.



fig6, ax1 = plt.subplots(1, 1, figsize=(12, 8))



sns.residplot(ols_model_fitted_y, train_df[dependent_var], lowess=False, ax=ax1, \

              scatter_kws={'alpha': 0.6}, line_kws={'color': 'red', 'lw': 2, 'alpha': 0.5})



ax1.set_title('Residuals vs Fitted')

ax1.set_xlabel('Fitted Values')

ax1.set_ylabel('Residuals');



fig7, ax1 = plt.subplots(1, 1, figsize=(12, 8))



sns.residplot(ols_model_fitted_y, y_train[dependent_var], lowess=False, ax=ax1, \

              scatter_kws={'alpha': 0.8}, line_kws={'color': 'red', 'lw': 2, 'alpha': 0.5})



ax1.set_title('Residuals vs Fitted - (Magnified View)')

ax1.set_xlabel('Fitted Values')

ax1.set_ylabel('Residuals');

ax1.set_xbound(min([x for x in ols_model_fitted_y if x>1])-3, \

               max(y_train[dependent_var])+2)
# Scale-Location plot.



fig8, ax1 = plt.subplots(1, 1, figsize=(12, 8))



sns.regplot(ols_model_fitted_y, ols_model_norm_residuals_abs_sqrt, ci=False, \

            lowess=False, scatter_kws={'alpha': 0.6}, fit_reg=False, ax=ax1, \

            line_kws={'color': 'red', 'lw': 2, 'alpha': 0.5})



ax1.set_title('Scale-Location')

ax1.set_xlabel('Fitted Values')

ax1.set_ylabel('Standardized Residuals')



fig9, ax1 = plt.subplots(1, 1, figsize=(12, 8))



sns.regplot(ols_model_fitted_y, ols_model_norm_residuals_abs_sqrt, ci=False, \

            lowess=False, scatter_kws={'alpha': 0.8}, fit_reg=False, ax=ax1, \

            line_kws={'color': 'red', 'lw': 2, 'alpha': 0.5})   



ax1.set_title('Scale-Location - (Magnified View)')

ax1.set_xlabel('Fitted Values')

ax1.set_ylabel('Standardized Residuals')

ax1.set_xbound(60, 74)
# Residuals vs Leverage plot.



fig10, ax1 = plt.subplots(figsize=(12,8))



plt.scatter(ols_model_leverage, ols_model_norm_residuals, alpha=0.5)



sns.regplot(ols_model_leverage, ols_model_norm_residuals, ax=ax1, \

              scatter=False, ci=False, lowess=False, fit_reg=False, \

              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



ax1.set_xlim(0, max(ols_model_leverage)+0.001)

ax1.set_title('Residuals vs Leverage')

ax1.set_xlabel('Leverage')

ax1.set_ylabel('Standardized Residuals')
new_train_df = train_df.copy()



for idx, row in new_train_df.iterrows():

    if row[1] <= 0:

        new_train_df.drop(idx, axis=0, inplace=True)



new_x = pd.DataFrame(new_train_df[independent_var])

new_y = pd.DataFrame(new_train_df[dependent_var])



new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_x, new_y, test_size=0.2)



new_X_train = sm.add_constant(new_x_train)



new_ols_model = sm.OLS(new_y_train, new_X_train).fit()



print(new_ols_model.summary())
# Cross validate with the test set of data.



new_X_test = sm.add_constant(new_x_test)



new_ols_model_test = sm.OLS(new_y_test, new_X_test).fit()



print(new_ols_model_test.summary())
# Set variables for info from the model, to use for analysis.



# Model fitted values.

new_ols_model_fitted_y = new_ols_model.fittedvalues



# Model residuals.

new_ols_model_residuals = new_ols_model.resid



# Normalized residuals.

new_ols_model_norm_residuals = new_ols_model.get_influence().resid_studentized_internal



# Absolute squared normalized residuals.

new_ols_model_norm_residuals_abs_sqrt = np.sqrt(np.abs(new_ols_model_norm_residuals))



# Leverage.

new_ols_model_leverage = new_ols_model.get_influence().hat_matrix_diag
# New Regression plot.



fig11, ax1 = plt.subplots(1, 1, figsize=(12, 8))

fig11.suptitle('NEW')



ax1.scatter(new_x_train, new_y_train, alpha=0.8)

ax1.plot(new_x_train, new_ols_model_fitted_y, color='red', linewidth=2)



ax1.set_title('Regression')

ax1.set_xlabel(independent_var)

ax1.set_ylabel(dependent_var)

ax1.text(275, 61,'y = 0.06x + 47.45', fontsize=20)



# Compare with the original regression plot.

fig2
# New Residuals density plot



mean = np.mean(new_ols_model_residuals)

std = np.std(new_ols_model_residuals)



kde = gaussian_kde(new_ols_model_residuals)

covf = kde.covariance_factor()

bw = covf * std

     

fig12, ax1 = plt.subplots(1, 1, figsize=(12, 8))

fig12.suptitle('NEW')



sns.distplot(new_ols_model_residuals, kde_kws={'bw': bw}, ax=ax1)



ax1.text(5, 0.15, "mean = {:.4f}\nstd = {:.4f}".format(mean, std), fontsize=18)

ax1.set_title('Residual Density')



# Compare with original residual density plot.

fig4
# Test if residuals are normally distributed using a test that factors skew and kurtosis.



s, pval = stats.normaltest(new_ols_model_residuals)



if pval < 0.05:

    print('new_ols_model.resid is not normally distributed.')

else:

    print('new_ols_model.resid is normally distributed.')
# Residuals vs Fitted plot.



fig13, ax1 = plt.subplots(1, 1, figsize=(12, 8))



sns.residplot(new_ols_model_fitted_y, new_y_train, lowess=True, ax=ax1, \

              scatter_kws={'alpha': 0.6}, line_kws={'color': 'red', 'lw': 2, 'alpha': 0.5})



ax1.set_title('Residuals vs Fitted')

ax1.set_xlabel('Fitted Values')

ax1.set_ylabel('Residuals');



# Compare with the original residuals vs fitted plot

fig6
# Use Breush-Pagan Test to check for heteroskedasticity.



test = smd.het_breuschpagan(new_ols_model_residuals, new_ols_model.model.exog)



if test[1] > 0.05:

    print('There is not enough evidence to conclude that there is heteroskedasticity in the data.')

else:

    print('There is enough evidence to conclude that there is heteroskedasticity in the data.')
# New Normal Q-Q plot.



fig13, ax1 = plt.subplots(figsize=(12, 8))





QQ = ProbPlot(new_ols_model_residuals)



QQ.qqplot(line='45', alpha=0.3, lw=1, color='#4c72b0', ax=ax1)



ax1.set_title('Normal Q-Q')

ax1.set_xlabel('Theoretical Quantiles')

ax1.set_ylabel('Standardized Residuals')



# Compare to original Normal QQ plot

fig5
# New Scale-Location plot.



fig14, ax1 = plt.subplots(1, 1, figsize=(12, 8))

fig14.suptitle('NEW')



sns.regplot(new_ols_model_fitted_y, new_ols_model_norm_residuals_abs_sqrt, ci=False, \

            lowess=True, scatter_kws={'alpha': 0.6}, ax=ax1, \

            line_kws={'color': 'red', 'lw': 2, 'alpha': 0.5})



ax1.set_title('Scale-Location')

ax1.set_xlabel('Fitted Values')

ax1.set_ylabel('Standardized Residuals')



# Compare with original scale-location plot.

fig8
# Residuals vs Leverage plot.



fig15, ax1 = plt.subplots(figsize=(12,8))

fig15.suptitle('NEW')



plt.scatter(new_ols_model_leverage, new_ols_model_norm_residuals, alpha=0.6)



sns.regplot(new_ols_model_leverage, new_ols_model_norm_residuals, ax=ax1, \

              scatter=False, ci=False, lowess=True, \

              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



ax1.set_xlim(0, max(new_ols_model_leverage)+0.001)

ax1.set_title('Residuals vs Leverage')

ax1.set_xlabel('Leverage')

ax1.set_ylabel('Standardized Residuals')



# Compare with original residuals vs leverage plot.

fig10