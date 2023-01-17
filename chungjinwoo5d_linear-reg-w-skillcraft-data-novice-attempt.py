# Importing Libraries
import numpy as np
import pandas as pd
import urllib.request as urllib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.stats.api as sms
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy import stats
from statsmodels.stats.outliers_influence import summary_table
from numpy.polynomial.polynomial import polyfit
''' getting data '''
df_main = pd.read_csv('../input/SkillCraft.csv')
''' skimming data '''
df_main.head(20)
''' splitting data '''
# using seed 111 for controlled test
df_train, df_test = train_test_split(df_main, test_size=0.2, random_state=111)
''' checking if data looks like a normal distribution '''
df_league_index_distr = (df_train['LeagueIndex'].value_counts()).to_frame()
df_league_index_distr.sort_index(inplace=True)
df_league_index_distr.plot(kind='bar',y='LeagueIndex',colormap='Paired')
plt.show()
''' Linear Aggression : Full Model and exlcuding unsignificant p-value '''
# Dependent Variable
y = df_train[['LeagueIndex']]

# 1st Model : Full Model
removed_cols = ['GameID','LeagueIndex']
x_1 = df_train.drop(removed_cols, axis=1)
model_1 = sm.OLS(y,x_1).fit()
print(model_1.summary())
# 2nd Model : Excluding Independent Variable w/ p-values lower than 0.05
removed_cols = removed_cols + ['TotalHours','MinimapRightClicks','UniqueUnitsMade',
                   'ComplexUnitsMade','ComplexAbilitiesUsed']
x_2 = df_train.drop(removed_cols, axis=1)
model_2 = sm.OLS(y,x_2).fit()
print(model_2.summary())
''' checking for outliers and try to reduce outliers that are big in size and a bit extreme'''
# checking for outliers
dep_var = list(y)[0]
indep_var_list = list(x_2)

for indep_var in indep_var_list:
    df_train.boxplot(column=indep_var,by=dep_var)
    plt.title(dep_var + ' vs ' + indep_var)
    plt.suptitle('')
plt.show()

# # removing 2 variables : SelectByHotkeys and MinimapAttacks
# removed_cols = removed_cols + ['SelectByHotkeys','MinimapAttacks']

# # 3rd Model : Excluding certain variable with outliers that seem to be extreme
# x_3 = df_train.drop(removed_cols, axis=1)
# model_3 = sm.OLS(y,x_3).fit()
# print(model_3.summary())

# # 4th Model : Removing insignificant p-values
# removed_cols = removed_cols + ['ActionLatency','TotalMapExplored']
# x_4 = df_train.drop(removed_cols, axis=1)
# model_4 = sm.OLS(y,x_4).fit()
# print(model_4.summary())
''' Checking for multicollinearity (using VIF) '''
# adding a constant to correct output
mc_x = add_constant(x_2)
pd.Series([variance_inflation_factor(mc_x.values, i) for i in range(mc_x.shape[1])],index=mc_x.columns)
# 5th Model : using Model 2 and reducing variables
removed_cols = removed_cols + ['APM']
x_5 = df_train.drop(removed_cols, axis=1)
model_5 = sm.OLS(y,x_5).fit()
# testing with VIF again
mc_x = add_constant(x_5)
pd.Series([variance_inflation_factor(mc_x.values, i) for i in range(mc_x.shape[1])],index=mc_x.columns)
# reducing NumberOfPACs
model_6_removed_cols = removed_cols + ['NumberOfPACs']
x_6 = df_train.drop(model_6_removed_cols, axis=1)
model_6 = sm.OLS(y,x_6).fit()
# testing with VIF again
mc_x = add_constant(x_6)
pd.Series([variance_inflation_factor(mc_x.values, i) for i in range(mc_x.shape[1])],index=mc_x.columns)
# reducing the other variable
model_7_removed_cols = removed_cols + ['ActionLatency']
x_7 = df_train.drop(model_7_removed_cols, axis=1)
model_7 = sm.OLS(y,x_7).fit()
# testing with VIF again
mc_x = add_constant(x_7)
pd.Series([variance_inflation_factor(mc_x.values, i) for i in range(mc_x.shape[1])],index=mc_x.columns)
# print(model_6.summary())
# print(model_7.summary())

''' Comparing Linear Models with ANOVA '''
anova_result = anova_lm(model_6, model_7)
print(anova_result)
print(model_7.summary())
# Fitted vs Residual (weird...)
fr_plot = plt.figure(1)
fr_plot.axes[0] = sns.residplot(x=model_7.fittedvalues, y=model_7.resid,
                          lowess=True, scatter_kws={'alpha': 0.5},
                          line_kws={'color':'red', 'lw':1, 'alpha':0.8})

fr_plot.axes[0].set_title('Residuals vs Fitted')
fr_plot.axes[0].set_xlabel('Fitted values')
fr_plot.axes[0].set_ylabel('Residuals')
plt.show()
# Normal QQ Plot
QQ = ProbPlot(model_7.resid)
qq_plot = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

qq_plot.axes[0].set_title('Normal QQ')
qq_plot.axes[0].set_xlabel('Theoretical Quantiles')
qq_plot.axes[0].set_ylabel('Standardized Residuals')
plt.show()
anderson_result = stats.anderson(model_7.resid)
anderson_result[1][2]
# getting predicted values
x_predict = df_test.drop(model_7_removed_cols, axis=1)
prediction = model_7.predict(x_predict)
prediction.head()
# adding the difference between actual and predicted values in a new column
df_final = pd.concat([df_test['LeagueIndex'],prediction], axis=1)
df_final.columns = ['ActualLeagueIndex','PredictedLeagueIndex']
df_final = df_final.assign(Difference=abs(df_final.ActualLeagueIndex-df_final.PredictedLeagueIndex))
df_final.head(20)
# skimming how much the average is
df_final['Difference'].mean()
plt.scatter(df_final['ActualLeagueIndex'], df_final['PredictedLeagueIndex'], edgecolors=(0, 0, 0))
plt.plot([df_final['ActualLeagueIndex'].min(), df_final['ActualLeagueIndex'].max()], 
         [df_final['ActualLeagueIndex'].min(), df_final['ActualLeagueIndex'].max()], 'k-', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.show()
st, data, ss2 = summary_table(model_7, alpha=0.05)

fitted_values = data[:, 2]
predict_mean_low, predict_mean_upp = data[:, 4:6].T
predict_low, predict_upp = data[:, 6:8].T

plt.scatter(df_final['PredictedLeagueIndex'], df_final['ActualLeagueIndex'])
# Fit with polyfit
n_fit, m_fit = polyfit(df_train['LeagueIndex'], fitted_values, 1)
plt.plot(df_train['LeagueIndex'], n_fit + m_fit * df_train['LeagueIndex'], '-', color='black', lw=1)
n_low, m_low = polyfit(df_train['LeagueIndex'], predict_low, 1)
plt.plot(df_train['LeagueIndex'], n_low + m_low * df_train['LeagueIndex'], ':', color='red', lw=1)
n_upp, m_upp = polyfit(df_train['LeagueIndex'], predict_upp, 1)
plt.plot(df_train['LeagueIndex'], n_upp + m_upp * df_train['LeagueIndex'], ':', color='red', lw=1)
n_mean_low, m_mean_low = polyfit(df_train['LeagueIndex'], predict_mean_low, 1)
plt.plot(df_train['LeagueIndex'], n_mean_low + m_mean_low * df_train['LeagueIndex'], ':', color='#ffa500', lw=1)
n_mean_upp, m_mean_upp = polyfit(df_train['LeagueIndex'], predict_mean_upp, 1)
plt.plot(df_train['LeagueIndex'], n_mean_upp + m_mean_upp * df_train['LeagueIndex'], ':', color='#ffa500', lw=1)
plt.show()