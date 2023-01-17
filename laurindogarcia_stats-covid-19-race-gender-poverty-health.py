import pandas as pd



all_data_5 = pd.read_csv("../input/covid-19-race-gender-poverty-risk-us-county/covid_data_log_200922.csv")
all_data_5.corr()
all_data_5.info()
import seaborn as sns

sns.heatmap(all_data_5.corr())
corr = all_data_5.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
# sns.scatterplot(all_data_5["Risk_Index"], all_data_5["Cases"])



sns.jointplot(all_data_5["Risk_Index"], all_data_5["Cases"], data=all_data_5);

# sns.scatterplot(all_data_5["Risk_Index"], all_data_5["Deaths"])



sns.jointplot(all_data_5["Risk_Index"], all_data_5["Deaths"], data=all_data_5);
# sns.scatterplot(all_data_5["Risk_Index"], all_data_5["B_Female"])



sns.jointplot(all_data_5["Risk_Index"], all_data_5["B_Female"], data=all_data_5);
# sns.scatterplot(all_data_5["Risk_Index"], all_data_5["B_Female"])



sns.jointplot(all_data_5["Risk_Index"], all_data_5["H_Female"], data=all_data_5);
# sns.scatterplot(all_data_5["Risk_Index"], all_data_5["Poverty"])



sns.jointplot(all_data_5["Risk_Index"], all_data_5["Poverty"], data=all_data_5);
sns.distplot(all_data_5["Risk_Index"]);
all_data_5.loc[:,["Risk_Index"]].describe()
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Data 

data = pd.read_csv('../input/covid-19-race-gender-poverty-risk-us-county/covid_data_log_200922.csv') 

df = pd.DataFrame(data)



# Fit Model & Output Regression Results Summary



# Import Package

import statsmodels.api as sm

from statsmodels.api import add_constant



# Build Model

X = data.loc[:,["Risk_Index"]]

y = data.loc[:,["Cases"]]



X = sm.add_constant(X)

model1 = sm.OLS(y,X)

results = model1.fit()



# MSE of the residuals

print(f"MSE: {results.mse_resid}")



# Output Results

results.summary()
# Define function to output plot of the model coefficients



def coefplot(results):

    '''

    Takes in results of OLS model and returns a plot of 

    the coefficients with 95% confidence intervals.

    

    Removes intercept, so if uncentered will return error.

    '''

    # Create dataframe of results summary 

    coef_df = pd.DataFrame(results.summary().tables[1].data)

    

    # Add column names

    coef_df.columns = coef_df.iloc[0]



    # Drop the extra row with column labels

    coef_df=coef_df.drop(0)



    # Set index to variable names 

    coef_df = coef_df.set_index(coef_df.columns[0])



    # Change datatype from object to float

    coef_df = coef_df.astype(float)



    # Get errors; (coef - lower bound of conf interval)

    errors = coef_df['coef'] - coef_df['[0.025']

    

    # Append errors column to dataframe

    coef_df['errors'] = errors



    # Drop the constant for plotting

    coef_df = coef_df.drop(['const'])



    # Sort values by coef ascending

    coef_df = coef_df.sort_values(by=['coef'])



    ### Plot Coefficients ###



    # x-labels

    variables = list(coef_df.index.values)

    

    # Add variables column to dataframe

    coef_df['variables'] = variables

    

    # Set sns plot style back to 'poster'

    # This will make bars wide on plot

    sns.set_context("poster")



    # Define figure, axes, and plot

    fig, ax = plt.subplots(figsize=(15, 10))

    

    # Error bars for 95% confidence interval

    # Can increase capsize to add whiskers

    coef_df.plot(x='variables', y='coef', kind='bar',

                 ax=ax, color='none', fontsize=22, 

                 ecolor='steelblue',capsize=0,

                 yerr='errors', legend=False)

    

    # Set title & labels

    plt.title('Coefficients of Features w/ 95% Confidence Intervals',fontsize=30)

    ax.set_ylabel('Coefficients',fontsize=22)

    ax.set_xlabel('',fontsize=22)

    

    # Coefficients

    ax.scatter(x=pd.np.arange(coef_df.shape[0]), 

               marker='o', s=80, 

               y=coef_df['coef'], color='steelblue')

    

    # Line to define zero on the y-axis

    ax.axhline(y=0, linestyle='--', color='red', linewidth=1)

    

    return plt.show()
coefplot(results)
sns.scatterplot(all_data_5["Poverty"], all_data_5["Cases"])
import statsmodels.formula.api as smf

fit1 = smf.ols("Cases ~ Poverty", data=all_data_5).fit()



fit1.summary()
print(f"MSE: {fit1.mse_resid}")
sns.scatterplot(fit1.fittedvalues, fit1.resid)

plt.xlabel("Fitted Values")

plt.ylabel("Residuals")

plt.show()
fit_all = smf.ols("Cases ~ Poverty + Population + W_Male + W_Female + B_Male + B_Female + H_Male + H_Female + I_Male + I_Female + A_Male + A_Female + NH_Male + NH_Female + Risk_Index", data=all_data_5).fit()



fit_all.summary()
print(f"MSE: {fit_all.mse_resid}")
sns.scatterplot(fit_all.fittedvalues, fit_all.resid)

plt.xlabel("Fitted Values")

plt.ylabel("Residuals")

plt.show()
fit_cases_v1 = smf.ols("Cases ~ Poverty + W_Male + W_Female + B_Male + B_Female + H_Male + H_Female", data=all_data_5).fit()



fit_cases_v1.summary()
print(f"MSE: {fit_cases_v1.mse_resid}")
fit_cases_v2a = smf.ols("Cases ~ Poverty + B_Male + B_Female", data=all_data_5).fit()



fit_cases_v2a.summary()
print(f"MSE: {fit_cases_v2a.mse_resid}")
fit_cases_v2b = smf.ols("Cases ~ Poverty + H_Male + H_Female", data=all_data_5).fit()



fit_cases_v2b.summary()
print(f"MSE: {fit_cases_v2b.mse_resid}")
fit_cases_v3 = smf.ols("Cases ~ Poverty + W_Female + B_Female + H_Female + I_Female + A_Female + NH_Female", data=all_data_5).fit()



fit_cases_v3.summary()
print(f"MSE: {fit_cases_v3.mse_resid}")
sns.set(color_codes=True)





sns.lmplot(x="Risk_Index", y="Cases", data=all_data_5);
fit_cases_v6 = smf.ols("Cases ~ W_Female + B_Female + H_Female + I_Female + A_Female + NH_Female", data=all_data_5).fit()



fit_cases_v6.summary()
print(f"MSE: {fit_cases_v6.mse_resid}")
fit_deaths_v1 = smf.ols("Deaths ~ Poverty", data=all_data_5).fit()



fit_deaths_v1.summary()
fit_deaths_v2a = smf.ols("Deaths ~ B_Male + B_Female", data=all_data_5).fit()



fit_deaths_v2a.summary()
print(f"MSE: {fit_deaths_v2a.mse_resid}")
fit_deaths_v2b = smf.ols("Deaths ~ H_Male + H_Female", data=all_data_5).fit()



fit_deaths_v2b.summary()
print(f"MSE: {fit_deaths_v2b.mse_resid}")
fit_deaths_v3 = smf.ols("Deaths ~ Cases + Poverty", data=all_data_5).fit()



fit_deaths_v3.summary()
print(f"MSE: {fit_deaths_v3.mse_resid}")
fit_deaths_v4 = smf.ols("Deaths ~ Cases + Risk_Index", data=all_data_5).fit()



fit_deaths_v4.summary()
print(f"MSE: {fit_deaths_v4.mse_resid}")
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Data 

data = pd.read_csv('../input/covid-19-race-gender-poverty-risk-us-county/covid_data_log_200922.csv') 

df = pd.DataFrame(data)



# Fit Model & Output Regression Results Summary



# Import Package

import statsmodels.api as sm

from statsmodels.api import add_constant



# Build Model

X = data.loc[:,["Cases", "Poverty", "B_Male", "B_Female"]]

y = data.loc[:,["Deaths"]]



X = sm.add_constant(X)

model1 = sm.OLS(y,X)

results = model1.fit()



# MSE of the residuals

print(f"MSE: {results.mse_resid}")



# Output Results

results.summary()
coefplot(results)
import statsmodels.api as sm



fig = sm.graphics.plot_partregress_grid(results)
# Fit Model & Output Regression Results Summary



# Import Package

import statsmodels.api as sm

from statsmodels.api import add_constant



# Build Model

X = data.loc[:,["Cases", "Risk_Index", "B_Male", "B_Female"]]

y = data.loc[:,["Deaths"]]



X = sm.add_constant(X)

model2 = sm.OLS(y,X)

results = model2.fit()



# MSE of the residuals

print(f"MSE: {results.mse_resid}")



# Output Results

results.summary()
coefplot(results)
# Fit Model & Output Regression Results Summary



# Import Package

import statsmodels.api as sm

from statsmodels.api import add_constant



# Build Model

X = data.loc[:,["W_Female", "B_Female", "H_Female", "I_Female", "A_Female", "NH_Female"]]

# X = data.loc[:,["W_Female", "B_Female", "H_Female"]]

y = data.loc[:,["Deaths"]]



X = sm.add_constant(X)

model3 = sm.OLS(y,X)

results = model3.fit()



# MSE of the residuals

print(f"MSE: {results.mse_resid}")



# Output Results

results.summary()
coefplot(results)