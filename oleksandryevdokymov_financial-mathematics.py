!pip install numpy-financial
import numpy_financial as npf



initialInvestment_a = -600

initialInvestment_b = -1500

cashFlows_a = [initialInvestment_a, 650, 750]

cashFlows_b = [initialInvestment_b, 1200, 2300, 2700]



# Calculate the IRRs

irr_a = npf.irr(cashFlows_a)

irr_b = npf.irr(cashFlows_b)



print("a) Internal rate of return: {}", irr_a)

print("b) Internal rate of return: {}", irr_b)
initital_investment = -1900

cash_flows = [initital_investment, 700, 600, 500, 400]

discount_rates = [0, 0.01, 0.01, 0.01, 0.02]



def npv(cash_flows, discount_rates):

    NPV=0

    for i in range(len(cash_flows)):

        NPV+=cash_flows[i]/(1+discount_rates[i])**i

        print('NPV in {} period: {}'.format(i, NPV))

    return NPV



print('The result Net Present Value:', npv(cash_flows, discount_rates))
face_value = 500

purchase_value = 485

days_to_maturity = 91



discount_yield = (face_value - purchase_value)/face_value*360/days_to_maturity*100

annual_rate=discount_rate=discount_yield

print("Annual rate : {}%".format(annual_rate))
!pip install statsmodels
import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.sandbox.regression.predstd import wls_prediction_std



CPRI = pd.read_csv('../input/cpri-dataset/CPRI.csv', parse_dates=True, index_col='Date',)

sp_500 = pd.read_csv('../input/sp-500/GSPC.csv', parse_dates=True, index_col='Date',)



monthly_prices = pd.concat([CPRI['Close'], sp_500['Close']], axis=1)

monthly_prices.columns = ['CPRI', 'GSPC']

print(monthly_prices.head())



# calculate monthly returns

monthly_returns = monthly_prices.pct_change(1)

clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row

print(clean_monthly_returns.head())



# split dependent and independent variable

X = clean_monthly_returns['GSPC']

y = clean_monthly_returns['CPRI']



# Add a constant to the independent value

X1 = sm.add_constant(X)



# make regression model 

CAPM_model = sm.OLS(y, X1)



# fit model and print results

results = CAPM_model.fit()

print(results.summary())



fig, ax = plt.subplots(figsize=(8,6))



ax.grid(True)

ax.plot(X, y, 'o', label="data")

ax.plot(X, results.fittedvalues, 'b-', label="CAPM")

ax.legend(loc='best');