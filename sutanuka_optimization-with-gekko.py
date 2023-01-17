import pandas as pd

import numpy as np
#Get Quarterly Historical Returns

quarterly_hist_rets = pd.read_excel("/kaggle/input/irg-datasets/Alts_Quarterly.xlsx", sheet_name="Historic Returns", header=0, index_col=0, parse_dates=True)/100

quarterly_hist_rets.index = pd.to_datetime(quarterly_hist_rets.index, format="%Y%m").to_period('Q')

quarterly_hist_rets.columns = quarterly_hist_rets.columns.str.strip()

#Get Quarterly Equilibrium Weights

quarterly_mktweights = pd.read_excel("/kaggle/input/irg-datasets/Alts_Quarterly.xlsx", sheet_name="Weights", header=0)
#Get HNW Views matrix

hnw_picks = pd.read_excel("/kaggle/input/irg-datasets/hnw_views.xlsx", sheet_name="picks", index_col=0, header=0)

hnw_views = pd.read_excel("/kaggle/input/irg-datasets/hnw_views.xlsx", sheet_name="views", header=0)

hnw_alpha = pd.read_excel("/kaggle/input/irg-datasets/hnw_views.xlsx", sheet_name="alpha", header=0)

#Cash Return (Not included in Black-Litterman calculations, set it here manually)

cash = pd.DataFrame([.0125])

cash.index = ["Cash"]



#Set Minimum, Maximum risk, Number of Trials for Efficient Frontier. This allows for same point comparison between different frontiers.

min_risk = .04

max_risk = .18

num_trials = 100

target_risks = np.linspace(min_risk, max_risk, num_trials)



#Set Black Litterman Inputs

rf = .0225

mrp = .03406
# Identify the Black Litterman Returns for the High Net Worth Asset Classes (excluding Cash)

# Set the Asset Classes to include in BL

# Based on Quarterly Returns

gaac_hnw =["US Gov",

              "US Securitized",

              "US Invest Grade",

              "US High Yield",

              "Foreign Dev Bond",

              "Foreign EM Bond",

              "US Large Value",

              "US Large Growth",

              "US Mid Value",

              "US Mid Growth",

              "US Small Value",

              "US Small Growth",

              "Foreign Dev Equity",

              "Foreign EM Equity",

              "Commodities",

              "Precious Metals",

              "Equity Alternatives",

              "Fixed Income Alternatives",

              "Global Macro Alternatives",

              "Private REIT",

              "Private Equity",

              "Private Credit",

          ]



# Pull the covariance and weights that correspond to the asset classes selected above

quarterly_cov_hnw = quarterly_hist_rets[gaac_hnw].cov()*4

quarterly_weights_hnw = (quarterly_mktweights[gaac_hnw].T/quarterly_mktweights[gaac_hnw].sum(axis=1)).T

# Calculate Black Litterman 

def black_litterman(cov, weights, rf=.031, tau=1, mrp=.033):

    """

    Calculate the Black-Litterman implied returns given the following:

    - Covariance matrix

    - Equilibrium Weights

    - Risk Free Rate

    - Tau

    - Risk Premium

    """

    portfolio_deviation=((weights.dot(cov)).dot(weights.T))**.5

    bl_lambda = (mrp/(portfolio_deviation**2))

    bl_implied=cov.dot(weights.T)*bl_lambda.iloc[0].item()+rf

    return bl_implied



bl_quarterly_hnw = black_litterman(quarterly_cov_hnw, quarterly_weights_hnw, rf=rf, mrp=mrp)

print(type(bl_quarterly_hnw))

bl_quarterly_hnw.shape

# Append forward cash return to the dataframe

bl_quarterly_hnw = cash.append(bl_quarterly_hnw)

print(bl_quarterly_hnw.shape)

print(bl_quarterly_hnw)



## Calculate covariance matrix that includes cash

bl_quarterly_hnw_cov = quarterly_hist_rets[bl_quarterly_hnw.index.values].cov()*4
!pip install gekko
from gekko import GEKKO
bl_quarterly_hnw_cov_ar = np.array(bl_quarterly_hnw_cov)

bl_quarterly_hnw_ar = np.array(bl_quarterly_hnw)
init_guess = np.repeat(1/bl_quarterly_hnw.shape[0], bl_quarterly_hnw.shape[0])

C = bl_quarterly_hnw_cov_ar

R = bl_quarterly_hnw_ar

R = np.reshape(R,23)

R.shape

init_guess


# Initialize Model

m = GEKKO(remote=True)



#initialize variables

w=m.Array(m.Var,23)





for i in range(23):

    w[i].value = 0.04347826

    w[i].lower = 0.0

    w[i].upper = 1.0





#Equations

Cw = [m.sum([m.Intermediate(C[i,j]*w[j]) for j in range(23)]) for i in range(23)]

wCw = m.sum([Cw[i]*w[i] for i in range(23)])

m.Equation(0.18 >= (wCw)**0.5)

m.Equation(m.sum(w[0:23]) == 1.0)

m.Equation(w[7]+w[9]+w[11] >= (m.sum(w[7:13])*0.3))

m.Equation(w[7]+w[9]+w[11] <= (m.sum(w[7:13])*0.7))

m.Equation(w[11]+w[12] >= 0.0)

m.Equation(w[11]+w[12] <= (m.sum(w[7:13])*0.15))





#Objective

# m.Maximize(m.sum([w[i]*R[i] for i in range(23)]))

for i in range(23):

    m.Maximize(w[i]*R[i])



#Solve simulation

m.solve() # solve on public server





#Results

for i in range(23):

    print('w['+str(i)+']='+str(w[i].value))


# Initialize Model

m = GEKKO(remote=True)



#initialize variables

w=m.Array(m.Var,23)





for i in range(23):

    w[i].value = 0.04347826

    w[i].lower = 0.0

    w[i].upper = 1.0





#Equations

C = np.random.rand(23,23)

Cw = [m.sum([m.Intermediate(C[i,j]*w[j]) for j in range(23)]) for i in range(23)]

wCw = m.sum([Cw[i]*w[i] for i in range(23)])

m.Equation(0.18 >= (wCw)**0.5)

m.Equation(m.sum(w[0:23]) == 1.0)

m.Equation(w[7]+w[9]+w[11] >= (m.sum(w[7:13])*0.3))

m.Equation(w[7]+w[9]+w[11] <= (m.sum(w[7:13])*0.7))

m.Equation(w[11]+w[12] >= 0.0)

m.Equation(w[11]+w[12] <= (m.sum(w[7:13])*0.15))





#Objective

# m.Maximize(m.sum([w[i]*R[i] for i in range(23)]))

R = np.ones(23)

for i in range(23):

    m.Maximize(w[i]*R[i])



#Solve simulation

m.solve() # solve on public server





#Results

for i in range(23):

    print('w['+str(i)+']='+str(w[i].value))