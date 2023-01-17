# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/nhl-salary-v2/NHL_Salary_Final_v2.xlsx')
df.head()

X = df.values[:, 1:176]

Y = df['Salary']
from sklearn.model_selection import train_test_split





x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)

x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, test_size = 0.1, random_state = 42)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_val.shape)

print(y_val.shape)

#We will do this if we decide to standardize our data



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)



x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

x_val = scaler.transform(x_val)
import statsmodels

import statsmodels.api as sm

import statsmodels.stats.api as sms

from statsmodels.formula.api import ols



formula = ''' Salary ~ Year + Month + Day + Pr_St_Val + Country_Val + Nat_Val + Ht + Wt + DftYr + DftRd + Ovrl + Left + Right + LW + RW + C + D + GP + G + A + A1 + A2 + PTS + Plus_Minus + E_Plus_Minus + PIM + Shifts + TOI + TOIX + TOI_GP + TOI% + IPP% + SH% + SV% + PDO + F_60 + A_60 + Pct% + Diff + Diff_60 + iCF + iFF + iSF + ixG + iSCF + iRB + iRS + iDS + sDist + sDist + Pass + iHF + iHF_2 + iHA + iHDf + iMiss + iGVA + iTKA + iBLK + iGVA + iTKA + iBLK + BLK% + iFOW + iFOL + FO% + %FOT + dzFOW + dzFOL + nzFOW + nzFOL + ozFOW + ozFOL + FOW.Up + FOL.Up + FOW.Down + FOL.Down + FOW.Close + FOL.Close + OTG + 1G + GWG + ENG + PSG + PSA + G.Bkhd + G.Dflct + G.Slap + G.Snap + G.Tip + G.Wrap + G.Wrst + CBar + Post + Over + Wide + S.Bkhd + S.Dflct + S.Slap + S.Snap + S.Tip + S.Wrap + S.Wrst + iPenT + iPenD + iPENT_2 + iPEND_2 + iPenDf + NPD + Mins + Maj + Match + Misc + Game + CF + CA + FF + FA + SF + SA + xGF + xGA + SCF + SCA + GF + GA + RBF + RBA + RSF + RSA + DSF + DSA + FOW + FOL + HF + HA + GVA + TKA + PENT + PEND + PS + DPS + PS + OTOI + Grit + DAP + Pace + GS + GS_G + ANA + ARI + BOS + BUF + CAR + CBJ + CGY + CHI + COL + DAL + DET + EDM + FLA + L.A + MIN + MTL + N.J + NSH + NYI + NYR + OTT + PHI + PIT + S.J + STL + T.B + TOR + VAN + WPG + WSH

'''



X = sm.add_constant(x_train) # adding a constant

 

model = sm.OLS(y_train, X)



results = model.fit()

 

print_model = results.summary()

print(print_model)
import matplotlib.pyplot as plt

import seaborn as sns

corr = df.corr(method ='pearson') 

plt.figure(figsize=(15, 10))

sns.heatmap(corr)

plt.show()
df2 = df[['Salary','Year','Wt','DftYr','RW','G','A1','A2','PTS','TOI%','iRB','iRS','iDS',

'FOW.Up','FOL.Up','S.Bkhd','Game','xGF','RBA','FOL','GS_G','COL','DAL','PIT','TOR',]]



df2
X = df2.values[:, 1:25]

Y = df2['Salary']
x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)

x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, test_size = 0.1, random_state = 42)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_val.shape)

print(y_val.shape)
X = sm.add_constant(x_train) # adding a constant

 

model2 = sm.OLS(y_train, X)



results2 = model2.fit()

 

print_model2 = results2.summary()

print(print_model2)
df3 = df[['Salary','Year','DftYr','RW','G','A1','A2','PTS','TOI%',

'FOW.Up','S.Bkhd','TOR']]



df3
X = df3.values[:, 1:12]

Y = df3['Salary']
x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)

x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, test_size = 0.1, random_state = 42)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_val.shape)

print(y_val.shape)
X = sm.add_constant(x_train) # adding a constant

 

model3 = sm.OLS(y_train, X)



results3 = model3.fit()

 

print_model3 = results3.summary()

print(print_model3)
from patsy import dmatrices

from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

from statsmodels.stats.outliers_influence import variance_inflation_factor



X = df3[['Salary','Year','DftYr','RW','G','A1','A2','PTS','TOI%','FOW.Up','S.Bkhd','TOR']]

X['Intercept'] = 1





vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['Variables'] = X.columns
vif.round(1)
df4 = df[['Salary','Year','DftYr', 'RW', 'TOI%','FOW.Up','S.Bkhd','TOR']]



df4
X = df4.values[:, 1:8]

Y = df4['Salary']
x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)

x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, test_size = 0.1, random_state = 42)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_val.shape)

print(y_val.shape)
X = sm.add_constant(x_train) # adding a constant

 

model4 = sm.OLS(y_train, X)



results4 = model4.fit()

 

print_model4 = results4.summary()

print(print_model4)
X = df4[['Salary','Year','DftYr', 'RW', 'TOI%','FOW.Up','S.Bkhd','TOR']]

X['Intercept'] = 1





vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['Variables'] = X.columns



vif.round(1)
X = sm.add_constant(x_test) # adding a constant

 

model5 = sm.OLS(y_test, X)



results5 = model5.fit()

 

print_model5 = results5.summary()

print(print_model5)
df5 = df[['Salary','Year','TOI%','FOW.Up']]

X = df5.values[:, 1:4]

Y = df5['Salary']
x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)

x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, test_size = 0.1, random_state = 42)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_val.shape)

print(y_val.shape)
X = sm.add_constant(x_train) # adding a constant

 

model5 = sm.OLS(y_train, X)



results5 = model5.fit()

 

print_model5 = results5.summary()

print(print_model5)
#Let's run against our test dataset

X = sm.add_constant(x_test) # adding a constant

 

model5 = sm.OLS(y_test, X)



results5 = model5.fit()

 

print_model5 = results5.summary()

print(print_model5)
X = sm.add_constant(x_val) # adding a constant

 

model5 = sm.OLS(y_val, X)



results5 = model5.fit()

 

print_model5 = results5.summary()

print(print_model5)