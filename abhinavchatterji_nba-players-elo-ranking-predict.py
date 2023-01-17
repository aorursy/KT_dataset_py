import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
from sklearn.model_selection  import train_test_split
from sklearn import datasets, linear_model, metrics 
import numpy as np
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
attendance_valuation_elo_merge_salary_df = pd.merge(attendance_valuation_elo_df,salary_df, how = 'inner', on = ['TEAM'])
attendance_valuation_elo_merge_salary_df.rename(columns = {'NAME':'PLAYER'}, inplace = True)
attendance_valuation_elo_merge_salary_df.head(5)
attendance_valuation_elo_merge_salary_merge_pie_df = pd.merge(attendance_valuation_elo_merge_salary_df,pie_df, how = 'inner', on = ['PLAYER']) 
attendance_valuation_elo_merge_salary_merge_pie_df.head(5)
plt.subplots(figsize=(20,15))
sns.heatmap(attendance_valuation_elo_merge_salary_merge_pie_df.iloc[3:,3:].corr())
Y = attendance_valuation_elo_merge_salary_merge_pie_df[['ELO']].values
X = attendance_valuation_elo_merge_salary_merge_pie_df[['SALARY','AGE','GP','W','MIN','OFFRTG',
                                                       'DEFRTG','NETRTG','AST%','AST/TO',
                                                       'OREB%','DREB%','REB%','EFG%',
                                                       'TS%','USG%','PACE','PIE']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, 
                                                    random_state=1) 
reg = linear_model.LinearRegression()
model = reg.fit(X_train, Y_train)
reg.coef_
reg.score(X_test, Y_test)
plt.style.use('fivethirtyeight') 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - Y_train, 
            color = "green", s = 10, label = 'Train data') 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - Y_test, 
            color = "blue", s = 10, label = 'Test data') 
plt.hlines(y = 0, xmin = 1200, xmax = 2000, linewidth = 2) 

plt.legend(loc = 'upper right') 
plt.xlim((1200,2000))

plt.title("Residual errors") 
  

plt.show() 
lss = ['SALARY','AGE','GP','W','MIN','OFFRTG','DEFRTG','NETRTG','AST%','AST/TO','OREB%','DREB%','REB%','EFG%','TS%','USG%','PACE','PIE']
coeffs = np.append(lss, reg.coef_)
Betas = pd.DataFrame(coeffs.reshape(2,18)).transpose()
Betas.columns = ['Factors','Coeff']
Betas['Coeff'] = pd.to_numeric(Betas['Coeff'])
Betas.plot(kind='bar')
plt.xlabel(Betas[['Factors']])
