# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from pulp import * # Python package for Linear Programming
import re
import ast

import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/FPL_2018_19_Wk0.csv')
df = df[df['Points']>50] # For a first pass, only pick the players who scored >50 points last season
df.reset_index(inplace=True,drop=True)
df.head()
df.shape
# Create the decision variables.. all the players (so 371 for this initial try)
def create_dec_var(df):
    decision_variables = []
    
    for rownum, row in df.iterrows():
        variable = str('x' + str(rownum))
        variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat= 'Integer')
        decision_variables.append(variable)
                                  
    return decision_variables

# This is what we want to maximize (objective function)
def total_points(df,lst,prob):
    total_points = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                formula = row['Points']*player
                total_points += formula

    prob += total_points
    
    return prob

# Add constraint for cash
def cash(df,lst,prob,avail_cash):
    total_paid = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                formula = row['Cost']*player
                total_paid += formula
    prob += (total_paid <= avail_cash), "Cash"
    
    return prob

# Add constraint for number of goalkeepers
def team_gkp(df,lst,prob,avail_gk):
    total_gk = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'GKP':
                    formula = 1*player
                    total_gk += formula

    prob += (total_gk == avail_gk), "GK"
    
    return prob

# Add constraint for number of defenders
def team_def(df,lst,prob,avail_def):
    total_def = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'DEF':
                    formula = 1*player
                    total_def += formula

    prob += (total_def == avail_def), "DEF"
    
    return prob

# Add constraint for number of midfielders
def team_mid(df,lst,prob,avail_mid):
    total_mid = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'MID':
                    formula = 1*player
                    total_mid += formula

    prob += (total_mid == avail_mid), "MID"
    
    return prob

# Add constraint for number of forwards
def team_fwd(df,lst,prob,avail_fwd):
    total_fwd = ""
    for rownum, row in df.iterrows():
        for i, player in enumerate(lst):
            if rownum == i:
                if row['Position'] == 'FWD':
                    formula = 1*player
                    total_fwd += formula

    prob += (total_fwd == avail_fwd), "FWD"
    
    return prob
# Assemble the whole problem data
def find_prob(df,ca,gk,de,mi,fw):
    
    prob = pulp.LpProblem('FantasyTeam', pulp.LpMaximize)
    lst = create_dec_var(df)
    
    prob = total_points(df,lst,prob)
    prob = cash(df,lst,prob,ca)
    prob = team_gkp(df,lst,prob,gk)
    prob = team_def(df,lst,prob,de)
    prob = team_mid(df,lst,prob,mi)
    prob = team_fwd(df,lst,prob,fw)
    
    return prob
# Solve the problem
def LP_optimize(df, prob):
    prob.writeLP('FantasyTeam.lp')
    
    optimization_result = prob.solve()
    assert optimization_result == pulp.LpStatusOptimal

#     print("Status:", LpStatus[prob.status])
#     print("Optimal value:", pulp.value(prob.objective))
# Find the optimal team
def df_decision(df,prob):
    variable_name = []
    variable_value = []

    for v in prob.variables():
        variable_name.append(v.name)
        variable_value.append(v.varValue)

    df_vals = pd.DataFrame({'variable': variable_name, 'value': variable_value})
    for rownum, row in df_vals.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        df_vals.loc[rownum, 'variable'] = int(value[0])

    df_vals = df_vals.sort_index(by='variable')

    #append results
    for rownum, row in df.iterrows():
        for results_rownum, results_row in df_vals.iterrows():
            if rownum == results_row['variable']:
                df.loc[rownum, 'Decision'] = results_row['value']

    return df
prob = find_prob(df,1000,2,5,5,3)
LP_optimize(df,prob)
df_final = df_decision(df,prob)
print(df_final[df_final['Decision']==1.0].Cost.sum(), df_final[df_final['Decision']==1.0].Points.sum())
# The final 15
df_final[df_final['Decision']==1.0]
prob343 = find_prob(df,830,1,3,4,3)
prob352 = find_prob(df,830,1,3,5,2)
prob433 = find_prob(df,830,1,4,3,3)
prob442 = find_prob(df,830,1,4,4,2)
prob451 = find_prob(df,830,1,4,5,1)
prob532 = find_prob(df,830,1,5,3,2)
prob541 = find_prob(df,830,1,5,4,1)
def prob_formations(df,prob):
    LP_optimize(df,prob)
    df_final = df_decision(df,prob)
    
    print(df_final[df_final['Decision']==1.0]['Points'].sum())
    
    return(df_final[df_final['Decision']==1.0])
# 1. 3-4-3
prob_formations(df,prob343)
# 2. 3-5-2
prob_formations(df,prob352)
# 3. 4-3-3
prob_formations(df,prob433)
# 4. 4-4-2
prob_formations(df,prob442)
# 5. 4-5-1
prob_formations(df,prob451)
# 6. 5-3-2
prob_formations(df,prob532)
# 7. 5-4-1
prob_formations(df,prob541)
