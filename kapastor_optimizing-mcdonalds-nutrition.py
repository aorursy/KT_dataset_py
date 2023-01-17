import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pulp import *
from tabulate import tabulate
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt # matplotlib
import os
import plotly.figure_factory as ff
cf.set_config_file(offline=False, world_readable=True, theme='pearl')
McData = pd.read_csv('../input/menu.csv')
# Clean the data:
McData.loc[McData['Item'].str.contains('Dr'),'Protein']=0
McData = McData[~(McData['Calories'] == 1880)]

# McData.drop(McData['Item'].str.contains('Bacon, Egg & Cheese Bagel with Egg Whites').index,inplace=True)

def make_scatter(McData,category,x_cat,y_cat):
    return  go.Scatter(
                    x = McData[McData['Category'].isin([category])][x_cat],
                    y = McData[McData['Category'].isin([category])][y_cat],
                    mode = "markers",
                    name = category,
                    text=  McData.Item)

# I want to see the fat/sugar scatter per category
x_cat = 'Calories'
y_cat = 'Carbohydrates'
data = [make_scatter(McData,cat,x_cat,y_cat) for cat in McData.Category.unique().tolist()]
layout = dict(title = '',
              xaxis= dict(title= 'Calories',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Carbohydrates(g)',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# I want to see the fat/sugar scatter per category
x_cat = 'Total Fat'
y_cat = 'Sodium'
data = [make_scatter(McData,cat,x_cat,y_cat) for cat in McData.Category.unique().tolist()]
layout = dict(title = '',
              xaxis= dict(title= 'Total Fat',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Sodium',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
MenuItems = McData.Item.tolist()
Calories = McData.set_index('Item')['Calories'].to_dict()
TotalFat = McData.set_index('Item')['Total Fat'].to_dict()
SaturatedFat = McData.set_index('Item')['Saturated Fat'].to_dict()
Carbohydrates = McData.set_index('Item')['Carbohydrates'].to_dict()
Sugars = McData.set_index('Item')['Sugars'].to_dict()
Protein = McData.set_index('Item')['Protein'].to_dict()
Sodium = McData.set_index('Item')['Sodium'].to_dict()

# Energy: 8,400kJ/2,000kcal
# Total fat: less than 70g
# Saturates: less than 20g
# Carbohydrate: at least 260g
# Total sugars: 90g
# Protein: 50g
# Salt: less than 6g


prob = LpProblem("McOptimization Problem", LpMinimize)
MenuItems_vars = LpVariable.dicts("MenuItems",MenuItems,lowBound=0, upBound=10,cat='Integer')
prob += lpSum([Calories[i]*MenuItems_vars[i] for i in MenuItems]), "Calories"
prob += lpSum([TotalFat[i]*MenuItems_vars[i] for i in MenuItems]) <= 70, "TotalFat"
prob += lpSum([SaturatedFat[i]*MenuItems_vars[i] for i in MenuItems]) <= 20, "Saturated Fat"
prob += lpSum([Carbohydrates[i]*MenuItems_vars[i] for i in MenuItems]) >= 260, "Carbohydrates_lower"
# prob += lpSum([Carbohydrates[i]*MenuItems_vars[i] for i in MenuItems]) <= 360, "Carbohydrates_upper"
prob += lpSum([Sugars[i]*MenuItems_vars[i] for i in MenuItems]) >= 80, "Sugars_lower"
prob += lpSum([Sugars[i]*MenuItems_vars[i] for i in MenuItems]) <= 100, "Sugars_upper"

prob += lpSum([Protein[i]*MenuItems_vars[i] for i in MenuItems]) >= 45, "Protein_lower"
prob += lpSum([Protein[i]*MenuItems_vars[i] for i in MenuItems]) <= 55, "Protein_upper"

prob += lpSum([Sodium[i]*MenuItems_vars[i] for i in MenuItems]) <= 6000, "Sodium"


prob.writeLP("McOptimization.lp")
prob.solve()
data_matrix = []
data_matrix.append(['Item', 'Amount','Calories','Total Fat','Carbohydrates','Protein','Sodium'])

print("Status:", LpStatus[prob.status])
for v in prob.variables():
    if v.varValue > 0:
        Item = McData.loc[McData['Item'] == v.name.replace('MenuItems_','').replace('_',' ')]
        Item_Calories = Item['Calories'].values*v.varValue
        Item_TotalFat = Item['Total Fat'].values*v.varValue
        Item_Carbohydrates = Item['Carbohydrates'].values*v.varValue
        Item_Protein = Item['Protein'].values*v.varValue
        Item_Sodium = Item['Sodium'].values*v.varValue


        data_matrix.append([v.name.replace('MenuItems_','').replace('_',' '),v.varValue,Item_Calories[0], \
                           Item_TotalFat[0],Item_Carbohydrates[0],Item_Protein[0],Item_Sodium[0]])


print(' ')
results = {}
print("Total Calories = ", value(prob.objective))
for constraint in prob.constraints:
    s = 0
    for var, coefficient in prob.constraints[constraint].items():
        sum += var.varValue * coefficient
    results[prob.constraints[constraint].name.replace('_lower','').replace('_upper','')] = s  




table = ff.create_table(data_matrix)
print(data_matrix)
iplot(table, filename='simple_table')
fig = go.Figure()
fig.add_trace(go.Bar(
    name='Nutrition',
    x=["TotalFat","Saturated Fat","Carbohydrates","Sugars","Protein","Sodium"], \
    y=[y_data["TotalFat"],y_data["Saturated_Fat"],y_data["Carbohydrates"],y_data["Sugars"],y_data["Protein"],y_data["Sodium"]/1000],
    error_y=dict(
            type='data',
            symmetric=False,
            array=[70-y_data["TotalFat"], 20-y_data["Saturated_Fat"], 0, 100-y_data["Sugars"],55-y_data["Protein"],2-y_data["Sodium"]/1000],
            arrayminus=[y_data["TotalFat"]-0,y_data["Saturated_Fat"]-0,y_data["Carbohydrates"]-260,y_data["Sugars"]-80,y_data["Protein"]-45,y_data["Sodium"]/1000-0])
))

fig.layout.update(barmode='group')
iplot(fig, filename='r')

