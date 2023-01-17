import pandas as pd
import pulp
factories = pd.read_csv('../input/Cost_Analysis.csv', index_col=['Month', 'Factory'])
factories
demand = pd.read_csv('../input/Demand.csv', index_col=['Month'])
demand
production = pulp.LpVariable.dicts("production",
                                     ((month, factory) for month, factory in factories.index),
                                     lowBound=0,
                                     cat='Integer')
factory_status = pulp.LpVariable.dicts("factory_status",
                                     ((month, factory) for month, factory in factories.index),
                                     cat='Binary')
model = pulp.LpProblem("Cost minimising scheduling problem", pulp.LpMinimize)
print(model)
model += pulp.lpSum(
    [production[month, factory] * factories.loc[(month, factory), 'Variable_Costs'] for month, factory in factories.index]
    + [factory_status[month, factory] * factories.loc[(month, factory), 'Fixed_costs'] for month, factory in factories.index]
)
months = demand.index
for month in months:
    model += production[(month, 'A')] + production[(month, 'B')] == demand.loc[month, 'Demand']
for month, factory in factories.index:
    min_production = factories.loc[(month, factory), 'Min_Capacity']
    max_production = factories.loc[(month, factory), 'Max_Capacity']
    model += production[(month, factory)] >= min_production * factory_status[month, factory]
    model += production[(month, factory)] <= max_production * factory_status[month, factory]
model += factory_status[5, 'B'] == 0
model += production[5, 'B'] == 0
model.solve()
pulp.LpStatus[model.status]
output = []
for month, factory in production:
    var_output = {
        'Month': month,
        'Factory': factory,
        'Production': production[(month, factory)].varValue
        #'Factory Status': factory_status[(month, factory)].varValue
    }
    output.append(var_output)
output_df = pd.DataFrame.from_records(output).sort_values(['Month', 'Factory'])
output_df.set_index(['Month', 'Factory'], inplace=True)
#output_df.set_index(['Month', 'Factory'])
output_df
print (pulp.value(model.objective))
