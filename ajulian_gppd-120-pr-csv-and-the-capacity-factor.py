import pandas as pd

import numpy as np



source_capacity_factors = {"Coal": 0.55, "Hydro": 0.40, "Gas": 0.80, "Oil": 0.64, "Solar": 0.25, "Wind": 0.30, "Nuclear": 0.85}



# "force_fix": - if False, the source_capacity_factors dictionary values are applied only to the 

#                "estimated_generation_gwh" values whose Capacity Factor is > 1

#              - if True, all the "estimated_generation_gwh" values are fixed with the source_capacity_factors dictionary values

def fix_estimated_generation(gpp_df, source_capacity_factors, force_fix=False):

    gpp_df["capacity_factor"] = np.where(gpp_df["capacity_mw"] > 0, gpp_df["estimated_generation_gwh"] / (gpp_df["capacity_mw"]*24*365/1000), 0)

    for idx in range(gpp_df.shape[0]):

        if (gpp_df.loc[idx, 'capacity_factor'] > 1) or force_fix: 

            gpp_df.loc[idx, 'capacity_factor'] = source_capacity_factors[gpp_df.loc[idx, "primary_fuel"]]

            gpp_df.loc[idx, 'estimated_generation_gwh'] = gpp_df.loc[idx, "capacity_factor"] * gpp_df.loc[idx, "capacity_mw"] * 24*365/1000

    return gpp_df



# Uncomment next line if you have not read yet the Power Plants file

# global_power_plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

# global_power_plants = fix_estimated_generation(global_power_plants, source_capacity_factors)
import pandas as pd

global_power_plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

total_capacity_mw = global_power_plants['capacity_mw'].sum()

print('Total Installed Capacity: '+'{:.2f}'.format(total_capacity_mw) + ' MW')

capacity = (global_power_plants.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()

capacity = capacity.sort_values('capacity_mw',ascending=False)

capacity['percentage_of_total'] = (capacity['capacity_mw']/total_capacity_mw)*100

capacity['percentage_of_total'].plot(kind='bar',color=['orange', 'yellow', 'black', 'orange','cyan','blue'], 

                                     title="Capacity per fuel type (%)")
total_estimated_generation_gwh = global_power_plants['estimated_generation_gwh'].sum()

print('Total Estimated Generation per year: '+'{:.2f}'.format(total_estimated_generation_gwh) + ' GWh')

estimated_generation = (global_power_plants.groupby(['primary_fuel'])['estimated_generation_gwh'].sum()).to_frame()

estimated_generation = estimated_generation.sort_values('estimated_generation_gwh',ascending=False)

estimated_generation['percentage_of_total'] = (estimated_generation['estimated_generation_gwh']/total_estimated_generation_gwh)*100

estimated_generation['percentage_of_total'].plot(kind='bar',color=['orange', 'yellow', 'black', 'orange','cyan','blue'],

                                                title="Annual generation per fuel type (%)")
global_power_plants["capacity_factor"] = global_power_plants["estimated_generation_gwh"]/(global_power_plants["capacity_mw"]*24*365/1000)

global_power_plants[["name", "capacity_mw", "primary_fuel", "estimated_generation_gwh", "capacity_factor"]]
global_power_plants = fix_estimated_generation(global_power_plants, source_capacity_factors)

global_power_plants[["name", "capacity_mw", "primary_fuel", "estimated_generation_gwh", "capacity_factor"]]    
total_estimated_generation_gwh = global_power_plants['estimated_generation_gwh'].sum()

print('Total Estimated Generation per year: '+'{:.2f}'.format(total_estimated_generation_gwh) + ' GWh')

estimated_generation = (global_power_plants.groupby(['primary_fuel'])['estimated_generation_gwh'].sum()).to_frame()

estimated_generation = estimated_generation.sort_values('estimated_generation_gwh',ascending=False)

estimated_generation['percentage_of_total'] = (estimated_generation['estimated_generation_gwh']/total_estimated_generation_gwh)*100

estimated_generation['percentage_of_total'].plot(kind='bar',color=['orange', 'yellow', 'black', 'orange','cyan','blue'],

                                                title="Annual generation per fuel type (%)")