#imports 

import pandas as pd

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#load datasets

chilled = pd.read_csv("../input/bdgp2-further-cleaned-datasets/chilled_water_cleaned.csv")



electricity = pd.read_csv("../input/bdgp2-further-cleaned-datasets/electricity_cleaned_new.csv")



gas = pd.read_csv("../input/bdgp2-further-cleaned-datasets/gas_cleaned_new.csv")



hot = pd.read_csv("../input/bdgp2-further-cleaned-datasets/hot_water_cleaned.csv")



irrigation = pd.read_csv("../input/bdgp2-further-cleaned-datasets/interpolated_propogated_irrigation.csv")



solar = pd.read_csv("../input/bdgp2-further-cleaned-datasets/solar_cleaned2.csv")



steam = pd.read_csv("../input/bdgp2-further-cleaned-datasets/steam_cleaned2.csv")



water = pd.read_csv("../input/bdgp2-further-cleaned-datasets/water_cleaned.csv")
#irrigation cleaned dataset left in extra column with no usable data 

irrigation = irrigation.drop(irrigation.columns[0], axis=1)
#current timestamp is in objects for dataframe

#change timestamp object into DateTime format 

chilled["timestamp"] = pd.to_datetime(chilled["timestamp"], format = "%Y-%m-%d %H:%M:%S")



electricity["timestamp"] = pd.to_datetime(electricity["timestamp"], format = "%Y-%m-%d %H:%M:%S")



gas["timestamp"] = pd.to_datetime(gas["timestamp"], format = "%Y-%m-%d %H:%M:%S")



hot["timestamp"] = pd.to_datetime(hot["timestamp"], format = "%Y-%m-%d %H:%M:%S")



irrigation["timestamp"] = pd.to_datetime(irrigation["timestamp"], format = "%Y-%m-%d %H:%M:%S")



solar["timestamp"] = pd.to_datetime(solar["timestamp"], format = "%Y-%m-%d %H:%M:%S")



steam["timestamp"] = pd.to_datetime(steam["timestamp"], format = "%Y-%m-%d %H:%M:%S")



water["timestamp"] = pd.to_datetime(water["timestamp"], format = "%Y-%m-%d %H:%M:%S")
#change index to timestamp

chilled = chilled.set_index("timestamp")



electricity = electricity.set_index("timestamp")



gas = gas.set_index("timestamp")



hot = hot.set_index("timestamp")



irrigation = irrigation.set_index("timestamp")



solar = solar.set_index("timestamp")



steam = steam.set_index("timestamp")



water = water.set_index("timestamp")
#resample time series

#current time stamp is recorded for each hour over a two year period 

#want to average of the week

chilled =  chilled.resample("W").mean()



electricity = electricity.resample("W").mean()



gas = gas.resample("W").mean()



hot = hot.resample("W").mean()



irrigation = irrigation.resample("W").mean()



solar = solar.resample("W").mean()



steam = steam.resample("W").mean()



water = water.resample("W").mean()
#separate each meter dataframe by location 
#chilled 

peacock_chilled = pd.DataFrame()

P = [col for col in chilled.columns if 'Peacock' in col]

peacock_chilled[P] = chilled[P]



moose_chilled = pd.DataFrame()

M = [col for col in chilled.columns if 'Moose' in col]

moose_chilled[M] = chilled[M]

 

bull_chilled = pd.DataFrame()

B = [col for col in chilled.columns if 'Bull' in col]

bull_chilled[B] = chilled[B]



hog_chilled = pd.DataFrame()

H = [col for col in chilled.columns if 'Hog' in col]

hog_chilled[H] = chilled[H]



eagle_chilled = pd.DataFrame()

E = [col for col in chilled.columns if 'Eagle' in col]

eagle_chilled[E] = chilled[E]



cockatoo_chilled = pd.DataFrame()

C = [col for col in chilled.columns if 'Cockatoo' in col]

cockatoo_chilled[C] = chilled[C]



panther_chilled = pd.DataFrame()

pan = [col for col in chilled.columns if 'Panther' in col]

panther_chilled[pan] = chilled[pan]



fox_chilled = pd.DataFrame()

F = [col for col in chilled.columns if 'Fox' in col]

fox_chilled[F] = chilled[F]



bobcat_chilled = pd.DataFrame()

bob = [col for col in chilled.columns if 'Bobcat' in col]

bobcat_chilled[bob] = chilled[bob]



crow_chilled = pd.DataFrame()

cr = [col for col in chilled.columns if 'Crow' in col]

crow_chilled[cr] = chilled[cr]



sites_chilled = [peacock_chilled, moose_chilled, bull_chilled, hog_chilled, eagle_chilled, 

                 cockatoo_chilled, panther_chilled, fox_chilled, bobcat_chilled, crow_chilled]
#electricity

panther_electricity = pd.DataFrame()

P = [col for col in electricity.columns if 'Panther' in col]

panther_electricity[P] = electricity[P]



robin_electricity = pd.DataFrame()

R = [col for col in electricity.columns if 'Robin' in col]

robin_electricity[R] = electricity[R]



fox_electricity = pd.DataFrame()

F = [col for col in electricity.columns if 'Fox' in col]

fox_electricity[F] = electricity[F]



rat_electricity = pd.DataFrame()

R = [col for col in electricity.columns if 'Rat' in col]

rat_electricity[R] = electricity[R]



bear_electricity = pd.DataFrame()

B = [col for col in electricity.columns if 'Bear' in col]

bear_electricity[B] = electricity[B]



lamb_electricity = pd.DataFrame()

L = [col for col in electricity.columns if 'Lamb' in col]

lamb_electricity[L] = electricity[L]



peacock_electricity = pd.DataFrame()

p = [col for col in electricity.columns if 'Peacock' in col]

peacock_electricity[p] = electricity[p]



moose_electricity = pd.DataFrame()

M = [col for col in electricity.columns if 'Moose' in col]

moose_electricity[M] = electricity[M]



gator_electricity = pd.DataFrame()

G = [col for col in electricity.columns if 'Gator' in col]

gator_electricity[G] = electricity[G]



bull_electricity = pd.DataFrame()

B = [col for col in electricity.columns if 'Bull' in col]

bull_electricity[B] = electricity[B]



bobcat_electricity = pd.DataFrame()

b = [col for col in electricity.columns if 'Bobcat' in col]

bobcat_electricity[b] = electricity[b]



crow_electricity = pd.DataFrame()

cr = [col for col in electricity.columns if 'Crow' in col]

crow_electricity[cr] = electricity[cr]



shrew_electricity = pd.DataFrame()

S = [col for col in electricity.columns if 'Shrew' in col]

shrew_electricity[S] = electricity[S]



wolf_electricity = pd.DataFrame()

W = [col for col in electricity.columns if 'Wolf' in col]

wolf_electricity[W] = electricity[W]



hog_electricity = pd.DataFrame()

H = [col for col in electricity.columns if 'Hog' in col]

hog_electricity[H] = electricity[H]



eagle_electricity = pd.DataFrame()

E = [col for col in electricity.columns if 'Eagle' in col]

eagle_electricity[E] = electricity[E]



cockatoo_electricity = pd.DataFrame()

C = [col for col in electricity.columns if 'Cockatoo' in col]

cockatoo_electricity[C] = electricity[C]



mouse_electricity = pd.DataFrame()

M = [col for col in electricity.columns if 'Mouse' in col]

mouse_electricity[M] = electricity[M]



sites_electricity = [panther_electricity, robin_electricity, fox_electricity, rat_electricity, bear_electricity, 

                     lamb_electricity, peacock_electricity, moose_electricity, gator_electricity, bull_electricity,

                     bobcat_electricity, crow_electricity, shrew_electricity, wolf_electricity, hog_electricity,

                     eagle_electricity, cockatoo_electricity, mouse_electricity]
#gas 

panther_gas = pd.DataFrame()

P = [col for col in gas.columns if 'Panther' in col]

panther_gas[P] = gas[P]



lamb_gas = pd.DataFrame()

L = [col for col in gas.columns if 'Lamb' in col]

lamb_gas[L] = gas[L]



bobcat_gas = pd.DataFrame()

b = [col for col in gas.columns if 'Bobcat' in col]

bobcat_gas[b] = gas[b]



shrew_gas = pd.DataFrame()

S = [col for col in gas.columns if 'Shrew' in col]

shrew_gas[S] = gas[S]



wolf_gas = pd.DataFrame()

W = [col for col in gas.columns if 'Wolf' in col]

wolf_gas[W] = gas[W]



sites_gas = [panther_gas, lamb_gas, bobcat_gas, shrew_gas, wolf_gas]
#hot

moose_hot = pd.DataFrame()

M = [col for col in hot.columns if 'Moose' in col]

moose_hot[M] = hot[M]



eagle_hot = pd.DataFrame()

E = [col for col in hot.columns if 'Eagle' in col]

eagle_hot[E] = hot[E]



cockatoo_hot = pd.DataFrame()

C = [col for col in hot.columns if 'Cockatoo' in col]

cockatoo_hot[C] = hot[C]



fox_hot = pd.DataFrame()

f = [col for col in hot.columns if 'Fox' in col]

fox_hot[f] = hot[f]



bobcat_hot = pd.DataFrame()

bob = [col for col in hot.columns if 'Bobcat' in col]

bobcat_hot[bob] = hot[bob]



crow_hot = pd.DataFrame()

cr = [col for col in hot.columns if 'Crow' in col]

crow_hot[cr] = hot[cr]



robin_hot = pd.DataFrame()

R = [col for col in hot.columns if 'Robin' in col]

robin_hot[R] = hot[R]



sites_hot = [moose_hot, eagle_hot, cockatoo_hot, robin_hot, fox_hot, bobcat_hot, crow_hot]
#irrigation

panther_irrigation = pd.DataFrame()

P = [col for col in irrigation.columns if 'Panther' in col]

panther_irrigation[P] = irrigation[P]



sites_irrigation = [panther_irrigation]
#solar 

bobcat_solar = pd.DataFrame()

bob = [col for col in solar.columns if 'Bobcat' in col]

bobcat_solar[bob] = solar[bob]



sites_solar = [bobcat_solar]
#steam

peacock_steam = pd.DataFrame()

P = [col for col in steam.columns if 'Peacock' in col]

peacock_steam[P] = steam[P]



moose_steam = pd.DataFrame()

M = [col for col in steam.columns if 'Moose' in col]

moose_steam[M] = steam[M]

 

bull_steam = pd.DataFrame()

B = [col for col in steam.columns if 'Bull' in col]

bull_steam[B] = steam[B]



hog_steam = pd.DataFrame()

H = [col for col in steam.columns if 'Hog' in col]

hog_steam[H] = steam[H]



eagle_steam = pd.DataFrame()

E = [col for col in steam.columns if 'Eagle' in col]

eagle_steam[E] = steam[E]



cockatoo_steam = pd.DataFrame()

C = [col for col in steam.columns if 'Cockatoo' in col]

cockatoo_steam[C] = steam[C]



sites_steam = [peacock_steam, moose_steam, bull_steam, hog_steam, eagle_steam, cockatoo_steam]
#water

panther_water = pd.DataFrame()

P = [col for col in water.columns if 'Panther' in col]

panther_water[P] = water[P]



bobcat_water = pd.DataFrame()

bob = [col for col in water.columns if 'Bobcat' in col]

bobcat_water[bob] = water[bob]



wolf_water = pd.DataFrame()

W = [col for col in water.columns if 'Wolf' in col]

wolf_water[W] = water[W]



sites_water = [panther_water, bobcat_water, wolf_water]
#Sum up the uage of each location per week for each type of usage and create new column 
#chilled 

for site in sites_chilled:

    name = site.columns[0].split("_")[0]

    site["{}_chilled_sum".format(name)] = site.sum(axis = 1)
#electricity

for site in sites_electricity:

    name = site.columns[0].split("_")[0]

    site["{}_electricity_sum".format(name)] = site.sum(axis = 1)
#gas

for site in sites_gas:

    name = site.columns[0].split("_")[0]

    site["{}_gas_sum".format(name)] = site.sum(axis = 1)
#hot

for site in sites_hot:

    name = site.columns[0].split("_")[0]

    site["{}_hot_sum".format(name)] = site.sum(axis = 1)
#irrigation

for site in sites_irrigation:

    name = site.columns[0].split("_")[0]

    site["{}_irrigation_sum".format(name)] = site.sum(axis = 1)
#solar

for site in sites_solar:

    name = site.columns[0].split("_")[0]

    site["{}_solar_sum".format(name)] = site.sum(axis = 1)
#steam

for site in sites_steam:

    name = site.columns[0].split("_")[0]

    site["{}_steam_sum".format(name)] = site.sum(axis = 1)
#water 

for site in sites_water:

    name = site.columns[0].split("_")[0]

    site["{}_water_sum".format(name)] = site.sum(axis = 1)
#list of all sites 



sites_all = [peacock_chilled, moose_chilled, bull_chilled, hog_chilled, eagle_chilled, 

                   cockatoo_chilled, panther_chilled, fox_chilled, bobcat_chilled, crow_chilled,

                   panther_electricity, robin_electricity, fox_electricity, rat_electricity, bear_electricity, 

                   lamb_electricity, peacock_electricity, moose_electricity, gator_electricity, bull_electricity,

                   bobcat_electricity, crow_electricity, shrew_electricity, wolf_electricity, hog_electricity,

                   eagle_electricity, cockatoo_electricity, mouse_electricity, panther_gas, lamb_gas, bobcat_gas, 

                   shrew_gas, wolf_gas, moose_hot, eagle_hot, cockatoo_hot, robin_hot, fox_hot, bobcat_hot, crow_hot,

                   panther_irrigation, bobcat_solar, peacock_steam, moose_steam, bull_steam, hog_steam, eagle_steam, 

                   cockatoo_steam,panther_water, bobcat_water, wolf_water

                  ]
#create new dataframe with timestamp column

peacock_chilled = peacock_chilled.reset_index()

site_usage_all = peacock_chilled[["timestamp"]].copy()

site_usage_all = site_usage_all.set_index("timestamp")
site_usage_all.head()
#set index back to timestamp 

peacock_chilled = peacock_chilled.set_index("timestamp")
#chilled 

#function to add the sum column to new dataframe

for df in sites_chilled:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_chilled_sum"%(animal)] = df["%s_chilled_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
#electricity

#function to add the sum column to new dataframe

for df in sites_electricity:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_electricity_sum"%(animal)] = df["%s_electricity_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
#gas

#function to add the sum column to new dataframe

for df in sites_gas:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_gas_sum"%(animal)] = df["%s_gas_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
#hot

#function to add the sum column to new dataframe

for df in sites_hot:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_hot_sum"%(animal)] = df["%s_hot_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
#irrigation

#function to add the sum column to new dataframe

for df in sites_irrigation:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_irrigation_sum"%(animal)] = df["%s_irrigation_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
#solar

#function to add the sum column to new dataframe

for df in sites_solar:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_solar_sum"%(animal)] = df["%s_solar_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
#steam

#function to add the sum column to new dataframe

for df in sites_steam:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_steam_sum"%(animal)] = df["%s_steam_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
#water

#function to add the sum column to new dataframe

for df in sites_water:

    animal = df.columns[0].split("_")[0]

    site_usage_all["%s_water_sum"%(animal)] = df["%s_water_sum"%(animal)]

    

#make sure the only columns left contain the chilled sum

columns_to_keep = [x for x in site_usage_all.columns if '_sum' in x]

site_usage_all = site_usage_all[columns_to_keep]
site_usage_all.head()
site_usage_all.head()
site_usage_all.shape
cols = list(site_usage_all.columns)

panther = [x for x in cols if x.startswith('Panther')]

panther_df = site_usage_all[panther]
panther_df.head()
cols = list(site_usage_all.columns)

moose = [x for x in cols if x.startswith('Moose')]

moose_df = site_usage_all[moose]
moose_df.head()
cols = list(site_usage_all.columns)

eagle = [x for x in cols if x.startswith('Eagle')]

eagle_df = site_usage_all[eagle]
eagle_df.head()
cols = list(site_usage_all.columns)

cockatoo = [x for x in cols if x.startswith('Cockatoo')]

cockatoo_df = site_usage_all[cockatoo]
cockatoo_df.head()
cols = list(site_usage_all.columns)

fox = [x for x in cols if x.startswith('Fox')]

fox_df = site_usage_all[fox]
fox_df.head()
cols = list(site_usage_all.columns)

peacock = [x for x in cols if x.startswith('Peacock')]

peacock_df = site_usage_all[peacock]
peacock_df.head()
cols = list(site_usage_all.columns)

bull = [x for x in cols if x.startswith('Bull')]

bull_df = site_usage_all[bull]
bull_df.head()
cols = list(site_usage_all.columns)

hog = [x for x in cols if x.startswith('Hog')]

hog_df = site_usage_all[hog]
hog_df.head()
cols = list(site_usage_all.columns)

crow = [x for x in cols if x.startswith('Crow')]

crow_df = site_usage_all[crow]
crow_df.head()
cols = list(site_usage_all.columns)

bobcat = [x for x in cols if x.startswith('Bobcat')]

bobcat_df = site_usage_all[bobcat]
bobcat_df.head()
cols = list(site_usage_all.columns)

robin = [x for x in cols if x.startswith('Robin')]

robin_df = site_usage_all[robin]
robin_df.head()
cols = list(site_usage_all.columns)

bear = [x for x in cols if x.startswith('Bear')]

bear_df = site_usage_all[bear]
bear_df.head()
cols = list(site_usage_all.columns)

lamb = [x for x in cols if x.startswith('Lamb')]

lamb_df = site_usage_all[lamb]
lamb_df.head()
cols = list(site_usage_all.columns)

rat = [x for x in cols if x.startswith('Rat')]

rat_df = site_usage_all[rat]
rat_df.head()
cols = list(site_usage_all.columns)

gator = [x for x in cols if x.startswith('Gator')]

gator_df = site_usage_all[gator]
gator_df.head()
cols = list(site_usage_all.columns)

wolf = [x for x in cols if x.startswith('Wolf')]

wolf_df = site_usage_all[wolf]
wolf_df.head()
cols = list(site_usage_all.columns)

shrew = [x for x in cols if x.startswith('Shrew')]

shrew_df = site_usage_all[shrew]
shrew_df.head()
cols = list(site_usage_all.columns)

swan = [x for x in cols if x.startswith('Swan')]

swan_df = site_usage_all[swan]
swan_df.head()
cols = list(site_usage_all.columns)

mouse = [x for x in cols if x.startswith('Mouse')]

mouse_df = site_usage_all[mouse]
mouse_df.head()
panther_df.to_csv("panther_meter_sums.csv")

moose_df.to_csv("moose_meter_sums.csv")

eagle_df.to_csv("eagle_meter_sums.csv")

cockatoo_df.to_csv("cockatoo_meter_sums.csv")

fox_df.to_csv("fox_meter_sums.csv")

peacock_df.to_csv("peacock_meter_sums.csv")

bull_df.to_csv("bull_meter_sums.csv")

hog_df.to_csv("hog_meter_sums.csv")

crow_df.to_csv("crow_meter_sums.csv")

bobcat_df.to_csv("bobcat_meter_sums.csv")

robin_df.to_csv("robin_meter_sums.csv")

bear_df.to_csv("bear_meter_sums.csv")

lamb_df.to_csv("lamb_meter_sums.csv")

rat_df.to_csv("rat_meter_sums.csv")

gator_df.to_csv("gator_meter_sums.csv")

wolf_df.to_csv("wolf_meter_sums.csv")

shrew_df.to_csv("shrew_meter_sums.csv")

#nothing in swan

mouse_df.to_csv("mouse_meter_sums.csv")