import pandas as pd
import matplotlib.pyplot as plt

ghg_df = pd.read_csv("../input/us-ghg-inventory-2020/us-ghg-inventory-2020.csv", sep=";")
ghg_df
ghg_NOx = ghg_df[ghg_df["Gas"]=="NOx"].set_index("Activity")
ghg_NOx.T.drop(["Gas"]).plot.bar(stacked=True, title="Main NOx emitting activities in the U.S.")
ghg_NOx["Year 2018"].plot.pie(title="Main NOx emitting activities in the U.S. (2018)")
ghg_SO2 = ghg_df[ghg_df["Gas"]=="SO2"].set_index("Activity")
ghg_SO2.T.drop(["Gas"]).plot.bar(stacked=True, title="Main SO2 emitting activities in the U.S.")
ghg_SO2["Year 2018"].plot.pie(title="Main SO2 emitting activities in the U.S. (2018)")
ghg_CO = ghg_df[ghg_df["Gas"]=="CO"].set_index("Activity")
# fig, ax1 = plt.figure(1, 3, (1, 2))
# fig, ax2 = plt.figure(133)
ghg_CO.T.drop(["Gas"]).plot.bar(stacked=True, title="Main CO emitting activities in the U.S.")
ghg_CO["Year 2018"].plot.pie(title="Main CO emitting activities in the U.S. (2018)")
ghg_Statio = ghg_df[ghg_df["Activity"]=="Stationary Fossil Fuel combustion"].set_index("Gas")
ghg_Statio["Year 2018"].plot.pie(title="Gases emitted by Stationary Fossil Fuel combustion in the U.S. (2018)")
ghg_Mobile = ghg_df[ghg_df["Activity"]=="Mobile Fossil Fuel combustion"].set_index("Gas")
ghg_Mobile["Year 2018"].plot.pie(title="Gases emitted by Mobile Fossil Fuel combustion in the U.S. (2018)")
gasoline_cars = 83640; non_gasoline_cars = 440; ratio = non_gasoline_cars / gasoline_cars
print("Ratio of [non-gasoline cars]/[gasoline cars] sold in 2017 in Puerto Rico was", round(ratio, 4))

gasoline_2017 = 950920454; # gallons
nox_car_2006 = 38.2; # lb
gasoline_car = 581; # gallons
gasoline_ef = nox_car_2006 / gasoline_car # lb/gl
print("The estimated gasoline Emissions Factor in 2006 in Puerto Rico was", round(gasoline_ef, 4))
nox_2017 = gasoline_2017 * gasoline_ef
print("The estimated NOx emissions in 2017 were", round(nox_2017), "lb, or", round(nox_2017/2000), "ton")
