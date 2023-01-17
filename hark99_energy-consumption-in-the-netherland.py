# Importing liabraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None) # Max Display

pd.set_option('display.max_rows', None)  # Max Display

import matplotlib.pyplot as plt

#!pip install seaborn

import seaborn as sns

import os

print(os.listdir("../input"))
# Energy Companies ENEXIS, Liander and Stedin data (Electricity)



# Enexis



df10=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012010.csv")

df11=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012011.csv")

df12=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012012.csv")

df13=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012013.csv")

df14=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012014.csv")

df15=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012015.csv")

df16=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012016.csv")

df17=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012017.csv")

df18=pd.read_csv("../input/dutch_energy/Electricity/enexis_electricity_01012018.csv")



# Liander



df_09=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012009.csv")

df_10=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012010.csv")

df_11=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012011.csv")

df_12=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012012.csv")

df_13=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012013.csv")

df_14=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012014.csv")

df_15=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012015.csv")

df_16=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012016.csv")

df_17=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012017.csv")

df_18=pd.read_csv("../input/dutch_energy/Electricity/liander_electricity_01012018.csv")



# Stedin



df_st09=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2009.csv")

df_st10=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2010.csv")

df_st11=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2011.csv")

df_st12=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2012.csv")

df_st13=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2013.csv")

df_st14=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2014.csv")

df_st15=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2015.csv")

df_st16=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2016.csv")

df_st17=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2017.csv")

df_st18=pd.read_csv("../input/dutch_energy/Electricity/stedin_electricity_2018.csv")     



# Energy Companies ENEXIS, Liander and Stedin data (Gas)



# Enexis



df_exisgas_10=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012010.csv")

df_exisgas_11=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012011.csv")

df_exisgas_12=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012012.csv")

df_exisgas_13=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012013.csv")

df_exisgas_14=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012014.csv")

df_exisgas_15=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012015.csv")

df_exisgas_16=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012016.csv")

df_exisgas_17=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012017.csv")

df_exisgas_18=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012018.csv")



# Liander



df_liandergas_09=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012009.csv")

df_liandergas_10=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012010.csv")

df_liandergas_11=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012011.csv")

df_liandergas_12=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012012.csv")

df_liandergas_13=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012013.csv")

df_liandergas_14=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012014.csv")

df_liandergas_15=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012015.csv")

df_liandergas_16=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012016.csv")

df_liandergas_17=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012017.csv")

df_liandergas_18=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012018.csv")



# Stedin



df_stgas_09=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2009.csv")

df_stgas_10=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2010.csv")

df_stgas_11=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2011.csv")

df_stgas_12=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2012.csv")

df_stgas_13=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2013.csv")

df_stgas_14=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2014.csv")

df_stgas_15=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2015.csv")

df_stgas_16=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2016.csv")

df_stgas_17=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2017.csv")

df_stgas_18=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2018.csv")
# Enexis



df_exisgas_10=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012010.csv")

df_exisgas_11=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012011.csv")

df_exisgas_12=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012012.csv")

df_exisgas_13=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012013.csv")

df_exisgas_14=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012014.csv")

df_exisgas_15=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012015.csv")

df_exisgas_16=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012016.csv")

df_exisgas_17=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012017.csv")

df_exisgas_18=pd.read_csv("../input/dutch_energy/Gas/enexis_gas_01012018.csv")



# Liander



df_liandergas_09=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012009.csv")

df_liandergas_10=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012010.csv")

df_liandergas_11=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012011.csv")

df_liandergas_12=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012012.csv")

df_liandergas_13=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012013.csv")

df_liandergas_14=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012014.csv")

df_liandergas_15=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012015.csv")

df_liandergas_16=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012016.csv")

df_liandergas_17=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012017.csv")

df_liandergas_18=pd.read_csv("../input/dutch_energy/Gas/liander_gas_01012018.csv")



# Stedin



df_stgas_09=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2009.csv")

df_stgas_10=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2010.csv")

df_stgas_11=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2011.csv")

df_stgas_12=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2012.csv")

df_stgas_13=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2013.csv")

df_stgas_14=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2014.csv")

df_stgas_15=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2015.csv")

df_stgas_16=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2016.csv")

df_stgas_17=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2017.csv")

df_stgas_18=pd.read_csv("../input/dutch_energy/Gas/stedin_gas_2018.csv")
# concatenation for electricity data

elec_frames=[df10,df11,df12,df13,df14,df15,df16,df17,df18, df_09, df_10,df_11,df_12,df_13,df_14,df_15,df_16,df_17,df_18, \

             df_st09,df_st10, df_st11, df_st12,df_st13,df_st14,df_st15,df_st16,df_st17,df_st18]

electricity=pd.concat (elec_frames, axis=0 , sort=False)

electricity.head()
print (electricity.shape)

print (electricity.dtypes)

print (electricity.describe())
# missing values

electricity.isnull().sum()

electricity=electricity.fillna(0)

print (electricity.isnull().sum())
# concatenation for gas data

gas_frames = [df_exisgas_10, df_exisgas_11,df_exisgas_12,df_exisgas_13,df_exisgas_14,df_exisgas_15,df_exisgas_16,df_exisgas_17, df_exisgas_18, \

     df_liandergas_09, df_liandergas_10, df_liandergas_11, df_liandergas_12, df_liandergas_13, df_liandergas_14, df_liandergas_15, df_liandergas_16, \

     df_liandergas_17,df_liandergas_18, \

     df_stgas_09, df_stgas_10, df_stgas_11, df_stgas_12, df_stgas_13, df_stgas_14, df_stgas_15, df_stgas_16, df_stgas_17, df_stgas_18 ]

gas=pd.concat (gas_frames, axis=0, sort=False)

gas.head()
print (gas.shape)

print (gas.dtypes)

print (gas.describe())
# missing values

gas.isnull().sum()

gas=gas.fillna(0)

print (gas.isnull().sum())
# concatenation for all the data

#data_frames = [df10,df11,df12,df13,df14,df15,df16,df17,df18, df_09, df_10,df_11,df_12,df_13,df_14,df_15,df_16,df_17,df_18, \

             #df_st09,df_st10, df_st11, df_st12,df_st13,df_st14,df_st15,df_st16,df_st17,df_st18, \

              # df_exisgas_10, df_exisgas_11,df_exisgas_12,df_exisgas_13,df_exisgas_14,df_exisgas_15,df_exisgas_16,df_exisgas_17, df_exisgas_18, \

     #df_liandergas_09, df_liandergas_10, df_liandergas_11, df_liandergas_12, df_liandergas_13, df_liandergas_14, df_liandergas_15, df_liandergas_16, \

     #df_liandergas_17,df_liandergas_18, \

     #df_stgas_09, df_stgas_10, df_stgas_11, df_stgas_12, df_stgas_13, df_stgas_14, df_stgas_15, df_stgas_16, df_stgas_17, df_stgas_18 

              #]



data_frames = [electricity,gas]

data=pd.concat (data_frames, axis=0, sort=False)

data.head()
print (data.shape)

print (data.dtypes)

print (data.describe())
# missing values

data.isnull().sum()
# Memory Usage

data.info(memory_usage='deep')
# Corrleation

data.corr()
viz=data._get_numeric_data()



from scipy import stats



pearson_coef, p_value = stats.pearsonr(viz['annual_consume_lowtarif_perc'], viz['annual_consume'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



pearson_coef, p_value = stats.pearsonr(viz['delivery_perc'], viz['annual_consume'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  



pearson_coef, p_value = stats.pearsonr(viz['num_connections'], viz['annual_consume'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  



pearson_coef, p_value = stats.pearsonr(viz['type_conn_perc'], viz['annual_consume'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  



pearson_coef, p_value = stats.pearsonr(viz['smartmeter_perc'], viz['annual_consume'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  



# 2009 Years Data



# Identifing For Smart Meters

df_09_sm=df_09['smartmeter_perc']

df_09_sm =pd.DataFrame(df_09_sm)

df_st09_sm=df_st09['smartmeter_perc']

df_st09_sm=pd.DataFrame(df_st09_sm)

df_liandergas_09_sm=df_liandergas_09['smartmeter_perc']

df_liandergas_09_sm=pd.DataFrame(df_liandergas_09_sm)

df_stgas_09_sm=df_stgas_09['smartmeter_perc']

df_stgas_09_sm=pd.DataFrame(df_stgas_09_sm)

# Merging

sm0009=pd.merge(df_09_sm, df_st09_sm, left_index=True, right_index=True, how='left')

sm009=pd.merge(sm0009, df_liandergas_09_sm, left_index=True, right_index=True, how='left')

sm09=pd.merge(sm009, df_stgas_09_sm, left_index=True, right_index=True, how='left')

# One Data Frame for respective year

sm09['2009'] = sm09.sum(axis=1)

sm09.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm09.shape)

print (sm09.head(2))



# Identifing For Solar Energy

df_09_se=df_09['delivery_perc']

df_09_se =pd.DataFrame(df_09_se)

df_st09_se=df_st09['delivery_perc']

df_st09_se=pd.DataFrame(df_st09_se)

df_liandergas_09_se=df_liandergas_09['delivery_perc']

df_liandergas_09_se=pd.DataFrame(df_liandergas_09_se)

df_stgas_09_se=df_stgas_09['delivery_perc']

df_stgas_09_se=pd.DataFrame(df_stgas_09_se)

# Merging

se0009=pd.merge(df_09_se, df_st09_se, left_index=True, right_index=True, how='left')

se009=pd.merge(se0009, df_liandergas_09_se, left_index=True, right_index=True, how='left')

se09=pd.merge(se009, df_stgas_09_se, left_index=True, right_index=True, how='left')

# One Data Frame for respective year

se09['2009'] = se09.sum(axis=1)

se09.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se09.shape)

print (se09.head(2))



# 2010



# Identifing For Annual Consumption

df_09_ac=df_09['annual_consume']

df_09_ac =pd.DataFrame(df_09_ac)

df_st09_ac=df_st09['annual_consume']

df_st09_ac=pd.DataFrame(df_st09_ac)

df_liandergas_09_ac=df_liandergas_09['annual_consume']

df_liandergas_09_ac=pd.DataFrame(df_liandergas_09_ac)

df_stgas_09_ac=df_stgas_09['annual_consume']

df_stgas_09_ac=pd.DataFrame(df_stgas_09_ac)

# Merging

ac0009=pd.merge(df_09_ac, df_st09_ac, left_index=True, right_index=True, how='left')

ac009=pd.merge(ac0009, df_liandergas_09_ac, left_index=True, right_index=True, how='left')

ac09=pd.merge(ac009, df_stgas_09_ac, left_index=True, right_index=True, how='left')

# One Data Frame for respective year

ac09['2009'] = ac09.sum(axis=1)

ac09.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac09.shape)

print (ac09.head(2))



# Identifing For Smart Meters

df10_sm=df10['smartmeter_perc']

df10_sm =pd.DataFrame(df10_sm)



df_10_sm=df_10['smartmeter_perc']

df_10_sm =pd.DataFrame(df_10_sm)



df_st10_sm=df_st10['smartmeter_perc']

df_st10_sm=pd.DataFrame(df_st10_sm)



df_exisgas_10_sm=df_exisgas_10['smartmeter_perc']

df_exisgas_10_sm=pd.DataFrame(df_exisgas_10_sm)



df_liandergas_10_sm=df_liandergas_10['smartmeter_perc']

df_liandergas_10_sm=pd.DataFrame(df_liandergas_10_sm)



df_stgas_10_sm=df_stgas_10['smartmeter_perc']

df_stgas_10_sm=pd.DataFrame(df_stgas_10_sm)



# Merging

sm000010=pd.merge(df_10_sm, df10_sm, left_index=True, right_index=True, how='left')

sm00010=pd.merge(sm000010, df_st10_sm, left_index=True, right_index=True, how='left')

sm0010=pd.merge(sm00010, df_exisgas_10_sm, left_index=True, right_index=True, how='left')

sm010=pd.merge(sm0010, df_liandergas_10_sm, left_index=True, right_index=True, how='left')

sm10=pd.merge(sm010, df_stgas_10_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm10['2010'] = sm10.sum(axis=1)

sm10.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm10.shape)

print (sm10.head(2))



# Identifing For Solar Energy



df10_se=df10['delivery_perc']

df10_se =pd.DataFrame(df10_se)



df_10_se=df_10['delivery_perc']

df_10_se =pd.DataFrame(df_10_se)



df_st10_se=df_st10['delivery_perc']

df_st10_se=pd.DataFrame(df_st10_se)



df_exisgas_10_se=df_exisgas_10['delivery_perc']

df_exisgas_10_se=pd.DataFrame(df_exisgas_10_se)



df_liandergas_10_se=df_liandergas_10['delivery_perc']

df_liandergas_10_se=pd.DataFrame(df_liandergas_10_se)



df_stgas_10_se=df_stgas_10['delivery_perc']

df_stgas_10_se=pd.DataFrame(df_stgas_10_se)



# Merging

se000010=pd.merge(df_10_se, df10_se, left_index=True, right_index=True, how='left')

se00010=pd.merge(se000010, df_st10_se, left_index=True, right_index=True, how='left')

se0010=pd.merge(se00010, df_exisgas_10_se, left_index=True, right_index=True, how='left')

se010=pd.merge(se0010, df_liandergas_10_se, left_index=True, right_index=True, how='left')

se10=pd.merge(se010, df_stgas_10_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se10['2010'] = se10.sum(axis=1)

se10.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se10.shape)

print (se10.head(2))



# Identifing For Annual Consumption



df10_ac=df10['annual_consume']

df10_ac =pd.DataFrame(df10_ac)



df_10_ac=df_10['annual_consume']

df_10_ac =pd.DataFrame(df_10_ac)



df_st10_ac=df_st10['annual_consume']

df_st10_ac=pd.DataFrame(df_st10_ac)



df_exisgas_10_ac=df_exisgas_10['annual_consume']

df_exisgas_10_ac=pd.DataFrame(df_exisgas_10_ac)



df_liandergas_10_ac=df_liandergas_10['annual_consume']

df_liandergas_10_ac=pd.DataFrame(df_liandergas_10_ac)



df_stgas_10_ac=df_stgas_10['annual_consume']

df_stgas_10_ac=pd.DataFrame(df_stgas_10_ac)



# Merging

ac000010=pd.merge(df_10_ac, df10_ac, left_index=True, right_index=True, how='left')

ac00010=pd.merge(ac000010, df_st10_ac, left_index=True, right_index=True, how='left')

ac0010=pd.merge(ac00010, df_exisgas_10_ac, left_index=True, right_index=True, how='left')

ac010=pd.merge(ac0010, df_liandergas_10_ac, left_index=True, right_index=True, how='left')

ac10=pd.merge(ac010, df_stgas_10_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac10['2010'] = ac10.sum(axis=1)

ac10.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac10.shape)

print (ac10.head(2))



# 2011



# Identifing For Smart Meters

df11_sm=df11['smartmeter_perc']

df11_sm =pd.DataFrame(df11_sm)



df_11_sm=df_11['smartmeter_perc']

df_11_sm =pd.DataFrame(df_11_sm)



df_st11_sm=df_st11['smartmeter_perc']

df_st11_sm=pd.DataFrame(df_st11_sm)



df_exisgas_11_sm=df_exisgas_11['smartmeter_perc']

df_exisgas_11_sm=pd.DataFrame(df_exisgas_11_sm)



df_liandergas_11_sm=df_liandergas_11['smartmeter_perc']

df_liandergas_11_sm=pd.DataFrame(df_liandergas_11_sm)



df_stgas_11_sm=df_stgas_11['smartmeter_perc']

df_stgas_11_sm=pd.DataFrame(df_stgas_11_sm)



# Merging

sm000011=pd.merge(df_11_sm, df11_sm, left_index=True, right_index=True, how='left')

sm00011=pd.merge(sm000011, df_st11_sm, left_index=True, right_index=True, how='left')

sm0011=pd.merge(sm00011, df_exisgas_11_sm, left_index=True, right_index=True, how='left')

sm011=pd.merge(sm0011, df_liandergas_11_sm, left_index=True, right_index=True, how='left')

sm11=pd.merge(sm011, df_stgas_11_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm11['2011'] = sm11.sum(axis=1)

sm11.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm11.shape)

print (sm11.head(2))



# Identifing For Solar Energy



df11_se=df11['delivery_perc']

df11_se =pd.DataFrame(df11_se)



df_11_se=df_11['delivery_perc']

df_11_se =pd.DataFrame(df_11_se)



df_st11_se=df_st11['delivery_perc']

df_st11_se=pd.DataFrame(df_st11_se)



df_exisgas_11_se=df_exisgas_11['delivery_perc']

df_exisgas_11_se=pd.DataFrame(df_exisgas_11_se)



df_liandergas_11_se=df_liandergas_11['delivery_perc']

df_liandergas_11_se=pd.DataFrame(df_liandergas_11_se)



df_stgas_11_se=df_stgas_11['delivery_perc']

df_stgas_11_se=pd.DataFrame(df_stgas_11_se)



# Merging

se000011=pd.merge(df_11_se, df11_se, left_index=True, right_index=True, how='left')

se00011=pd.merge(se000011, df_st11_se, left_index=True, right_index=True, how='left')

se0011=pd.merge(se00011, df_exisgas_11_se, left_index=True, right_index=True, how='left')

se011=pd.merge(se0011, df_liandergas_11_se, left_index=True, right_index=True, how='left')

se11=pd.merge(se011, df_stgas_11_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se11['2011'] = se11.sum(axis=1)

se11.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se11.shape)

print (se11.head(2))



# Identifing For Annual Consumption



df11_ac=df11['annual_consume']

df11_ac =pd.DataFrame(df11_ac)



df_11_ac=df_11['annual_consume']

df_11_ac =pd.DataFrame(df_11_ac)



df_st11_ac=df_st11['annual_consume']

df_st11_ac=pd.DataFrame(df_st11_ac)



df_exisgas_11_ac=df_exisgas_11['annual_consume']

df_exisgas_11_ac=pd.DataFrame(df_exisgas_11_ac)



df_liandergas_11_ac=df_liandergas_11['annual_consume']

df_liandergas_11_ac=pd.DataFrame(df_liandergas_11_ac)



df_stgas_11_ac=df_stgas_11['annual_consume']

df_stgas_11_ac=pd.DataFrame(df_stgas_11_ac)



# Merging

ac000011=pd.merge(df_11_ac, df11_ac, left_index=True, right_index=True, how='left')

ac00011=pd.merge(ac000011, df_st11_ac, left_index=True, right_index=True, how='left')

ac0011=pd.merge(ac00011, df_exisgas_11_ac, left_index=True, right_index=True, how='left')

ac011=pd.merge(ac0011, df_liandergas_11_ac, left_index=True, right_index=True, how='left')

ac11=pd.merge(ac011, df_stgas_11_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac11['2011'] = ac11.sum(axis=1)

ac11.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac11.shape)

print (ac11.head(2))



# 2012



# Identifing For Smart Meters

df12_sm=df12['smartmeter_perc']

df12_sm =pd.DataFrame(df12_sm)



df_12_sm=df_12['smartmeter_perc']

df_12_sm =pd.DataFrame(df_12_sm)



df_st12_sm=df_st12['smartmeter_perc']

df_st12_sm=pd.DataFrame(df_st12_sm)



df_exisgas_12_sm=df_exisgas_12['smartmeter_perc']

df_exisgas_12_sm=pd.DataFrame(df_exisgas_12_sm)



df_liandergas_12_sm=df_liandergas_12['smartmeter_perc']

df_liandergas_12_sm=pd.DataFrame(df_liandergas_12_sm)



df_stgas_12_sm=df_stgas_12['smartmeter_perc']

df_stgas_12_sm=pd.DataFrame(df_stgas_12_sm)



# Merging

sm000012=pd.merge(df_12_sm, df12_sm, left_index=True, right_index=True, how='left')

sm00012=pd.merge(sm000012, df_st12_sm, left_index=True, right_index=True, how='left')

sm0012=pd.merge(sm00012, df_exisgas_12_sm, left_index=True, right_index=True, how='left')

sm012=pd.merge(sm0012, df_liandergas_12_sm, left_index=True, right_index=True, how='left')

sm12=pd.merge(sm012, df_stgas_12_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm12['2012'] = sm12.sum(axis=1)

sm12.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm12.shape)

print (sm12.head(2))





# Identifing For Solar Energy



df12_se=df12['delivery_perc']

df12_se =pd.DataFrame(df12_se)



df_12_se=df_12['delivery_perc']

df_12_se =pd.DataFrame(df_12_se)



df_st12_se=df_st12['delivery_perc']

df_st12_se=pd.DataFrame(df_st12_se)



df_exisgas_12_se=df_exisgas_12['delivery_perc']

df_exisgas_12_se=pd.DataFrame(df_exisgas_12_se)



df_liandergas_12_se=df_liandergas_12['delivery_perc']

df_liandergas_12_se=pd.DataFrame(df_liandergas_12_se)



df_stgas_12_se=df_stgas_12['delivery_perc']

df_stgas_12_se=pd.DataFrame(df_stgas_12_se)



# Merging

se000012=pd.merge(df_12_se, df12_se, left_index=True, right_index=True, how='left')

se00012=pd.merge(se000012, df_st12_se, left_index=True, right_index=True, how='left')

se0012=pd.merge(se00012, df_exisgas_12_se, left_index=True, right_index=True, how='left')

se012=pd.merge(se0012, df_liandergas_12_se, left_index=True, right_index=True, how='left')

se12=pd.merge(se012, df_stgas_12_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se12['2012'] = se12.sum(axis=1)

se12.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se12.shape)

print (se12.head(2))



# Identifing For Annual Consumption



df12_ac=df12['annual_consume']

df12_ac =pd.DataFrame(df12_ac)



df_12_ac=df_12['annual_consume']

df_12_ac =pd.DataFrame(df_12_ac)



df_st12_ac=df_st12['annual_consume']

df_st12_ac=pd.DataFrame(df_st12_ac)



df_exisgas_12_ac=df_exisgas_12['annual_consume']

df_exisgas_12_ac=pd.DataFrame(df_exisgas_12_ac)



df_liandergas_12_ac=df_liandergas_12['annual_consume']

df_liandergas_12_ac=pd.DataFrame(df_liandergas_12_ac)



df_stgas_12_ac=df_stgas_12['annual_consume']

df_stgas_12_ac=pd.DataFrame(df_stgas_12_ac)



# Merging

ac000012=pd.merge(df_12_ac, df12_ac, left_index=True, right_index=True, how='left')

ac00012=pd.merge(ac000012, df_st12_ac, left_index=True, right_index=True, how='left')

ac0012=pd.merge(ac00012, df_exisgas_12_ac, left_index=True, right_index=True, how='left')

ac012=pd.merge(ac0012, df_liandergas_12_ac, left_index=True, right_index=True, how='left')

ac12=pd.merge(ac012, df_stgas_12_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac12['2012'] = ac12.sum(axis=1)

ac12.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac12.shape)

print (ac12.head(2))



# 2013



# Identifing For Smart Meters

df13_sm=df13['smartmeter_perc']

df13_sm =pd.DataFrame(df13_sm)



df_13_sm=df_13['smartmeter_perc']

df_13_sm =pd.DataFrame(df_13_sm)



df_st13_sm=df_st13['smartmeter_perc']

df_st13_sm=pd.DataFrame(df_st13_sm)



df_exisgas_13_sm=df_exisgas_13['smartmeter_perc']

df_exisgas_13_sm=pd.DataFrame(df_exisgas_13_sm)



df_liandergas_13_sm=df_liandergas_13['smartmeter_perc']

df_liandergas_13_sm=pd.DataFrame(df_liandergas_13_sm)



df_stgas_13_sm=df_stgas_13['smartmeter_perc']

df_stgas_13_sm=pd.DataFrame(df_stgas_13_sm)



# Merging

sm000013=pd.merge(df_13_sm, df13_sm, left_index=True, right_index=True, how='left')

sm00013=pd.merge(sm000013, df_st13_sm, left_index=True, right_index=True, how='left')

sm0013=pd.merge(sm00013, df_exisgas_13_sm, left_index=True, right_index=True, how='left')

sm013=pd.merge(sm0013, df_liandergas_13_sm, left_index=True, right_index=True, how='left')

sm13=pd.merge(sm013, df_stgas_13_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm13['2013'] = sm13.sum(axis=1)

sm13.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm13.shape)

print (sm13.head(2))



# Identifing For Solar Energy



df13_se=df13['delivery_perc']

df13_se =pd.DataFrame(df13_se)



df_13_se=df_13['delivery_perc']

df_13_se =pd.DataFrame(df_13_se)



df_st13_se=df_st13['delivery_perc']

df_st13_se=pd.DataFrame(df_st13_se)



df_exisgas_13_se=df_exisgas_13['delivery_perc']

df_exisgas_13_se=pd.DataFrame(df_exisgas_13_se)



df_liandergas_13_se=df_liandergas_13['delivery_perc']

df_liandergas_13_se=pd.DataFrame(df_liandergas_13_se)



df_stgas_13_se=df_stgas_13['delivery_perc']

df_stgas_13_se=pd.DataFrame(df_stgas_13_se)



# Merging

se000013=pd.merge(df_13_se, df13_se, left_index=True, right_index=True, how='left')

se00013=pd.merge(se000013, df_st13_se, left_index=True, right_index=True, how='left')

se0013=pd.merge(se00013, df_exisgas_13_se, left_index=True, right_index=True, how='left')

se013=pd.merge(se0013, df_liandergas_13_se, left_index=True, right_index=True, how='left')

se13=pd.merge(se013, df_stgas_13_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se13['2013'] = se13.sum(axis=1)

se13.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se13.shape)

print (se13.head(2))



# Identifing For Annual Consumption



df13_ac=df13['annual_consume']

df13_ac =pd.DataFrame(df13_ac)



df_13_ac=df_13['annual_consume']

df_13_ac =pd.DataFrame(df_13_ac)



df_st13_ac=df_st13['annual_consume']

df_st13_ac=pd.DataFrame(df_st13_ac)



df_exisgas_13_ac=df_exisgas_13['annual_consume']

df_exisgas_13_ac=pd.DataFrame(df_exisgas_13_ac)



df_liandergas_13_ac=df_liandergas_13['annual_consume']

df_liandergas_13_ac=pd.DataFrame(df_liandergas_13_ac)



df_stgas_13_ac=df_stgas_13['annual_consume']

df_stgas_13_ac=pd.DataFrame(df_stgas_13_ac)



# Merging

ac000013=pd.merge(df_13_ac, df13_ac, left_index=True, right_index=True, how='left')

ac00013=pd.merge(ac000013, df_st13_ac, left_index=True, right_index=True, how='left')

ac0013=pd.merge(ac00013, df_exisgas_13_ac, left_index=True, right_index=True, how='left')

ac013=pd.merge(ac0013, df_liandergas_13_ac, left_index=True, right_index=True, how='left')

ac13=pd.merge(ac013, df_stgas_13_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac13['2013'] = ac13.sum(axis=1)

ac13.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac13.shape)

print (ac13.head(2))



# 2014



# Identifing For Smart Meters

df14_sm=df14['smartmeter_perc']

df14_sm =pd.DataFrame(df14_sm)



df_14_sm=df_14['smartmeter_perc']

df_14_sm =pd.DataFrame(df_14_sm)



df_st14_sm=df_st14['smartmeter_perc']

df_st14_sm=pd.DataFrame(df_st14_sm)



df_exisgas_14_sm=df_exisgas_14['smartmeter_perc']

df_exisgas_14_sm=pd.DataFrame(df_exisgas_14_sm)



df_liandergas_14_sm=df_liandergas_14['smartmeter_perc']

df_liandergas_14_sm=pd.DataFrame(df_liandergas_14_sm)



df_stgas_14_sm=df_stgas_14['smartmeter_perc']

df_stgas_14_sm=pd.DataFrame(df_stgas_14_sm)



# Merging

sm000014=pd.merge(df_14_sm, df14_sm, left_index=True, right_index=True, how='left')

sm00014=pd.merge(sm000014, df_st14_sm, left_index=True, right_index=True, how='left')

sm0014=pd.merge(sm00014, df_exisgas_14_sm, left_index=True, right_index=True, how='left')

sm014=pd.merge(sm0014, df_liandergas_14_sm, left_index=True, right_index=True, how='left')

sm14=pd.merge(sm014, df_stgas_14_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm14['2014'] = sm14.sum(axis=1)

sm14.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm14.shape)

print (sm14.head(2))



# Identifing For Solar Energy



df14_se=df14['delivery_perc']

df14_se =pd.DataFrame(df14_se)



df_14_se=df_14['delivery_perc']

df_14_se =pd.DataFrame(df_14_se)



df_st14_se=df_st14['delivery_perc']

df_st14_se=pd.DataFrame(df_st14_se)



df_exisgas_14_se=df_exisgas_14['delivery_perc']

df_exisgas_14_se=pd.DataFrame(df_exisgas_14_se)



df_liandergas_14_se=df_liandergas_14['delivery_perc']

df_liandergas_14_se=pd.DataFrame(df_liandergas_14_se)



df_stgas_14_se=df_stgas_14['delivery_perc']

df_stgas_14_se=pd.DataFrame(df_stgas_14_se)



# Merging

se000014=pd.merge(df_14_se, df14_se, left_index=True, right_index=True, how='left')

se00014=pd.merge(se000014, df_st14_se, left_index=True, right_index=True, how='left')

se0014=pd.merge(se00014, df_exisgas_14_se, left_index=True, right_index=True, how='left')

se014=pd.merge(se0014, df_liandergas_14_se, left_index=True, right_index=True, how='left')

se14=pd.merge(se014, df_stgas_14_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se14['2014'] = se14.sum(axis=1)

se14.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se14.shape)

print (se14.head(2))



# Identifing For Annual Consumption



df14_ac=df14['annual_consume']

df14_ac =pd.DataFrame(df14_ac)



df_14_ac=df_14['annual_consume']

df_14_ac =pd.DataFrame(df_14_ac)



df_st14_ac=df_st14['annual_consume']

df_st14_ac=pd.DataFrame(df_st14_ac)



df_exisgas_14_ac=df_exisgas_14['annual_consume']

df_exisgas_14_ac=pd.DataFrame(df_exisgas_14_ac)



df_liandergas_14_ac=df_liandergas_14['annual_consume']

df_liandergas_14_ac=pd.DataFrame(df_liandergas_14_ac)



df_stgas_14_ac=df_stgas_14['annual_consume']

df_stgas_14_ac=pd.DataFrame(df_stgas_14_ac)



# Merging

ac000014=pd.merge(df_14_ac, df14_ac, left_index=True, right_index=True, how='left')

ac00014=pd.merge(ac000014, df_st14_ac, left_index=True, right_index=True, how='left')

ac0014=pd.merge(ac00014, df_exisgas_14_ac, left_index=True, right_index=True, how='left')

ac014=pd.merge(ac0014, df_liandergas_14_ac, left_index=True, right_index=True, how='left')

ac14=pd.merge(ac014, df_stgas_14_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac14['2014'] = ac14.sum(axis=1)

ac14.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac14.shape)

print (ac14.head(2))



# 2015



# Identifing For Smart Meters

df15_sm=df15['smartmeter_perc']

df15_sm =pd.DataFrame(df15_sm)



df_15_sm=df_15['smartmeter_perc']

df_15_sm =pd.DataFrame(df_15_sm)



df_st15_sm=df_st15['smartmeter_perc']

df_st15_sm=pd.DataFrame(df_st15_sm)



df_exisgas_15_sm=df_exisgas_15['smartmeter_perc']

df_exisgas_15_sm=pd.DataFrame(df_exisgas_15_sm)



df_liandergas_15_sm=df_liandergas_15['smartmeter_perc']

df_liandergas_15_sm=pd.DataFrame(df_liandergas_15_sm)



df_stgas_15_sm=df_stgas_15['smartmeter_perc']

df_stgas_15_sm=pd.DataFrame(df_stgas_15_sm)



# Merging

sm000015=pd.merge(df_15_sm, df15_sm, left_index=True, right_index=True, how='left')

sm00015=pd.merge(sm000015, df_st15_sm, left_index=True, right_index=True, how='left')

sm0015=pd.merge(sm00015, df_exisgas_15_sm, left_index=True, right_index=True, how='left')

sm015=pd.merge(sm0015, df_liandergas_15_sm, left_index=True, right_index=True, how='left')

sm15=pd.merge(sm015, df_stgas_15_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm15['2015'] = sm15.sum(axis=1)

sm15.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm15.shape)

print (sm15.head(2))



# Identifing For Solar Energy



df15_se=df15['delivery_perc']

df15_se =pd.DataFrame(df15_se)



df_15_se=df_15['delivery_perc']

df_15_se =pd.DataFrame(df_15_se)



df_st15_se=df_st15['delivery_perc']

df_st15_se=pd.DataFrame(df_st15_se)



df_exisgas_15_se=df_exisgas_15['delivery_perc']

df_exisgas_15_se=pd.DataFrame(df_exisgas_15_se)



df_liandergas_15_se=df_liandergas_15['delivery_perc']

df_liandergas_15_se=pd.DataFrame(df_liandergas_15_se)



df_stgas_15_se=df_stgas_15['delivery_perc']

df_stgas_15_se=pd.DataFrame(df_stgas_15_se)



# Merging

se000015=pd.merge(df_15_se, df15_se, left_index=True, right_index=True, how='left')

se00015=pd.merge(se000015, df_st15_se, left_index=True, right_index=True, how='left')

se0015=pd.merge(se00015, df_exisgas_15_se, left_index=True, right_index=True, how='left')

se015=pd.merge(se0015, df_liandergas_15_se, left_index=True, right_index=True, how='left')

se15=pd.merge(se015, df_stgas_15_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se15['2015'] = se15.sum(axis=1)

se15.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se15.shape)

print (se15.head(2))



# Identifing For Annual Consumption



df15_ac=df15['annual_consume']

df15_ac =pd.DataFrame(df15_ac)



df_15_ac=df_15['annual_consume']

df_15_ac =pd.DataFrame(df_15_ac)



df_st15_ac=df_st15['annual_consume']

df_st15_ac=pd.DataFrame(df_st15_ac)



df_exisgas_15_ac=df_exisgas_15['annual_consume']

df_exisgas_15_ac=pd.DataFrame(df_exisgas_13_ac)



df_liandergas_15_ac=df_liandergas_15['annual_consume']

df_liandergas_15_ac=pd.DataFrame(df_liandergas_15_ac)



df_stgas_15_ac=df_stgas_15['annual_consume']

df_stgas_15_ac=pd.DataFrame(df_stgas_15_ac)



# Merging

ac000015=pd.merge(df_15_ac, df15_ac, left_index=True, right_index=True, how='left')

ac00015=pd.merge(ac000015, df_st15_ac, left_index=True, right_index=True, how='left')

ac0015=pd.merge(ac00015, df_exisgas_15_ac, left_index=True, right_index=True, how='left')

ac015=pd.merge(ac0015, df_liandergas_15_ac, left_index=True, right_index=True, how='left')

ac15=pd.merge(ac015, df_stgas_15_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac15['2015'] = ac15.sum(axis=1)

ac15.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac15.shape)

print (ac15.head(2))



# 2016



# Identifing For Smart Meters

df16_sm=df16['smartmeter_perc']

df16_sm =pd.DataFrame(df16_sm)



df_16_sm=df_16['smartmeter_perc']

df_16_sm =pd.DataFrame(df_16_sm)



df_st16_sm=df_st16['smartmeter_perc']

df_st16_sm=pd.DataFrame(df_st16_sm)



df_exisgas_16_sm=df_exisgas_16['smartmeter_perc']

df_exisgas_16_sm=pd.DataFrame(df_exisgas_16_sm)



df_liandergas_16_sm=df_liandergas_16['smartmeter_perc']

df_liandergas_16_sm=pd.DataFrame(df_liandergas_16_sm)



df_stgas_16_sm=df_stgas_16['smartmeter_perc']

df_stgas_16_sm=pd.DataFrame(df_stgas_16_sm)



# Merging

sm000016=pd.merge(df_16_sm, df16_sm, left_index=True, right_index=True, how='left')

sm00016=pd.merge(sm000016, df_st16_sm, left_index=True, right_index=True, how='left')

sm0016=pd.merge(sm00016, df_exisgas_16_sm, left_index=True, right_index=True, how='left')

sm016=pd.merge(sm0016, df_liandergas_16_sm, left_index=True, right_index=True, how='left')

sm16=pd.merge(sm016, df_stgas_16_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm16['2016'] = sm16.sum(axis=1)

sm16.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm16.shape)

print (sm16.head(2))

# Identifing For Solar Energy



df16_se=df16['delivery_perc']

df16_se =pd.DataFrame(df16_se)



df_16_se=df_16['delivery_perc']

df_16_se =pd.DataFrame(df_16_se)



df_st16_se=df_st16['delivery_perc']

df_st16_se=pd.DataFrame(df_st16_se)



df_exisgas_16_se=df_exisgas_16['delivery_perc']

df_exisgas_16_se=pd.DataFrame(df_exisgas_16_se)



df_liandergas_16_se=df_liandergas_16['delivery_perc']

df_liandergas_16_se=pd.DataFrame(df_liandergas_16_se)



df_stgas_16_se=df_stgas_16['delivery_perc']

df_stgas_16_se=pd.DataFrame(df_stgas_16_se)



# Merging

se000016=pd.merge(df_16_se, df16_se, left_index=True, right_index=True, how='left')

se00016=pd.merge(se000016, df_st16_se, left_index=True, right_index=True, how='left')

se0016=pd.merge(se00016, df_exisgas_16_se, left_index=True, right_index=True, how='left')

se016=pd.merge(se0016, df_liandergas_16_se, left_index=True, right_index=True, how='left')

se16=pd.merge(se016, df_stgas_16_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se16['2016'] = se16.sum(axis=1)

se16.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se16.shape)

print (se16.head(2))



# Identifing For Annual Consumption



df16_ac=df16['annual_consume']

df16_ac =pd.DataFrame(df16_ac)



df_16_ac=df_16['annual_consume']

df_16_ac =pd.DataFrame(df_16_ac)



df_st16_ac=df_st16['annual_consume']

df_st16_ac=pd.DataFrame(df_st16_ac)



df_exisgas_16_ac=df_exisgas_16['annual_consume']

df_exisgas_16_ac=pd.DataFrame(df_exisgas_16_ac)



df_liandergas_16_ac=df_liandergas_16['annual_consume']

df_liandergas_16_ac=pd.DataFrame(df_liandergas_16_ac)



df_stgas_16_ac=df_stgas_16['annual_consume']

df_stgas_16_ac=pd.DataFrame(df_stgas_16_ac)



# Merging

ac000016=pd.merge(df_16_ac, df16_ac, left_index=True, right_index=True, how='left')

ac00016=pd.merge(ac000016, df_st16_ac, left_index=True, right_index=True, how='left')

ac0016=pd.merge(ac00016, df_exisgas_16_ac, left_index=True, right_index=True, how='left')

ac016=pd.merge(ac0016, df_liandergas_16_ac, left_index=True, right_index=True, how='left')

ac16=pd.merge(ac016, df_stgas_16_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac16['2016'] = ac16.sum(axis=1)

ac16.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac16.shape)

print (ac16.head(2))



# 2017



# Identifing For Smart Meters

df17_sm=df17['smartmeter_perc']

df17_sm =pd.DataFrame(df17_sm)

df_17_sm=df_17['smartmeter_perc']

df_17_sm =pd.DataFrame(df_17_sm)

df_st17_sm=df_st17['smartmeter_perc']

df_st17_sm=pd.DataFrame(df_st17_sm)



df_exisgas_17_sm=df_exisgas_17['smartmeter_perc']

df_exisgas_17_sm=pd.DataFrame(df_exisgas_17_sm)



df_liandergas_17_sm=df_liandergas_17['smartmeter_perc']

df_liandergas_17_sm=pd.DataFrame(df_liandergas_17_sm)



df_stgas_17_sm=df_stgas_17['smartmeter_perc']

df_stgas_17_sm=pd.DataFrame(df_stgas_17_sm)



# Merging

sm000017=pd.merge(df_17_sm, df17_sm, left_index=True, right_index=True, how='left')

sm00017=pd.merge(sm000017, df_st17_sm, left_index=True, right_index=True, how='left')

sm0017=pd.merge(sm00017, df_exisgas_17_sm, left_index=True, right_index=True, how='left')

sm017=pd.merge(sm0017, df_liandergas_17_sm, left_index=True, right_index=True, how='left')

sm17=pd.merge(sm017, df_stgas_17_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm17['2017'] = sm17.sum(axis=1)

sm17.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm17.shape)

print (sm17.head(2))



# Identifing For Solar Energy



df17_se=df17['delivery_perc']

df17_se =pd.DataFrame(df17_se)



df_17_se=df_17['delivery_perc']

df_17_se =pd.DataFrame(df_17_se)



df_st17_se=df_st17['delivery_perc']

df_st17_se=pd.DataFrame(df_st17_se)



df_exisgas_17_se=df_exisgas_17['delivery_perc']

df_exisgas_17_se=pd.DataFrame(df_exisgas_17_se)



df_liandergas_17_se=df_liandergas_17['delivery_perc']

df_liandergas_17_se=pd.DataFrame(df_liandergas_17_se)



df_stgas_17_se=df_stgas_17['delivery_perc']

df_stgas_17_se=pd.DataFrame(df_stgas_17_se)



# Merging

se000017=pd.merge(df_17_se, df17_se, left_index=True, right_index=True, how='left')

se00017=pd.merge(se000017, df_st17_se, left_index=True, right_index=True, how='left')

se0017=pd.merge(se00017, df_exisgas_17_se, left_index=True, right_index=True, how='left')

se017=pd.merge(se0017, df_liandergas_17_se, left_index=True, right_index=True, how='left')

se17=pd.merge(se017, df_stgas_17_se, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

se17['2017'] = se17.sum(axis=1)

se17.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se17.shape)

print (se17.head(2))



# Identifing For Annual Consumption



df17_ac=df17['annual_consume']

df17_ac =pd.DataFrame(df17_ac)



df_17_ac=df_17['annual_consume']

df_17_ac =pd.DataFrame(df_17_ac)



df_st17_ac=df_st17['annual_consume']

df_st17_ac=pd.DataFrame(df_st17_ac)



df_exisgas_17_ac=df_exisgas_17['annual_consume']

df_exisgas_17_ac=pd.DataFrame(df_exisgas_17_ac)



df_liandergas_17_ac=df_liandergas_17['annual_consume']

df_liandergas_17_ac=pd.DataFrame(df_liandergas_17_ac)



df_stgas_17_ac=df_stgas_17['annual_consume']

df_stgas_17_ac=pd.DataFrame(df_stgas_17_ac)



# Merging

ac000017=pd.merge(df_17_ac, df17_ac, left_index=True, right_index=True, how='left')

ac00017=pd.merge(ac000017, df_st17_ac, left_index=True, right_index=True, how='left')

ac0017=pd.merge(ac00017, df_exisgas_17_ac, left_index=True, right_index=True, how='left')

ac017=pd.merge(ac0017, df_liandergas_17_ac, left_index=True, right_index=True, how='left')

ac17=pd.merge(ac017, df_stgas_17_ac, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

ac17['2017'] = ac17.sum(axis=1)

ac17.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac17.shape)

print (ac17.head(2))



# 2018



# Identifing For Smart Meters

df18_sm=df18['smartmeter_perc']

df18_sm =pd.DataFrame(df18_sm)



df_18_sm=df_18['smartmeter_perc']

df_18_sm =pd.DataFrame(df_18_sm)



df_st18_sm=df_st18['smartmeter_perc']

df_st18_sm=pd.DataFrame(df_st18_sm)



df_exisgas_18_sm=df_exisgas_18['smartmeter_perc']

df_exisgas_18_sm=pd.DataFrame(df_exisgas_18_sm)



df_liandergas_18_sm=df_liandergas_18['smartmeter_perc']

df_liandergas_18_sm=pd.DataFrame(df_liandergas_18_sm)



df_stgas_18_sm=df_stgas_18['smartmeter_perc']

df_stgas_18_sm=pd.DataFrame(df_stgas_18_sm)



# Merging

sm000018=pd.merge(df_18_sm, df18_sm, left_index=True, right_index=True, how='left')

sm00018=pd.merge(sm000018, df_st18_sm, left_index=True, right_index=True, how='left')

sm0018=pd.merge(sm00018, df_exisgas_18_sm, left_index=True, right_index=True, how='left')

sm018=pd.merge(sm0018, df_liandergas_18_sm, left_index=True, right_index=True, how='left')

sm18=pd.merge(sm018, df_stgas_18_sm, left_index=True, right_index=True, how='left')





# One Data Frame for respective year

sm18['2018'] = sm18.sum(axis=1)

sm18.drop(['smartmeter_perc_x','smartmeter_perc_y'], axis=1, inplace=True)

print (sm18.shape)

print (sm18.head(2))



# Identifing For Solar Energy



df18_se=df18['delivery_perc']

df18_se =pd.DataFrame(df18_se)



df_18_se=df_18['delivery_perc']

df_18_se =pd.DataFrame(df_18_se)



df_st18_se=df_st18['delivery_perc']

df_st18_se=pd.DataFrame(df_st18_se)



df_exisgas_18_se=df_exisgas_18['delivery_perc']

df_exisgas_18_se=pd.DataFrame(df_exisgas_18_se)



df_liandergas_18_se=df_liandergas_18['delivery_perc']

df_liandergas_18_se=pd.DataFrame(df_liandergas_18_se)



df_stgas_18_se=df_stgas_18['delivery_perc']

df_stgas_18_se=pd.DataFrame(df_stgas_18_se)



# Merging

se000018=pd.merge(df_18_se, df18_se, left_index=True, right_index=True, how='left')

se00018=pd.merge(se000018, df_st18_se, left_index=True, right_index=True, how='left')

se0018=pd.merge(se00018, df_exisgas_18_se, left_index=True, right_index=True, how='left')

se018=pd.merge(se0018, df_liandergas_18_se, left_index=True, right_index=True, how='left')

se18=pd.merge(se018, df_stgas_18_se, left_index=True, right_index=True, how='left')



# One Data Frame for respective year

se18['2018'] = se16.sum(axis=1)

se18.drop(['delivery_perc_x','delivery_perc_y'], axis=1, inplace=True)

print (se18.shape)

print (se18.head(2))



# Identifing For Annual Consumption



df18_ac=df18['annual_consume']

df18_ac =pd.DataFrame(df18_ac)



df_18_ac=df_18['annual_consume']

df_18_ac =pd.DataFrame(df_18_ac)



df_st18_ac=df_st18['annual_consume']

df_st18_ac=pd.DataFrame(df_st18_ac)



df_exisgas_18_ac=df_exisgas_18['annual_consume']

df_exisgas_18_ac=pd.DataFrame(df_exisgas_18_ac)



df_liandergas_18_ac=df_liandergas_18['annual_consume']

df_liandergas_18_ac=pd.DataFrame(df_liandergas_18_ac)

df_stgas_18_ac=df_stgas_18['annual_consume']

df_stgas_18_ac=pd.DataFrame(df_stgas_18_ac)



# Merging

ac000018=pd.merge(df_18_ac, df18_ac, left_index=True, right_index=True, how='left')

ac00018=pd.merge(ac000018, df_st18_ac, left_index=True, right_index=True, how='left')

ac0018=pd.merge(ac00018, df_exisgas_18_ac, left_index=True, right_index=True, how='left')

ac018=pd.merge(ac0018, df_liandergas_18_ac, left_index=True, right_index=True, how='left')

ac18=pd.merge(ac018, df_stgas_18_ac, left_index=True, right_index=True, how='left')



# One Data Frame for respective year

ac18['2018'] = ac18.sum(axis=1)

ac18.drop(['annual_consume_x', 'annual_consume_y'], axis=1, inplace=True)

print (ac18.shape)

print (ac18.head(2))
# Merging All the data to make one data frames for each feature

# For Smart Meters



sm_09=pd.merge(sm09, sm10, left_index=True, right_index=True, how='right')



sm_10=pd.merge(sm_09, sm11, left_index=True, right_index=True, how='right')



sm_11=pd.merge(sm_10, sm12, left_index=True, right_index=True, how='right')



sm_12=pd.merge(sm_11, sm13, left_index=True, right_index=True, how='right')



sm_13=pd.merge(sm_12, sm14, left_index=True, right_index=True, how='left')



sm_14=pd.merge(sm_13, sm15, left_index=True, right_index=True, how='right')



sm_15=pd.merge(sm_14, sm16, left_index=True, right_index=True, how='right')



sm_16=pd.merge(sm_15, sm17, left_index=True, right_index=True, how='right')



sm_17=pd.merge(sm_16, sm18, left_index=True, right_index=True, how='right')



print (sm_17.shape)

print (sm_17.head(2))



# For Solar Energy



se_09=pd.merge(se09, se10, left_index=True, right_index=True, how='right')



se_10=pd.merge(se_09, se11, left_index=True, right_index=True, how='right')



se_11=pd.merge(se_10, se12, left_index=True, right_index=True, how='right')



se_12=pd.merge(se_11, se13, left_index=True, right_index=True, how='right')



se_13=pd.merge(se_12, se14, left_index=True, right_index=True, how='left')



se_14=pd.merge(se_13, se15, left_index=True, right_index=True, how='right')



se_15=pd.merge(se_14, se16, left_index=True, right_index=True, how='right')



se_16=pd.merge(se_15, se17, left_index=True, right_index=True, how='right')



se_17=pd.merge(se_16, se18, left_index=True, right_index=True, how='right')



print (se_17.shape)

print (se_17.head(2))



# For Annual COnsumption



ac_09=pd.merge(ac09, ac10, left_index=True, right_index=True, how='right')



ac_10=pd.merge(ac_09, ac11, left_index=True, right_index=True, how='right')



ac_11=pd.merge(ac_10, ac12, left_index=True, right_index=True, how='right')



ac_12=pd.merge(ac_11, ac13, left_index=True, right_index=True, how='right')



ac_13=pd.merge(ac_12, ac14, left_index=True, right_index=True, how='left')



ac_14=pd.merge(ac_13, ac15, left_index=True, right_index=True, how='right')



ac_15=pd.merge(ac_14, ac16, left_index=True, right_index=True, how='right')



ac_16=pd.merge(ac_15, ac17, left_index=True, right_index=True, how='right')



ac_17=pd.merge(ac_16, ac18, left_index=True, right_index=True, how='right')



print (ac_17.shape)

print (ac_17.head(2))

# Now Adjusting Dataframe with features 'Year' and the 'Total' 

# Smart Meters

# we can use the sum() method to get the total population per year

df_tot_sm = pd.DataFrame(sm_17.sum(axis=0))



# change the years to type int (useful for regression later on)

df_tot_sm.index = map(int, df_tot_sm.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot_sm.reset_index(inplace = True)



# rename columns

df_tot_sm.columns = ['Year', 'SM-Total']



# view the final dataframe

print (df_tot_sm.head(2))



# Solar Energy

# we can use the sum() method to get the total population per year

df_tot_se = pd.DataFrame(se_17.sum(axis=0))



# change the years to type int (useful for regression later on)

df_tot_se.index = map(int, df_tot_se.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot_se.reset_index(inplace = True)



# rename columns

df_tot_se.columns = ['Year', 'SE-Total']



# view the final dataframe

print (df_tot_se.head(2))



# Annual Consumption

# we can use the sum() method to get the total population per year

df_tot_ac = pd.DataFrame(ac_17.sum(axis=0))



# change the years to type int (useful for regression later on)

df_tot_ac.index = map(int, df_tot_ac.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot_ac.reset_index(inplace = True)



# rename columns

df_tot_ac.columns = ['Year', 'AC-Total']



# view the final dataframe

print (df_tot_ac.head(2))
# Smart Meter Distribution(Histogram)

sm_17.plot(kind='hist',  figsize=(10, 6))

plt.title('Histogram of Smart Meters from 2009 - 2018')

plt.ylabel('Total Percentage Value')

plt.xlabel('Spreading')



plt.show()



# Smart Meter Spreading Over the Years

df_tot_sm.set_index('Year', inplace=True)

df_tot_sm.plot(kind='bar',figsize=(12,8))



plt.title('Smart Meters Spreading over the years from 2009-2018')

plt.ylabel('Total Percentage Value')

plt.xlabel('Years')

#plt.text(8.1,2.45e7, '2018 Reading') # Note

plt.xticks(rotation=360)

plt.show() # need this line to show the updates made to the figure



# Solar Energy Trend

df_tot_se.plot(kind='line', x='Year', y='SE-Total', figsize=(10, 6), color='darkblue')



plt.title('Solar Energy Trend from 2009 - 2018')

plt.xlabel('Year')

plt.ylabel('Total Percentage Value')



plt.show()
pd.set_option('display.float_format', lambda x: '%.1f' % x)

plt.figure(figsize=(8,5))

x_data = df_tot_ac['Year'].values

y_data =df_tot_ac[['AC-Total']].values

plt.plot(x_data,y_data, 'ro')

plt.ylabel('Annual Consumption')

plt.xlabel('Year')

plt.show()
import scipy.optimize as optimize

from scipy.optimize import curve_fit

x=df_tot_ac['Year'].values # Data

y=df_tot_ac['AC-Total'].values # Data

plt.figure(figsize=(12,5))

plt.plot(x, y, 'ro', label="original data")

def log(x, a, b): # For non linear 

    return    np.array(a) +  np.array(b) * (np.log(x)) # u can use too np.array(a)

popt, pcov = curve_fit(log, x, y)#,  maxfev=1000)

#print the final parameters

print(" a = %f, b = %f" % (popt[0], popt[1]))

t = np.linspace(1,1009*3, 10)

plt.plot(t, log(t, *popt), label="Fitted Curve")

plt.legend(loc='lower right')

plt.show()
# Normalizing

xdata =x/max(x)

ydata =y/max(y)
# split data into train/test

msk=np.random.rand(len(df_tot_ac)) < .8

train_x=xdata[msk]

test_x=ydata[~msk]

train_y=xdata[msk]

test_y=ydata[~msk]



# build the model using train set

popt, pcov = curve_fit(log, train_x, train_y, maxfev=1000)



# predict using test set

y_hat = log(test_x.flatten(), *popt)
from sklearn.metrics import r2_score

# evaluation

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(y_hat , test_y) )


