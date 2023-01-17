import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Reading the data

data1a = pd.read_csv('/kaggle/input/GSDP.csv')
data1a.head()
# Basic info regarding the data

data1a.info()
# Observe the various columns in the dataset

data1a.columns
# Remove the rows: (% Growth over the previous year)' and 'GSDP - CURRENT PRICES (in Crore) for the year 2016-17.

data1a = data1a[data1a['Duration'] != '2016-17']

data1a
# Check the total number of null values in each columns

data1a.isnull().sum()
# Check if any column has all the values as NAN

data1a.isnull().all(axis=0)
# removing West Bengal as the whole column is NAN

data1a = data1a.drop('West Bengal1', axis = 1)
data1a
data1a.iloc[6:].isnull().sum() # since there are at max. only 1 missing value we can take the average of the other two numbers
avg_growth = data1a.iloc[6:]
avg_growth #dataframe to find the average growth of states
avg_growth.columns
# Taking only the values for the states

average_growth_values = avg_growth[avg_growth.columns[2:34]].mean()  
# Sorting the average growth rate values and then making a dataframe for all the states

average_growth_values = average_growth_values.sort_values()

average_growth_rate = average_growth_values.to_frame(name='Average growth rate')

average_growth_rate
# plotting the average growth rate for all the states

plt.figure(figsize=(12,10), dpi = 300)



sns.barplot(x = average_growth_rate['Average growth rate'], y = average_growth_values.index,palette='viridis')

plt.xlabel('Average Growth Rate', fontsize=12)

plt.ylabel('States', fontsize=12)

plt.title('Average Growth Rate for all the states',fontsize=13)

plt.show()
# top 5 states as per average growth rate



average_growth_rate['Average growth rate'][-5:]
# top 5 states as per average growth rate for the years 2013-14, 2014-15, 2015-16



avg_growth[['Mizoram','Tripura','Nagaland','Manipur','Arunachal Pradesh']]
#create a dataframe to store the mean and the standard deviation of the growth rate for various states



describe = pd.DataFrame(avg_growth.describe())

describe = describe.T

describe
# states having mean growth rate greater than 12 and standard deviation less than 2



describe[(describe['mean']>12) & (describe['std']<2)]
# states having mean growth rate greater than 13 and standard deviation greater than 2



describe[(describe['mean']<12) & (describe['std']>2)]
data1a.head()
# filtering out the data for the year 2015-16 and storing it in a dataframe

total_GDP_15_16 = data1a[(data1a['Items  Description'] == 'GSDP - CURRENT PRICES (` in Crore)') & (data1a['Duration'] == '2015-16')]

total_GDP_15_16
# carrying out necessary transformation to make the data ready for plotting



total_GDP_15_16_states = total_GDP_15_16[total_GDP_15_16.columns[2:34]].transpose()

total_GDP_15_16_states = total_GDP_15_16_states.rename(columns={4: 'Total GDP of States 2015-16'})

total_GDP_15_16_states = total_GDP_15_16_states.dropna()

total_GDP_15_16_states = total_GDP_15_16_states.sort_values('Total GDP of States 2015-16',ascending=True)

total_GDP_15_16_states
plt.figure(figsize=(10,8), dpi = 600)



sns.barplot(x = total_GDP_15_16_states['Total GDP of States 2015-16'], y = total_GDP_15_16_states.index,palette='plasma')

plt.xlabel('Total GDP of States for 2015-16', fontsize=12)

plt.ylabel('States', fontsize=12)

plt.title('Total GDP of States 2015-16 for all the states',fontsize=12)

plt.show()
top_5_eco = total_GDP_15_16_states[-5:]

top_5_eco
bottom_5_eco = total_GDP_15_16_states[:5]

bottom_5_eco
Andhra_Pradesh = pd.read_csv('/kaggle/input/NAD-Andhra_Pradesh-GSVA_cur_2016-17.csv')
Arunachal_Pradesh = pd.read_csv('/kaggle/input/NAD-Arunachal_Pradesh-GSVA_cur_2015-16.csv')
Assam = pd.read_csv('/kaggle/input/NAD-Assam-GSVA_cur_2015-16.csv')
Bihar = pd.read_csv('/kaggle/input/NAD-Bihar-GSVA_cur_2015-16.csv')
Chhattisgarh = pd.read_csv('/kaggle/input/NAD-Chhattisgarh-GSVA_cur_2016-17.csv')
Goa = pd.read_csv('/kaggle/input/NAD-Goa-GSVA_cur_2015-16.csv')
Gujarat = pd.read_csv('/kaggle/input/NAD-Gujarat-GSVA_cur_2015-16.csv')
Haryana = pd.read_csv('/kaggle/input/NAD-Haryana-GSVA_cur_2016-17.csv')
Himachal_Pradesh = pd.read_csv('/kaggle/input/NAD-Himachal_Pradesh-GSVA_cur_2014-15.csv')
Jharkhand = pd.read_csv('/kaggle/input/NAD-Jharkhand-GSVA_cur_2015-16.csv')
Karnataka = pd.read_csv('/kaggle/input/NAD-Karnataka-GSVA_cur_2015-16.csv')
Kerala = pd.read_csv('/kaggle/input/NAD-Kerala-GSVA_cur_2015-16.csv')
Madhya_Pradesh = pd.read_csv('/kaggle/input/NAD-Madhya_Pradesh-GSVA_cur_2016-17.csv')
Maharashtra = pd.read_csv('/kaggle/input/NAD-Maharashtra-GSVA_cur_2014-15.csv')
Manipur = pd.read_csv('/kaggle/input/NAD-Manipur-GSVA_cur_2014-15.csv')
Meghalaya = pd.read_csv('/kaggle/input/NAD-Meghalaya-GSVA_cur_2016-17.csv')
Mizoram = pd.read_csv('/kaggle/input/NAD-Mizoram-GSVA_cur_2014-15.csv')
Nagaland = pd.read_csv('/kaggle/input/NAD-Nagaland-GSVA_cur_2014-15.csv')
Odisha = pd.read_csv('/kaggle/input/NAD-Odisha-GSVA_cur_2016-17.csv')
Punjab = pd.read_csv('/kaggle/input/NAD-Punjab-GSVA_cur_2014-15.csv')
Rajasthan = pd.read_csv('/kaggle/input/NAD-Rajasthan-GSVA_cur_2014-15.csv')
Sikkim = pd.read_csv('/kaggle/input/NAD-Sikkim-GSVA_cur_2015-16.csv')
Tamil_Nadu = pd.read_csv('/kaggle/input/NAD-Tamil_Nadu-GSVA_cur_2016-17.csv')
Telangana = pd.read_csv('/kaggle/input/NAD-Telangana-GSVA_cur_2016-17.csv')
Tripura = pd.read_csv('/kaggle/input/NAD-Tripura-GSVA_cur_2014-15.csv')
Uttar_Pradesh = pd.read_csv('/kaggle/input/NAD-Uttar_Pradesh-GSVA_cur_2015-16.csv')
Uttarakhand = pd.read_csv('/kaggle/input/NAD-Uttarakhand-GSVA_cur_2015-16.csv')
andhra_pradesh = Andhra_Pradesh[['S.No.','Item', '2014-15']]

andhra_pradesh = andhra_pradesh.rename(columns={'2014-15': 'Andhra_Pradesh'})



arunachal_pradesh = Arunachal_Pradesh[['S.No.','Item', '2014-15']]

arunachal_pradesh = arunachal_pradesh.rename(columns={'2014-15': 'Arunachal_Pradesh'})



assam = Assam[['S.No.','Item', '2014-15']]

assam = assam.rename(columns={'2014-15': 'Assam'})



bihar = Bihar[['S.No.','Item', '2014-15']]

bihar = bihar.rename(columns={'2014-15': 'Bihar'})



chhattisgarh = Chhattisgarh[['S.No.','Item', '2014-15']]

chhattisgarh = chhattisgarh.rename(columns={'2014-15': 'Chhattisgarh'})



goa = Goa[['S.No.','Item', '2014-15']]

goa = goa.rename(columns={'2014-15': 'Goa'})



gujarat = Gujarat[['S.No.','Item', '2014-15']]

gujarat = gujarat.rename(columns={'2014-15': 'Gujarat'})



haryana = Haryana[['S.No.','Item', '2014-15']]

haryana = haryana.rename(columns={'2014-15': 'Haryana'})



himachal_Pradesh = Himachal_Pradesh[['S.No.','Item', '2014-15']]

himachal_Pradesh = himachal_Pradesh.rename(columns={'2014-15': 'Himachal_Pradesh'})



jharkhand = Jharkhand[['S.No.','Item', '2014-15']]

jharkhand = jharkhand.rename(columns={'2014-15': 'Jharkhand'})



karnataka = Karnataka[['S.No.','Item', '2014-15']]

karnataka = karnataka.rename(columns={'2014-15': 'Karnataka'})



kerala = Kerala[['S.No.','Item', '2014-15']]

kerala = kerala.rename(columns={'2014-15': 'Kerala'})



madhya_pradesh = Madhya_Pradesh[['S.No.','Item', '2014-15']]

madhya_pradesh = madhya_pradesh.rename(columns={'2014-15': 'Madhya_Pradesh'})



maharashtra = Maharashtra[['S.No.','Item', '2014-15']]

maharashtra = maharashtra.rename(columns={'2014-15': 'Maharashtra'})



manipur = Manipur[['S.No.','Item', '2014-15']]

manipur = manipur.rename(columns={'2014-15': 'Manipur'})



meghalaya = Meghalaya[['S.No.','Item', '2014-15']]

meghalaya = meghalaya.rename(columns={'2014-15': 'Meghalaya'})



mizoram = Mizoram[['S.No.','Item', '2014-15']]

mizoram = mizoram.rename(columns={'2014-15': 'Mizoram'})



nagaland = Nagaland[['S.No.','Item', '2014-15']]

nagaland = nagaland.rename(columns={'2014-15': 'Nagaland'})



odisha = Odisha[['S.No.','Item', '2014-15']]

odisha = odisha.rename(columns={'2014-15': 'Odisha'})



punjab = Punjab[['S.No.','Item', '2014-15']]

punjab = punjab.rename(columns={'2014-15': 'Punjab'})



rajasthan = Rajasthan[['S.No.','Item', '2014-15']]

rajasthan = rajasthan.rename(columns={'2014-15': 'Rajasthan'})



sikkim = Sikkim[['S.No.','Item', '2014-15']]

sikkim = sikkim.rename(columns={'2014-15': 'Sikkim'})



tamil_nadu = Tamil_Nadu[['S.No.','Item', '2014-15']]

tamil_nadu = tamil_nadu.rename(columns={'2014-15': 'Tamil_Nadu'})



telangana = Telangana[['S.No.','Item', '2014-15']]

telangana = telangana.rename(columns={'2014-15': 'Telangana'})



tripura = Tripura[['S.No.','Item', '2014-15']]

tripura = tripura.rename(columns={'2014-15': 'Tripura'})



uttar_pradesh = Uttar_Pradesh[['S.No.','Item', '2014-15']]

uttar_pradesh = uttar_pradesh.rename(columns={'2014-15': 'Uttar_Pradesh'})



uttarakhand = Uttarakhand[['S.No.','Item', '2014-15']]

uttarakhand = uttarakhand.rename(columns={'2014-15': 'Uttarakhand'})
# Merging all the tables for different states into a single dataframe



dfs = [andhra_pradesh,arunachal_pradesh, assam, bihar, chhattisgarh, goa, gujarat, haryana,himachal_Pradesh,

       jharkhand, karnataka,kerala,madhya_pradesh, maharashtra,manipur,meghalaya,mizoram, nagaland,odisha,

       punjab,rajasthan,sikkim,tamil_nadu,telangana,tripura,uttarakhand, uttar_pradesh]





from functools import reduce

df_final = reduce(lambda left,right: pd.merge(left,right,how ='left',on=['S.No.', 'Item']), dfs)
df_final.columns
# Renaming some of the state names for merging data at a later stage



df_final = df_final.rename(columns={'Andhra_Pradesh':'Andhra Pradesh', 'Arunachal_Pradesh':'Arunachal Pradesh',

                                   'Himachal_Pradesh':'Himachal Pradesh','Madhya_Pradesh':'Madhya Pradesh',

                                   'Tamil_Nadu':'Tamil Nadu','Uttar_Pradesh':'Uttar Pradesh',

                                   'Chhattisgarh':'Chhatisgarh','Uttarakhand':'Uttrakhand'})
# Final dataframe having the data for all the states for all the sectors and subsectors of the economy



df_final
gdp_per_capita = df_final.iloc[32][2:].sort_values()

gdp_per_capita = gdp_per_capita.to_frame(name = 'GDP per capita')

gdp_per_capita
plt.figure(figsize=(12,8), dpi=600)                             



sns.barplot(x = gdp_per_capita['GDP per capita'], y =gdp_per_capita.index, palette='Reds' )

plt.xlabel('GDP per capita', fontsize=12)

plt.ylabel('States', fontsize=12)

plt.title('GDP per capita vs States',fontsize=12)

plt.show()
top_5_gdp_per_capita = gdp_per_capita[-5:]

top_5_gdp_per_capita
bottom_5_gdp_per_capita = gdp_per_capita[:5]

bottom_5_gdp_per_capita
ratio = gdp_per_capita['GDP per capita'].max()/gdp_per_capita['GDP per capita'].min()

print('The Ratio of highest per capita GDP to the lowest per capita GDP is: ',ratio)
# Identifying the Primary, Secondary and the tertiary sectors and concating these to form a dataframe



primary = df_final[df_final['Item']=='Primary']

secondary = df_final[df_final['Item']=='Secondary']

tertiary = df_final[df_final['Item']=='Tertiary']

gdp = df_final[df_final['Item']=='Gross State Domestic Product']



pst = pd.concat([primary, secondary,tertiary,gdp], axis = 0).reset_index()

pst =  pst.drop(['index','S.No.'], axis = 1).set_index('Item')
pst
# calculating the percentage contribution of each sector to the Gross State Domestic Product for each state



pst.loc['primary_percentage'] = pst.loc['Primary'] / pst.loc['Gross State Domestic Product'] * 100

pst.loc['secondary_percentage'] = pst.loc['Secondary'] / pst.loc['Gross State Domestic Product'] * 100

pst.loc['tertiary_percentage'] = pst.loc['Tertiary'] / pst.loc['Gross State Domestic Product'] * 100
pst
# Transposing the dataframe for better readability



pst = pst.T

pst = pst.sort_values('Gross State Domestic Product')

pst
plt.figure(figsize=(12,10), dpi =600)



bars1 = pst['primary_percentage']

bars2 = pst['secondary_percentage']

bars3 = pst['tertiary_percentage']

 

legends = ['Primary %', 'Secondary %', 'Tertiary %']



bars = np.add(bars1, bars2).tolist()

 

r = np.arange(0,len(pst.index))

 

names = pst.index

barWidth = 1

 

# Create red bars

plt.bar(r, bars1, color='red', edgecolor='white')

# Create green bars (middle), on top of the firs ones

plt.bar(r, bars2, bottom=bars1, color='green', edgecolor='white')

# Create blue bars (top)

plt.bar(r, bars3, bottom=bars, color='blue', edgecolor='white')

 

plt.xticks(r, names,rotation=90)

plt.xlabel('States',fontsize=12)

plt.ylabel('Percentage contribution to GDP',fontsize=12)

plt.title('Percentage contribution of the Primary, Secondary and Tertiary sectors as a percentage of the total GDP for all the states')



plt.legend(legends)



plt.tight_layout()

gdp_per_capita
# States between the 85th and 100th percentile



C1 = gdp_per_capita[gdp_per_capita['GDP per capita'] > gdp_per_capita['GDP per capita'].quantile(0.85)]

C1
# States between the 50th and 85th percentile



C2 = gdp_per_capita[(gdp_per_capita['GDP per capita'] > gdp_per_capita['GDP per capita'].quantile(0.50)) & (gdp_per_capita['GDP per capita'] < gdp_per_capita['GDP per capita'].quantile(0.85))]

C2
# States between the 20th and 50th percentile



C3 = gdp_per_capita[(gdp_per_capita['GDP per capita'] > gdp_per_capita['GDP per capita'].quantile(0.20)) & (gdp_per_capita['GDP per capita'] <= gdp_per_capita['GDP per capita'].quantile(0.50))]

C3
# States below the 20th percentile



C4 = gdp_per_capita[gdp_per_capita['GDP per capita'] < gdp_per_capita['GDP per capita'].quantile(0.20)]

C4
C1_df = df_final[['S.No.','Item']+list(states for states in C1.index)]

C2_df = df_final[['S.No.','Item']+list(states for states in C2.index)]

C3_df = df_final[['S.No.','Item']+list(states for states in C3.index)]

C4_df = df_final[['S.No.','Item']+list(states for states in C4.index)]
C1_df = C1_df.iloc[[0,5,7,8,9,11,14,22,23,24,25,30,32]]

C2_df = C2_df.iloc[[0,5,7,8,9,11,14,22,23,24,25,30,32]]

C3_df = C3_df.iloc[[0,5,7,8,9,11,14,22,23,24,25,30,32]]

C4_df = C4_df.iloc[[0,5,7,8,9,11,14,22,23,24,25,30,32]]
C1_df.reset_index(drop=True, inplace=True)

C2_df.reset_index(drop=True, inplace=True)

C3_df.reset_index(drop=True, inplace=True)

C4_df.reset_index(drop=True, inplace=True)
C1_df
# Creating the column for Total values for all sub-sectors for all the states and the column for the percentage contribution

# to the total GSDP by each of the sub-sectors for all the states



C1_df['Total for all states'] = C1_df['Kerala']+C1_df['Haryana']+C1_df['Sikkim']+C1_df['Goa']

C1_df['Percentage of Total GDP'] = C1_df['Total for all states']/C1_df['Total for all states'][11] * 100

C1_df
# Identifying the major sub-sectors contributing more to the GSDP  by finding the cumulative sum



C1_contributor = C1_df[['Item','Percentage of Total GDP']][:-2].sort_values(by='Percentage of Total GDP', ascending=False)

C1_contributor.reset_index(drop=True, inplace=True)

C1_contributor['Cumulative sum'] = C1_contributor['Percentage of Total GDP'].cumsum()

C1_contributor
plt.figure(figsize=(6,4), dpi=600)

sns.barplot(y=C1_contributor['Item'], x = C1_contributor['Percentage of Total GDP'], palette='inferno')

plt.xlabel("Percentage of Total GSDP for C1 States")

plt.ylabel('Sub-sectors')

plt.title('Percentage of Total GSDP for C1 States vs Sub-sectors')

plt.savefig("Percentage of Total GSDP for C1 States vs Sub-sectors.png", bbox_inches='tight', dpi=600)



plt.show()
C2_df['Total for all states']=list(C2_df[list(states for states in C2_df.columns)[2:]].sum(axis=1))

C2_df['Percentage of Total GDP'] = C2_df['Total for all states']/C2_df['Total for all states'][11] * 100

C2_contributor = C2_df[['Item','Percentage of Total GDP']][:-2].sort_values(by='Percentage of Total GDP', ascending=False)

C2_contributor.reset_index(drop=True, inplace=True)

C2_contributor['Cumulative sum'] = C2_contributor['Percentage of Total GDP'].cumsum()

C2_contributor
plt.figure(figsize=(6,4), dpi=600)

sns.barplot(y=C2_contributor['Item'], x = C2_contributor['Percentage of Total GDP'],palette='hot')

plt.xlabel("Percentage of Total GSDP for C2 States")

plt.ylabel('Sub-sectors')

plt.title('Percentage of Total GSDP for C2 States vs Sub-sectors')

plt.show()
C3_df['Total for all states']=list(C3_df[list(states for states in C3_df.columns)[2:]].sum(axis=1))

C3_df['Percentage of Total GDP'] = C3_df['Total for all states']/C3_df['Total for all states'][11] * 100

C3_contributor = C3_df[['Item','Percentage of Total GDP']][:-2].sort_values(by='Percentage of Total GDP', ascending=False)

C3_contributor.reset_index(drop=True, inplace=True)

C3_contributor['Cumulative sum'] = C3_contributor['Percentage of Total GDP'].cumsum()

C3_contributor
plt.figure(figsize=(6,4), dpi=600)

sns.barplot(y=C3_contributor['Item'], x = C3_contributor['Percentage of Total GDP'], palette='autumn')

plt.xlabel("Percentage of Total GSDP for C3 States")

plt.ylabel('Sub-sectors')

plt.title('Percentage of Total GSDP for C3 States vs Sub-sectors')



plt.show()
C4_df['Total for all states']=list(C4_df[list(states for states in C4_df.columns)[2:]].sum(axis=1))

C4_df['Percentage of Total GDP'] = C4_df['Total for all states']/C4_df['Total for all states'][11] * 100

C4_contributor = C4_df[['Item','Percentage of Total GDP']][:-2].sort_values(by='Percentage of Total GDP', ascending=False)

C4_contributor.reset_index(drop=True, inplace=True)

C4_contributor['Cumulative sum'] = C4_contributor['Percentage of Total GDP'].cumsum()

C4_contributor
plt.figure(figsize=(6,4), dpi=600)

sns.barplot(y=C4_contributor['Item'], x = C4_contributor['Percentage of Total GDP'], palette='spring')

plt.xlabel("Percentage of Total GSDP for C4 States")

plt.ylabel('Sub-sectors')

plt.title('Percentage of Total GSDP for C4 States vs Sub-sectors')



plt.show()
# Reading the data and selecting the data for the year 2014-14 and the education level for Primary, Upper Primary and Secondary



data2 = pd.read_csv('/kaggle/input/droupout rate.csv')

data2 = data2[['Level of Education - State','Primary - 2014-2015.1','Upper Primary - 2014-2015','Secondary - 2014-2015']]

data2
# Dropping rows of data which we don not need like Union Territories and for which we don't have GDP per-capita available like West Bengal



data2 =  data2.drop([0,5,7,8,9,14,18,26,35,36])

data2 = data2.reset_index(drop = True)

data2=data2.rename(columns={'Level of Education - State': 'State'})
# Necessary transformation like resetting the index and renaming the column name for merging with another dataframe



states_gdp_per_capita = gdp_per_capita.reset_index()

states_gdp_per_capita=states_gdp_per_capita.rename(columns={'index':'State'})
# Merging the above dataframe with the GDP per-capita dataframe



data2_final = pd.merge(data2,states_gdp_per_capita,how='left',on=['State'])
data2_final = data2_final.rename(columns={'State':'Level of education - State'})
# Final dataframe having the education level dropout rates for all the states and the GDP per capita



data2_final
data2_final.describe()
# Primary - 2014-2015.1



plt.figure(figsize=(8,6), dpi= 600)



sns.regplot(y=data2_final['GDP per capita'],x=data2_final['Primary - 2014-2015.1'])

plt.xlabel('Primary Drop out rate')

plt.ylabel('Per capita GDP')

plt.title('Per capita GDP vs Primary Drop out rate')

plt.show()
# Upper Primary - 2014-2015



plt.figure(figsize=(8,6), dpi= 600)



sns.regplot(y=data2_final['GDP per capita'],x=data2_final['Upper Primary - 2014-2015'])

plt.xlabel('Upper Primary Drop out rate')

plt.ylabel('Per capita GDP')

plt.title('Per capita GDP vs Upper Primary Drop out rate')

plt.show()
# Secondary - 2014-2015



plt.figure(figsize=(8,6), dpi= 100)



sns.regplot(y=data2_final['GDP per capita'],x=data2_final['Secondary - 2014-2015'])

plt.xlabel('Secondary Drop out rate')

plt.ylabel('Per capita GDP')

plt.title('Per capita GDP vs Secondary Drop out rate')

plt.show()