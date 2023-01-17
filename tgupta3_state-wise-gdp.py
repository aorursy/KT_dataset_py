import pandas as pd
import numpy as np
import matplotlib as plt
import warnings
warnings.filterwarnings("ignore")
path =r'../input/statesdata/State_wise_GDP.csv' #path of the file for question1
df                =    pd.read_csv(path)
df                =    df.dropna(axis=1, how='all') # remove columns whose all values are null
df_new            =    df[df['Items  Description'] == '(% Growth over previous year)'] #remove all the rows with item description other than growth over previous year
df_new            =    df_new[(df_new['Duration'] !="2016-17") ]#this is to remove all the rows for the duration 2016-17
df_new            =    df_new[(df_new['Duration'] !="2011-12")]# this is to remove all the rows with values 2011-12
df_new            =    df_new[(df_new['Duration'] !="2012-13")]# this is to remove all the rows with values 2012-13
df_new.loc['mean']=    df_new.mean() # this is to add a new row in the dataframe
df_new            =    df_new.reset_index(drop=True)
df_avg            =    df_new.loc[0,:] #taking all the columns of state names
df_avg            =    df_new.iloc[-1,:] # taking all the mean values
df_avg            =    df_avg.iloc[2:-1] # this is to exclude, duration and items as the row names from the new dataframes which includes averages
df_avg_sorted     =    df_avg.sort_values(ascending=False)# the data has been captured

#plotting the dataframe with average values for three years in a descending order
df_avg_sorted.plot(kind="bar",figsize=(10,7))
plt.pyplot.xlabel("States")
plt.pyplot.ylabel("(% Growth over previous year)")
plt.pyplot.title("average percentage growth over a period of 2013-16")
df_curosity             =    df[['Jammu & Kashmir','All_India GDP','Items  Description','Duration']]
df_curosity             =    df_curosity[df_curosity['Items  Description'] == '(% Growth over previous year)'] #remove all the rows with item description other than growth over previous year
df_curosity             =    df_curosity[(df_curosity['Duration'] !="2016-17") ]#this is to remove all the rows for the duration 2016-17
df_curosity             =    df_curosity[(df_curosity['Duration'] !="2011-12")]# this is to remove all the rows with values 2011-12
df_curosity             =    df_curosity[(df_curosity['Duration'] !="2012-13")]
df_curosity.loc['mean'] =    df_curosity.mean()
df_curosity_plt         =   df_curosity.iloc[-1,:2] 
df_curosity_plt.plot(kind="bar")
plt.pyplot.title("J&K vs ALL_INDIA_GDP")
plt.pyplot.ylabel("Mean GDP over three years")
#Plot the total GDP of the states for the year 2015-16:
df_2015_16         = df[df['Duration'] == "2015-16"] #here the data frame used was created in the first step from the csv             
df_2015_16         = df_2015_16[df_2015_16['Items  Description'] =="GSDP - CURRENT PRICES (` in Crore)"]
#df_2015_16 = df_2015_16.T
df_2015_16         = df_2015_16.reset_index(drop=True)
df_2015_16_sorted  = df_2015_16.loc[0,:]
df_2015_16_sorted  = df_2015_16_sorted.iloc[2:-1].sort_values(ascending=False)
df_2015_16_sorted.plot(kind="bar",figsize=(10,5))
plt.pyplot.xlabel("States")
plt.pyplot.ylabel("GSDP - CURRENT PRICES (` in Crore)")
plt.pyplot.title("total GDP of the states")
#pat1 b 
import glob # this library is used to import all the files with a particular regula expression
path      = r'../input/states-data/'
all_files = glob.glob(path+"*")
lst       = []
for file in all_files:
    state = file.split("/")[-1].split('-')[1] #i have used - to split to get the proper name of the state
    if(state == "Chandigarh" or state =="Delhi" or state == "Puducherry"):#this step will remove UT's
        continue
    df         =   pd.read_csv(file,encoding = "ISO-8859-1") #encoding was done to remove the error while fethcing the values during read_csv
    df         =   df [['Item','2014-15']]
    df         =   df.T
    new_header =   df.iloc[0]
    new_header =   new_header.str.replace("*","") #replacing all the stars from the column names
    df.columns =   new_header
    df =df [1:]
    df.loc[:,'States'] = state
    lst.append(df)
frame       =  pd.concat(lst, axis=0,ignore_index = True)#the total dataframe for all the files
plot_gsdp   =  frame[['States','Per Capita GSDP (Rs.)']] #partial subset of the main data containign two columns
plot_gsdp   =  plot_gsdp.set_index(['States']).sort_values(by = 'Per Capita GSDP (Rs.)',ascending=False)
plot_gsdp.plot(kind="bar",figsize=(10,7))
plt.pyplot.title("Per Capita in Rs for all the states for the year 2014-2015")
plt.pyplot.ylabel("Per Capita GSDP (Rs.)")
#Find the ratio of the highest per capita GDP to the lowest per capita GDP.
ratio = plot_gsdp[['Per Capita GSDP (Rs.)']].max()/plot_gsdp[['Per Capita GSDP (Rs.)']].min()
print(ratio)
# the ratio is found to be 8.004742
#per_contri is a datframe which will show the percentage contribution of all the sectors, primary, secondary and tertiary over all the states
per_contri                  = frame[['States','Primary','Secondary','Tertiary','Gross State Domestic Product']]
per_contri                  = per_contri.sort_values(by = 'Gross State Domestic Product',ascending = False)

#take the only columns with the name as primary, secondary and tertiary and name them as primary_per,secondary_per,tertiary_per
per_contri['primary_per']   = per_contri['Primary']/per_contri['Gross State Domestic Product']
per_contri['secondary_per'] = per_contri['Secondary']/per_contri['Gross State Domestic Product']
per_contri['tertiary_per']  = per_contri['Tertiary']/per_contri['Gross State Domestic Product']

#set the index to states for better and easy understanding and representation
per_contri                  = per_contri.set_index(['States'])

#loc is used to fetch only the columns which have 'primary_per','secondary_per','tertiary_per' as column names
per_contri.loc[:,['primary_per','secondary_per','tertiary_per']].plot.bar(stacked="True",figsize=(10,7))
plt.pyplot.title("percentage of all the three sectors for the year 2014-2015")
plt.pyplot.ylabel("percentage contribution of primary,secondary and tertiary sectors")
#frame.columns, in the below line we are fetching all the column names except for the sector names and population etc

quant = frame[['Agriculture, forestry and fishing',
       'Mining and quarrying', 'Manufacturing',
       'Electricity, gas, water supply & other utility services',
       'Construction', 'Trade, repair, hotels and restaurants',
       'Transport, storage, communication & services related to broadcasting',
       'Financial services',
       'Real estate, ownership of dwelling & professional services',
       'Public administration', 'Other services','Gross State Domestic Product','Per Capita GSDP (Rs.)']]
#quant = frame[['Gross State Domestic Product','Per Capita GSDP (Rs.)','Agriculture, forestry and fishing','Mining and quarrying','Manufacturing','Electricity, gas, water supply & other utility services', 'Construction','Trade & repair services','Transport, storage, communication & services related to broadcasting','Financial services','Real estate, ownership of dwelling & professional services','Public administration','Other services']]
quant['Per Capita GSDP (Rs.)'] =  quant[['Per Capita GSDP (Rs.)','Gross State Domestic Product']].astype('float')
quantile_frame                 =  quant.quantile(q=[0.2, 0.5, 0.85,1])
quantile_ranks                 = []

#itrrows is used to loop through the all the rows in pandas dataframe and it has been used to calculate the quartile and 
#categorise them in c1,c2,c3,c4 categories etc.

for index, row in quant.iterrows():
    if (row['Per Capita GSDP (Rs.)'] <= quantile_frame.loc[0.2]['Per Capita GSDP (Rs.)']):
        quantile_ranks.append("c4")
    elif (row['Per Capita GSDP (Rs.)'] > quantile_frame.loc[0.2]['Per Capita GSDP (Rs.)'] and row['Per Capita GSDP (Rs.)'] <= quantile_frame.loc[0.5]['Per Capita GSDP (Rs.)']):
        quantile_ranks.append("c3")
    elif (row['Per Capita GSDP (Rs.)'] > quantile_frame.loc[0.5]['Per Capita GSDP (Rs.)'] and row['Per Capita GSDP (Rs.)'] <= quantile_frame.loc[0.85]['Per Capita GSDP (Rs.)']):
        quantile_ranks.append("c2")
    else:
        quantile_ranks.append("c1")
        
#in the below line a new column is created which will append the categories in the dataframe        
quant['quartile'] = quantile_ranks
#four dataframes are created here for each category
quant_c1 = quant[quant['quartile'] =="c1"].set_index(['quartile'])
quant_c2 = quant[quant['quartile'] =="c2"].set_index(['quartile'])
quant_c3 = quant[quant['quartile'] =="c3"].set_index(['quartile'])
quant_c4 = quant[quant['quartile'] =="c4"].set_index(['quartile'])
# plot the graph for category c1
quant_c1              = quant_c1.append(quant_c1.agg(['sum']))
val                   = quant_c1.loc['sum']['Gross State Domestic Product'] # value to fnd the ratio of each subsector
quant_c1.loc['ratio'] = quant_c1.loc['sum']/val
quant_c1              = quant_c1.loc['ratio'].sort_values(ascending=False)
quant_c1              = quant_c1.drop('Gross State Domestic Product')
quant_c1              = quant_c1[quant_c1.cumsum() <= 0.8]
#plot the graph with the below code for c1 category
quant_c1.plot(kind="bar" )
plt.pyplot.title("Types of subsectors which includes to top 80 % for category 1")
plt.pyplot.xlabel("types of subsectors")
plt.pyplot.ylabel("percentage contibution of Subsectors")
print(val)
# plot the graph for category c2
quant_c2               = quant_c2.append(quant_c2.agg(['sum']))
val                    = quant_c2.loc['sum']['Gross State Domestic Product'] # value to fnd the ratio of each subsector
quant_c2.loc['ratio']  = quant_c2.loc['sum']/val
quant_c2               = quant_c2.loc['ratio'].sort_values(ascending=False)
quant_c2               = quant_c2.drop('Gross State Domestic Product')
quant_c2               = quant_c2[quant_c2.cumsum() <= 0.8]

#c2 categroy dataframe is created , plot the same with the below code
quant_c2.plot(kind="bar")
plt.pyplot.title("Types of subsectors which includes to top 80 % for category c2")
plt.pyplot.xlabel("types of subsectors")
plt.pyplot.ylabel("percentage contibution of Subsectors")
# plot the graph for category c3
quant_c3              = quant_c3.append(quant_c3.agg(['sum']))
val                   = quant_c3.loc['sum']['Gross State Domestic Product'] # value to fnd the ratio of each subsector
quant_c3.loc['ratio'] = quant_c3.loc['sum']/val
quant_c3              = quant_c3.loc['ratio'].sort_values(ascending=False)
quant_c3              = quant_c3.drop('Gross State Domestic Product')
quant_c3              = quant_c3[quant_c3.cumsum() <= 0.8]

#plot the graph for c3 category 
quant_c3.plot(kind="bar")
plt.pyplot.title("Types of subsectors which includes to top 80 % for category c3")
plt.pyplot.xlabel("types of subsectors")
plt.pyplot.ylabel("percentage contibution of Subsectors")
# plot the graph for category c4
quant_c4              = quant_c4.append(quant_c4.agg(['sum']))
val                   = quant_c4.loc['sum']['Gross State Domestic Product'] # value to fnd the ratio of each subsector
quant_c4.loc['ratio'] = quant_c4.loc['sum']/val
quant_c4              = quant_c4.loc['ratio'].sort_values(ascending=False)
quant_c4              = quant_c4.drop('Gross State Domestic Product')
quant_c4              = quant_c4[quant_c4.cumsum() <= 0.8]

#cumulative sum for the category has been obtained, now its turn to plot the graph for c4 category
quant_c4.plot(kind="bar")
plt.pyplot.title("Types of subsectors which includes to top 80 % for category c4")
plt.pyplot.xlabel("types of subsectors")
plt.pyplot.ylabel("percentage contibution of Subsectors")
new = pd.merge(quant_c1,quant_c2, how ="outer", on ='Item')
new = new.rename(columns={'ratio_x':'c1','ratio_y':'c2'})
new1 = pd.merge(quant_c3,quant_c3, how ="outer", on ='Item')
new1 = new1.rename(columns={'ratio_x':'c3','ratio_y':'c4'})
req = pd.merge(new,new1,how="outer",on="Item")

req.plot.bar(rot=90,width=0.7,figsize=(10,7))
plt.pyplot.xlabel("Types of subsectors")
plt.pyplot.ylabel("percentage contribution of each subsector")
path                                = r'../input/datafiles/rs_session243_au570_1.1.csv'
df_2                                = pd.read_csv(path)
#df.rename is an inbuilt function used to rename the columns and this takes values in the dictionary format
df_2.rename(columns = {'Primary - 2014-2015': 'Primary - 2013-2014','Primary - 2014-2015.1': 'Primary - 2014-2015','Level of Education - State':'States'},inplace = True)
df_req                              = df_2[['Primary - 2014-2015','Upper Primary - 2014-2015','Senior Secondary - 2014-2015']]
df_req['States']                    = df_2['States'].str.replace(" ","_")
df_chk                              = pd.merge(df_req,frame[['States','Per Capita GSDP (Rs.)']],on="States",how = 'inner')
df_primary                          = df_chk[['Primary - 2014-2015','Per Capita GSDP (Rs.)','States']]
df_primary['Per Capita GSDP (Rs.)'] = df_primary['Per Capita GSDP (Rs.)']/10000
df_primary                          = df_primary.sort_values(by = 'Per Capita GSDP (Rs.)',ascending=False)
x                                   = df_primary['Primary - 2014-2015'].values
y                                   = df_primary['Per Capita GSDP (Rs.)'].values
values                              = ['Primary - 2014-2015','Per Capita GSDP (Rs.)/10000']
df_primary                          = df_primary.set_index(['States'])
df_primary.plot.bar(rot=90,figsize=(10,8),width = 0.8)
plt.pyplot.legend(values,loc=2)
df_upper_primary                          = df_chk[['Upper Primary - 2014-2015','Per Capita GSDP (Rs.)','States']]
df_upper_primary['Per Capita GSDP (Rs.)'] = df_upper_primary['Per Capita GSDP (Rs.)']/10000
df_upper_primary                          = df_upper_primary.sort_values(by = 'Per Capita GSDP (Rs.)',ascending=False)
x                                         = df_upper_primary['Upper Primary - 2014-2015'].values
y                                         = df_upper_primary['Per Capita GSDP (Rs.)'].values
values                                    = ['Upper Primary - 2014-2015','Per Capita GSDP (Rs.) in thousands']
df_upper_primary                          = df_upper_primary.set_index(['States'])
df_upper_primary.plot.bar(rot=90,figsize=(10,8),width = 1.2)
plt.pyplot.legend(values,loc=2)
df_senior_secondary                          = df_chk[['Senior Secondary - 2014-2015','Per Capita GSDP (Rs.)','States']]
df_senior_secondary['Per Capita GSDP (Rs.)'] = df_senior_secondary['Per Capita GSDP (Rs.)']/10000
df_senior_secondary                          = df_senior_secondary.sort_values(by = 'Per Capita GSDP (Rs.)',ascending=False)
x                                            = df_senior_secondary['Senior Secondary - 2014-2015'].values
y                                            = df_senior_secondary['Per Capita GSDP (Rs.)'].values
values                                       = ['Senior Secondary - 2014-2015','Per Capita GSDP (Rs.) in thousands']
df_senior_secondary                          = df_senior_secondary.set_index(['States'])
df_senior_secondary.plot.bar(rot=90,figsize=(10,8),width = 1.2)
plt.pyplot.legend(values,loc=2)