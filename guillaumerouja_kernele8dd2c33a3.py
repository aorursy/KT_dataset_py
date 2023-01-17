import pandas as pd

import numpy as np

import glob

import datetime 

import math 

import time



start_time = time.time()



station = 68241

    

def Datastations (station,path): # Collect and compile all weather csv files for a given period

    allFiles = glob.glob(path + "/*.csv") 

    list_ = []

    array = ['T', 'MinT','MaxT','Precip','AIR_TEMP','AIR_TEMP_MIN','AIR_TEMP_MAX','PRCP'] #create array of rows to be kept

    for file_ in allFiles:

        df = pd.read_csv(file_,index_col=None, header=0)

        df = df[df.station_number==station]

        df = df.loc[df['parameter'].isin(array)] #Keep the rows we need (NB:useless columns will be droped at cleaning stage)

        list_.append(df)  

    frame = pd.concat(list_,ignore_index=True)

    return (frame)



df1 = Datastations(station,"../input/public-data/public-data/refdata/obs") # Run function to extract and combine csv files for 2017/2018

df2 = Datastations(station,"../input/public-data/public-data/refdata/BoM_ETA_20160501-20170430/obs") # Run function to extract and combine csv files for 2016/2017



print (len(df1)) 

print (df1.head()) 

print (len(df2)) 

print (df2.head()) 



print("--- %s seconds ---" % (time.time() - start_time))

#for df1 



df1["valid_start"] = df1["valid_start"].apply(pd.to_numeric)

df1["valid_end"] = df1["valid_end"].apply(pd.to_numeric)



df1["valid_start"]= df1["valid_start"]+36000 # add 10h to convert GMT to Australia time

df1["valid_end"]= df1["valid_end"]+36000



df1["valid_start"] = pd.to_datetime(df1["valid_start"],unit='s') # convert epoch time in valid time

df1["valid_end"] = pd.to_datetime(df1["valid_end"],unit='s') 



print(df1['valid_start'].values[1],df1['valid_end'].values[1])



#for df2



df2["valid_start"] = df2["valid_start"].apply(pd.to_numeric)

df2["valid_end"] = df2["valid_end"].apply(pd.to_numeric)



df2["valid_start"]= df2["valid_start"]+36000 # add 10h to convert GMT to Australia time

df2["valid_end"]= df2["valid_end"]+36000



df2.loc[df2.parameter =="AIR_TEMP", "valid_end"] = df2["valid_end"]+3600 # add 1h to AIR_TEMP (appears as instantaneous)

df2.loc[df2.parameter =="PRCP", "valid_end"] = df2["valid_end"]+3000 # Precip. are only reported on the first 10 min of each hour so add 50 min.

 

df2["valid_start"] = pd.to_datetime(df2["valid_start"],unit='s') #convert epoch time in normal time

df2["valid_end"] = pd.to_datetime(df2["valid_end"],unit='s') 



print(df2['valid_start'].values[1],df2['valid_end'].values[1])
#for df1

df1['T_Celsius'] = np.where(df1['parameter']=='T', df1['value'], '') # create new columns before dropping unncessary ones

df1['MinT_Celsius'] = np.where(df1['parameter']=='MinT', df1['value'], '')

df1['MaxT_Celsius'] = np.where(df1['parameter']=='MaxT', df1['value'], '')

df1['Precip_mm'] = np.where(df1['parameter']=='Precip', df1['value'], '')



df1= df1.drop(["area_code", "unit", "statistic", "level","qc_valid_minutes","parameter","value","qc_valid_start","qc_valid_end"], axis=1) # drop unncessary columns 

df1 = df1.groupby(['valid_start','valid_end','station_number'])['T_Celsius','MinT_Celsius','MaxT_Celsius','Precip_mm'].sum().reset_index()



print (len(df1)) #-> result should be close to (365*24=8760)

print(df1.head())



#for df2 ->repeat same steps 



df2['T_Celsius'] = np.where(df2['parameter']=='AIR_TEMP', df2['value'], '')

df2['MinT_Celsius'] = np.where(df2['parameter']=='AIR_TEMP_MIN', df2['value'], '')

df2['MaxT_Celsius'] = np.where(df2['parameter']=='AIR_TEMP_MAX', df2['value'], '')

df2['Precip_mm'] = np.where(df2['parameter']=='PRCP', df2['value'], '')



df2= df2.drop(["area_code", "unit", "statistic", "level","qc_valid_minutes","parameter","value","instantaneous","qc_valid_minutes_start","qc_valid_minutes_end"], axis=1) 

df2 = df2.groupby(['valid_start','valid_end','station_number'])['T_Celsius','MinT_Celsius','MaxT_Celsius','Precip_mm'].sum().reset_index()



print (len(df2)) 

print(df2.head())



#a/ Merge dataframes



df3 = df2.append(df1, ignore_index=True)



print(len(df3)) #Check nb of lines after merge -> should be close to 8760*2= 17,520



#b/ Insert missing rows and drop rows outside year 2017



df3 = df3.resample('60Min', on='valid_start').first().drop('valid_start', 1).reset_index()

df4= df3.drop(df3[(df3.valid_start < "2017-01-01 00:00:00")|(df3.valid_start > "2017-12-31 23:00:00")].index)



if (len(df4))!=8760:

    print('Too many missing data, check your data')

    

print (df4.head()) #cross check data integrity

print (df4.tail())



#c/ Fill missing valid_end dates & station numbers



df4['valid_end']=df4['valid_start']+ datetime.timedelta(0,3600)

df4['station_number']=station



#d/ Check number of missing values and change format of empty data so they can be addressed at e/



print(len(df4))

df4 = df4.replace('', np.nan, regex=True)

print(df4.isna().sum())



#e/ Create a warning line each time a period of at least 5 consecutive days without data is identified



df4["T_Celsius"] = df4["T_Celsius"].apply(pd.to_numeric)

df4["MinT_Celsius"] = df4["MinT_Celsius"].apply(pd.to_numeric)

df4["MaxT_Celsius"] = df4["MaxT_Celsius"].apply(pd.to_numeric)

df4["Precip_mm"] = df4["Precip_mm"].apply(pd.to_numeric)



def Datagap (parameter):

    k=0

    for i in range(len(df4)):

        if (math.isnan(df4[parameter].values[i])):

            k=k+1

            if (k>=120):

                print ('Warning datagap >= 5 days: check your data',parameter)

                k=0

        else:

            k=0

        

Datagap("T_Celsius")

Datagap("MinT_Celsius")

Datagap("MaxT_Celsius")

Datagap("Precip_mm")

#a/ Filling missing weather data based on x previous/following day(s)



def Datafilling (parameter):

    for i in range(len(df4)):

        j=0 #initialize counter of loops to find data at a given hour

        if (math.isnan(df4[parameter].values[i])): #if a data is missing      

            while (j<6): #We allow a maximum of 5 missing data at a given hour either before or after current time

                j=j+1 #increment nb of 

                if (math.isnan(df4[parameter].values[i-j*24]))== False: #if a data is found:

                    df4[parameter].values[i] = df4[parameter].values[i-j*24] #missing data is filled with data from x previous day

                    j=6 #Exit the loop and start a new count

                else: #if data not found at D-1 we look at D+1 then D-2, D+2, etc.

                    if (math.isnan(df4[parameter].values[i+j*24]))== False: #if a data is found 

                        df4[parameter].values[i] = df4[parameter].values[i+j*24] #missing data is filled with data from x following day

                        j=6 #Exit the loop and start a new count

    

Datafilling("T_Celsius")

Datafilling("MinT_Celsius")

Datafilling("MaxT_Celsius")

Datafilling("Precip_mm")



#b/ Final check and file export

                                                   

print(df4.isna().sum()) # Check nb of missing data

    

if (df4.isnull().values.any()):

    print ('Warning: datagap for a given hour > 5 days: check your data') # Warning if too many missing data at a given hour.

    

df4=df4.round({'T_Celsius': 1,'MinT_Celsius':1,'MaxT_Celsius':1,'Precip_mm':1}) # Final cleaning of decimals.

            

df4.to_csv(r'../input/public-data/public-data/csv files/dapto_test2.csv',index=False)  # Export final csv file         

            