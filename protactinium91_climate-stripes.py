import pandas as pd



import os, io, requests



import functools



import numpy as np



import matplotlib.pyplot as plt



from matplotlib.colors import Normalize







# Original file forked from: https://github.com/hausfath/scrape_global_temps 

# This follows with the same license, it is completely free for copy, modification, and reuse to all interested parties.







#File URLs



gistemp_file = 'https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts+dSST.csv'



noaa_file = 'https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/p12/12/1880-2050.csv'



hadley_file = 'https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.monthly_ns_avg.txt'



berkeley_file = 'http://berkeleyearth.lbl.gov/auto/Global/Land_and_Ocean_complete.txt'



cowtan_way_file = 'http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.txt'







#Common baseline period (inclusive)



start_year = 1971



end_year = 2000







def import_gistemp(filename):



    '''



    Import NASA's GISTEMP, reformatting it to long



    '''



    urlData = requests.get(filename).content



    df = pd.read_csv(io.StringIO(urlData.decode('utf-8')), skiprows=1)



    df.drop(['J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON'], axis=1, inplace=True)



    for num in range(1, 13):



        df.columns.values[num] = 'month_'+str(num)



    df_long = pd.wide_to_long(df, ['month_'], i='Year', j='month').reset_index()



    df_long.columns = ['year', 'month', 'gistemp']



    df_long = df_long.apply(pd.to_numeric, errors='coerce')



    df_long.sort_values(by=['year', 'month'], inplace=True)



    df_long.reset_index(inplace=True, drop=True)



    return df_long











def import_noaa(filename):



    '''



    Import NOAA's GlobalTemp



    '''



    df = pd.read_csv(filename, skiprows=5, names=['date', 'noaa'])



    df['year'] = df['date'].astype(str).str[:4]



    df['month'] = df['date'].astype(str).str[4:6]



    df.drop(['date'], axis=1, inplace=True)



    df = df[['year', 'month', 'noaa']].apply(pd.to_numeric)



    return df











def import_hadley(filename):



    '''



    Import Hadley's HadCRUT4



    '''



    urlData = requests.get(hadley_file).content



    df = pd.read_csv(io.StringIO(urlData.decode('utf-8')), header=None, delim_whitespace=True, usecols=[0,1], names=['date', 'hadcrut4'])



    df['year'] = df['date'].astype(str).str[:4]



    df['month'] = df['date'].astype(str).str[5:7]



    df = df[['year', 'month', 'hadcrut4']].apply(pd.to_numeric)



    return df











def import_berkeley(filename):



    '''



    Import Berkeley Earth. Keep only the operational air-over-sea-ice varient from the file.



    '''



    df = pd.read_csv(berkeley_file, skiprows=76, delim_whitespace=True, header=None, usecols=[0,1,2], names=['year', 'month', 'berkeley'])



    end_pos = df.index[df['month'] == 'Global'].tolist()[0]



    return df[0:end_pos].apply(pd.to_numeric)











def import_cowtan_way(filename):



    '''



    Import Cowtan and Way's temperature record.



    '''



    df = pd.read_csv(filename, delim_whitespace=True, header=None, usecols=[0,1], names=['date', 'cowtan_way'])



    df['year'] = df['date'].astype(str).str[:4].astype(int)



    df['month'] = ((df['date'] - df['year']) * 12 + 1).astype(int)



    df = df[['year', 'month', 'cowtan_way']].apply(pd.to_numeric)



    return df











def combined_global_temps(start_year, end_year):



    '''



    Merge all the files together, rebaselining them all to a common period.



    '''



    hadley = rebaseline(import_hadley(hadley_file), start_year, end_year)



    gistemp = rebaseline(import_gistemp(gistemp_file), start_year, end_year)



    noaa = rebaseline(import_noaa(noaa_file), start_year, end_year)



    berkeley = rebaseline(import_berkeley(berkeley_file), start_year, end_year)



    cowtan_and_way = rebaseline(import_cowtan_way(cowtan_way_file), start_year, end_year)







    dfs = [hadley, gistemp, noaa, berkeley, cowtan_and_way]



    df_final = functools.reduce(lambda left,right: pd.merge(left,right,on=['year', 'month'], how='outer'), dfs)



    return df_final.round(3)











def rebaseline(temps, start_year, end_year):



    '''



    Rebaseline data by subtracting the mean value between the start and



    end years from the series.



    '''



    mean = temps[



        temps['year'].between(start_year, end_year, inclusive=True)



    ].iloc[:, 2].mean()



    temps.iloc[:, 2] -= mean



    return temps







berkeley = rebaseline(import_berkeley(berkeley_file), start_year, end_year)

berkeley




temp_data = berkeley



savename = 'global_temps_line'







temps = temp_data.iloc[:,2]

temps
temps_normed = ((temps - temps.min(0)) / temps.ptp(0)) * (len(temps) - 1)

temps_normed








elements = len(temps)







x_lbls = np.arange(elements)



y_vals = temps_normed / (len(temps) - 1)



y_vals2 = np.full(elements, 1)



bar_wd  = 1







my_cmap = plt.cm.RdBu_r #choose colormap to use for bars



norm = Normalize(vmin=0, vmax=elements - 1)
def colorval(num):



    return my_cmap(norm(num))







fig=plt.figure(figsize=(6,5))



plt.axis('off')



plt.axis('tight')







#Plot warming stripes. Change y_vals2 to y_vals to plot stripes under the line only.



plt.bar(x_lbls, y_vals2, color = list(map(colorval, temps_normed)), width=1.0)







#Plot temperature timeseries. Comment out to only plot stripes



#plt.plot(x_lbls, y_vals - 0.002, color='black', linewidth=2)







plt.xticks( x_lbls + bar_wd, x_lbls)



plt.ylim(0, 1)



fig.subplots_adjust(bottom = 0)



fig.subplots_adjust(top = 1)



fig.subplots_adjust(right = 1.005)



fig.subplots_adjust(left = 0)



fig.savefig(savename+'.png', dpi=300)