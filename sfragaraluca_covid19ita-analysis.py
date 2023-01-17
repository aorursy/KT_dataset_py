



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





data=pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')



def date_format(x):

    x=x[5:10]

    month=x[:2]

    day = x[3:]

    x= day+'-'+month

    return x



data['Date']=data['Date'].apply(date_format)

print(data.columns)



def regionData(dataset,regionName):

    regionindex=dataset.index[dataset['RegionName']==regionName].tolist()

    region_df= pd.DataFrame(dataset.iloc[regionindex])

    return region_df



#for _,row in regionData(data,'Veneto').iterrows():

def CFR(regionname):

    CFR_values=[]

    CFR_values.append((regionData(data,regionname)['Deaths']/regionData(data,regionname)['TotalPositiveCases'])*100)

    return(CFR_values)



fig, plt1= plt.subplots(figsize=(15,10))



plt1.plot(regionData(data,'Veneto')['Date'],regionData(data, 'Veneto')['TotalPositiveCases'], color='orange',label = 'Confirmed Cases Veneto')

plt1.plot(regionData(data,'Lombardia')['Date'],regionData(data, 'Lombardia')['TotalPositiveCases'], color='red',label = 'Confirmed Cases Lombardia')



plt3 = plt1.twinx()

plt3.plot(np.array(CFR('Veneto'))[0], label='Veneto CFR',linestyle='dashed', color='orange')

plt3.plot(np.array(CFR('Lombardia'))[0], label='Lombardia CFR', linestyle='dashed', color='red')

plt3.legend(loc="upper left", bbox_to_anchor=(0, 0.88))

plt3.set_ylabel('CFR %')

plt1.plot(regionData(data,'Veneto')['Date'],regionData(data,'Veneto')['TestsPerformed'], color='lightgreen', label ='Tests Performed Veneto')

plt1.plot(regionData(data,'Lombardia')['Date'],regionData(data,'Lombardia')['TestsPerformed'], color='darkgreen', label= 'Tests Performed Lombardia')

#plt2.set_ylabel("Tests Performed")

plt1.set_xlabel("Date")

#plt1.set_ylabel("Confirmed cases")

plot_title='Confirmed cases vs Tests Performed (at {})'.format(data['Date'].iloc[-1])

plt1.set_title(plot_title)

plt1.legend(loc='upper left')

plt1.xaxis.set_major_locator(plt.MaxNLocator(25))



#plt1.set_yscale('log')

plt1.grid(which='major')

plt1.grid(which='minor')

plt.savefig('Covid19_plot_2803.png')

plt.show()