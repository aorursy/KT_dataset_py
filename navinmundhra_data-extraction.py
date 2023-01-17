import numpy as np
import pandas as pd
! pip install tabula-py
import tabula as tb
from scipy import stats
import datetime as dt
datapath = '../input/daily-power-generation-in-india-20172020'
base = dt.datetime(2017,9,1) ## 1st September, 2017
end = dt.datetime(2020,3,18) ## 18th March, 2020 

numdays = (end-base).days
date_list = [str((base + dt.timedelta(days=x)).date()) for x in range(numdays+1)] ## list of dates sorted from start to the end date

def rev(currdate):
    currdate = currdate.split('-'); currdate = str(currdate[-1]+'-'+currdate[1]+'-'+currdate[0])
    return currdate

# date_list
numdays
finaldf = pd.DataFrame()
finaldf.insert(0, 'Date', 'NaN')
finaldf.insert(1, 'Hydro Generation Estimated (in MU)', 'NaN')
finaldf.insert(1, 'Hydro Generation Actual (in MU)', 'NaN')
finaldf.insert(1, 'Nuclear Generation Estimated (in MU)', 'NaN')
finaldf.insert(1, 'Nuclear Generation Actual (in MU)', 'NaN')
finaldf.insert(1, 'Thermal Generation Estimated (in MU)', 'NaN')
finaldf.insert(1, 'Thermal Generation Actual (in MU)', 'NaN')
finaldf.insert(1, 'Region', 'NaN')

s, f = 0, 129
notavailable = []

finaldf
for t in range(numdays):
    ## GETTING THE DATE FOR WHICH WE NEED TO GET THE REPORT FOR
    currdate = date_list[t]
    ## CREATING THE LINK
    link1 = 'https://npp.gov.in/public-reports/cea/daily/dgr/{}/dgr1-{}.pdf'.format(rev(currdate), currdate)
    ## TRY OPENING AND READING THE DATA. IF "HTMLERROR" i.e. FILE DOES NOT EXIST(NATIONAL HOLIDAY), STORE THE DATE IN A LIST
    try:
        df = tb.read_pdf(link1, stream=False)[0]
    except:
        notavailable.append(currdate)
        continue
        
    def getdata(region, val):
        """    UTILITY FUNCTION TO EXTRACT CERTAIN ROWS FOR DATA. RETURNS A LIST FOR THE  """
        temp = list()
        temp.append(currdate)
        df.iloc[:,0] = df.iloc[:,0].apply(lambda x: x.replace(" ", "")if type(x)!=float else x)
        ind = df[df.iloc[:,0]==region].index[0]
        temp.append(df.iloc[ind, 0]) 
        temp.extend(df.iloc[ind+1:ind+1+val,4:6].transpose().reset_index(drop=True).melt()['value'].to_list())
        return temp
    
    ## GETTING ROW-WISE DATA FOR REGIONS
    north = getdata('Northern', 3)
    west = getdata('Western', 3)
    south = getdata('Southern', 3)
    east = getdata('Eastern', 2); east.insert(4, 'NaN'); east.insert(4, 'NaN')
    neast = getdata('NorthEastern', 2); neast.insert(4, 'NaN'); neast.insert(4, 'NaN')
    
    ## INSERTING THE RESULTS IN THE TEMPLATE OF FINAL DATAFRAME CREATED
    finaldf = finaldf.append(pd.Series(north, index=finaldf.columns), ignore_index=True)
    finaldf = finaldf.append(pd.Series(west, index=finaldf.columns), ignore_index=True)
    finaldf = finaldf.append(pd.Series(south, index=finaldf.columns), ignore_index=True)
    finaldf = finaldf.append(pd.Series(east, index=finaldf.columns), ignore_index=True)
    finaldf = finaldf.append(pd.Series(neast, index=finaldf.columns), ignore_index=True)
    
finaldf.to_csv(datapath+'/file.csv', index=False)
notavailable 
import IPython.display as ipd

ipd.FileLink(datapath+'/file.csv')