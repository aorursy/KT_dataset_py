# %autosave 600
import os
import numpy as np
import pandas as pd
from glob import glob
wkbks = glob(os.path.join(os.pardir, 'input', 'xls_files_all', 'WIC*.xls'))
sorted(wkbks)
shts = ['Total Infants', 'Children Participating', 'Food Costs']

wic = pd.DataFrame()
for w in wkbks:
    for s in shts:
        frame = pd.read_excel(w, sheetname=s, skiprows=4
                  , skip_footer=4)
        frame['Type'] = s
        frame = frame.melt(id_vars=['Type', 'State Agency or Indian Tribal Organization'])
        wic = wic.append(frame, ignore_index=True)

print(wic.shape)
print(wic.columns)
# clean up
wic.columns=['Type', 'Area', 'Month', 'Value']
wic = wic[(wic.Area.str.contains('Region') == False)].copy()
wic['Month'] = pd.to_datetime(wic['Month'], errors='coerce')
wic = wic[wic.Month.notnull()]
wic['Value'] = pd.to_numeric(wic['Value'], downcast='integer')
wic = wic[wic.Value.notnull()]
wic.head()
wkbks = glob(os.path.join(os.pardir, 'input', 'xls_files_all', 'est*.xls'))
sorted(wkbks)
pov = pd.DataFrame()
y=2013
for w in wkbks:
        frame = pd.read_excel(w, skiprows=3, usecols=[2,26])
        frame['Year'] = y
        pov = pov.append(frame, ignore_index=True)
        y=y+1
        
wic.to_csv('wicdata.csv', index=False)
pov.to_csv('povertydata.csv', index=False)
# get a list of all excel files in the folder
wkbks = glob(os.path.join(os.pardir, 'input', 'xls_files_all', 'WIC*.xls'))
sorted(wkbks)
shts =['Pregnant Women Participating',
     'Women Fully Breastfeeding',
     'Women Partially Breastfeeding',
     'Total Breastfeeding Women', 
     'Postpartum Women Participating',
     'Total Women', 
     'Infants Fully Breastfed',
     'Infants Partially Breastfed',
     'Infants Fully Formula-fed',
     'Total Infants', 
     'Children Participating',
     'Total Number of Participants', 
     'Average Food Cost Per Person',
     'Food Costs', 
     'Rebates Received',
     'Nut. Services & Admin. Costs']

year = 2013

for w in wkbks:
    savedir = os.path.join(os.pardir, str(year))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    year = year+1
    for s in shts:
        frame = pd.read_excel(w, sheet_name=s, skiprows=4
                  , skip_footer=4)
        frame.to_csv(os.path.join(savedir, '{0}.csv'.format(s)), index=False)
        
        
