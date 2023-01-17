import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
df = pd.read_csv('../input/merklesokratasgmt2/Data Analyst Assignment (1).xlsx - Assignment-2.csv')
df.head()
dates = df.columns
new_column = []
for x in dates:
    if x == 'SKU':
        new_column.append('SKU')
    else:
        datess = datee = datetime.strptime(x, "%m/%d/%Y")
        new_column.append(datess.month)
col_rename_dict = {i:j for i,j in zip(dates,new_column)}
df.rename(columns=col_rename_dict, inplace=True)
df.head()
newDf = df.groupby(df.columns, axis=1).sum()
jan = newDf[1].sum()
feb = newDf[2].sum()
mar = newDf[3].sum()
apr = newDf[4].sum()
may = newDf[5].sum()
jun = newDf[6].sum()
jul = newDf[7].sum()
aug = newDf[8].sum()
sep = newDf[9].sum()
otb = newDf[10].sum()
nov = newDf[11].sum()
dec = newDf[12].sum()
data = {'aug 19': aug, 'sep 19': sep, 'oct 19': otb,  
        'nov 19': nov, 'dec 19': dec, 'jan 20': jan, 'feb 20': feb, 'mar 20': mar, 'apr 20': apr, 'may 20': may,  'jun 20': jun, 'jul 20': jul} 
month = list(data.keys()) 
values = list(data.values()) 
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(month, values, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Month") 
plt.ylabel("Total Sales") 
plt.title("Monthly sales") 
plt.show() 
#part one of the 1st task
print('Sales in August 2019 are '+ str(aug))
print('Sales in September 2019 are '+ str(sep))
print('Sales in October 2019 are '+ str(otb))
print('Sales in November 2019 are '+ str(nov))
print('Sales in December 2019 are '+ str(dec))
print('Sales in January 2020 are '+ str(jan))
print('Sales in Febuary 2020 are '+ str(feb))
print('Sales in March 2020 are '+ str(mar))
print('Sales in April 2020 are '+ str(apr))
print('Sales in May 2020 are '+ str(may))
print('Sales in June 2020 are '+ str(jun))
print('Sales in July 2020 are '+ str(jul))
#part 2 of first question
print('Sales in August-November are '+ str(aug + sep + otb + nov))
print('Sales in December-March are '+ str(dec + jan + feb + mar))
print('Sales in April-July are '+ str(apr + may + jun + jul))