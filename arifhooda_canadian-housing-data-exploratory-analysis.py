# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#Ref_Date,GEO,Geographical classification,HOUSING,UNIT,Vector,Coordinate,Value
df = pd.read_csv('../input/data.csv', encoding = 'ISO-8859-1',na_values={'Geographical classification': []},low_memory=False)
df.head()

# Any results you write to the current directory are saved as output.
df.shape
# identify datatypes of the 11 columns, add the stats to the datadict
datadict = pd.DataFrame(df.dtypes)
datadict
# identify missing values of the 8 columns,add the stats to the datadict
datadict['MissingVal'] = df.isnull().sum()
datadict
# Identify number of unique values, For object nunique will the number of levels
# Add the stats the data dict
datadict['NUnique']=df.nunique()
datadict
# Identify the count for each variable, add the stats to datadict
datadict['Count']=df.count()
datadict
# rename the 0 column
datadict = datadict.rename(columns={0:'DataType'})
datadict
# get discriptive statistcs on "object" datatypes
df.describe(include=['object'])
# get discriptive statistcs on "number" datatypes
df.describe(include=['number'])
#Remove rows with unit=total units and geo=Census metropolitan areas because pre-aggregation data
df=df[df.UNIT != 'Total units']
df=df[df.GEO != 'Census metropolitan areas']
#Split Ref_Date in Year and Month
df[['Year','Month']]=df['Ref_Date'].str.split('/',expand=True)
df.Year = df.Year.astype(int)
df.Month = df.Month.astype(int)
#Split GEO into Location and Province
df['GEO']=df['GEO'].str.replace(r'\,(?=.*?\,)', '')
df[['Location','Province']]=df['GEO'].str.split(',',expand=True)
df
#df.GEO.unique()
#pivot table on housing attribute
df1=pd.pivot_table(df,index=["Ref_Date","Year",'Month','Province','Location','UNIT'],columns=['HOUSING'],values=["Value"],aggfunc=np.sum)
df1=pd.DataFrame(df1)
df1=pd.DataFrame(df1.to_records())
df1.columns = [hdr.replace("('Value', '", "").replace("')", "") for hdr in df1.columns]
df1
#Aggregated by Year
dfyear=pd.pivot_table(df1,index=["Year"],values=["Housing completions","Housing starts","Housing under construction"]
    ,aggfunc={np.sum})
dfyear["Housing under construction"]=dfyear["Housing under construction"]/12
dfyear.describe()
dfyear.corr()
dfyear.hist(figsize=(15,15))  
axe=dfyear.plot(figsize=(15,8),x=dfyear.index.values,xticks=range(dfyear.index.values.min(),dfyear.index.values.max(),2)
                ,grid=(True,'minor','X'),legend=True) 
axe.set_title('Canadian Housing Starts, Under Construction and Completion',fontsize=20)
axe.legend(["Housing completions","Housing starts","Housing under construction"])
plt.show()
#Aggregated by unit type
dfunit=pd.pivot_table(df1,index=["Year"],columns=["UNIT"],values=["Housing under construction"],aggfunc={np.sum})
axe=dfunit.plot(figsize=(15,8),x=dfyear.index.values,xticks=range(dfyear.index.values.min(),dfyear.index.values.max(),2), \
          kind='bar',stacked=True) 
axe.set_title('Housing under construction by Unit Type',fontsize=20)
axe.legend(["Apartment and other unit types","Row units","Semi-detached units",'Single-detached units'])
plt.show()
#Aggregated by unit type
dfunit=pd.pivot_table(df1,index=["Year"],columns=["UNIT"],values=["Housing starts"],aggfunc={np.sum})
axe=dfunit.plot(figsize=(15,8),x=dfyear.index.values,xticks=range(dfyear.index.values.min(),dfyear.index.values.max(),2), \
          kind='bar',stacked=True,) 
axe.set_title('Housing starts by Unit Type',fontsize=20)
axe.legend(["Apartment and other unit types","Row units","Semi-detached units",'Single-detached units'])
plt.show()
dfseason=pd.pivot_table(df1.loc[df1['Year'] >= 2014],index=["Year","Month"],values=["Housing completions"],aggfunc={np.sum})
g = dfseason.groupby([dfseason.index.get_level_values('Year'), dfseason.index.get_level_values('Month')]).mean()
axe=dfseason.plot(figsize=(15,8))
axe.set_xticks(range(len(g)))
axe.set_xticklabels(["%s-%02d" % item for item in g.index.tolist()], rotation=90);
axe.set_title('Seasonal Trend - Housing Completions',fontsize=20)
axe.legend(["Housing completions"])
