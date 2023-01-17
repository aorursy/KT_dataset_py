from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import seaborn as sns
from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))
df = pd.read_csv("../input/bus-breakdown-and-delays.csv")
print(df.head())
print(df.info())
for index, value in df[['School_Year','Busbreakdown_ID','Run_Type','Bus_No','Route_Number','Reason','Schools_Serviced','Occurred_On','Boro']][0:1].iterrows():
    print(index," : ",value)
df.describe(include='all')
print(df.corr())
#correlation map
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
df.head(8)
df.columns
df.plot()
df.Busbreakdown_ID.plot(kind = 'line', color = 'blue',label = 'Busbreakdown_ID',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.Number_Of_Students_On_The_Bus.plot(kind = 'line', color = 'brown',label = 'Number_Of_Students_On_The_Bus',linewidth=1, alpha = 0.5,grid = True, linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('Busbreakdown ID')            
plt.ylabel('Number Of Students On The Bus')
plt.title('School Bus Correlation Line Plot')            
plt.show()
df.Busbreakdown_ID.plot(kind = 'hist',orientation='horizontal', cumulative=True, label='Busbreakdown_ID', bins = 10, color='blue',figsize = (12,12))
plt.show()
df.Number_Of_Students_On_The_Bus.plot(kind = 'hist',orientation='horizontal', cumulative=True, label='Number_Of_Students_On_The_Bus', bins = 10, color='blue',figsize = (12,12))
plt.show()
plt.figure()
print(df.keys())
print(type(df))
# User Defined Function
def tuble_ex():
    """return defined Bus_No tuble"""
    t = df.Bus_No
    return t
print(tuble_ex())
#%%
# lets return NY bus data csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. our threshold is avarage 
threshold = sum(df.Number_Of_Students_On_The_Bus)/len(df.Number_Of_Students_On_The_Bus)
print("threshold: ",threshold)
df["Number_Of_Students_On_The_Bus_level"]=["high" if i > threshold else "low" for i in df.Number_Of_Students_On_The_Bus]
df.loc[:10,["Number_Of_Students_On_The_Bus_level","Number_Of_Students_On_The_Bus"]]
print(df['Bus_Company_Name'].value_counts(dropna=False))
df.plot(kind='box',color='red', vert=False, sym='r+')
#tidy data
df_new = df.head()
df_new
#lets melt id_vars, value_vars
melted = pd.melt(frame=df_new,id_vars ='Route_Number',value_vars= ['How_Long_Delayed','Number_Of_Students_On_The_Bus'])
melted
#reverse of melting
melted.pivot(index='Route_Number',columns = 'variable', values ='value')
#concatenating data
df1 = df.head()
df2 = df.tail()
conc_df_colm = pd.concat([df1,df2],axis =0,ignore_index=True) # axis = 0 : adds dataframes in column
conc_df_colm
df1 = df['How_Long_Delayed'].head()
df2 = df['Number_Of_Students_On_The_Bus'].head()
conc_df_col = pd.concat([df1,df2],axis =1) # axis = 0 : adds dataframes in column
conc_df_col
df.dtypes
df['Reason'] = df['Reason'].astype('category')
df.dtypes
#missing data

df["How_Long_Delayed"].value_counts(dropna = False)
df1=df
df1["How_Long_Delayed"].dropna(inplace = True)
assert df["How_Long_Delayed"].notnull().all()  #returns nothing becuse we drop NaN values
df["How_Long_Delayed"].fillna('empty',inplace = True)
assert df['How_Long_Delayed'].notnull().all()
assert df.How_Long_Delayed.dtypes == np.object
df1 = df['Has_Contractor_Notified_Schools'].head()
df2 = df['Has_Contractor_Notified_Parents'].head()
list_label = ["Has_Contractor_Notified_Schools","Has_Contractor_Notified_Parents"]
list_col = [df1,df2]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
dataf = pd.DataFrame(data_dict)
dataf
#add new columns
dataf["Busbreakdown_ID"] = ["1212699","1212700","1212701","1212703","1212704"]
dataf["income"] = 0 #broadcasting entire column
dataf