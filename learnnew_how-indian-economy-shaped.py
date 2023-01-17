#Importing the required modules
import numpy as np 
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/World_Bank_Data_India.csv')
data
data.dropna(thresh=10,axis=1,inplace=True,how='any')
data
fig, axes1 = plt.subplots(figsize=(20,8))
plt.plot(data['Years'],data['POP_TOTL'], '*-')
plt.plot(data['Years'],data['POP_014'],'*-' )
plt.plot(data['Years'],data['POP_1564'], '*-')
plt.plot(data['Years'],data['POP_65'], '*-')
plt.legend(prop={'size': 16})
plt.subplots(figsize=(20,3))
plt.plot(data['Years'],data['EMP_TOTL'], '*-')
plt.title("Total Employment growth")

fig, axes1 = plt.subplots(figsize=(20,6))
#plt.plot(data['Years'],data['EMP_TOTL'], '*-')
plt.plot(data['Years'],data['EMP_SELF'],'*-' ,label="Self Employed")
plt.plot(data['Years'],data['EMP_SRV'], '-',label="Service Sector")
plt.plot(data['Years'],data['EMP_IND'], '*-',label="Industrial Sector")
plt.plot(data['Years'],data['EMP_AGR'], '*-',label="Agriculture Sector")
plt.title("Employment Growth Sector wise")
plt.legend(prop={'size': 16})
fig, axes1 = plt.subplots(figsize=(20,4))
plt.plot(data['Years'],data['GDP_IND'], '*-')
plt.plot(data['Years'],data['GDP_AGR'],'*-' )
plt.legend(prop={'size': 16})
fig, axes1 = plt.subplots(figsize=(20,4))
plt.plot(data['Years'],data['FRTL'], '*-')
plt.plot(data['Years'],data['BR'],'*-' )
plt.plot(data['Years'],data['DR'],'*-' )
plt.legend(prop={'size': 16})
