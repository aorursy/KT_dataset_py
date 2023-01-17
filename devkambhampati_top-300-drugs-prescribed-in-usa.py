import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#Reading Data table from website (web address serves as input below)
A=pd.read_html('https://clincalc.com/DrugStats/Top300Drugs.aspx')
A[0]   #Dataframe listing the table contents 
Drug_DF=A[0]   #assigning it to a new DataFrame, Drug_DF
Drug_DF.columns  #List of all columns within the DataFrame
Drug_DF.drop('Annual Change',axis=1,inplace=True)  #dropping annual change column, not relevant for current analysis
Drug_DF  #current DataFrame after above changes
Drug_DF.rename(columns={'Total Prescriptions (2017)':'2017_Total_Prescriptions'},inplace=True)  #Renaming Column name
Drug_DF.describe()  #High Level Dataframe Statistics
Drug_DF.size  #number of elements in DataFrame
Drug_DF.shape   #number of rows and columns in DataFrame: 300 rows, 3 columns
Drug_DF.ndim   #Two dimensional DataFrame
Drug_DF.columns  #DataFrame Column Names
Prescriptions=Drug_DF['2017_Total_Prescriptions'].sum()
Drug_DF['Total_Percentage']=Drug_DF['2017_Total_Prescriptions']/Prescriptions*100
# Total Percentage column indicated the prescription percentage for each drug among the Top 300 prescribed drugs in 2017
Drug_DF['Total_Percentage'].sum()
Drug_DF['Total_Percentage'][0:20].sum()
#Slicing DataFrame for Top 10 Analysis

B=Drug_DF[0:10]
plt.figure(figsize=(14,7))
axes = plt.gca()
plt.bar(B['Drug Name'],B['2017_Total_Prescriptions'],color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('Pharmaceutical Drugs')
plt.xticks (fontsize=14)
plt.ylabel('Quantity of Prescriptions')
plt.yticks (fontsize=14)
plt.title('TOP 10 Drugs Prescribed in USA (2017)')
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
B=Drug_DF[0:10]
plt.figure(figsize=(14,7))
axes = plt.gca()
plt.bar(B['Drug Name'],B['Total_Percentage'],color='orange')
plt.xticks(rotation=45)
plt.xlabel('Pharmaceutical Drugs')
plt.xticks (fontsize=14)
plt.ylabel('Percentage of Top 300 Prescribed Drugs')
plt.yticks (fontsize=14)
plt.title('TOP 10 Drugs Prescribed in USA (2017)-Percentage Contribution')
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
Drug_DF[0:50]  
#Total Prescriptions of Top 50 Prescribed Drugs in USA
Drug_DF['2017_Total_Prescriptions'][0:50].sum() 
#Total Percentage of Top 50 Prescribed Drugs in USA (within Top 300 sample set)
Drug_DF['Total_Percentage'][0:50].sum()