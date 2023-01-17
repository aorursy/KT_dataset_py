# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/h1b_kaggle.csv')

#print(data.head(5))

New_data= data

data=data.loc[(data['YEAR'] == 2016) &(data['CASE_STATUS']== 'CERTIFIED')]

data=data.drop(['SOC_NAME','FULL_TIME_POSITION','lon','lat','PREVAILING_WAGE','CASE_STATUS'],axis=1)

print(data.tail(5))

#data.describe()
df_infy=data.loc[(data['EMPLOYER_NAME']== 'INFOSYS LIMITED')]

df_Cogi=data.loc[(data['EMPLOYER_NAME']=='COGNIZANT TECHNOLOGY SOLUTIONS U.S. CORPORATION')]

df_syntel=data.loc[(data['EMPLOYER_NAME']=='SYNTEL CONSULTING INC.')]

df_tata=data.loc[(data['EMPLOYER_NAME']=='TATA CONSULTANCY SERVICES LIMITED')]



index = np.arange(2)

columns = ['INFY', 'CTS', 'SYNTEL' ,'TATA']



df_cns =pd.DataFrame(columns=columns, index = index)

df_cns['INFY']=df_infy.EMPLOYER_NAME.count()

df_cns['CTS']=df_Cogi.EMPLOYER_NAME.count()

df_cns['SYNTEL']=df_syntel.EMPLOYER_NAME.count()

df_cns['TATA']=df_tata.EMPLOYER_NAME.count()





df_cns.tail(-1).plot(kind='bar')

plt.ylabel('# of H1B Visas')

plt.xlabel('Companies')

plt.show()
#print(data[data['WORKSITE'].str.contains("ARIZONA")==True])

df_Phx=data.loc[(data['WORKSITE']== 'PHOENIX, ARIZONA')]

print(df_Phx.count())
#print (New_data.head())

EMP_data=New_data.groupby('EMPLOYER_NAME').size().nlargest(10)

#print(EMP_data)

EMP_data.plot(kind='bar')
New_data=New_data.drop(['SOC_NAME','FULL_TIME_POSITION','lon','lat','PREVAILING_WAGE','CASE_STATUS'],

                       axis=1)

New_data1=New_data.groupby(['EMPLOYER_NAME','YEAR']).size().nlargest(10)

print(New_data1)

#Result = pd.pivot_table(New_data1, index='EMPLOYER_NAME', columns='YEAR',  aggfunc=np.size)

#print(Result)

#sns.heatmap(Result, annot=False, fmt="g" ,cbar_kws={"orientation": "horizontal"})

#plt.show()
#print (New_data.head())

EMP_data=New_data.groupby('EMPLOYER_NAME').size().nlargest(10)

print(EMP_data)

EMP_data.plot(kind='pie')