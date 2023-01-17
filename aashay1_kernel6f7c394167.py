# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#import the required Python libraries
import datetime
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
#read the data in csv file to a dataframe
filename="/kaggle/input/covid19-in-india/covid_19_india.csv"
df=pd.read_csv(filename)
df.head()
#Create a dataframe for daywise Covid19 cases in Maharashtra
dfmah=df[df['State/UnionTerritory']=='Maharashtra'].loc[1496:,['Date','Confirmed']].reset_index(drop=True)
dfmah.head()
#Calculate the growth column
cf=dfmah['Confirmed']
l=len(cf)
gr=list()
prev_case=df[(df['State/UnionTerritory']=='Maharashtra') & (df['Date']=='30/04/20')].Confirmed
init_growth=int(cf[0]-prev_case)
gr.append(init_growth)
for i in range(1,l):
    gr.append((cf[i]-cf[i-1]))
dfmah['Growth']=gr
dfmah.head(10) #Covid19 cases in Maharashtra
#Create the variables x and y 
y=np.array(gr)
x=dfmah.index.values
x=x.reshape(-1,1)
#Build the Linear regression model
model=LinearRegression().fit(x,y)
#Initialize the values before prediction
start_date=datetime.datetime.strptime(dfmah.iloc[-1,0],'%d/%m/%y')
l=len(cf)
confirmed=cf[l-1]
listout=[]
#Generate the prediction and values for everyday for next 15 days 
for i in range(15):
    start_date=start_date+datetime.timedelta(1)
    new_date=start_date
    k=l+i-1
    z=model.predict(np.array([[k,]]))
    growth=int(z[0])
    confirmed=confirmed+growth
    listout.append([new_date, confirmed, growth])
dfout=pd.DataFrame(listout,columns=['Date','Confirmed','Growth'])
dfout 
#Covid19 cases Maharashtra prediction for next 15 days using LinearRegression
plt.figure(figsize=(15,6))
plt.title("Maharashtra Covid-19 cases prediction for next 15 days")
sns.lineplot(x=dfout['Date'],y=dfout['Confirmed'])