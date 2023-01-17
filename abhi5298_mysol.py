

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#no. of left and right footred player
dataset=pd.read_csv("../input/fifa19/data.csv")
pf=dataset.iloc[:,14:15].values

print("left footed player")
lfp=np.count_nonzero(pf == 'Left')
rfp=np.count_nonzero(pf == 'Right')
print(lfp)
print("Right footed player")
print(rfp)

#cleaning data
dataset['Value'] = dataset['Value'].map(lambda x: x.lstrip('€'))
dataset['Wage'] = dataset['Wage'].map(lambda x: x.lstrip('€'))#for removing euro sign
repl_dict = {'[kK]': '*1e3', '[mM]': '*1e6', '[bB]': '*1e9', }
dataset['Value']=dataset['Value'].replace(repl_dict, regex=True).map(pd.eval)#for conversion of k into numerical value
dataset['Wage']=dataset['Wage'].replace(repl_dict, regex=True).map(pd.eval)

#naming columns
name=dataset["Name"]
val=dataset["Value"]
#top 10 players according to their values

namev = list(zip(name, val)) 
df = pd.DataFrame(namev, columns = ['Name', 'values']) 

df=df.sort_values(by='values',ascending=False)
pten=df.head(10)
rere=list(pten["Name"])
jeje=list(pten["values"])
fig = plt.figure(figsize = (20, 5)) 
  
# creating the bar plot 
plt.bar(rere,jeje , color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Players") 
plt.ylabel("Value in Euros") 
plt.show() 
