#Data Wrangling
import numpy as np
import pandas as pd
df1=pd.DataFrame({'key':['a','b'],'val1':[1,2]})
df2=pd.DataFrame({'key':['a','b'],'val2':[4,5]})

print ("df1")
print (df1)

print ("######")

print ("df2")
print (df2)

print (pd.merge(df1,df2,on = 'key'))

#print(pd.concat([df1,df2],axis=0,sort=True)) 
#vertical stacking df2 below of df1
print(pd.concat([df1,df2],axis=0,sort=True))


# print(pd.concat([df1,df2],axis=1,sort=True))
#Horizontal stacking df2 to left of df1
print(pd.concat([df1,df2],axis=1,sort=True))



