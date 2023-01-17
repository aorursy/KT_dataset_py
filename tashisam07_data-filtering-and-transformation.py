import pandas as pd

import numpy as np
name=['tashi','Akash','shamu','tenzin','may','landa','khenpo']

age=[23,43,12,13,53,23,19]

rating=[23,54.65,34.56,123,23.45,43.56,12.3]

print(name,age,rating)
d=int(input("Enter the window size:"))

r=int(d/2)

print(r)
#finding mean of age data filter

q=int(input("Enter the size:"))

age=[20,21,24,20,30,21,60,19,21]

age_filter=age[:]

start=int(q/2)

end=len(age)-start

print(start)

print(end)

for i in range(start,end):

    age_filter[i]=np.mean(age[(i-start):(i+start+1)])

print(age)

print(age_filter)
#finding median data filter

a=int(input("Enter the size:"))

age=[20,21,24,20,30,21,60,19,21,30]



age_filter=age[:]

start=int(a/2)

end=len(age)-start

for i in range(start,end):

    age_filter[i]=np.median(age[(i-start):(i+start+1)])

print(age)

print(age_filter)
g={'age':pd.Series([18,19,20,18,23,25,15,16,45,12]),

   'rating':pd.Series([4.23,3.24,3.98,4.56,3.65,3.8,3.78,2.98,4.80,4.10])}

#Create a DataFrame

df=pd.DataFrame(g)

print(df)
s=int(input("print the value you wanna transform: "))

min=df['age'].min()

max=df['age'].max()

y=((s-min)/(max-min))



print(y)