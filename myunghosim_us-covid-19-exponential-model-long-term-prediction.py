import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math
days=[x for x in range(1,210)]

rates=[1+(25-(1.03**x))/100 for x in range(len(days))]

rates = [num if num>1 else 1.005 for num in rates]

plt.plot(days,rates)

plt.xlabel('nth day')

plt.ylabel('rate of growth on nth day')

plt.title('Rate of growth')

plt.figure(figsize=(20, 10))

plt.show()
# confirmed = [85000*1.05**x for x in days]

confirmed=[0]*len(days)

confirmed[0]=85000

for i in range(1,len(days)):

    confirmed[i] =confirmed[i-1]*rates[i] 

#cap confirmed at 300 million

confirmed = [num if num<3*10**8 else 3*10**8 for num in confirmed]

# recovered = [1500*1.05**x for x in days]

recovered=[0]*len(days)

recovered[0]=1700

for i in range(1,len(days)):

    recovered[i] =recovered[i-1]*rates[i] 

recovered = [num if num<3*10**8 else 3*10**8 for num in recovered]

active = [confirmed[i]-recovered[i] for i in range(len(days))]

#cap active below at 0

active = [num if num>0 else 0 for num in active]
plt.plot(days,confirmed)

plt.xlabel('nth day')

plt.ylabel('# of confirmed cases')

plt.title('US confirmed cases prediction')

plt.figure(figsize=(20, 10))

plt.show()

plt.plot(days,recovered)

plt.xlabel('nth day')

plt.ylabel('# of recovered cases')

plt.title('US recovered cases prediction')

plt.show()

plt.plot(days, active)

plt.xlabel('nth day')

plt.ylabel('# of active cases')

plt.title('US Active cases prediction (Confirmed-recovered)')

plt.show()
