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
# so the rate_of_spread  0.11988 based on 3 days increase 
# This gives us the most probable predication of the spread 

cuurent_count = 100 # based on March 15 early morning count

rate_of_spread =  0.140

thislist = [cuurent_count]
last_count = thislist[0]

for x in range(113):
    last_count = last_count + last_count*rate_of_spread
    thislist.append(last_count)

    
print(thislist)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np



#30days 
plt.plot(thislist)
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()



#### real data 


#### Real Data 
#real_data  = pd.read_csv("/kaggle/input/covid19april6/complete_april6.csv")
real_data  = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/complete.csv")
#real_data['total_count'] = real_data['Total Confirmed cases (Indian National)'] + real_data['Total Confirmed cases ( Foreign National )']
real_data.head()

#real = real_data[real_data['Date']>='2020-03-15'].groupby(['Date'])['total_count'].agg('sum')


real = real_data[real_data['Date']>='2020-03-15'].groupby(['Date'])['Total Confirmed cases'].agg('sum')

print(real)

real_daily_count = real.to_numpy()
print(real_daily_count)


daily_increase =  np.copy(real_daily_count)

prev= real_daily_count[0]


for index, value in np.ndenumerate(daily_increase):
        id = index[0]
        daily_increase[id] = value - prev
        prev = value

print(daily_increase)

#15days 
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))


ax = plt.subplot(111)
ax.plot(np.arange(113), thislist[:-1], label='Predicted')
ax.plot(np.arange(113), real_daily_count, label='Real')
plt.title('Prediction vs Real')
ax.legend()
plt.show()




#15days 
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
ax = plt.subplot(111)
ax.plot(np.arange(113), daily_increase, label='Daily Increase')
plt.title('Prediction vs Real')
ax.legend()
plt.show()
#### Real Data 
#real_data  = pd.read_csv("/kaggle/input/covid19april6/complete_april6.csv")
test_data  = pd.read_excel("/kaggle/input/covid19-india-complete-data/COVID19 India Complete Dataset April 2020.xlsx","State-Wise Testing Data")
#real_data['total_count'] = real_data['Total Confirmed cases (Indian National)'] + real_data['Total Confirmed cases ( Foreign National )']
#test_data = test_data[test_data["State"] == "West bengal"]

test_data = test_data[test_data["State"] ==  "West Bengal"]

test_data = test_data.sort_values(by="Updated On")

test_data["Total Tested"].plot()
cuurent_count = 100 # based on March 15 early morning count

rate_of_spread =  .05

thislist = [cuurent_count]
last_count = thislist[0]

for x in range(30):
    last_count = last_count + last_count*rate_of_spread
    thislist.append(last_count)

    
print(thislist)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#5days 
plt.plot(thislist[:-25])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#10days 
plt.plot(thislist[:-20])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#15days 
plt.plot(thislist[:-15])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#20days 
plt.plot(thislist[:-10])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#25days 
plt.plot(thislist[:-5])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#30days 
plt.plot(thislist)
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()
cuurent_count = 100 # based on March 15 early morning count

rate_of_spread =  .1

thislist = [cuurent_count]
last_count = thislist[0]

for x in range(30):
    last_count = last_count + last_count*rate_of_spread
    thislist.append(last_count)

    
print(thislist)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#5days 
plt.plot(thislist[:-25])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#10days 
plt.plot(thislist[:-20])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#15days 
plt.plot(thislist[:-15])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#20days 
plt.plot(thislist[:-10])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#25days 
plt.plot(thislist[:-5])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#30days 
plt.plot(thislist)
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()
cuurent_count = 100 # based on March 15 early morning count

rate_of_spread =  .25

thislist = [cuurent_count]
last_count = thislist[0]

for x in range(30):
    last_count = last_count + last_count*rate_of_spread
    thislist.append(last_count)

    
print(thislist)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#5days 
plt.plot(thislist[:-25])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#10days 
plt.plot(thislist[:-20])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#15days 
plt.plot(thislist[:-15])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#20days 
plt.plot(thislist[:-10])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#25days 
plt.plot(thislist[:-5])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#30days 
plt.plot(thislist)
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()
cuurent_count = 100 # based on March 15 early morning count

rate_of_spread =  .5

thislist = [cuurent_count]
last_count = thislist[0]

for x in range(30):
    last_count = last_count + last_count*rate_of_spread
    thislist.append(last_count)

    
print(thislist)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#5days 
plt.plot(thislist[:-25])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#10days 
plt.plot(thislist[:-20])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#15days 
plt.plot(thislist[:-15])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#20days 
plt.plot(thislist[:-10])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#25days 
plt.plot(thislist[:-5])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#30days 
plt.plot(thislist)
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()
cuurent_count = 100 # based on March 15 early morning count

rate_of_spread =  1

thislist = [cuurent_count]
last_count = thislist[0]

for x in range(30):
    last_count = last_count + last_count*rate_of_spread
    thislist.append(last_count)

    
print(thislist)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#5days 
plt.plot(thislist[:-25])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#10days 
plt.plot(thislist[:-20])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#15days 
plt.plot(thislist[:-15])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#20days 
plt.plot(thislist[:-10])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#25days 
plt.plot(thislist[:-5])
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()

#30days 
plt.plot(thislist)
plt.ylabel('infected count')
plt.xlabel('Days from  March  15th')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-999907374182400,999907374182400))
plt.show()