## Loading Json Data

import json
f = open("../input/db.json",'r')
data = json.load(f)


## Extracting utc timestamp in a list    

utc = []
for i in range(1,3226):
    utc.append(data['_default'][str(i)]['created_utc'])
    

### Using the timestamp list to get memes on a certain day
import datetime
days={'Monday':0,'Tuesday':0,'Wednesday':0,'Thursday':0,'Friday':0,'Saturday':0,'Sunday':0}

for i in utc:
    day=datetime.datetime.fromtimestamp(int(i)).weekday()
    if day == 0:
        days['Monday'] +=1
    elif day ==1:
        days['Tuesday'] +=1
    elif day ==2:
        days['Wednesday'] += 1
    elif day ==3:
        days['Thursday'] += 1
    elif day ==4:
        days['Friday'] += 1
    elif day ==5:
        days['Saturday'] += 1
    elif day ==6:
        days['Sunday'] += 1    
        

## Plotting the graph


import numpy as np
import matplotlib.pyplot as plt
 
weekday = ('Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun')
y_pos = np.arange(len(weekday))
no_of_memes = [days['Monday'] ,days['Tuesday'],days['Wednesday'],days['Thursday'],days['Friday'],days['Saturday'],days['Sunday']]
 
plt.bar(y_pos, no_of_memes, align='center', alpha=0.5)
plt.xticks(y_pos, weekday)
plt.ylabel('No Of Memes')
plt.xlabel('WeekDay')
plt.title('Memes Per Day')
 
plt.show()