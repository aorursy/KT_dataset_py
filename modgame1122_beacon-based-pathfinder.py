import rssi
import csv
import os.path
import time
import random
import numpy as np
from tqdm import tqdm

interface = raw_input('Enter interface name: ')
rssi_scanner = rssi.RSSI_Scan(interface)
fileCheck = 'n'

if (os.path.isfile("rss.csv") and os.path.isfile("Data.npy")):
    fileCheck = raw_input("Do you want to continue with the exisitng files? [y/n]: ")
    if(fileCheck == 'y'):
        Data = np.load('Data.npy', allow_pickle=True)
        wh = Data[0]
        Map = Data[1]
        ssids = Data[2]

if fileCheck == 'n':
    wh = input('Enter height and width (h,w): ')
    Map = np.ones(wh)
    ssids = input("Enter AP SSIDS ['BBPFAP1','BBPFAP2','BBPFAP3','BBPFAP4', ...]: ")
    ssids.sort()
    
values = []


while 1:
    if(raw_input('Do ypu want to take a measurement? [y/n]: ') != 'y'):
        break
    sample = input('Enter number of samples to take: ')
    xy = input('Enter Map coordinates b/w (0,0) to (' + str(wh[0] - 1) + ',' + str(wh[1] - 1) + '): ')
    loc = raw_input('Enter RSS name: ') + '=' + str(xy)
    if (xy >= (0,0)) and (xy < wh):
        raw_input("Press Enter to start training")
        for z in tqdm(range(sample)):
            result = rssi_scanner.getAPinfo(sudo=True, networks=ssids)
            for x in [item for item in ssids if item not in [i['ssid'] for i in result]]:
                result.append({'ssid':x, 'quality':'0', 'signal':-1000})
            result.sort(key=lambda i: i['ssid'])
            value = {'LOC':loc}
            for x in result:
                value.update({x['ssid']:int(x['signal'])})
            values.append(value)
            time.sleep(0.3)
        Map[xy] = 0
    else:
        print('Coordinates out of range.')

keys = ssids + ['LOC']
if(fileCheck == 'y'):
    with open('rss.csv', 'ab') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writerows(values)
else:
    with open('rss.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(values)
np.save('Data', np.array([wh, Map, ssids]))
import rssi
import csv
import os.path
import time
from tqdm import tqdm

interface = 'wlan0'
loc = "RSS01"
count = 10
values = []
ssids = ['BBPFAP1','BBPFAP2','BBPFAP3','BBPFAP4']
rssi_scanner = rssi.RSSI_Scan(interface)

for i in tqdm(range(count)):
    result = rssi_scanner.getAPinfo(sudo=True, networks=ssids)
    result.sort(key=lambda i: i['ssid'])
    value = {'LOC':loc}
    for x in result:
        value.update({x['ssid']:int(x['signal'])})
    values.append(value)
    time.sleep(0.3)
    
if(os.path.isfile("rss.csv")):
    with open('rss.csv', 'ab') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writerows(values)
else:
    keys = values[0].keys()
    with open('rss.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(values)

eval(loc.split('=')[1])
import rssi
import csv
import os.path
import time
import random
import numpy as np
from tqdm import tqdm
wh = (3,3)
Map = np.ones(wh) 
sample = 20
values = []
ssids = ['BBPFAP1','BBPFAP2','BBPFAP3','BBPFAP4']
xy = (1,2)
loc = "RSS06" + '=' + str(xy)
if (xy >= (0,0)) and (xy < wh):   
    for z in tqdm(range(sample)):
        result = [
            {
                'ssid':'BBPFAP2',
                'quality':'43/70',
                'signal':random.randint(-45, -40)
            },
            {
                'ssid':'BBPFAP4',
                'quality':'30/70',
                'signal':random.randint(-45, -40)
            },
            {
                'ssid':'BBPFAP3',
                'quality':'30/70',
                'signal':random.randint(-65, -60)
            },
            {
                'ssid':'BBPFAP1',
                'quality':'30/70',
                'signal':random.randint(-65, -60)
            }
        ] 
        for x in [item for item in ssids if item not in [i['ssid'] for i in result]]:
            result.append({'ssid':x, 'quality':'0', 'signal':-1000})
        result.sort(key=lambda i: i['ssid'])
        value = {'LOC':loc}
        for x in result:
            value.update({x['ssid']:int(x['signal'])})
        values.append(value)
    Map[xy] = 0
else:
    print('Coordinates out of range.')
if(os.path.isfile("rss.csv")):
    keys = ssids + ['LOC']
    with open('rss.csv', 'ab') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writerows(values)
else:
    keys = ssids + ['LOC']
    with open('rss.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(values)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pickle

dataset = pd.read_csv("rss.csv")
modelName = raw_input("Enter model name: ")
x = dataset.iloc[:,0:-1]
y = dataset['LOC']
X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y,test_size=0.2)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
pickle.dump(gnb, open(modelName, 'wb'))
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle
import rssi

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) 

def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return False

interface = raw_input('Enter interface name: ')
rssi_scanner = rssi.RSSI_Scan(interface)
Data = np.load('Data.npy', allow_pickle=True)
model = pickle.load(open(raw_input('Enter model name: '), 'rb'))
grid =  Data[1]
ssids =  Data[2]
prev = None
curr = None
route = [None]
move = None

while(1):
    goal = input('Enter goal (x, y): ')
    while (1):
        prev = curr
        result = rssi_scanner.getAPinfo(sudo=True, networks=ssids)
        into = into + 1
        for x in [item for item in ssids if item not in [i['ssid'] for i in result]]:
            result.append({'ssid':x, 'quality':'0', 'signal':-1000})
        result.sort(key=lambda i: i['ssid'])
        value = []
        for x in result:
            value.append(int(x['signal']))
        curr = [value]
        curr = eval(model.predict(curr)[0].split('=')[1])
        #print curr
        if(curr == goal):
            break
        if(curr not in route):
            route = astar(grid, curr, goal)
            route = route + [curr]
            route = route[::-1]
        #print(route)
        nex = route[route.index(curr) + 1]
        if (prev != None):
            cal = (curr[1] - prev[1], -1 * (curr[0] - prev[0]))
            calr = (curr[0] + cal[0], curr[1] + cal[1])
            call = (curr[0] - cal[0], curr[1] - cal[1])
            if(nex == calr):
                move = 'right'
            elif(nex == call):
                move = 'left'
            else:
                move = 'straight'
        else:
            move = 'straight'
        print(move)
        
    print('Goal Reached')
    if(raw_input('Do you want to restart? [y/n]: ') != 'y'):
        break
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle
#import rssi

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) 

def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return False

#interface = raw_input('Enter interface name: ')
#rssi_scanner = rssi.RSSI_Scan(interface)
Data = np.load('../input/Dataaa/Data.npy', allow_pickle=True)
model = pickle.load(open('../input/Dataaa/model.out', 'rb'))
grid =  Data[1]
ssids =  Data[2]

!pip install rssi
result = [
            {
                'ssid':'BBPFAP2',
                'quality':'43/70',
                'signal':random.randint(-45, -40)
            },
            {
                'ssid':'BBPFAP4',
                'quality':'30/70',
                'signal':random.randint(-45, -40)
            },
            {
                'ssid':'BBPFAP3',
                'quality':'30/70',
                'signal':random.randint(-65, -60)
            },
            {
                'ssid':'BBPFAP1',
                'quality':'30/70',
                'signal':random.randint(-35, -30)
            }
        ] 
for x in [item for item in ssids if item not in [i['ssid'] for i in result]]:
    result.append({'ssid':x, 'quality':'0', 'signal':-1000})
result.sort(key=lambda i: i['ssid'])
value = []
for x in result:
    value.append(int(x['signal']))

start = [value]
goal = [[-60,-57,-31,-70]]

i = 0
results = [
    [
        {
        'ssid':'BBPFAP1',
        'quality':'43/70',
        'signal':-34
        },
        {
            'ssid':'BBPFAP2',
            'quality':'30/70',
            'signal':-55
        },
        {
            'ssid':'BBPFAP3',
            'quality':'30/70',
            'signal':-58
        },
        {
            'ssid':'BBPFAP4',
            'quality':'30/70',
            'signal':-68
        }
    ],
    [
        {
        'ssid':'BBPFAP1',
        'quality':'43/70',
        'signal':-40
        },
        {
            'ssid':'BBPFAP2',
            'quality':'30/70',
            'signal':-61
        },
        {
            'ssid':'BBPFAP3',
            'quality':'30/70',
            'signal':-45
        },
        {
            'ssid':'BBPFAP4',
            'quality':'30/70',
            'signal':-63
        }
    ],
    [
        {
        'ssid':'BBPFAP1',
        'quality':'43/70',
        'signal':-60
        },
        {
            'ssid':'BBPFAP2',
            'quality':'30/70',
            'signal':-67
        },
        {
            'ssid':'BBPFAP3',
            'quality':'30/70',
            'signal':-30
        },
        {
            'ssid':'BBPFAP4',
            'quality':'30/70',
            'signal':-57
        }
    ],
    [
        {
        'ssid':'BBPFAP1',
        'quality':'43/70',
        'signal':-60
        },
        {
            'ssid':'BBPFAP2',
            'quality':'30/70',
            'signal':-67
        },
        {
            'ssid':'BBPFAP3',
            'quality':'30/70',
            'signal':-30
        },
        {
            'ssid':'BBPFAP4',
            'quality':'30/70',
            'signal':-57
        }
    ],
    [
        {
        'ssid':'BBPFAP1',
        'quality':'43/70',
        'signal':-65
        },
        {
            'ssid':'BBPFAP2',
            'quality':'30/70',
            'signal':-63
        },
        {
            'ssid':'BBPFAP3',
            'quality':'30/70',
            'signal':-41
        },
        {
            'ssid':'BBPFAP4',
            'quality':'30/70',
            'signal':-40
        }
    ],
    [
        {
        'ssid':'BBPFAP1',
        'quality':'43/70',
        'signal':-65
        },
        {
            'ssid':'BBPFAP2',
            'quality':'30/70',
            'signal':-55
        },
        {
            'ssid':'BBPFAP3',
            'quality':'30/70',
            'signal':-55
        },
        {
            'ssid':'BBPFAP4',
            'quality':'30/70',
            'signal':-32
        }
    ]
]
prev = None
curr = None
route = [None]
move = None

while(1):
    goal = input('Enter goal (x, y): ')
    into = 0
    while (1):
        prev = curr
        result = results
        into = into + 1
        for x in [item for item in ssids if item not in [i['ssid'] for i in result]]:
            result.append({'ssid':x, 'quality':'0', 'signal':-1000})
        result.sort(key=lambda i: i['ssid'])
        value = []
        for x in result:
            value.append(int(x['signal']))
        curr = [value]
        curr = eval(model.predict(curr)[0].split('=')[1])
        #print curr
        if(curr == goal):
            break
        if(curr not in route):
            route = astar(grid, curr, goal)
            route = route + [curr]
            route = route[::-1]
        #print(route)
        nex = route[route.index(curr) + 1]
        if (prev != None):
            cal = (curr[1] - prev[1], -1 * (curr[0] - prev[0]))
            calr = (curr[0] + cal[0], curr[1] + cal[1])
            call = (curr[0] - cal[0], curr[1] - cal[1])
            if(nex == calr):
                move = 'right'
            elif(nex == call):
                move = 'left'
            else:
                move = 'straight'
        else:
            move = 'straight'
        print(move)
        
    print('Goal Reached')
    if(raw_input('Do you want to restart? [y/n]: ') != 'y'):
        break
ssids

start = eval(model.predict(start)[0].split('=')[1])
goal = eval(model.predict(goal)[0].split('=')[1])

goal = (2,2)

route = astar(grid, start, goal)
route = route + [start]
route = route[::-1]
print(route)
(0,0) in route

##############################################################################
# plot the path
##############################################################################

 

#extract x and y coordinates from route list
x_coords = []
y_coords = []

for i in (range(0,len(route))):
    x = route[i][0]
    y = route[i][1]
    x_coords.append(x)
    y_coords.append(y)

# plot map and path
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid, cmap=plt.cm.Dark2)
ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)
ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)
ax.plot(y_coords,x_coords, color = "black")
plt.show()
p = (1,1)
c = (0,1)
n = (0,0)
cal = (c[1] - p[1], -1 * (c[0] - p[0]))
calr = (c[0] + cal[0], c[1] + cal[1])
call = (c[0] - cal[0], c[1] - cal[1])

print('Right = ' + str(calr == n) + ', Left = ' + str(call == n))