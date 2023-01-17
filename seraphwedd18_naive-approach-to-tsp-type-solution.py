import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
with open("../input/hashcode-drone-delivery/busy_day.in") as f:

    data = f.read().splitlines()
def dist(A, B):

    return np.ceil(((A.x - B.x)**2 + (A.y - B.y)**2) ** 0.5)



class Warehouse():

    

    def __init__(self, id_no, row, col, data):

        '''Personal data of each warehouse.'''

        self.idn = id_no

        self.y = row

        self.x = col

        self.inv = np.array(data) #An array

    

    def __repr__(self):

        '''A warehouse is represented by its total inventory.'''

        return "%i" %sum(self.inv)

    

    def set_distances_orders(self, orders):

        '''Sets the fixed weights for travel to each orders.'''

        self.order_distances = [dist(self, o) for o in orders]

    

    def set_distances_warehouses(self, warehouses):

        '''Sets the fixed weights for travel between warehouses.'''

        self.warehouse_distances = [dist(self, w) for w in warehouses]

    

    def check_pickup(self, item, n):

        '''Check if n inventory item is available. Returns True if available else False.'''

        if self.inv[item] >= n:

            self.inv[item] -= n

            return True

        

        else:

            return False





class Order():

    

    def __init__(self, id_no, row, col, data):

        '''Personal data of each order.'''

        self.idn = id_no

        self.y = row

        self.x = col

        self.order = data #A list

        self.finished = False

        self.finish_time = np.inf

    

    def __repr__(self):

        '''An order is represented by the length of its order.'''

        return "%i" %len(self.order)

    

    def set_distances_warehouses(self, warehouses):

        '''Sets the fixed weights for travel between warehouses.'''

        self.warehouse_distances = [dist(self, w) for w in warehouses]

    

    def check_remaining_order(self):

        '''Checks if order is still unfinished.'''

        try:

            return self.order[0]

        except:

            return None

    

    def check_drop(self, item, n):

        '''Updates order for each n units of item type.'''

        for _ in range(n):

            self.order.remove(item)



            

class Drone():

    

    def __init__(self, id_no, row, col):

        '''Personal properties of each drone.'''

        self.idn = id_no

        self.y = row

        self.x = col

        self.status = 'wait'

        

        self.item_held = [0 for _ in range(NUM_PRODUCTS)] #An array

        self.action_timer = 0

        self.actions = []

    

    def __repr__(self):

        "A drone is represented by its unique ID."

        return "%i" %self.idn

    

    def get_job(self, orders, warehouses):

        '''Picks a job to do if status is "wait".'''

        

        #First, check if not idle:

        if self.status != 'wait':

            if self.action_timer <= 0:

                try: #Try to fetch action

                    msg = self.actions.pop(0)

                    

                except: #If no more actions available, idle

                    self.status = 'wait'

                    return

                

                _, t, idn, _, _ = msg.split()

                

                if t == 'L':

                    d = dist(self, warehouses[int(idn)])

                    self.y, self.x = warehouses[int(idn)].y, warehouses[int(idn)].x

                    

                else:

                    self.status = 'deliver'

                    d = dist(self, orders[int(idn)])

                    self.y, self.x = orders[int(idn)].y, orders[int(idn)].x

                    self.item_held = [0 for _ in range(NUM_PRODUCTS)]

                    

                self.action_timer = 1 + d

                return msg

            else:

                self.action_timer -= 1

                

            return None

        

        #Second, if idle, check for nearby order to fulfill

        d = np.inf

        o = None

        for order in orders:

            dd = dist(self, order)

            if dd < d:

                finished = False

                #Check if item is already being attended to

                try:

                    if len(order.order) == len(CURRENT_FINISHED_DELIVERY[order.idn]):

                        finished = True

                except:

                    pass

                    

                if not finished:

                    d = dd

                    o = order

        

        #If no more order

        if o == None or o.finished:

            return None

                

        #Third, choose items to pickup within capacity

        picked = {}

        for item in o.order:

            #Check if item already attended to

            try:

                if o.order.count(item) > CURRENT_FINISHED_DELIVERY[o.idn].count(item):

                    pass

                else:

                    continue #Next item

            except:

                pass

            

            #Check if any item can still fit

            if MAX_WEIGHT - sum(self.item_held * PROD_WEIGHTS) < MIN_WEIGHT:

                break #Escape if full



            #If it can, check if next item can fit

            self.item_held[item] += 1

            if sum(self.item_held * PROD_WEIGHTS) > MAX_WEIGHT:

                self.item_held[item] -= 1

            else:

                picked[item] = -1 #Mutates to a list of (dist, wrh.idn, item)

                try:

                    CURRENT_FINISHED_DELIVERY[o.idn].append(item)

                except:

                    CURRENT_FINISHED_DELIVERY[o.idn] = [item,]

                

                

        #Fourth, check for the nearest path from warehouse pickup to order delivery

        d = np.inf

        w = None

        for w_dist, ids in sorted(zip(o.warehouse_distances, range(10))):

            wrh = warehouses[ids]

            dd = dist(self, wrh)

            

            #Initialize nearest warehouse to load from

            for i in picked.keys():

                if (wrh.inv[i]) and (picked[i] == -1):

                    picked[i] = [dd, wrh.idn, i]

            

            if dd + w_dist < d:

                d = dd

                w = wrh

                #Update nearest warehouse if has stock of item

                for i in picked.keys():

                    if wrh.inv[i] >= self.item_held[i]:

                        picked[i] = [dd, wrh.idn, i]

        

        #Save all Load actions

        for d, idn, i in sorted(picked.values()):

            self.actions.append("%i L %i %i %i" %(self.idn, idn, i, self.item_held[i]))

        #Save all Drop actions

        for d, idn, i in sorted(picked.values()):

            self.actions.append("%i D %i %i %i" %(self.idn, o.idn, i, self.item_held[i]))

        

        self.status = 'load' #To start action again

        return None
NUM_ROWS, NUM_COLS, DRONES, DEADLINE, MAX_WEIGHT = map(int, data.pop(0).split())

NUM_PRODUCTS = int(data.pop(0))

PROD_WEIGHTS = np.array(list(map(int, data.pop(0).split())))

MIN_WEIGHT = np.min(PROD_WEIGHTS)

WAREHOUSE_NUM = int(data.pop(0))



print("ROWS: %i, COLS: %i, DRONES: %i, DEADLINE: %i, MAX_WEIGHT: %i, NUM_PRODUCTS: %i, WAREHOUSE: %i" %(

    NUM_ROWS, NUM_COLS, DRONES, DEADLINE, MAX_WEIGHT, NUM_PRODUCTS, WAREHOUSE_NUM))



all_warehouses = []

all_orders = []



for q in range(WAREHOUSE_NUM):

    r, c = map(int, data.pop(0).split())

    all_warehouses.append(Warehouse(q, r, c, list(map(int, data.pop(0).split()))))

    

CUSTOMERS = int(data.pop(0))



for cid in range(CUSTOMERS):

    r, c = map(int, data.pop(0).split())

    n = int(data.pop(0))

    orders = list(map(int, data.pop(0).split()))

    all_orders.append(Order(cid, r, c, orders))



for w in all_warehouses:

    w.set_distances_warehouses(all_warehouses)

    w.set_distances_orders(all_orders)



for o in all_orders:

    o.set_distances_warehouses(all_warehouses)

    

CURRENT_FINISHED_DELIVERY = {} #For tracking of active deliveries
print("Warehouses:", all_warehouses)

print("Total inventory: %i" %sum([sum(x.inv) for x in all_warehouses]))

print("Total Orders: %i" %sum([len(x.order) for x in all_orders]))
all_drones = []

for i in range(DRONES):

    all_drones.append(Drone(i,

                            all_warehouses[i%WAREHOUSE_NUM].y,

                            all_warehouses[i%WAREHOUSE_NUM].x)

                     )



LIMIT = 5 #Order limit

all_msg = []

all_timers = []

all_done = False

for t in range(DEADLINE):

    #For each loop, update all drones

    #print("Loop: ", t, [x.action_timer for x in all_drones])

    all_timers.append([x.action_timer for x in all_drones])

    for i in range(DRONES):

        msg = all_drones[i].get_job(all_orders[:LIMIT], all_warehouses)

        

        if msg:

            #Check for finished orders:

            

            all_done = True

            for o in all_orders[:LIMIT]:

                if len(o.order) == len(CURRENT_FINISHED_DELIVERY.get(o.idn, [])):

                    o.finished = True

                    o.finish_time = t

                else:

                    all_done = False

                

            #print(CURRENT_FINISHED_DELIVERY)

            #print(sum(o.finished for o in all_orders), [o.idn for o in all_orders if o.finished])

            #print(t, '\t', all_done, msg)

            all_msg.append(msg)

            

    if all_done:

        if all(dr.status == 'wait' for dr in all_drones):

            break

    
with open('submission.csv', 'w') as f:

    f.write("%i\n" %len(all_msg))

    f.write("\n".join(all_msg))
print(len(all_msg))

print('\n'.join(all_msg[:200]))
import matplotlib.pyplot as plt



plt.plot(all_timers[:5000])

plt.show()
s = 0

for k in CURRENT_FINISHED_DELIVERY.keys():

    s += len(CURRENT_FINISHED_DELIVERY[k])

print('Total Delivered Items:', s)

print("Total inventory: %i" %sum([sum(x.inv) for x in all_warehouses]))

print("Total Orders: %i" %sum([len(x.order) for x in all_orders[:LIMIT]]))