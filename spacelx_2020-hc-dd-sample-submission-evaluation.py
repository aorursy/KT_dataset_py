import pandas as pd

import numpy as np

from tqdm import tqdm
print('Extracting data')



# =============================================================================

# load problem file

# =============================================================================

with open('/kaggle/input/hashcode-drone-delivery/busy_day.in') as file:

    line_list = file.read().splitlines()

    

# =============================================================================

# problem parameters

# =============================================================================

ROWS, COLS, DRONES, TURNS, MAXLOAD = map(int, line_list[0].split())

   

# =============================================================================

# load product information

# =============================================================================

weights = line_list[2].split()

products_df = pd.DataFrame({'weight': weights})



wh_count = int(line_list[3])

wh_endline = (wh_count*2)+4



wh_invs = line_list[5:wh_endline+1:2]

for i, wh_inv in enumerate(wh_invs):

    products_df[f'wh{i}_inv'] = wh_inv.split()



# products_df has shape [400,11]

# (# of products, [weight, wh0_inv, wh1_inv,...])

products_df = products_df.astype(int)



# =============================================================================

# load warehouse locations

# =============================================================================

wh_locs = line_list[4:wh_endline:2]

wh_rows = [wl.split()[0] for wl in wh_locs]

wh_cols = [wl.split()[1] for wl in wh_locs]



warehouse_df = pd.DataFrame(

    {'wh_row': wh_rows, 'wh_col': wh_cols}).astype(np.uint16)



# =============================================================================

# load order information

# =============================================================================

order_locs = line_list[wh_endline+1::3]

o_rows = [ol.split()[0] for ol in order_locs]

o_cols = [ol.split()[1] for ol in order_locs]



orders_df = pd.DataFrame({'row': o_rows, 'col': o_cols})



orders_df[orders_df.duplicated(keep=False)].sort_values('row')



orders_df['product_count'] = line_list[wh_endline+2::3]



order_array = np.zeros((len(orders_df), len(products_df)), dtype=np.uint16)

orders = line_list[wh_endline+3::3]

for i,order in enumerate(orders):

    products = [int(prod) for prod in order.split()]

    for p in products:

        order_array[i, p] += 1



df = pd.DataFrame(data=order_array,

                  columns=['p_'+ str(i) for i in range(400)],

                  index=orders_df.index)



orders_df = orders_df.astype(int).join(df)



print('... success')
# get list of drone commands

submission = pd.read_csv('/kaggle/input/2020hcdd-sample-submission/submission.csv')

allcommands = submission[submission.columns[0]].values



# delivery_times will store the timestamp of the last delivery for each order

order_completion_times = np.full((len(orders_df)), -1)

# missing_items holds the number of each product required to complete an order

missing_items = orders_df.copy()

# inventory_ops simply lists all inventory changes of all warehouses

inventory_ops = pd.DataFrame(columns=['action', 'wh', 'item', 'count', 'turn'])



# iterate through all drones

for ddd in tqdm(range(DRONES)):

    

    # get only commands for this specific drone

    dronecommands = [iii for iii in allcommands if iii.split()[0] == str(ddd)]

    

    # all drones start at warehouse 0 at timestep 0 with 0 weight loaded

    currentloc = warehouse_df.loc[0].values

    currenttime = 0

    currentweight = 0

    

    # go through commands in order

    for cmd in dronecommands:

        

        # split command into separate components

        _, action, locidx, prod, count = cmd.split(' ')

        

        # for "wait" commands

        # add given number of turns to timer and continue

        if action == 'W':

            currenttime += locidx

            continue



        # get target location

        if action == 'L' or action == 'U':

            newloc = warehouse_df.loc[int(locidx)].values

        elif action == 'D':

            newloc = orders_df.loc[int(locidx), ['row', 'col']].values

        # calculate distance, round up to the next integer and add to timer

        dist = int(np.ceil(np.sqrt(np.sum((currentloc-newloc)**2))))

        currenttime += dist

        # add one step for loading / unloading / delivery itself

        currenttime += 1

        # check if end of simulation is reached

        if currenttime > TURNS:

            raise Exception('Maximum simulation time exceeded')

        # update current location

        currentloc = np.copy(newloc)

        

        # update drone weight

        if action == 'L':

            currentweight += int(count) * products_df.loc[int(prod),'weight']

        elif action == 'D' or action == 'U':

            currentweight -= int(count) * products_df.loc[int(prod),'weight']

        # check if drone weight limit exceeded  

        if currentweight > MAXLOAD:

            raise Exception('Maximum drone load exceeded')

        

        # for deliveries

        if action == 'D':

            # check number of items delivered and update required items

            if missing_items.at[int(locidx), 'p_{}'.format(prod)] >= int(count):

                missing_items.at[int(locidx), 'p_{}'.format(prod)] -= int(count)

                # note latest delivery for each order

                if currenttime > order_completion_times[int(locidx)]:

                     order_completion_times[int(locidx)] = currenttime

            else:

                raise Exception('Too many items delivered')

                

        # save list of loading / unloading operations for checking warehouse inventory

        if action == 'L' or action == 'U':

            inventory_ops = inventory_ops.append({

                'action': action,

                'wh': int(locidx),

                'item': int(prod),

                'count': int(count),

                'turn': currenttime

            }, ignore_index=True)
for wh in range(len(warehouse_df)):

    for item in tqdm(range(len(products_df))):

        # all inventory operations at this warehouse involving this product

        tmp = inventory_ops[

            (inventory_ops['wh'] == wh) &

            (inventory_ops['item'] == item)            

        ]

        if not len(tmp):

            continue

        

        # sort chronologically

        tmp = tmp.sort_values(by='turn')

        # get initial stock

        inv = products_df.loc[item, f'wh{wh}_inv']

        # if overall fewer or just as many products are removed as are stored

        # in the warehouse, no further checks needed

        if len(tmp[tmp['action'] == 'L']) <= inv:

            continue

        # otherwise, "simulate" loading and unloading to see

        # whether inventory goes negative

        for iii in tmp.index:

            if tmp.loc[iii, 'action'] == 'L':

                inv -= tmp.loc[iii, 'count']

            else:

                inv += tmp.loc[iii, 'count']

            # check inventory after each step

            if inv < 0:

                raise Exception('Removal of unstocked product attempted')
# check which orders are still missing items

completed = np.max(missing_items.iloc[:,3:].values > 0, axis=1) <= 0

print('Orders completed:', len(np.where(completed)[0]))

print('Orders not completed:', len(np.where(completed == False)[0]))

# sum up scores of all completed orders

order_scores = np.ceil(100 * (TURNS - order_completion_times[np.where(completed)[0]]) / TURNS)

print('Score:', int(np.sum(order_scores)))
import shutil



shutil.copyfile('/kaggle/input/2020hcdd-sample-submission/submission.csv', 'submission.csv')