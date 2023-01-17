

f = open("/kaggle/input/hashcode-drone-delivery/busy_day.in", "r")



parameters = list(map(int,f.readline().split()))



rows = parameters[0]

cols = parameters[1]

num_drones = parameters[2]

max_turns = parameters[3]

max_weight = parameters[4]



print(f" rows -> {rows}")

print(f" cols -> {cols}")

print(f" num_drones -> {num_drones}")

print(f" max_turns -> {max_turns}")

print(f" max_weight -> {max_weight}")
#Products dictionary



num_products = int(f.readline())



products = dict()

weights_list = list(map(int,f.readline().split()))

for product_no, product_weight in enumerate(weights_list):

    products[product_no] = product_weight



                                                 

print(products)



#warehouse inventory dictionary



num_warehouse = int(f.readline())

warehouses = dict()



for warehouse_no in range(num_warehouse):



    warehouse = dict()

    warehouse["location"] = list(map(int,f.readline().split())) 

    inventory = dict()

    inventory_list = list(map(int,f.readline().split()))

    for inventory_serial, num_inventory_product in enumerate(inventory_list):

        inventory[inventory_serial] = num_inventory_product



    warehouse["inventory"] = inventory

    

    warehouses[warehouse_no] = warehouse

    

print(warehouses)
%%time

print(warehouses[4]['inventory'][7])


num_orders = int(f.readline())

orders = dict()



for order_no in range(num_orders):



    order = dict()

    order["location"] = list(map(int,f.readline().split())) 

    ordered_items = dict()

    num_ordered_items = int(f.readline())

    ordered_items_list = list(map(int,f.readline().split()))

    for ordered_items_serial,ordered_items_type  in enumerate(ordered_items_list):

        ordered_items[ordered_items_serial] = ordered_items_type



    order["items"] = ordered_items

    

    orders[order_no] = order

    

print(orders)
print(orders[999]["location"])