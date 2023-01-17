# catalog has prices
catalog = [
    ["P1", 100],
    ["P2", 324],
    ["P3", 2],
]

#order has quantities
order = [
    ["P1", 12],
    ["P3", 100],
]

#make it easy to look up catalog items by converting to a dict
catalog_dict = {}
for c in catalog:
    catalog_dict[c[0]] = c

#WHAT IS THE TOTAL AMOUNT FOR THE ORDER?

total = 0
for item in order:
    catalog_item = catalog_dict[item[0]]
    #TODO adjust total using the order item and the catalog_item
    
print(total) 

#1440
# catalog has prices
catalog = [
    ["P1", 100],
    ["P2", 324],
    ["P3", 2],
]

#order has quantities
order = [
    ["P1", 12],
    ["P3", 100],
]

#make it easy to look up catalog items by converting to a dict
catalog_dict = {}
for c in catalog:
    catalog_dict[c[0]] = c

#WHAT IS THE TOTAL AMOUNT FOR THE ORDER?

total = 0
for item in order:
    part_number = item[0]
    catalog_item = catalog_dict[part_number]
    price = catalog_item[1]
    quantity = item[1]
    total += price * quantity
    
print(total) 


