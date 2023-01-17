# required packages: install aprior algorithm package

!pip install apyori
import apyori as ap
def display_rules(association_results):

    for item in association_results:

        # first index of the inner list

        # Contains base item and add item

        pair = item[0] 

#         print("pair", len(pair))

        if len(pair) == 1:

            continue

        items = [x for x in pair]

#         print(items)

        if 'nan' in items:

            continue

        

        print("item",item)

#         print('item[2][0][1]',str(item[2][0][1]))

#         print('item[2][0][0]',str(item[2][0][0]))

        LL = len(item[2])

#         print("lenght",LL)

        for tt in range(0,LL):

            A_items = [x for x in item[2][tt][0]]

            B_items = [x for x in item[2][tt][1]]

#             print("A_items",A_items)

#             print("B_items",B_items)

            print("Rule: " + str(A_items) + " -> " + str(B_items))

            #         print("Rule: " + item[2][0][0] + " -> " + item[2][0][1])



            #second index of the inner list

            print("Support: " + str(item[1]))



            #third index of the list located at 0th

            #of the third index of the inner list



            print("Confidence: " + str(item[2][tt][2]))

            print("Lift: " + str(item[2][tt][3]))

            print("=====================================")
#Toy example 1

transactions = [

    ['apple', 'orange'],

    ['apple', 'banana'],

]

association_rules = ap. apriori(transactions, min_support=0.5, min_confidence=0.5, min_lift=1, min_length=2)

association_results = list(association_rules)

print(association_results)

print("------------list of rules------------")

display_rules(association_results)
#Toy example 2 

transactions_2 = [

    ['Bread', 'Milk', 'Chips', 'Mustard'],

    ['Beer', 'Diaper', 'Bread', 'Eggs'],

    ['Beer', 'Coke', 'Diaper', 'Milk'],

    ['Beer', 'Bread', 'Diaper', 'Milk','Chips'],

    ['Coke', 'Bread', 'Diaper', 'Milk'],

    ['Beer', 'Bread', 'Diaper', 'Milk','Mustard'],

    ['Coke', 'Bread', 'Diaper', 'Milk'],



]

print("transactions_2",transactions_2)
association_rules = ap. apriori(transactions_2, min_support=0.4, min_confidence=0.75, min_lift=1, min_length=2)

association_results = list(association_rules)

# print(association_results)

print(association_results[5])

print("------------list of rules------------")

display_rules(association_results)
#Toy example 3

transactions_3 = [

    ['Beer', 'Diaper', 'Baby Powder', 'Bread', 'Umbrella'],

    ['Diaper', 'Baby Powder'],

    ['Beer', 'Diaper', 'Milk'],

    ['Diaper', 'Beer','Detergent'],

    ['Beer','Milk','Coke'],

]
association_rules = ap. apriori(transactions_3, min_support=0.4, min_confidence=0.7, min_lift=0, min_length=2)

association_results = list(association_rules)

# print(association_results)

print("------------list of rules------------")

display_rules(association_results)
import pandas as pd

store_data = pd.read_csv('..//input//store_data.csv',header=None)

store_data.head()
records = []

for i in range(0, 7501):

    records.append([str(store_data.values[i,j]) for j in range(0, 20)])
association_rules = ap.apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)

association_results = list(association_rules)

print("------------list of rules------------")

display_rules(association_results)