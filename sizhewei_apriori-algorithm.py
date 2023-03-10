import numpy as np 

import pandas as pd 
data = pd.read_csv('/kaggle/input/supermarket/GroceryStoreDataSet.csv', header=None)

data.head()

transactions = []

for i in range(len(data)):

    transactions.append(data.values[i, 0].split(','))

print(transactions)
class Apriori:

    

    def __init__(self, transactions, min_support, min_confidence):

        self.transactions = transactions

        self.min_support = min_support # The minimum support.

        self.min_confidence = min_confidence # The minimum confidence.

        self.support_data = {} # A dictionary. The key is frequent itemset and the value is support.      

        

    def create_C1(self):

        """

        create frequent candidate 1-itemset C1 by scaning data set.

        Input:

            None 

        Output:

            C1: A set which contains all frequent candidate 1-itemsets

        """

        C1 = set()

        for transaction in self.transactions:

            for item in transaction:

                C1.add(frozenset([item]))

        return C1

    

    def isapriori(self, Ck_item, Lksub1):

        for item in Ck_item:

            sub_Ck = Ck_item - frozenset([item])

            if sub_Ck not in Lksub1:

                return False

        return True

    

    def create_Ck(self, Lksub1, k):

        """

        Create Ck.

        Input:

            Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.

            k: the item number of a frequent itemset.

        Output:

            Ck: A set which contains all all frequent candidate k-itemsets.

        """

        

        Ck = set()

        len_Lksub1 = len(Lksub1)

        list_Lksub1 = list(Lksub1)

        for i in range(len_Lksub1):

#             for j in range(1, len_Lksub1):# Ask: if there should be i+1 instead of 1 ???

            for j in range(i+1, len_Lksub1):# Ask: if there should be i+1 instead of 1 ???

                l1 = list(list_Lksub1[i])

                l2 = list(list_Lksub1[j])

                l1.sort()

                l2.sort()

#                 is_Apriori = True

                if l1[0:k-2] == l2[0:k-2]:

                    # TODO: self joining Lk-1

                    Ck_item = list_Lksub1[i] | list_Lksub1[j]

                    # TODO: pruning

                    if self.isapriori(Ck_item, Lksub1):

                        Ck.add(Ck_item)

#                     for item in Ck_item:

#                         sub_Ck_item = Ck_item-frozenset([item])

#                         if sub_Ck_item not in Lksub1:

#                             is_Apriori = False

#                             break

#                     if is_Apriori :

#                         Ck.add(frozenset([Ck_item]))

                    



        return Ck

    

    def generate_Lk_from_Ck(self, Ck):

        """

        Generate Lk by executing a delete policy from Ck.

        Input:

            Ck: A set which contains all all frequent candidate k-itemsets.

        Output:

            Lk: A set which contains all all frequent k-itemsets.

        """

        

        Lk = set()

        item_count = {}

        for transaction in self.transactions:

            for item in Ck:

                if item.issubset(transaction):

                    if item not in item_count:

                        item_count[item] = 1

                    else:

                        item_count[item] += 1

        t_num = float(len(self.transactions))

        for item in item_count:

            support = item_count[item] / t_num

            if support >= self.min_support:

                Lk.add(item)

                self.support_data[item] = support

        return Lk

        

    def generate_L(self):

        """

        Generate all frequent item sets..

        Input:

            None

        Output:

            L: The list of Lk.

        """        

        self.support_data = {}

        

        C1 = self.create_C1()

        L1 = self.generate_Lk_from_Ck(C1)

        Lksub1 = L1.copy()

        L = []

        L.append(Lksub1)

        i = 2

        while True:

            Ci = self.create_Ck(Lksub1, i)

            Li = self.generate_Lk_from_Ck(Ci)

            if Li:

                Lksub1 = Li.copy()

                L.append(Lksub1)

                i += 1

            else:

                break

        return L

        

        

    def generate_rules(self):

        """

        Generate association rules from frequent itemsets.

        Input:

            None

        Output:

            big_rule_list: A list which contains all big rules. Each big rule is represented

                       as a 3-tuple.

        """

        L = self.generate_L()

        

        big_rule_list = []

        sub_set_list = []

        for i in range(0, len(L)):

            for freq_set in L[i]:

                for sub_set in sub_set_list:

                    if sub_set.issubset(freq_set):

                        # TODO : compute the confidence

                        conf =  self.support_data[freq_set] / self.support_data[freq_set - sub_set]

                        big_rule = (freq_set - sub_set, sub_set, conf)

                        if conf >= self.min_confidence and big_rule not in big_rule_list:

                            big_rule_list.append(big_rule)

                sub_set_list.append(freq_set)

        return big_rule_list

        
model = Apriori(transactions, min_support=0.1, min_confidence=0.75)
L = model.generate_L()



for Lk in L:

    print('frequent {}-itemsets???\n'.format(len(list(Lk)[0])))



    for freq_set in Lk:

        print(freq_set, 'support:', model.support_data[freq_set])

    

    print()
rule_list = model.generate_rules()



for item in rule_list:

    print(item[0], "=>", item[1], "confidence: ", item[2])
# transactions: [['MILK', 'BREAD', 'BISCUIT'], ['BREAD', 'MILK', 'BISCUIT', 'CORNFLAKES'], ['BREAD', 'TEA', 'BOURNVITA'], ['JAM', 'MAGGI', 'BREAD', 'MILK'], ['MAGGI', 'TEA', 'BISCUIT'], ['BREAD', 'TEA', 'BOURNVITA'], ['MAGGI', 'TEA', 'CORNFLAKES'], ['MAGGI', 'BREAD', 'TEA', 'BISCUIT'], ['JAM', 'MAGGI', 'BREAD', 'TEA'], ['BREAD', 'MILK'], ['COFFEE', 'COCK', 'BISCUIT', 'CORNFLAKES'], ['COFFEE', 'COCK', 'BISCUIT', 'CORNFLAKES'], ['COFFEE', 'SUGER', 'BOURNVITA'], ['BREAD', 'COFFEE', 'COCK'], ['BREAD', 'SUGER', 'BISCUIT'], ['COFFEE', 'SUGER', 'CORNFLAKES'], ['BREAD', 'SUGER', 'BOURNVITA'], ['BREAD', 'COFFEE', 'SUGER'], ['BREAD', 'COFFEE', 'SUGER'], ['TEA', 'MILK', 'COFFEE', 'CORNFLAKES']]

test_c = set()

transactions = [['a','c','d'],['b','c','e'],['a','b','c','e'],['b','e']]

for transaction in transactions:

    for item in transaction:

        test_c.add(frozenset([item]))

print(test_c)

lentestc = len(test_c)

list_testc = list(test_c)

print(list_testc)

k=1

Ck=set()

for i in range(lentestc):

    for j in range(i+1,lentestc):

#         print(list_testc[0])

        l1 = list(list_testc[i])

        l2 = list(list_testc[j])

#         print(l1)

#         print(l2)

        l1.sort()

        l2.sort()

#         print(l1)

#         print(l2)

#         print(l1[0:k-2])

        if l1[0:k-2] == l2[0:k-2]:

            Ck.add(frozenset(list_testc[i]|list_testc[j]))

#             print(Ck)

#         if j==1:

#             print(j)

#             break

#     if i==0:

#         print(i)

#         break

print(Ck)

print(len(Ck))
Lk = set()

item_count = {}

for transaction in transactions:

    for item in test_c:

        if item.issubset(transaction):

            if item not in item_count:

                item_count[item] = 1

            else:

                item_count[item] += 1

                

print(item_count)
count = 0

for i in range(5):

    for j in range(1,5):

        print(i,j)

        count +=1 

print(count)