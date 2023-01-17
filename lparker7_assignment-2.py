import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



fpath = '../input/santa-2019-revenge-of-the-accountants/family_data.csv'

data = pd.read_csv(fpath, index_col='family_id')

fpath = '../input/santa-2019-revenge-of-the-accountants/sample_submission.csv'

submission = pd.read_csv(fpath, index_col='family_id')

family_size_dict = data[['n_people']].to_dict()['n_people'] 

cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].to_dict()

days = list(range(100, 0, -1))
def calc_penalty(table):

    penalty = 0    # 1

    people_scheduled = {k: 0 for k in days}    # n

    for family_id, day in enumerate(table):    # n

        number_of_people = family_size_dict[family_id]    # 2

        people_scheduled[day] += number_of_people    # 3

        # At most, this if statement has to do 10 evaluations, which all take 3

        if day == choice_dict['choice_0'][family_id]:

            penalty += 0    # 2

        elif day == choice_dict['choice_1'][family_id]:

            penalty += 50    # 2

        elif day == choice_dict['choice_2'][family_id]:

            penalty += 50 + 9 * number_of_people    # 4

        elif day == choice_dict['choice_3'][family_id]:

            penalty += 100 + 9 * number_of_people    # 4

        elif day == choice_dict['choice_4'][family_id]:

            penalty += 200 + 9 * number_of_people    # 4

        elif day == choice_dict['choice_5'][family_id]:

            penalty += 200 + 18 * number_of_people    # 4

        elif day == choice_dict['choice_6'][family_id]:

            penalty += 300 + 18 * number_of_people    # 4

        elif day == choice_dict['choice_7'][family_id]:

            penalty += 300 + 36 * number_of_people    # 4

        elif day == choice_dict['choice_8'][family_id]:

            penalty += 400 + 36 * number_of_people    # 4

        elif day == choice_dict['choice_9'][family_id]:

            penalty += 500 + 36 * number_of_people + 199 * number_of_people    # 6

        else:

            penalty += 500 + 36 * number_of_people + 398 * number_of_people    # 6



    for _, occupancy in people_scheduled.items():    # n

        if (occupancy < 125) or (occupancy > 300):    # 2 at worst

            # Use occupancy in penalty to incentivise picking under-occupied days

            penalty += (9999999999 + occupancy*10000)    # 4



    # Calculate the accounting cost

    accounting_cost = (people_scheduled[days[0]] - 125.0) / 400.0 * people_scheduled[days[0]] ** (0.5)    # 9

    accounting_cost = max(0, accounting_cost)    # 3

    yesterday_count = people_scheduled[days[0]]    # 3

    for day in days[1:]: # n

        today_count = people_scheduled[day]    # 2

        diff = abs(today_count - yesterday_count)    # 3

        accounting_cost += max(0, (people_scheduled[day] - 125.0) / 400.0 * people_scheduled[day] ** (0.5 + diff / 50.0))    # 12

        yesterday_count = today_count    # 1



    penalty += accounting_cost    # 2



    return penalty # 1

# Generic tree node class

class TreeNode(object):

    def __init__(self, val, choice_id):

        self.val = val

        self.choice_id = choice_id

        self.left = None

        self.right = None

        self.height = 1



# Binary Tree Class

# The below code was adapted from Bhavya Jain's work

# @ https://www.geeksforgeeks.org/binary-search-tree-set-1-search-and-insertion/

class BinaryTree(object):

    # A utility function to insert a new node with the given key

    def insert(self, root, key, choice_id):

        if not root:

            return TreeNode(key, choice_id)

        elif key < root.val:

            root.left = self.insert(root.left, key, choice_id)

        else:

            root.right = self.insert(root.right, key, choice_id)



        return root



    def inorder(self, root):

        for p in self._subtree_inorder(root):

            yield p



    def _subtree_inorder(self, root):

        if root.left is not None:  # if left child exists, traverse its subtree

            for other in self._subtree_inorder(root.left):

                yield other

        yield root  # visit p between its subtrees

        if root.right is not None:  # if right child exists, traverse its subtree

            for other in self._subtree_inorder(root.right):

                yield other
submission = pd.read_csv(fpath, index_col='family_id')



start = time.process_time()    # 1



table = submission['assigned_day'].tolist()    # n

new2 = table.copy()    # n

for fam_id, _ in enumerate(new2):    # n

    trial = new2.copy()    # n

    new_scores = list(range(0, len(choice_dict)))    # 5 (3 for range, 1 for len, 1 for assignment)

    myTree = BinaryTree()    # 1

    root = None    # 1

    root = myTree.insert(None, calc_penalty(trial), 10)    # 67n + 20



    for i in new_scores:    # n

        trial[fam_id] = choice_dict[f'choice_{i}'][fam_id]    # 4

        root = myTree.insert(root, calc_penalty(trial), i)    # 67n + 20



    min_id = list(myTree.inorder(root))[0].choice_id    # 2n + 3

    if min_id < 10:    # 1

        new2[fam_id] = choice_dict[f'choice_{min_id}'][fam_id]    # 4



submission['assigned_day'] = new2    # n

score = calc_penalty(new2)    # 66n + 20

print(f'Binary Tree Score: {score}')

print(f'Binary Tree Time: {time.process_time() - start}')

submission.to_csv(f'submission_{score}.csv')
# AVL tree class

# The below code was adapted from Ajitesh Pathak's work

# @ https://www.geeksforgeeks.org/avl-tree-set-1-insertion/

class AVLTree(object):

    def insert(self, root, key, choice_id):



        # Step 1 - Perform normal BST

        if not root:

            return TreeNode(key, choice_id)

        elif key < root.val:

            root.left = self.insert(root.left, key, choice_id)

        else:

            root.right = self.insert(root.right, key, choice_id)



        # Step 2 - Update the height of the ancestor node

        root.height = 1 + max(self.getHeight(root.left),

                              self.getHeight(root.right))



        # Step 3 - Get the balance factor

        balance = self.getBalance(root)



        # Step 4 - If the node is unbalanced, then try out the 4 cases

            # Case 1 - Left Left

        if balance > 1 and key < root.left.val:

            return self.rightRotate(root)



            # Case 2 - Right Right

        if balance < -1 and key > root.right.val:

            return self.leftRotate(root)



            # Case 3 - Left Right

        if balance > 1 and key > root.left.val:

            root.left = self.leftRotate(root.left)

            return self.rightRotate(root)



            # Case 4 - Right Left

        if balance < -1 and key < root.right.val:

            root.right = self.rightRotate(root.right)

            return self.leftRotate(root)



        return root



    def getHeight(self, root):

        if not root:

            return 0



        return root.height



    def getBalance(self, root):

        if not root:

            return 0



        return self.getHeight(root.left) - self.getHeight(root.right)



    def leftRotate(self, z):



        y = z.right

        T2 = y.left



        # Perform rotation

        y.left = z

        z.right = T2



        # Update heights

        z.height = 1 + max(self.getHeight(z.left),

                           self.getHeight(z.right))

        y.height = 1 + max(self.getHeight(y.left),

                           self.getHeight(y.right))



        # Return the new root

        return y



    def rightRotate(self, z):



        y = z.left

        T3 = y.right



        # Perform rotation

        y.right = z

        z.left = T3



        # Update heights

        z.height = 1 + max(self.getHeight(z.left),

                           self.getHeight(z.right))

        y.height = 1 + max(self.getHeight(y.left),

                           self.getHeight(y.right))



        # Return the new root

        return y



    def preOrder(self, root):



        if not root:

            return



        print("{0} ".format(root.val), end="")

        self.preOrder(root.left)

        self.preOrder(root.right)





    def inorder(self, root):

        for p in self._subtree_inorder(root):

            yield p



    def _subtree_inorder(self, root):

        if root.left is not None:  # if left child exists, traverse its subtree

            for other in self._subtree_inorder(root.left):

                yield other

        yield root  # visit p between its subtrees

        if root.right is not None:  # if right child exists, traverse its subtree

            for other in self._subtree_inorder(root.right):

                yield other

                

    # FOR TEST 3           

    def find_min(self, root):

        if root.left is None:

            return root

        return self.find_min(root.left)
submission = pd.read_csv(fpath, index_col='family_id')



start = time.process_time()    # 1



table = submission['assigned_day'].tolist()    # n

new2 = table.copy()    # n

for fam_id, _ in enumerate(new2):    # n

    trial = new2.copy()    # n

    new_scores = list(range(0, len(choice_dict)))    # 5 (3 for range, 1 for len, 1 for assignment)

    myTree = AVLTree()    # 1

    root = None    # 1

    root = myTree.insert(None, calc_penalty(trial),  10)    # 66n + log n + 20



    for i in new_scores:    # n

        trial[fam_id] = choice_dict[f'choice_{i}'][fam_id]    # 4

        root = myTree.insert(root, calc_penalty(trial), i)    # 66n + log n + 20



    min_id = list(myTree.inorder(root))[0].choice_id    # 2n + 3

    if min_id < 10:    # 1

        new2[fam_id] = choice_dict[f'choice_{min_id}'][fam_id]    # 4



submission['assigned_day'] = new2    # n

score = calc_penalty(new2)    # 66n + 20

print(f'AVL Tree Score: {score}')

print(f'AVL Tree Time: {time.process_time() - start}')

submission.to_csv(f'submission_{score}.csv')
submission = pd.read_csv(fpath, index_col='family_id')



start = time.process_time()    # 1



table = submission['assigned_day'].tolist()    # n

new2 = table.copy()    # n

for fam_id, _ in enumerate(new2):    # n

    trial = new2.copy()    # n

    new_scores = list(range(0, len(choice_dict)))    # 5 (3 for range, 1 for len, 1 for assignment)

    myTree = AVLTree()    # 1

    root = None    # 1

    root = myTree.insert(None, calc_penalty(trial),  10)    # 66n + log n + 20



    for i in new_scores:    # n

        trial[fam_id] = choice_dict[f'choice_{i}'][fam_id]    # 4

        root = myTree.insert(root, calc_penalty(trial), i)    # 66n + log n + 20



    min_id = myTree.find_min(root).choice_id    # log n + 2

    if min_id < 10:    # 1

        new2[fam_id] = choice_dict[f'choice_{min_id}'][fam_id]    # 4



submission['assigned_day'] = new2    # n

score = calc_penalty(new2)    # 48n + 3

print(f'AVL Tree Score: {score}')

print(f'AVL Tree Time: {time.process_time() - start}')

submission.to_csv(f'submission_{score}.csv')
# Import our modules that we are using

import matplotlib.pyplot as plt

from math import log2



# Create the vectors X and Y

x = np.array(range(100),dtype='int64')

y_1 = np.array(range(100),dtype='int64')

y_2 = np.array(range(100),dtype='int64')

y_3 = np.array(range(100),dtype='int64')



for j in x:

    if j != 0:

        y_1[j] = 67*(j**3) + 94*(j**2) + 105*j + 21

        y_2[j] = 66*(j**3) + (j**2)*log2(j) + 93*(j**2) + j*log2(j) + 105*j + 21

        y_3[j] = 66*(j**3) + (j**2)*log2(j) + 91*(j**2) + 2*j*log2(j) + 104*j + 21





# Create the plot

plt.plot(x,y_1,label='Binary Search Tree')

plt.plot(x,y_2,label='AVL Tree')

plt.plot(x,y_3,label='AVL Tree (find_min)')





# Add a title

plt.title('Runtime comparison')



# Add X and y Label

plt.xlabel('Inputs')

plt.ylabel('# of primitive operations')



# Add a Legend

plt.legend()
# Estimated units of time taken for each algorithm for an n of 6000

y_1 = 67*(6000**3) + 94*(6000**2) + 105*6000 + 21

print("Binary serach tree: "+str(y_1))

y_2 = 66*(6000**3) + (6000**2)*log2(6000) + 93*(6000**2) + 6000*log2(6000) + 105*6000 + 21

print("AVL tree: "+str(y_2))

y_3 = 66*(6000**3) + (6000**2)*log2(6000) + 91*(6000**2) + 2*6000*log2(6000) + 104*6000 + 21

print("AVL tree (find_min): "+str(y_3))