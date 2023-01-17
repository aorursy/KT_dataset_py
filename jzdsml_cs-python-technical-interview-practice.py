# rotate list

# no time/space requirements

# return "rotated" version of input list



def rotate(my_list, num_rotations):

  index = num_rotations%(len(my_list))

  return my_list[-index:]+my_list[:(len(my_list)-index)]













#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 1), ['f', 'a', 'b', 'c', 'd', 'e'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 1) == ['f', 'a', 'b', 'c', 'd', 'e']))



print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 2), ['e', 'f', 'a', 'b', 'c', 'd'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 2) == ['e', 'f', 'a', 'b', 'c', 'd']))



print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 3), ['d', 'e', 'f', 'a', 'b', 'c'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 3) == ['d', 'e', 'f', 'a', 'b', 'c']))



print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 4), ['c', 'd', 'e', 'f', 'a', 'b'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 4) == ['c', 'd', 'e', 'f', 'a', 'b']))
# rotate list

# Constant space requirement

# return input list "rotated"



def rotate(my_list, num_rotations):

  num_rotations %= len(my_list)

  

  # define a reverse helper function

  def rev(lst, low, high):

    while low < high:

      lst[low], lst[high] = lst[high], lst[low]

      high -= 1

      low += 1

      

  # reverse first num_rotation elements

  rev(my_list, 0, num_rotations - 1)

  # reverse later elements

  rev(my_list, num_rotations, len(my_list) - 1)

  # reverse the whole new my_list

  rev(my_list, 0, len(my_list) - 1)

    

  return my_list



#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 1), ['b', 'c', 'd', 'e', 'f', 'a'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 1) == ['b', 'c', 'd', 'e', 'f', 'a']))



print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 2), ['c', 'd', 'e', 'f', 'a', 'b'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 2) == ['c', 'd', 'e', 'f', 'a', 'b']))



print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 3), ['d', 'e', 'f', 'a', 'b', 'c'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 3) == ['d', 'e', 'f', 'a', 'b', 'c']))



print("{0}\n should equal \n{1}\n {2}\n".format(rotate(['a', 'b', 'c', 'd', 'e', 'f'], 4), ['e', 'f', 'a', 'b', 'c', 'd'], rotate(['a', 'b', 'c', 'd', 'e', 'f'], 4) == ['e', 'f', 'a', 'b', 'c', 'd']))
# find rotation point 

# No time/space requirements

# return index of "rotation point" element



def rotation_point(rotated_list):

  for i in range(len(rotated_list)-1):

    if rotated_list[i] > rotated_list[i+1]:

      return i+1

  return 0







#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(rotation_point(['a', 'b', 'c', 'd', 'e', 'f']), 0, rotation_point(['a', 'b', 'c', 'd', 'e', 'f']) == 0))



print("{0}\n should equal \n{1}\n {2}\n".format(rotation_point(['c', 'd', 'e', 'f', 'a']), 4, rotation_point(['c', 'd', 'e', 'f', 'a']) == 4))



print("{0}\n should equal \n{1}\n {2}\n".format(rotation_point([13, 8, 9, 10, 11]), 1, rotation_point([13, 8, 9, 10, 11]) == 1))
# find rotation point 

# O(logN) time requirement

# return index of "rotation point" element



def rotation_point(rotated_list):

  left = 0

  right = len(rotated_list) - 1

  mid = (left + right)//2

  if rotated_list[left] < rotated_list[right]:

    return 0

  while left <= right:

    mid = (left + right)//2

    if rotated_list[mid-1] > rotated_list[mid]:

      return mid

    # edge case: final two elements, a>b, mid always point to a

    # will go into a infinite loop, so need to check mid+1 too!

    if rotated_list[mid] > rotated_list[mid+1]:

      return mid+1

    if rotated_list[left] > rotated_list[mid]:

      right = mid

    if rotated_list[right] < rotated_list[mid]:

      left = mid

  













#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(rotation_point(['a', 'b', 'c', 'd', 'e', 'f']), 0, rotation_point(['a', 'b', 'c', 'd', 'e', 'f']) == 0))



print("{0}\n should equal \n{1}\n {2}\n".format(rotation_point(['c', 'd', 'e', 'f', 'a']), 4, rotation_point(['c', 'd', 'e', 'f', 'a']) == 4))



print("{0}\n should equal \n{1}\n {2}\n".format(rotation_point([13, 8, 9, 10, 11]), 1, rotation_point([13, 8, 9, 10, 11]) == 1))
# remove duplicates 

# no time/space requirements

# return a list with duplicates removed



def remove_duplicates(dupe_list):

  new_list = []

  for item in dupe_list:

    if item not in new_list:

      new_list.append(item)

  return new_list







#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(remove_duplicates(['a', 'a', 'x', 'x', 'x', 'g', 't', 't']), ['a', 'x', 'g', 't'], remove_duplicates(['a', 'a', 'x', 'x', 'x', 'g', 't', 't']) == ['a', 'x', 'g', 't']))



print("{0}\n should equal \n{1}\n {2}\n".format(remove_duplicates(['c', 'c', 'd', 'd', 'e', 'e', 'f', 'a', 'a']), ['c', 'd', 'e', 'f', 'a'], remove_duplicates(['c', 'c', 'd', 'd', 'e', 'e', 'f', 'a', 'a']) == ['c', 'd', 'e', 'f', 'a']))



print("{0}\n should equal \n{1}\n {2}\n".format(remove_duplicates([13, 13, 13, 13, 13, 42]), [13, 42], remove_duplicates([13, 13, 13, 13, 13, 42]) == [13, 42]))
# remove duplicates 

# constant space

# return index of last unique element



def move_duplicates(dupe_list):

  # two pointers: unique, check

  # first time seeing an value at "check": swap value at "unique" with value at "check", unique +=1

  # seeing a duplicate value at "check": unique stays the same, check+=1 until firs seeing a new value. swap value at current "check" location with value at "unique", check+=1

  unique = 0

  check = 1

  while check <= len(dupe_list) -1:

    if dupe_list[unique] != dupe_list[check]:

      unique += 1

      dupe_list[unique], dupe_list[check] = dupe_list[check], dupe_list[unique]

      check += 1

    else:   

      check += 1

  return unique



#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(move_duplicates(['a', 'a', 'g', 't', 't', 'x', 'x', 'x']), 3, move_duplicates(['a', 'a', 'g', 't', 't', 'x', 'x', 'x']) == 3))



print("{0}\n should equal \n{1}\n {2}\n".format(move_duplicates(['a', 'a', 'c', 'c', 'd', 'd', 'e', 'e', 'f']), 4, move_duplicates(['a', 'a', 'c', 'c', 'd', 'd', 'e', 'e', 'f']) == 4))



print("{0}\n should equal \n{1}\n {2}\n".format(move_duplicates([13, 13, 13, 13, 13, 42]), 1, move_duplicates([13, 13, 13, 13, 13, 42]) == 1))
# max sub sum

# no time/space requirements

# return maximum contiguous sum in list



def maximum_sub_sum(my_list):

  if len(my_list)==0:

    return 0

  if len(my_list)==1:

    return my_list[0]

  n = len(my_list)

  lst =[]

  for size in range(1, n+1):

    for i in range(0, n-size+1):

      sum = 0

      for j in range(i, i+size):

        sum += my_list[j]

      lst.append(sum)

  return max(lst)











#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(maximum_sub_sum([1, 2, 3, 4, 5]), 15, maximum_sub_sum([1, 2, 3, 4, 5]) == 15))



print("{0}\n should equal \n{1}\n {2}\n".format(maximum_sub_sum([-1, -1, -2, -4, -5, -9, -12, -13]), -1, maximum_sub_sum([-1, -1, -2, -4, -5, -9, -12, -13]) == -1))



print("{0}\n should equal \n{1}\n {2}\n".format(maximum_sub_sum([1, -7, 2, 15, -11, 2]), 17, maximum_sub_sum([1, -7, 2, 15, -11, 2]) == 17))
# max sub sum

# linear time, constant space requirements

# return maximum contiguous sum in list



def maximum_sub_sum(my_list):

  i = 0

  sum = my_list[0]

  max_sum_seen = my_list[0]

  for i in range(1,len(my_list)):

    if sum + my_list[i] > 0:

      sum += my_list[i]

      max_sum_seen = max(max_sum_seen, sum)

    else:

      sum = 0

  return max_sum_seen



#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal \n{1}\n {2}\n".format(maximum_sub_sum([1, 2, 3, 4, 5]), 15, maximum_sub_sum([1, 2, 3, 4, 5]) == 15))



print("{0}\n should equal \n{1}\n {2}\n".format(maximum_sub_sum([-1, -1, -2, -4, -5, -9, -12, -13]), -1, maximum_sub_sum([-1, -1, -2, -4, -5, -9, -12, -13]) == -1))



print("{0}\n should equal \n{1}\n {2}\n".format(maximum_sub_sum([1, -7, 2, 15, -11, 2]), 17, maximum_sub_sum([1, -7, 2, 15, -11, 2]) == 17))
# pair sum

# no time/space requirements

# return list of indices that sum to target







def pair_sum(nums, target):

  for i in range(len(nums)-1):

    for j in range(i,len(nums)):

      if nums[i] + nums[j] == target:

        return [i,j]

  return None





#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal any of \n{1}\n {2}\n".format(pair_sum([1, 2, 3, 4, 5], 6), [[1, 3], [0, 4]], pair_sum([1, 2, 3, 4, 5], 6) in [[1, 3], [0,4]]))



print("{0}\n should equal any of \n{1}\n {2}\n".format(pair_sum([-1, -1, -2, -4, -5, -9, -12, -13], -21), [[5, 6]], pair_sum([-1, -1, -2, -4, -5, -9, -12, -13], -21) in [[5, 6]]))



print("{0}\n should equal \n{1}\n {2}\n".format(pair_sum([1, -7, 2, 15, -11, 2], 42), None, pair_sum([1, -7, 2, 15, -11, 2], 42) == None))



print("{0}\n should equal any of \n{1}\n {2}\n".format(pair_sum([0, -7, 2, 15, -11, 2], 2), [[0, 2], [0,5]], pair_sum([0, -7, 2, 15, -11, 2], 2) in [[0, 2], [0, 5]]))
# pair sum

# linear time, linear space requirement

# return list of indices that sum to target



def pair_sum(nums, target):

  comlements={}

  idxs={}

  for i in range(len(nums)):

    idxs[nums[i]]=i

    comlements[i] = target - nums[i]

  for i in range(len(nums)):

    if idxs.get(comlements[i],None):

      return [i,idxs[comlements[i]]]

  return None





#### TESTS SHOULD ALL BE TRUE ####

print("{0}\n should equal any of \n{1}\n {2}\n".format(pair_sum([1, 2, 3, 4, 5], 6), [[1, 3], [0,4]], pair_sum([1, 2, 3, 4, 5], 6) in [[1, 3], [0,4]]))



print("{0}\n should equal any of \n{1}\n {2}\n".format(pair_sum([-1, -1, -2, -4, -5, -9, -12, -13], -21), [[5, 6]], pair_sum([-1, -1, -2, -4, -5, -9, -12, -13], -21) in [[5, 6]]))



print("{0}\n should equal \n{1}\n {2}\n".format(pair_sum([1, -7, 2, 15, -11, 2], 42), None, pair_sum([1, -7, 2, 15, -11, 2], 42) == None))



print("{0}\n should equal any of \n{1}\n {2}\n".format(pair_sum([0, -7, 2, 15, -11, 2], 2), [[0, 2], [0,5]], pair_sum([0, -7, 2, 15, -11, 2], 2) in [[0, 2], [0, 5]]))
from timeit import timeit



## Rotate

def rotate(lst, degree):

  rotation = degree % len(lst)

  return lst[-rotation:] + lst[:-rotation]



def rev(lst, low, high):

  while low < high:

    lst[low], lst[high] = lst[high], lst[low]

    high -= 1

    low += 1

  return lst



def rotate_optimal(my_list, num_rotations):

  rev(my_list, 0, num_rotations - 1)

  rev(my_list, num_rotations, len(my_list) - 1)

  rev(my_list, 0, len(my_list) - 1)

  return my_list



print("NAIVE ROTATE: ")

print(timeit('rotate([i for i in range(1000)], 100)', number = 10, setup="from __main__ import rotate"))

print("SPACE OPTIMAL ROTATE: ")

print(timeit('rotate_optimal([i for i in range(1000)], 100)', number = 10, setup="from __main__ import rotate_optimal"))



## Rotate Point

def rotation_point(rotated_list):

  rotation_idx = 0

  for i in range(len(rotated_list)):

    if rotated_list[i] < rotated_list[rotation_idx]:

      rotation_idx = i

  return rotation_idx



def rotation_point_optimal(rotated_list):

  low = 0

  high = len(rotated_list) - 1

  while low <= high:

    mid = (low + high) // 2

    mid_next = (mid + 1) % len(rotated_list)

    mid_previous = (mid - 1) % len(rotated_list)

    

    if (rotated_list[mid] < rotated_list[mid_previous]) and (rotated_list[mid] < rotated_list[mid_next]):

      return mid

    elif rotated_list[mid] < rotated_list[high]:

      high = mid - 1

    else:

      low = mid + 1



print("\nNAIVE ROTATE POINT: ")

print(timeit('rotation_point([i for i in range(1000)])', number = 10, setup="from __main__ import rotation_point"))

print("TIME OPTIMAL ROTATE POINT: ")

print(timeit('rotation_point_optimal([i for i in range(1000)])', number = 10, setup="from __main__ import rotation_point_optimal"))



## Duplicates

def remove_duplicates(dupe_list):

  unique_values = []

  for el in dupe_list:

    if el not in unique_values:

      unique_values.append(el)

  return unique_values





def move_duplicates_optimal(dupe_list):

  unique_idx = 0

  for i in range(len(dupe_list) - 1):

    if dupe_list[i] != dupe_list[i + 1]:

      dupe_list[i], dupe_list[unique_idx] = dupe_list[unique_idx], dupe_list[i]

      unique_idx += 1

  dupe_list[unique_idx], dupe_list[len(dupe_list) - 1] = dupe_list[len(dupe_list) - 1], dupe_list[unique_idx]

  return unique_idx



print("\nNAIVE REMOVE DUPLICATES: ")

print(timeit('remove_duplicates([i + i for i in range(1000)])', number = 10, setup="from __main__ import remove_duplicates"))

print("SPACE OPTIMAL REMOVE DUPLICATES: ")

print(timeit('move_duplicates_optimal([i + i for i in range(1000)])', number = 10, setup="from __main__ import move_duplicates_optimal"))



## Max Sub Sum

def maximum_sub_sum(my_list):

  max_sum = my_list[0]

  for i in range(len(my_list)):

    for j in range(i, len(my_list)):

      sub_sum = sum(my_list[i:j + 1])

      if sub_sum > max_sum:

        max_sum = sub_sum

  return max_sum



def maximum_sub_sum_optimal(my_list):

  if max(my_list) < 0:

    return max(my_list)

  

  max_sum = 0

  max_sum_tracker = 0

  for i in range(len(my_list)):

    max_sum_tracker += my_list[i]

    if max_sum_tracker < 0:

      max_sum_tracker = 0

    if max_sum_tracker > max_sum:

      max_sum = max_sum_tracker

    

  return max_sum



print("\nNAIVE MAX SUB-SUM: ")

print(timeit('maximum_sub_sum([i for i in range(200)])', number = 10, setup="from __main__ import maximum_sub_sum"))

print("SPACE OPTIMAL REMOVE DUPLICATES: ")

print(timeit('maximum_sub_sum_optimal([i for i in range(200)])', number = 10, setup="from __main__ import maximum_sub_sum_optimal"))



## Pair Sum

def pair_sum(my_list, target):

  for i in range(len(my_list)):

    for j in range(i, len(my_list)):

      if my_list[i] + my_list[j] == target:

        return [i, j]

  return None



def pair_sum_optimal(my_list, target):

  comp = {}

  index = {}

  for i in range(len(my_list)):

    half = comp.get(my_list[i], None)

    if half is not None:

      return [index[half], i]

    comp[target - my_list[i]] = my_list[i]

    index[my_list[i]] = i

    

    

print("\nNAIVE PAIR SUM: ")

print(timeit('pair_sum([i for i in range(1000)], -1)', number = 10, setup="from __main__ import pair_sum"))

print("TIME OPTIMAL PAIR SUM: ")

print(timeit('pair_sum_optimal([i for i in range(1000)], -1)', number = 10, setup="from __main__ import pair_sum_optimal"))
def rain_water(lst):

  left_wall = [lst[0]]

  right_wall = [lst[-1]]

  for i in range(1,len(lst)):

    ## naive solution: copy a sub-list and find the max at each iteration 

    ## results in O(N^2) for the whole function

    #left_wall.append(max(lst[:i+1]))

    #right_wall.append(max(lst[-i-1:]))

    

    # optimize to O(N)

    left_wall.append(max(left_wall[-1], lst[i]))

    right_wall.append(max(right_wall[-1], lst[-i-1]))

  right_wall.reverse()

  print(left_wall, right_wall)

  # water only rises to the smaller of left_wall and right_wall

  # get full capacity at each index (including submerged bars)

  capacity = [min(left, right) for left,right in zip(left_wall, right_wall)] 

  print(capacity)

  # subtract submerged bars

  water = [cap - bar for cap, bar in zip(capacity, lst)]

  print(water, sum(water))

  return sum(water)



# test

normal_case = [4, 2, 1, 3, 0, 1, 2]

edge_case = [0, 2, 0]

print(rain_water(normal_case), rain_water(edge_case))
class Node:

  def __init__(self, val, next = None):

    self.val = val

    self.next = next
class LinkedList:

  def __init__(self, head_node = None):

    self.head = head_node



  def add(self, val):

    new_head = Node(val)

    new_head.next = self.head

    self.head = new_head

    

  def traverse(self):

    head = self.head

    print("Starting traversal from head")

    while head:

      print("visiting node: {0}".format(head.val))

      head = head.next

    print("Traversal complete")

    

  def size(self):

    node_count = 0

    current_node = self.head

    while current_node:

      node_count += 1

      current_node = current_node.next

    return node_count

  

  def __repr__(self):

    text = ''

    head = self.head

    while head:

      text += str(head.val) + ' -> '

      head = head.next

    return text



  def n_from_last(self, n):

    node_count = 0

    current_node = self.head

    while node_count < (self.size() - 1 - n):

      node_count += 1

      current_node = current_node.next

    return current_node



  def insert(self, node_value, location):

    new_node = Node(node_value)

    if location == 0:

      new_node.next = self.head

      self.head = new_node

      return 

    node_count = 0

    current_node = self.head

    while node_count < (location - 1):

      current_node = current_node.next

      node_count += 1

    new_node.next = current_node.next

    current_node.next = new_node

    return 

        

  def remove_duplicates(self):

    current_node = self.head

    

    while current_node:

      while current_node.next and current_node.next.val == current_node.val:

        current_node.next = current_node.next.next

      current_node = current_node.next

    return self
test_1 = LinkedList()

test_1.add('d')

test_1.add('c')

test_1.add('b')

test_1.add('a')

test_1.insert('x', 2)



test_result = test_1.head

for i in range(2):

  test_result = test_result.next

print("Result node's value should be 'x': {0}".format(test_result.val))



test_1.insert('t', 0)

test_result = test_1.head

print("Result node's value should be 't': {0}".format(test_result.val))
test_list = LinkedList()

test_list.add('e')

test_list.add('d')

test_list.add('c')

test_list.add('b')

test_list.add('a')



test_result = test_list.n_from_last(0)



print("Result node's value should be 'e': {0}".format(test_result.val))



test_result = test_list.n_from_last(3)

print("Result node's value should be 'b': {0}".format(test_result.val))
test_linked_list = LinkedList()



test_linked_list.add('d')

test_linked_list.add('c')

test_linked_list.add('c')

test_linked_list.add('c')

test_linked_list.add('b')

test_linked_list.add('a')

test_linked_list.add('a')



test_linked_list.remove_duplicates()



duplicates = {}

duplicate_found = False

current_node = test_linked_list.head

while current_node:

  if duplicates.get(current_node.val, False):

    duplicate_found = True

    break

  else:

    duplicates[current_node.val] = True

    current_node = current_node.next



if duplicate_found:

  print("Not all duplicates removed, try again!")

else:

  print("Duplicates removed, nice work!")
linked_list_a = LinkedList()

linked_list_b = LinkedList()

linked_list_a.add('z')

linked_list_a.add('x')

linked_list_a.add('c')

linked_list_a.add('a')

linked_list_b.add('u')

linked_list_b.add('g')

linked_list_b.add('b')





def merge(linked_list_a, linked_list_b):

  if linked_list_a.head.val >= linked_list_b.head.val:

    big = linked_list_a

    small = linked_list_b

  else:

    big = linked_list_b

    small = linked_list_a

  current = small.head

  node1 = small.head.next

  node2 = big.head

  while True:

    if node1.val <= node2.val:

      current.next = node1

      current = current.next

      if node1.next:

        node1 = node1.next

      else:

        break

    else:

      current.next = node2

      current = current.next

      if node2.next:

        node2 = node2.next

      else:

        break

  if not node1.next and node2.next:

    node1.next = node2

  if node1.next and not node2.next:

    node2.next = node1

  return small

      

    

 



merged_linked_list = merge(linked_list_a, linked_list_b)



print("Merged list should contain all nodes in sorted order: a -> b -> c -> g -> u -> x -> z")

print("Your function returned: {0}".format(merged_linked_list))
def set_up_test_case():

  head_node_1 = Node('x')

  head_node_2 = Node('d')

  current_node_1 = head_node_1

  current_node_2 = head_node_2

  

  for letter in ['a', 'b']:

    current_node_1.next = Node(letter)

    current_node_1 = current_node_1.next

    

  current_node_2.next = Node('f')

  current_node_2 = current_node_2.next

  

  for shared_node in [Node('q'), Node('e')]:

  	current_node_1.next = shared_node

  	current_node_2.next = shared_node

  	current_node_1 = current_node_1.next

  	current_node_2 = current_node_2.next

    

  linked_list_1 = LinkedList(head_node_1)

  linked_list_2 = LinkedList(head_node_2)

  return linked_list_1, linked_list_2



linked_list_1, linked_list_2 = set_up_test_case()

print(linked_list_1)

print(linked_list_2)



def merge_point(linked_list_a, linked_list_b):

  if linked_list_a.size() >= linked_list_b.size():

    long = linked_list_a

    short = linked_list_b

  else:

    long = linked_list_b

    short = linked_list_a

  diff = long.size() - short.size()

  lgnode = long.head

  stnode = short.head

  while diff:

    lgnode = lgnode.next

    diff -= 1

  while lgnode.next and stnode.next:

    if lgnode == stnode:

      return lgnode

    lgnode = lgnode.next

    stnode = stnode.next

  return None



test_result = merge_point(linked_list_1, linked_list_2)



print("Function should return merge point node holding 'q': {0}".format(test_result.val))
test_linked_list = LinkedList()



test_linked_list.add('d')

test_linked_list.add('c')

test_linked_list.add('b')

test_linked_list.add('a')





def reverse(linked_list):

  prev = None

  current = linked_list.head

  nxt = current.next

  current.next = None

  while nxt:

    nxtnxt = nxt.next

    nxt.next = current

    current = nxt

    nxt = nxtnxt

  return LinkedList(current)





print("Pre-reverse: {0}".format(test_linked_list))



reversed_linked_list = reverse(test_linked_list)



print("Post-reverse: {0}".format(reversed_linked_list))
def build_cycle_linked_list():

  start_node = Node('a')

  head = start_node

  b = Node('b')

  c = Node('c')

  d = Node('d')

  for letter_node in [b, c, d]:

    start_node.next = letter_node

    start_node = start_node.next

  start_node.next = b

  return LinkedList(head)

  

def build_linked_list_no_cycle():

  start_node = Node('a')

  head = start_node

  

  for letter in ['b', 'c', 'd', 'b']:

    start_node.next = Node(letter)

    start_node = start_node.next

  return LinkedList(head)



cycle_linked_list = build_cycle_linked_list()

no_cycle_linked_list = build_linked_list_no_cycle()



def has_cycle_baseline(linked_list):

  lst = []

  current = linked_list.head

  while current:

    if current not in lst:

      lst.append(current)

      current = current.next

    else:

      return True

  return False





def has_cycle(linked_list):

  slow = linked_list.head

  fast = linked_list.head

  while slow and fast:

    slow = slow.next

    fast = fast.next

    if fast:

      fast = fast.next

    else:

      return False

    if slow == fast:

      return True

  return False



cycle_result = has_cycle(cycle_linked_list)

no_cycle_result = has_cycle(no_cycle_linked_list)



print("Should return True when a cycle exists: {0}".format(cycle_result))



print("Should return False when a cycle does not exist: {0}".format(no_cycle_result))





  
def build_test_case():

  linked_list_a = LinkedList()

  linked_list_a.add(9)

  linked_list_a.add(1)

  linked_list_a.add(3)

  linked_list_a.add(2)

  linked_list_b = LinkedList()

  linked_list_b.add(8)

  linked_list_b.add(6)

  linked_list_b.add(8)

  return linked_list_a, linked_list_b



linked_list_a, linked_list_b = build_test_case()





def add_two(linked_list_a, linked_list_b):

  if linked_list_a.size() >=linked_list_b.size():

    long = linked_list_a

    short = linked_list_b

  else:

    short = linked_list_a

    long = linked_list_b

    

  lgnode = long.head

  stnode = short.head

  carry = 0

  while lgnode and stnode:

    lgnode.val += (stnode.val + carry)

    carry = lgnode.val//10

    lgnode.val %= 10

    lgnode = lgnode.next

    stnode = stnode.next

  while lgnode:

    lgnode.val += carry

    carry = lgnode.val//10

    lgnode.val %= 10

    if not lgnode.next and carry > 0:

      lgnode.next = Node(carry)

      break

    lgnode = lgnode.next

  # need to consider edge case when the highest digit in long is 9 and there is carry=1 from earlier, otherwise the highest 1 will be truncated!!!

  return long



def add_two_another_option(linked_list_a, linked_list_b):

  

  result = LinkedList()

  carry = 0

  

  a_node = linked_list_a.head

  b_node = linked_list_b.head

  

  while a_node or b_node:

    

    if b_node:

      b_val = b_node.val

      b_node = b_node.next

    else:

      b_val = 0

      

    if a_node:

      a_val = a_node.val

      a_node = a_node.next

    else:

      a_val = 0

      

    to_sum = a_val + b_val + carry

    

    if to_sum > 9:

      carry = 1

      to_sum %= 10

    else:

      carry = 0



    

    if not result.head:

      result.head = Node(to_sum)

      tmp = result.head

    else:

      tmp.next = Node(to_sum)

      tmp = tmp.next

      

  if carry:

    tmp.next = Node(carry)



  return result

    





print("Adding linked list:\n{0}\nto linked list\n{1}\n".format(linked_list_a, linked_list_b))

result = add_two(linked_list_a, linked_list_b)

print("Result should be: 0 -> 0 -> 0 -> 0 -> 1\nFunction returned:\n{0}".format(result))


num_in_fibonacci = 9

arguments_count = {}

def fibonacci(num):

  count = arguments_count.get(num,0)

  count += 1

  arguments_count[num] = count

  if num < 0:

    print("Not a valid number")

  if num <= 1:

    return num

  else:

    return fibonacci(num - 1) + fibonacci(num - 2)

  



print("Number {0} in the fibonacci sequence is {1}.".format(num_in_fibonacci, fibonacci(num_in_fibonacci)))

      

for num, count in arguments_count.items():

  print("Argument {0} seen {1} time(s)!".format(num, count))



print("Fibonacci function called {0} total times!".format(sum(arguments_count.values())))
num_in_fibonacci = 9

function_calls = []

memo = {}

def fibonacci(num, memo):

  function_calls.append(1)

  

  if num < 0:

    print("Not a valid number")

  if num <= 1:

    return num

  elif memo.get(num):

    return memo.get(num)

  else:

    memo[num] = fibonacci(num - 1, memo) + fibonacci(num - 2, memo)

    return memo[num]

  

  

fibonacci_result = fibonacci(num_in_fibonacci, {})

print("Number {0} in the fibonacci sequence is {1}.".format(num_in_fibonacci, fibonacci_result))



print("Fibonacci function called {0} total times!".format(len(function_calls)))
# write powerset functin that returns the powerset of the input list

from itertools import chain, combinations

def powerset(iterable):

  s = list(iterable)

  return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))



fruits = ['apple', 'orange', 'grape']

fruit_combinations = powerset(fruits)

print(fruit_combinations)
# write a Loot class

class Loot:

  def __init__(self, name, weight, value):

    self.name = name

    self.weight = weight

    self.value = value



  def __repr__(self):

    return "{0}:\n  weighs {1},\n  valued at {2}.".format(self.name, self.weight, self.value)



computer = Loot("Computer", 2, 12)

print(computer)
def knapsack(loot, weight_limit):

  best_value = None

  all_combo = powerset(loot)

  for combo in all_combo:

    combo_weight = sum([item.weight for item in combo])

    combo_value = sum([item.value for item in combo])

    if combo_weight <= weight_limit:

      if not best_value or best_value < combo_value:

        best_value = combo_value

  if not best_value:

    print("knapsack couldn't fit any items")

  return best_value

    

available_loot = [Loot("Clock", 3, 8), Loot("Vase", 4, 12), Loot("Diamond", 1, 7)]

weight_capacity = 4

best_combo = knapsack(available_loot, weight_capacity)

print("The ideal loot given capacity {0} is\n{1}".format(weight_capacity, best_combo))
def knapsack(loot, weight_limit):

  grid = [[0 for col in range(weight_limit + 1)] for row in range(len(loot) + 1)]

  for row, item in enumerate(loot):

    row = row + 1

    for col in range(weight_limit + 1):

      weight_capacity = col

      if item.weight <= weight_capacity:

        item_value = item.value

        item_weight = item.weight

        previous_best_less_item_weight = grid[row - 1][weight_capacity - item_weight]

        capacity_value_with_item = item_value + previous_best_less_item_weight

        capacity_value_without_item = grid[row - 1][col]

        grid[row][col] = max(capacity_value_with_item, capacity_value_without_item)

      else:

        grid[row][col] = grid[row - 1][col]

  return grid[len(loot)][weight_limit]

    

available_loot = [Loot("Clock", 3, 8), Loot("Vase", 4, 12), Loot("Diamond", 1, 7)]

weight_capacity = 4

best_combo = knapsack(available_loot, weight_capacity)

print("The ideal loot given capacity {0} is\n{1}".format(weight_capacity, best_combo))
dna_1 = "ACCGTT"

dna_2 = "CCAGCA"



def longest_common_subsequence(string_1, string_2):

  print("Finding longest common subsequence of {0} and {1}".format(string_1, string_2))

  grid = [[0 for col in range(len(string_1)+1)] 

          for row in range(len(string_2)+1)]

  for row in range(1, len(string_2) + 1):

    print("Comparing: {0}".format(string_2[row - 1]))

    for col in range(1, len(string_1) + 1):

      print("Against: {0}".format(string_1[col - 1]))

      if string_1[col - 1] == string_2[row - 1]:

        grid[row][col] = grid[row - 1][col - 1] + 1

      else:

        grid[row][col] = max(grid[row - 1][col], grid[row][col - 1])

  print("The grid is:")

  for row_line in grid:

    print(row_line)

    

  # fining the subsequence

  result = []

  row = len(string_2)

  col = len(string_1)

  while row > 0 and col > 0:

    if string_1[col - 1] == string_2[row - 1]:

      result.append(string_1[col - 1])

      row -= 1

      col -= 1

    else:

      if grid[row - 1][col] > grid[row][col - 1]:

        row -= 1

      else:

        col -= 1

  print("The longest common subsequence is: {}".format(result[::-1]))

        

        

longest_common_subsequence(dna_1, dna_2)

    

  
