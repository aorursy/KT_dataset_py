class Node:

  def __init__(self, value, link_node=None):

    self.value = value

    self.link_node = link_node

    

  def set_link_node(self, link_node):

    self.link_node = link_node

    

  def get_link_node(self):

    return self.link_node

  

  def get_value(self):

    return self.value



# Add your code below:

yacko = Node("likes to yak")

wacko = Node("has a penchant for hoarding snacks")

dot = Node("enjoys spending time in movie lots")



yacko.set_link_node(dot)

dot.set_link_node(wacko)



dots_data = yacko.get_link_node().get_value()

wackos_data=dot.get_link_node().get_value()



print(dots_data)

print(wackos_data)
class Node:

  def __init__(self, value, next_node=None):

    self.value = value

    self.next_node = next_node

    

  def get_value(self):

    return self.value

  

  def get_next_node(self):

    return self.next_node

  

  def set_next_node(self, next_node):

    self.next_node = next_node



class LinkedList:

  def __init__(self, value=None):

    self.head_node = Node(value)

  

  def get_head_node(self):

    return self.head_node

  

  def insert_beginning(self, new_value):

    new_node = Node(new_value)

    new_node.set_next_node(self.head_node)

    self.head_node = new_node

    

  def stringify_list(self):

    string_list = ""

    current_node = self.get_head_node()

    while current_node:

      if current_node.get_value() != None:

        string_list += str(current_node.get_value()) + "\n"

      current_node = current_node.get_next_node()

    return string_list

  

  def remove_node(self, value_to_remove):

    current_node = self.get_head_node()

    if current_node.get_value() == value_to_remove:

      self.head_node = current_node.get_next_node()

    else:

      while current_node:

        next_node = current_node.get_next_node()

        if next_node.get_value() == value_to_remove:

          current_node.set_next_node(next_node.get_next_node())

          current_node = None

        else:

          current_node = next_node
class Node:

  def __init__(self, value, next_node=None):

    self.value = value

    self.next_node = next_node

    

  def set_next_node(self, next_node):

    self.next_node = next_node

    

  def get_next_node(self):

    return self.next_node

  

  def get_value(self):

    return self.value



class Stack:

  def __init__(self, limit=1000):

    self.top_item = None

    self.size = 0

    self.limit = limit

  

  def push(self, value):

    if self.has_space():

      item = Node(value)

      item.set_next_node(self.top_item)

      self.top_item = item

      self.size += 1

      print("Adding {} to the pizza stack!".format(value))

    else:

      print("No room for {}!".format(value))



  def pop(self):

    if not self.is_empty():

      item_to_remove = self.top_item

      self.top_item = item_to_remove.get_next_node()

      self.size -= 1

      print("Delivering " + item_to_remove.get_value())

      return item_to_remove.get_value()

    print("All out of pizza.")



  def peek(self):

    if not self.is_empty():

      return self.top_item.get_value()

    print("Nothing to see here!")



  def has_space(self):

    return self.limit > self.size



  def is_empty(self):

    return self.size == 0

  

# Defining an empty pizza stack

pizza_stack = Stack(6)



# Adding pizzas as they are ready until we have 

pizza_stack.push("pizza #1")

pizza_stack.push("pizza #2")

pizza_stack.push("pizza #3")

pizza_stack.push("pizza #4")

pizza_stack.push("pizza #5")

pizza_stack.push("pizza #6")





pizza_stack.push("pizza #7")



# Delivering pizzas from the top of the stack down

print("The first pizza to deliver is " + pizza_stack.peek())

pizza_stack.pop()

pizza_stack.pop()

pizza_stack.pop()

pizza_stack.pop()

pizza_stack.pop()

pizza_stack.pop()



pizza_stack.pop()
class Node:

  def __init__(self, value, next_node=None):

    self.value = value

    self.next_node = next_node

    

  def set_next_node(self, next_node):

    self.next_node = next_node

    

  def get_next_node(self):

    return self.next_node

  

  def get_value(self):

    return self.value





class Queue:

  def __init__(self, max_size=None):

    self.head = None

    self.tail = None

    self.max_size = max_size

    self.size = 0

    

  def enqueue(self, value):

    if self.has_space():

      item_to_add = Node(value)

      print("Adding " + str(item_to_add.get_value()) + " to the queue!")

      if self.is_empty():

        self.head = item_to_add

        self.tail = item_to_add

      else:

        self.tail.set_next_node(item_to_add)

        self.tail = item_to_add

      self.size += 1

    else:

      print("Sorry, no more room!")

         

  def dequeue(self):

    if self.get_size() > 0:

      item_to_remove = self.head

      print(str(item_to_remove.get_value()) + " is served!")

      if self.get_size() == 1:

        self.head = None

        self.tail = None

      else:

        self.head = self.head.get_next_node()

      self.size -= 1

      return item_to_remove.get_value()

    else:

      print("The queue is totally empty!")

  

  def peek(self):

    if self.size > 0:

      return self.head.get_value()

    else:

      print("No orders waiting!")

  

  def get_size(self):

    return self.size

  

  def has_space(self):

    if self.max_size == None:

      return True

    else:

      return self.max_size > self.get_size()

    

  def is_empty(self):

    return self.size == 0



print("Creating a deli line with up to 10 orders...\n------------")

deli_line = Queue(10)

print("Adding orders to our deli line...\n------------")

deli_line.enqueue("egg and cheese on a roll")

deli_line.enqueue("bacon, egg, and cheese on a roll")

deli_line.enqueue("toasted sesame bagel with butter and jelly")

deli_line.enqueue("toasted roll with butter")

deli_line.enqueue("bacon, egg, and cheese on a plain bagel")

deli_line.enqueue("two fried eggs with home fries and ketchup")

deli_line.enqueue("egg and cheese on a roll with jalapeos")

deli_line.enqueue("plain bagel with plain cream cheese")

deli_line.enqueue("blueberry muffin toasted with butter")

deli_line.enqueue("bacon, egg, and cheese on a roll")

# ------------------------ #

deli_line.enqueue("western omelet with home fries")

# ------------------------ #

print("------------\nOur first order will be " + deli_line.peek())

print("------------\nNow serving...\n------------")

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

deli_line.dequeue()

# ------------------------ #

deli_line.dequeue()

# ------------------------ #
class HashMap:

  def __init__(self, array_size):

    self.array_size = array_size

    self.array = [None for item in range(array_size)]



  def hash(self, key, count_collisions=0):

    key_bytes = key.encode()

    hash_code = sum(key_bytes)

    return hash_code + count_collisions



  def compressor(self, hash_code):

    return hash_code % self.array_size



  def assign(self, key, value):

    array_index = self.compressor(self.hash(key))

    current_array_value = self.array[array_index]



    if current_array_value is None:

      self.array[array_index] = [key, value]

      return



    if current_array_value[0] == key:

      self.array[array_index] = [key, value]

      return



    # Collision!



    number_collisions = 1



    while(current_array_value[0] != key):

      new_hash_code = self.hash(key, number_collisions)

      new_array_index = self.compressor(new_hash_code)

      current_array_value = self.array[new_array_index]



      if current_array_value is None:

        self.array[new_array_index] = [key, value]

        return



      if current_array_value[0] == key:

        self.array[new_array_index] = [key, value]

        return



      number_collisions += 1



    return



  def retrieve(self, key):

    array_index = self.compressor(self.hash(key))

    possible_return_value = self.array[array_index]



    if possible_return_value is None:

      return None



    if possible_return_value[0] == key:

      return possible_return_value[1]



    retrieval_collisions = 1



    while (possible_return_value != key):

      new_hash_code = self.hash(key, retrieval_collisions)

      retrieving_array_index = self.compressor(new_hash_code)

      possible_return_value = self.array[retrieving_array_index]



      if possible_return_value is None:

        return None



      if possible_return_value[0] == key:

        return possible_return_value[1]



      retrieval_collisions += 1



    return

  

  

hash_map = HashMap(15)

hash_map.assign("gabbro", "igneous")

hash_map.assign("sandstone", "sedimentary")

hash_map.assign("gneiss", "metamorphic")





print(hash_map.retrieve("gabbro"))

print(hash_map.retrieve("sandstone"))

print(hash_map.retrieve("gneiss"))

  

class Node:

  def __init__(self, value):

    self.value = value

    self.next_node = None

    

  def get_value(self):

    return self.value

  

  def get_next_node(self):

    return self.next_node

  

  def set_next_node(self, next_node):

    self.next_node = next_node



class LinkedList:

  def __init__(self, head_node=None):

    self.head_node = head_node

  

  def insert(self, new_node):

    current_node = self.head_node



    if not current_node:

      self.head_node = new_node



    while(current_node):

      next_node = current_node.get_next_node()

      if not next_node:

        current_node.set_next_node(new_node)

      current_node = next_node



  def __iter__(self):

    current_node = self.head_node

    while(current_node):

      yield current_node.get_value()

      current_node = current_node.get_next_node()



        

flower_definitions = [['begonia', 'cautiousness'], ['chrysanthemum', 'cheerfulness'], ['carnation', 'memories'], ['daisy', 'innocence'], ['hyacinth', 'playfulness'], ['lavender', 'devotion'], ['magnolia', 'dignity'], ['morning glory', 'unrequited love'], ['periwinkle', 'new friendship'], ['poppy', 'rest'], ['rose', 'love'], ['snapdragon', 'grace'], ['sunflower', 'longevity'], ['wisteria', 'good luck']]





# HashMap with Separate Chaining

# At array index: a linkedlist, each node.value is [key, value]. Note the first value is the value of the node, which is a 2-element list; 

# the second value is the value (at the 2nd index in the 2-element list) for the key-value pair.

class HashMap:

  def __init__(self, size):

    self.array_size = size

    self.array = [LinkedList() for i in range(size)]

    

  def hash(self, key):

    return sum(key.encode())

  

  def compress(self, hash_code):

    return hash_code%self.array_size

  

  def assign(self, key, value):

    hash_code = self.hash(key)

    array_index = self.compress(hash_code)

    payload = Node([key, value])

    list_at_index = self.array[array_index]

    current = list_at_index.head_node

    while current: 

        # note that since we defined __iter__ method for LinkedList, we can still use for item in list_at_index to access each node's value as item,

        # but here we provide another way that doesn't use this feature to traverse the linkedlist.

      if current.value[0] == key:

        current.value[1] = value

        return

      current = current.next_node

    list_at_index.insert(payload)

    

  

  def retrieve(self, key):

    hash_code = self.hash(key)

    array_index = self.compress(hash_code)

    list_at_index = self.array[array_index]

    current = list_at_index.head_node

    while current:

      if current.value[0] == key:

        return current.value[1]

      current = current.next_node

    return None



blossom = HashMap(len(flower_definitions))



# Construct the hashmap

for item in flower_definitions:

  blossom.assign(item[0], item[1])

  print("key:", item[0], ", hash:", blossom.hash(item[0]), ", index:", blossom.compress(blossom.hash(item[0])))



# Search in the hashmap

for item in flower_definitions:

  print(item[0], blossom.retrieve(item[0]))

class TreeNode:

  def __init__(self, value):

    self.value = value

    self.children = []



  def __repr__(self, level=0):

    # HELPER METHOD TO PRINT TREE!

    ret = "--->" * level + repr(self.value) + "\n"

    for child in self.children:

      ret += child.__repr__(level+1)

    return ret



  def add_child(self, child_node):

    self.children.append(child_node) 



### TEST CODE TO PRINT TREE

company = [

  "Monkey Business CEO", 

  "VP of Bananas", 

  "VP of Lazing Around", 

  "Associate Chimp", 

  "Chief Bonobo", "Produce Manager", "Tire Swing R & D"]

root = TreeNode(company.pop(0))

for count in range(2):

  child = TreeNode(company.pop(0))

  root.add_child(child)



root.children[0].add_child(TreeNode(company.pop(0)))

root.children[0].add_child(TreeNode(company.pop(0)))

root.children[1].add_child(TreeNode(company.pop(0)))

root.children[1].add_child(TreeNode(company.pop(0)))

print("MONKEY BUSINESS, LLC.")

print("=====================")

print(root)



class TreeNode:

  def __init__(self, value):

    self.value = value # data

    self.children = [] # references to other nodes



  def add_child(self, child_node):

    # creates parent-child relationship

    print("Adding " + child_node.value)

    self.children.append(child_node) 

    

  def remove_child(self, child_node):

    # removes parent-child relationship

    print("Removing " + child_node.value + " from " + self.value)

    self.children = [child for child in self.children 

                     if child is not child_node]



  def traverse(self):

    # moves through each node referenced from self downwards

    nodes_to_visit = [self]

    while len(nodes_to_visit) > 0:

      current_node = nodes_to_visit.pop()

      print(current_node.value)

      nodes_to_visit += current_node.children

class MinHeap:

  def __init__(self):

    self.heap_list = [None]

    self.count = 0



  def parent_idx(self, idx):

    return idx // 2



  def left_child_idx(self, idx):

    return idx * 2



  def right_child_idx(self, idx):

    return idx * 2 + 1



  def child_present(self, idx):

    return self.left_child_idx(idx) <= self.count

  

  def retrieve_min(self):

    if self.count == 0:

      print("No items in heap")

      return None

    

    min = self.heap_list[1]

    self.heap_list[1] = self.heap_list[self.count]

    self.count -= 1

    self.heap_list.pop()

    self.heapify_down()

    return min



  def add(self, element):

    self.count += 1

    self.heap_list.append(element)

    self.heapify_up()





  def get_smaller_child_idx(self, idx):

    if self.right_child_idx(idx) > self.count:

      return self.left_child_idx(idx)

    else:

      left_child = self.heap_list[self.left_child_idx(idx)]

      right_child = self.heap_list[self.right_child_idx(idx)]

      if left_child < right_child:

        return self.left_child_idx(idx)

      else:

        return self.right_child_idx(idx)

    

  def heapify_up(self):

    idx = self.count

    swap_count = 0

    while self.parent_idx(idx) > 0:

      if self.heap_list[self.parent_idx(idx)] > self.heap_list[idx]:

        swap_count += 1

        tmp = self.heap_list[self.parent_idx(idx)]

        self.heap_list[self.parent_idx(idx)] = self.heap_list[idx]

        self.heap_list[idx] = tmp

      idx = self.parent_idx(idx)



    element_count = len(self.heap_list)

    if element_count > 10000:

      print("Heap of {0} elements restored with {1} swaps"

            .format(element_count, swap_count))

      print("")    

      

  def heapify_down(self):

    idx = 1

    # starts at 1 because we swapped first and last elements

    swap_count = 1

    while self.child_present(idx):

      smaller_child_idx = self.get_smaller_child_idx(idx)

      if self.heap_list[idx] > self.heap_list[smaller_child_idx]:

        swap_count += 1

        tmp = self.heap_list[smaller_child_idx]

        self.heap_list[smaller_child_idx] = self.heap_list[idx]

        self.heap_list[idx] = tmp

      idx = smaller_child_idx



    element_count = len(self.heap_list)

    if element_count >= 10000:

      print("Heap of {0} elements restored with {1} swaps"

            .format(element_count, swap_count))

      print("")  



        

        

# import random number generator

from random import randrange



# make an instance of MinHeap

min_heap = MinHeap()



# populate min_heap with descending numbers

descending_nums = [n for n in range(10001, 1, -1)]

print("ADDING!")

for el in descending_nums:

  min_heap.add(el)



print("REMOVING!")

# remove minimum until min_heap is empty

min_heap.retrieve_min()

class MinHeap:

  def __init__(self):

    self.heap_list = [None]

    self.count = 0



  # HEAP HELPER METHODS

  # DO NOT CHANGE!

  def parent_idx(self, idx):

    return idx // 2



  def left_child_idx(self, idx):

    return idx * 2



  def right_child_idx(self, idx):

    return idx * 2 + 1



  def child_present(self, idx):

    return self.left_child_idx(idx) <= self.count



  # END OF HEAP HELPER METHODS

  



  def retrieve_min(self):

    if self.count == 0:

      print("No items in heap")

      return None

    

    min = self.heap_list[1]

    print("Removing: {0} from {1}".format(min, self.heap_list))

    self.heap_list[1] = self.heap_list[self.count]

    self.count -= 1

    self.heap_list.pop()

    print("Last element moved to first: {0}".format(self.heap_list))    

    self.heapify_down()

    return min



  def add(self, element):

    self.count += 1

    print("Adding: {0} to {1}".format(element, self.heap_list))

    self.heap_list.append(element)

    self.heapify_up()





  def get_smaller_child_idx(self, idx):

    if self.right_child_idx(idx) > self.count:

      print("There is only a left child")

      return self.left_child_idx(idx)

    else:

      left_child = self.heap_list[self.left_child_idx(idx)]

      right_child = self.heap_list[self.right_child_idx(idx)]

      if left_child < right_child:

        print("Left child is smaller")

        return self.left_child_idx(idx)

      else:

        print("Right child is smaller")

        return self.right_child_idx(idx)

    

  def heapify_up(self):

    idx = self.count

    while self.parent_idx(idx) > 0:

      if self.heap_list[self.parent_idx(idx)] > self.heap_list[idx]:

        tmp = self.heap_list[self.parent_idx(idx)]

        print("swapping {0} with {1}".format(tmp, self.heap_list[idx]))

        self.heap_list[self.parent_idx(idx)] = self.heap_list[idx]

        self.heap_list[idx] = tmp

      idx = self.parent_idx(idx)

    print("HEAP RESTORED! {0}".format(self.heap_list))

    print("")

      

  def heapify_down(self):

    idx = 1

    while self.child_present(idx):

      smaller_child_idx = self.get_smaller_child_idx(idx)

      if self.heap_list[idx] > self.heap_list[smaller_child_idx]:

        tmp = self.heap_list[smaller_child_idx]

        print("swapping {0} with {1}".format(self.heap_list[idx], tmp))

        self.heap_list[smaller_child_idx] = self.heap_list[idx]

        self.heap_list[idx] = tmp



      idx = smaller_child_idx

    print("HEAP RESTORED! {0}".format(self.heap_list))

    print("")    



# import random number generator

from random import randrange



# make an instance of MinHeap

min_heap = MinHeap()



# populate min_heap with random numbers

random_nums = [randrange(1, 101) for n in range(6)]

for el in random_nums:

  min_heap.add(el)



# remove minimum until min_heap is empty

while min_heap.count != 0:

  min_heap.retrieve_min()

from random import randrange



class Vertex:

  def __init__(self, value):

    self.value = value

    self.edges = {}



  def add_edge(self, vertex, weight = 0):

    self.edges[vertex] = weight



  def get_edges(self):

    return list(self.edges.keys())



class Graph:

  def __init__(self, directed = False):

    self.graph_dict = {}

    self.directed = directed



  def add_vertex(self, vertex):

    self.graph_dict[vertex.value] = vertex



  def add_edge(self, from_vertex, to_vertex, weight = 0):

    self.graph_dict[from_vertex.value].add_edge(to_vertex.value, weight)

    if not self.directed:

      self.graph_dict[to_vertex.value].add_edge(from_vertex.value, weight)



  def find_path(self, start_vertex, end_vertex):

    start = [start_vertex]

    seen = {}

    while len(start) > 0:

      current_vertex = start.pop(0)

      seen[current_vertex] = True

      print("Visiting " + current_vertex)

      if current_vertex == end_vertex:

        return True

      else:

        vertices_to_visit = set(self.graph_dict[current_vertex].edges.keys())

        start += [vertex for vertex in vertices_to_visit if vertex not in seen]

    return False



def print_graph(graph):

  for vertex in graph.graph_dict:

    print("")

    print(vertex + " connected to")

    vertex_neighbors = graph.graph_dict[vertex].edges

    if len(vertex_neighbors) == 0:

      print("No edges!")

    for adjacent_vertex in vertex_neighbors:

      print("=> " + adjacent_vertex)





def build_graph(directed):

  g = Graph(directed)

  vertices = []

  for val in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:

    vertex = Vertex(val)

    vertices.append(vertex)

    g.add_vertex(vertex)



  for v in range(len(vertices)):

    v_idx = randrange(0, len(vertices) - 1)

    v1 = vertices[v_idx]

    v_idx = randrange(0, len(vertices) - 1)

    v2 = vertices[v_idx]

    g.add_edge(v1, v2, randrange(1, 10))



  print_graph(g)



build_graph(False)

# Define build_bst() below...

def build_bst(my_list):

  if not my_list:

    return "No Child"

  middle_idx = len(my_list)//2

  middle_value = my_list[middle_idx]

  print("Middle index: " + str(middle_idx))

  print("Middle value: " + str(middle_value))

  tree_node = {"data": middle_value}

  tree_node["left_child"] = build_bst(my_list[:middle_idx])

  tree_node["right_child"] = build_bst(my_list[middle_idx+1:])

  return tree_node



# For testing

sorted_list = [12, 13, 14, 15, 16]

binary_search_tree = build_bst(sorted_list)

print(binary_search_tree)



# fill in the runtime as a string

# 1, logN, N, N*logN, N^2, 2^N, N!

runtime = "N*logN"
