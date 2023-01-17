nums = [9, 8, 7, 6, 5, 4, 3, 2, 1]

print("PRE SORT: {0}".format(nums))



def swap(arr, index_1, index_2):

  temp = arr[index_1]

  arr[index_1] = arr[index_2]

  arr[index_2] = temp



def bubble_sort_unoptimized(arr):

  iteration_count = 0

  for el in arr:

    for index in range(len(arr) - 1):

      iteration_count += 1

      if arr[index] > arr[index + 1]:

        swap(arr, index, index + 1)



  print("PRE-OPTIMIZED ITERATION COUNT: {0}".format(iteration_count))



def bubble_sort(arr):

  iteration_count = 0

  for i in range(len(arr)):

    # iterate through unplaced elements

    for idx in range(len(arr) - i - 1):

      iteration_count += 1

      if arr[idx] > arr[idx + 1]:

        # replacement for swap function

        arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]

        

  print("POST-OPTIMIZED ITERATION COUNT: {0}".format(iteration_count))



bubble_sort_unoptimized(nums.copy())

bubble_sort(nums)

print("POST SORT: {0}".format(nums))
def merge_sort(items):

  if len(items) <= 1:

    return items



  middle_index = len(items) // 2

  left_split = items[:middle_index]

  right_split = items[middle_index:]



  left_sorted = merge_sort(left_split)

  right_sorted = merge_sort(right_split)



  return merge(left_sorted, right_sorted)



def merge(left, right):

  result = []



  while (left and right):

    if left[0] < right[0]:

      result.append(left[0])

      left.pop(0)

    else:

      result.append(right[0])

      right.pop(0)



  if left:

    result += left

  if right:

    result += right



  return result



unordered_list1 = [356, 746, 264, 569, 949, 895, 125, 455]

unordered_list2 = [787, 677, 391, 318, 543, 717, 180, 113, 795, 19, 202, 534, 201, 370, 276, 975, 403, 624, 770, 595, 571, 268, 373]

unordered_list3 = [860, 380, 151, 585, 743, 542, 147, 820, 439, 865, 924, 387]







ordered_list1 =merge_sort(unordered_list1)

print(ordered_list1)

ordered_list2 =merge_sort(unordered_list2)

print(ordered_list2)

ordered_list3 =merge_sort(unordered_list3)

print(ordered_list3)

from random import randrange, shuffle

# in-place implementation

def quicksort(list, start, end):

  # this portion of list has been sorted

  if start >= end:

    return

  print("Running quicksort on {0}".format(list[start: end + 1]))

  # select random element to be pivot

  pivot_idx = randrange(start, end + 1)

  pivot_element = list[pivot_idx]

  print("Selected pivot {0}".format(pivot_element))

  # swap random element with last element in sub-lists

  list[end], list[pivot_idx] = list[pivot_idx], list[end]



  # tracks all elements which should be to left (lesser than) pivot

  less_than_pointer = start

  

  for i in range(start, end):

    # we found an element out of place

    if list[i] < pivot_element:

      # swap element to the right-most portion of lesser elements

      print("Swapping {0} with {1}".format(list[i], pivot_element))

      list[i], list[less_than_pointer] = list[less_than_pointer], list[i]

      # tally that we have one more lesser element

      less_than_pointer += 1

  # move pivot element to the right-most portion of lesser elements

  list[end], list[less_than_pointer] = list[less_than_pointer], list[end]

  print("{0} successfully partitioned".format(list[start: end + 1]))

  # recursively sort left and right sub-lists

  quicksort(list, start, less_than_pointer - 1)

  quicksort(list, less_than_pointer + 1, end)





    

    

  

list = [5,3,1,7,4,6,2,8]

shuffle(list)

print("PRE SORT: ", list)

print(quicksort(list, 0, len(list) -1))

print("POST SORT: ", list)

# Another quick sort



def qs(arr):

  if len(arr) <= 1:

    return arr



  smaller = []

  larger = []

  

  pivot = 0

  pivot_element = arr[pivot]

  

  for i in range(1, len(arr)):

    if arr[i] > pivot_element:

      larger.append(arr[i])

    else:

      smaller.append(arr[i])



  sorted_smaller = qs(smaller)

  sorted_larger = qs(larger)



  return sorted_smaller + [pivot_element] + sorted_larger
def radix_sort(to_be_sorted):

  maximum_value = max(to_be_sorted)

  max_exponent = len(str(maximum_value))

  being_sorted = to_be_sorted[:]



  for exponent in range(max_exponent):

    position = exponent + 1

    index = -position



    digits = [[] for i in range(10)]



    for number in being_sorted:

      number_as_a_string = str(number)

      try:

        digit = number_as_a_string[index]

      except IndexError:

        digit = 0

        

      digit = int(digit)



      digits[digit].append(number)



    being_sorted = []

    for numeral in digits:

      being_sorted.extend(numeral)



  return being_sorted



unsorted_list = [830, 921, 163, 373, 961, 559, 89, 199, 535, 959, 40, 641, 355, 689, 621, 183, 182, 524, 1]





print(radix_sort(unsorted_list))





def binary_search(sorted_list, left_pointer, right_pointer, target):

  # this condition indicates we've reached an empty "sub-list"

  if left_pointer >= right_pointer:

    return "value not found"

	

  # We calculate the middle index from the pointers now

  mid_idx = (left_pointer + right_pointer) // 2

  mid_val = sorted_list[mid_idx]



  if mid_val == target:

    return mid_idx

  if mid_val > target:

    # we reduce the sub-list by passing in a new right_pointer

    return binary_search(sorted_list, left_pointer, mid_idx, target)

  if mid_val < target:

    # we reduce the sub-list by passing in a new left_pointer

    return binary_search(sorted_list, mid_idx + 1, right_pointer, target)

  

values = [77, 80, 102, 123, 288, 300, 540]

start_of_values = 0

end_of_values = len(values)

result = binary_search(values, start_of_values, end_of_values, 288)



print("element {0} is located at index {1}".format(288, result))
# iteration



def binary_search(sorted_list, target):

  left_pointer = 0

  right_pointer = len(sorted_list)

  

  # fill in the condition for the while loop

  while left_pointer < right_pointer:

    # calculate the middle index using the two pointers

    mid_idx = (left_pointer + right_pointer)//2

    mid_val = sorted_list[mid_idx]

    if mid_val == target:

      return mid_idx

    if target < mid_val:

      # set the right_pointer to the appropriate value

      right_pointer = mid_idx

    if target > mid_val:

      # set the left_pointer to the appropriate value

      left_pointer = mid_idx + 1

  

  return "Value not in list"



# test cases

print(binary_search([5,6,7,8,9], 9))

print(binary_search([5,6,7,8,9], 10))

print(binary_search([5,6,7,8,9], 8))

print(binary_search([5,6,7,8,9], 4))

print(binary_search([5,6,7,8,9], 6))
def sparse_search(data, search_val):

  print("Data: " + str(data))

  print("Search Value: " + str(search_val))

  first = 0

  last = len(data)-1

  while first<=last:

    mid = (first+last)//2

    if not data[mid]:

      left = mid - 1

      right = mid + 1

      while(True):

        if left < first and right > last:

          print("{0} is not in the dataset".format(search_val))

          return

        elif right <= last and data[right]:

          mid = right

          break

        elif left >= first and data[left]:

          mid = left

          break

        right += 1

        left -= 1

    if data[mid] == search_val:

        print("{0} found at position {1}".format(search_val, mid))

        return

    if search_val < data[mid]:

      last = mid - 1

    if search_val > data[mid]:

      first = mid + 1

  print("{0} is not in the dataset".format(search_val))

     
sparse_search(["A", "", "", "", "B", "", "", "", "C", "", "", "D"], "C")
sparse_search(["A", "B", "", "", "E"], "A")
sparse_search(["", "X", "", "Y", "Z"], "Z")
sparse_search(["A", "", "", "", "B", "", "", "", "C"], "D")
sparse_search(["Apple", "", "Banana", "", "", "", "", "Cow"], "Banana")
sparse_search(["Alex", "", "", "", "", "Devan", "", "", "Elise", "", "", "", "Gary", "", "", "Mimi", "", "", "Parth", "", "", "", "Zachary"], "Parth")
the_most_dangerous_graph = {

    'lava': set(['sharks', 'piranhas']),

    'sharks': set(['lava', 'bees', 'lasers']),

    'piranhas': set(['lava', 'crocodiles']),

    'bees': set(['sharks']),

    'lasers': set(['sharks', 'crocodiles']),

    'crocodiles': set(['piranhas', 'lasers'])

  }



def dfs(graph, current_vertex, target_value, visited = None):

  if visited is None:

    visited = []

  visited.append(current_vertex)

  if current_vertex is target_value:

    return visited

  

  for neighbor in graph[current_vertex]:

    if neighbor not in visited:

      path = dfs(graph, neighbor, target_value, visited)

      if path:

        return path



# Call dfs() below and print the result:

print(dfs(the_most_dangerous_graph, "crocodiles", "bees"))
the_most_dangerous_graph = {

    'lava': set(['sharks', 'piranhas']),

    'sharks': set(['lava', 'bees', 'lasers']),

    'piranhas': set(['lava', 'crocodiles']),

    'bees': set(['sharks']),

    'lasers': set(['sharks', 'crocodiles']),

    'crocodiles': set(['piranhas', 'lasers'])

  }



def bfs(graph, start_vertex, target_value):

  path = [start_vertex]

  vertex_and_path = [start_vertex, path]

  bfs_queue = [vertex_and_path]

  visited = set()

  

  while bfs_queue:

    current_vertex, path = bfs_queue.pop(0)

    visited.add(current_vertex)

    

    for neighbor in graph[current_vertex]:

      if neighbor not in visited:

        if neighbor == target_value:

          return path + [neighbor]

        else:

          bfs_queue.append([neighbor, path + [neighbor]])

          

        



# Call bfs() below and print the result:

print(bfs(the_most_dangerous_graph, "crocodiles", "bees"))
