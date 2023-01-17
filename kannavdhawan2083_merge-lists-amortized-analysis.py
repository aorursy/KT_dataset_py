def merge_lists(list_1,list_2):
    
    """Merges the two lists without any duplicates
    
    Args:
        list_1(List)
        list_2(List) 
    
    Attributes:
        list_1(List): List of elements of length n
        list_2(List): List of elements of length m
    
    Returns:
        new_merged_list: Merged list with no duplicates.
    
    """
    list_3= list_1 + list_2           # New list containing elements from both lists.
    set_merged=set(list_3)            # Set conversion eliminating duplicates
    new_merged_list=list(set_merged)  # Set to list conversion

    # Similar solution

    # new_merged_list = list(set(list1+list2))
    
    return new_merged_list
m_list=merge_lists(['Allison', 'Brian', 'Peter'],['Jason', 'Peter', 'Sara']) # function call
print("Merged List is: {}".format(m_list))
# importing libraries
import random   # To generate the lists
import timeit   # To calculate the time taken by the function
len_l1=100000   # length of list 1
len_l2=100000   # length if list 2
random_list_1 = random.sample(range(10, 1000000),len_l1)  # Generating random list of size 100000
random_list_2 = random.sample(range(10, 1000000),len_l2)  
start=timeit.default_timer()                    # start time
m_list=merge_lists(random_list_1,random_list_2) # function call
stop=timeit.default_timer()                     # stop time

print("Time taken: {}".format(stop-start))
# print("Merged List is {}".format(m_list))
print("Length of First list is {}".format(len(random_list_1)))
print("Length of Second list is {}".format(len(random_list_2)))
print("Length of Merged list is {}".format(len(m_list)))
def merge_lists_brute(list_1,list_2):

    """Merges the two lists without any duplicates
    
    Args:
        list_1(List)
        list_2(List) 
    
    Attributes:
        list_1(List): List of elements of length n
        list_2(List): List of elements of length m
    
    Returns:
        new_merged_list: Merged list with no duplicates.
    
    """
    
    list_1=list(set(list_1))    # Removes duplicates from first list
    
    # Adds new elements into First list from Second list
    new_merged_list = list_1 + [x for x in list_2 if x not in list_1]
    
    return new_merged_list
m_list_brute=merge_lists_brute(['Allison', 'Brian', 'Peter'],['Jason', 'Peter', 'Sara']) # function call
print("Merged List is: {}".format(m_list_brute))
start=timeit.default_timer()                    # start time
m_list=merge_lists_brute(random_list_1,random_list_2) # function call
stop=timeit.default_timer()                     # stop time

print("Time taken: {}".format(stop-start))
# print("Merged List is {}".format(m_list))
print("Length of First list is {}".format(len(random_list_1)))
print("Length of Second list is {}".format(len(random_list_2)))
print("Length of Merged list is {}".format(len(m_list)))