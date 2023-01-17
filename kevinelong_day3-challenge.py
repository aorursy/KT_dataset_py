# Create a function that combines two lists into one list

def combine_lists(list1, list2):
    output_list = []
    #TODO PUT CODE HERE - loop through both and append
    
    return output_list

print(combine_lists( [9,5,3], [87,3,1]))
# Create a function that combines two lists into one list

def combine_lists(list1, list2):
    output_list = []
    #TODO PUT CODE HERE - loop through both and append

    for item in list1:
        output_list.append(item)
    
    for item in list2:
        output_list.append(item)
    
    return output_list

print(combine_lists( [9,5,3], [87,3,1]))