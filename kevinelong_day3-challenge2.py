# Create a function that combines two dictionaries into one dictionary

def combine_dicts(dict1, dict2):
    output_dict = {}
    #TODO PUT CODE HERE - loop through both and add values using keys and square brakets on both sides of =
    
    return output_dict

print(combine_dicts( {"A": 9, "B": 5, "C": 3}, {"D": 87, "E": 3, "F": 1 }))
#L2 How can we merge/sum values if there are duplicate keys - using an "if" and the "in" keyword.
# Create a function that combines two dictionaries into one dictionary

def combine_dicts(dict1, dict2):
    output_dict = {}
    #TODO PUT CODE HERE - loop through both and add values using keys and square brakets
    
    for k in dict1:
        output_dict[k] = dict1[k]

    for k in dict2:
        output_dict[k] = dict2[k]
        
    return output_dict

print(combine_dicts( {"A": 9, "B": 5, "C": 3}, {"D": 87, "E": 3, "F": 1 }))
# Create a function that combines two dictionaries into one dictionary

def combine_dicts(dict1, dict2):
    output_dict = {}
    #TODO PUT CODE HERE - loop through both and add values using keys and square brakets
    
    for k in dict1:
        output_dict[k] = dict1[k]

    for k in dict2:
        if k in output_dict:
            output_dict[k] = output_dict[k] + dict2[k]
#             output_dict[k] += dict2[k] # SAME BUT SHORTER

        else:
            output_dict[k] = dict2[k]
        
    return output_dict

print(combine_dicts( {"A": 9, "B": 5, "C": 3}, {"A": 87, "B": 3, "F": 1 }))