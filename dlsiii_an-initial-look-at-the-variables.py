'''
    Kaggle Homesite Competition: initial categorization of variable types
    =======================================================================
    - List of dicts (items in list are variables, keys are values, values are counts)
    - Then figure out what proportion are numbers for each variable
        -- And if there seems to be a common value for missing values (for example, god help me if it's 9999)
        -- And whether they look like they should actually be treated like numbers (based on the number of distinct values)

    AUTHOR: Dan Sweeney
    DATE:   28 December 2015
'''

import csv

#filepaths
in_train = '../input/train.csv'
out_chars = '../output/var_chars.csv'

#==============================================================================
# read in data to list of dicts
#==============================================================================

f_in = csv.reader(open(in_train,'r'),delimiter=',')
header = next(f_in)

#initialize the variable list with empty dictionaries
var_list = [{} for _ in range(len(header))]


for curr_vars in f_in:

    for i, var in enumerate(curr_vars):
             
        #want to set as floats if possible, either way add new distinct
        #values as keys to dictionary and update count of values
        try: 
            y = float(var)
            var_list[i][y] = var_list[i][y] + 1 if y in var_list[i] else 1
            
        except:
            var_list[i][var] = var_list[i][var] + 1 if var in var_list[i] else 1

        
#==============================================================================
# get a little info about vars        
#==============================================================================
obs = sum(var_list[0].values())
group_size = 0.05*obs
var_chars = []

for var in var_list:

    #initialize temporary variables
    nflt = 0.0
    flt_set = set()
    avg_numer = 0.0
    flt_mode = 0.0
    flt_mode_cnt = 0.0
    all_mode = 'na'
    all_mode_cnt = 0.0
    all_mode_perc = 0.0
    sum_miss = 0.0
    group_cnt = 0.0
    group_set = set()

    flt_mode_perc = 0.0
    avg_flt = 0.0
    max_flt = 0.0
    min_flt = 0.0

    #get number of distinct values
    dist_vals = len(var)        

    for k in var:

        #update mode (among all values) 
        if var[k] > all_mode_cnt:
            all_mode = k
            all_mode_cnt = var[k]
        
        #update number of meaningful groups
        if var[k] > group_size:
            group_cnt += 1
            group_set.add(k)
        
        #update missing values
        sum_miss += var[k] if k == '' else 0.0
        
        if type(k) == float:
            flt_set.add(k)
            avg_numer += k*var[k]
            nflt += var[k]
            
            if var[k] > flt_mode_cnt:
                flt_mode = k
                flt_mode_cnt = var[k]

    if nflt > 0.0:
        avg_flt = avg_numer / nflt
        max_flt = max(flt_set)
        min_flt = min(flt_set)
        flt_mode_perc = flt_mode_cnt / nflt

    #append variable characteristics to list
    var_chars.append([dist_vals,
                     sum_miss / obs, 
                     nflt / obs,
                     all_mode,
                     all_mode_cnt / obs,
                     group_cnt,
                     group_set,
                     avg_flt,
                     flt_mode,
                     flt_mode_perc,
                     max_flt,
                     min_flt])

#==============================================================================
# print variable characteristics to file
#==============================================================================

#list of elements to print
print_list = [0,1,2,3,4,5,7,8,9,10,11]
print_string = ''
for i, var in enumerate(var_chars):
    print_string += str(header[i]).rstrip('\n')
    for j in print_list:
        print_string += ','+str(var[j]).rstrip('\n')
    print_string += '\n'
    

f_out=open(out_chars,'w')

f_out.write('variable,dist_vals,p_miss,p_flt,mode,p_mode,groups,avg_flt,mode_flt,p_mode_flt,max_flt,min_flt\n')
f_out.write(print_string)

f_out.close()

