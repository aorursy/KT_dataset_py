import pandas as pd
import numpy as np
from multiprocessing import Pool # For multiprocessing
#read climate file through pd.read_csv()
climate_data = pd.read_csv("../input/climate-data/ClimateData.csv")
#view climate data information
climate_data.info()
#show all column names
climate_data.columns
#check the type of " Date" column
type(climate_data[' Date'][1])
#format columns name
climate_data.columns = list(map(str.strip,list(climate_data.columns)))
# Covert date column from string to datetime type
climate_data['Date'] = pd.to_datetime(climate_data['Date'])
# Sort data frame by date
climate_data.sort_values("Date", inplace = True)
climate_data['Date'] = climate_data['Date'].apply(lambda x: x.date())
#check type of "Date column again
type(climate_data['Date'][1])
#reset index 
climate_data = climate_data.reset_index(drop = True)
# Define a hash function.
def s_hash(x, n):
    
   
    #The rows in dataframe are assigned into different sub dataframes according to the month value of "Data" column.
    #The rows in Jan - March are assigned to hash with value 1, rows in Apri-Jun are assigned to hash with value 2,
    #Rows in July-Sep are assigned to hash with value 3, rows in Oct-Dec are assigende to hash with value 4 
    #ceil function is used to as round-up for the hash value
    hash_value = np.ceil(x/(12/n))     # Use ceil() function to round up the result
    

    return hash_value

# Hash data partitionining function. 
# We will use the "s_hash" function defined above to realise this partitioning
def h_partition(data, n):
    # n is the number of processors
    
    #get the number of rows and columns
    nrow,ncol = data.shape
    
    dic = {} # We will use a dictionary to store the keys and values to map hash_value and corresponding records
    for x in range(nrow): # For each data record, perform the following
        h = s_hash(data.iloc[x,1].month, n) # Get the hash key of each row based on month
        if (h in dic.keys()): # If the key exists
            dic[h] = dic[h].append(data.iloc[x,:])
        else: # If the key does not exist
            s = pd.DataFrame(columns=data.columns) # Create an empty value set/Data Frame
            s = s.append(data.iloc[x,:]) # Insert a new key and value pair
            dic[h] = s # Add the value set to the key

    return dic
#search function
def binary_search(data, key):
    data.sort_values('Date', inplace = True)
    key = pd.to_datetime(key).date()
       
    matched_record = None
    position = -1 # not found position
    
    lower = 0
    middle = 0
    upper = len(data)-1
    
    ### START CODE HERE ### 
    while lower <= upper:
        # calculate middle: the half of lower and upper
        middle = int((lower + upper)/2)
        
        if data.iloc[middle,1] == key:
            matched_record = data.iloc[middle,:]
            position = middle
            df = pd.DataFrame(columns=data.columns)
            return df.append(matched_record)
        elif data.iloc[middle,1] < key:
            lower = middle
        else:
            upper = middle
      
    ### END CODE HERE ###
    
# Test the binary search algorithm
binary_search(climate_data, "2017/1/5")
# Parallel searching algorithm for exact match
def parallel_search(data, query, n_processor, m_partition, m_search):
  
    results = pd.DataFrame(columns=data.columns)
    
    #enabling parallel processing. 
    pool = Pool(processes=n_processor)

    print("data partitioning:" + str(m_partition.__name__))
    print("searching method:" + str(m_search.__name__))
    
    DD = m_partition(data, n_processor) 
    
    # Each element in DD has a pair (hash key: s_hash(date))
    query_hash_value = s_hash(pd.to_datetime(query).date().month, n_processor)
    d = DD[query_hash_value]
    result = pool.apply(m_search, [d, query])
    results = pd.concat([results, result])

    return results.reset_index(drop =True)
#Test the parallel_search function
parallel_search(climate_data,"2017-12-15",4,h_partition,binary_search)
#read fire_data file through pd.read_csv()
fire_data = pd.read_csv("../input/fire-incident-data/FireData.csv")
#print the min and max value of
fire_data.describe()
#view columns name of fire_data
fire_data.columns
# Range data partitionining function
def range_partition(data, range_indices):
  
    result = []
    
    # First, we sort the dataset according dates  
    new_data = data
    new_data.sort_values(new_data.columns[-1],inplace = True)
    new_data.reset_index(drop = True)
    
    #get numbers of rows and columns
    nrow, ncol = new_data.shape
    
    # Calculate the number of sub dataframe
    n_bin = len(range_indices) 

    #last_element_index
    last_index = 0
    
    # For each dataframe, perform the following
    for i in range(n_bin): 
        x = 0
        while x < nrow:
            if new_data.iloc[x,-1] >= range_indices[i]:
                result.append(new_data.iloc[last_index:x,:].reset_index(drop = True))
                last_index = x
                x = nrow
            else:
                x +=1

    # Append the last remaining data rows
    result.append(new_data.iloc[last_index:].reset_index(drop = True))
    
    return result
#range search in sorted data
def range_search(data, value_range):
    #get total number of rows and columns in data
    nrow, ncol = data.shape
    first_index = None
    last_index = None
    for i in range(nrow):
        #data.iloc[i,-1] is the surface temp(celclus)
        if data.iloc[i,-1] >= value_range[0] and first_index == None:
            first_index = i
        if data.iloc[i,-1] > value_range[1]:
            last_index = i
            break
    if first_index != None and last_index != None:
        return data.iloc[first_index:last_index,].reset_index(drop=True)
    else:
        if first_index != None:
            return data.iloc[first_index:,].reset_index(drop=True)
        else:
            return None
# Parallel searching algorithm for exact match
def parallel_search_range(data, query, n_processor, m_partition, m_search):

    results = pd.DataFrame(columns=data.columns)

    # enabling parallel processing. 
    pool = Pool(processes=n_processor)
      
    print("data partitioning:" + str(m_partition.__name__))
    print("searching method:" + str(m_search.__name__))
        
    DD = m_partition(data, [50, 100]) 
    for d in DD: # Find the range that may contain the query 
        if (query[0] >= d.iloc[0,-1] and query[0] <= d.iloc[-1,-1]) or (query[1] >= d.iloc[0,-1] and query[1] <= d.iloc[-1,-1]): 
            result = pool.apply(m_search, [d,query])
            results = pd.concat([results, result])
    
    return results.reset_index(drop =True)
parallel_search_range(fire_data,[65,100],4,range_partition,range_search)
climate_data.head()
fire_data.head()
#view columns type
climate_data.info()
#view columns type
fire_data.info()
#view the type of data column of fire_data
print(fire_data.iloc[0,-2])
type(fire_data.iloc[0,-2])
#transform type into data
fire_data['Date'] = pd.to_datetime(fire_data['Date']).apply(lambda x: x.date())
print(fire_data.iloc[0,-2])
type(fire_data.iloc[0,-2])
# Range data partitionining function
def range_partition_join(data, range_indices):
 
    result = []
    
    # First, we sort the dataset according their date 
    new_data = data
    new_data.sort_values("Date",inplace = True)
    new_data = new_data.reset_index(drop = True)
    
    #get numbers of rows and columns
    nrow, ncol = new_data.shape
    
    # Calculate the number of bins
    n_bin = len(range_indices) 

    #last_element_index
    last_index = 0
    
    # For each bin, perform the following
    for i in range(n_bin): 
        x = 0
        while x < nrow:
            if new_data.loc[x,"Date"]> range_indices[i]:
                result.append(new_data.iloc[last_index:x,:].reset_index(drop =True))
                last_index = x
                x = nrow
            else:
                x +=1

    # Append the last remaining data rows
    record = new_data.iloc[last_index:,:].reset_index(drop=True)
    result.append(record)
    
    return result
def NL_join(T1, T2):
    #initialize result  
    result = pd.DataFrame(columns=(list(T1.columns)+(list(T2.columns))))
    
    #Get the number of rows and columns
    nrow_T1,ncol_T1 = T1.shape
    nrow_T2,ncol_T2 = T2.shape

    # For each row of T1
    for i in range(nrow_T1):
        # For each row of T2
        for x in range(nrow_T2):
            #If matched Then            
            if T1.loc[i,"Date"] == T2.loc[x,"Date"]:
                # Store the joined records into the result dataframe
                match_record = T1.iloc[i,:].append(T2.iloc[x,:])
                result = result.append(match_record, ignore_index= True)
    return result
#create a date list to split dataframe
date_list = pd.Series(["2017-3-31","2017-6-30","2017-9-30"])
date_list = pd.to_datetime(date_list).apply(lambda x : x.date())
def DPBP_join(T1, T2, Join_function, n_processor):
      
    result = pd.DataFrame(columns=(list(T1.columns)+(list(T2.columns))))

    # Partition T1 & T2 into sub-tables using rr_partition().
    # The number of the sub-tables must be the equal to the n_processor
    T1_subsets = range_partition_join(T1, date_list)
    T2_subsets = range_partition_join(T2, date_list)
    
    # enabling parallel processing. 
    pool = Pool(processes = n_processor)
    
    for i in range(len(T1_subsets)):
        # Apply a join on each processor
        result = result.append(pool.apply(Join_function, [T1_subsets[i], T2_subsets[i]]))

    return result.reset_index(drop = True)
result_count = DPBP_join(fire_data,climate_data,NL_join,4)
result_count.head()
result_count[["Surface Temperature (Celcius)","Air Temperature(Celcius)","Relative Humidity","Max Wind Speed"]]
def NL_join_condition(T1, T2):
    
    #initialize result
    result = pd.DataFrame(columns=(list(T1.columns)+(list(T2.columns))))
    nrow_T1,ncol_T1 = T1.shape
    nrow_T2,ncol_T2 = T2.shape

    # For each row of T1
    for i in range(nrow_T1):
        # For each row of T2
        for x in range(nrow_T2):
            #If matched Then            
            if T1.loc[i,"Date"] == T2.loc[x,"Date"]:
                if T1.loc[i,"Confidence"] >=80 and T1.loc[i,"Confidence"] <= 100: 
                    # Store the joined row into the result dataframe
                    match_record = T1.iloc[i,:].append(T2.iloc[x,:])
                    result = result.append(match_record, ignore_index= True)

    return result
result_condition = DPBP_join(fire_data,climate_data,NL_join_condition,4)
result_condition.head()
DPBP_join(fire_data,climate_data,NL_join_condition,4)[["Datetime","Surface Temperature (Celcius)","Confidence","Air Temperature(Celcius)"]]
# Display the Temprature column name
fire_data.columns.values[-1]
# Check the rows and columns of dataframe
print("Length of dataset:", len(fire_data)) # Same as print(fire.shape[0])
print("Number of columns:", fire_data.shape[1])
# Round-robin data partitionining function
def rb_partition(data, n):    
    # Declare abd initialize a list to store result 
    result = []
    
    # Create n empty lists inside result, each list stores a chunk of data
    for i in range(n):
        result.append([])
    
    # Calculate the number of the rows in the dataframe to be allocated to each chunk
    #n_bin = data.shape[0]/n
    n_chunk = len(data)/n
   
    # For each chunk, add the rows from the dataframe based on its index to the lists inside result 
    for index, row in data.iterrows(): 
        # Calculate the index of the bin that the current data point will be assigned
        index_chunk = (int) (index % n)
        result[index_chunk].append(row) 
                
    return result
# -- Define Sort and Merge functions

# Define a quick sort function which sort the list according to the temperature
def qsort(arr): 
    if len(arr) <= 1:
        return arr
    else:
        return qsort([x for x in arr[1:] if x['Surface Temperature (Celcius)'] < arr[0]['Surface Temperature (Celcius)']]) \
                + [arr[0]] \
                + qsort([x for x in arr[1:] if x['Surface Temperature (Celcius)'] >= arr[0]['Surface Temperature (Celcius)']])
# Testing the qsort function
qsort(rb_partition(fire_data, 4)[1])
# Define a K-way Merge function  
# Find the smallest record's index
def find_min(subset):    
    min_record = subset[0]
    index = 0
    for i in range(len(subset)):
        if(subset[i] < min_record):  
            index = i
            min_record = subset[i]
    return index



def k_way_merge(subsets):
    
    # indexes stores the indexes of sorted records in a temp list
    indexes = []
    for x in subsets:
        indexes.append(0) # initialisaze all indexes to 0

    # final result will be stored in this variable
    result = []  
    
    # the temp list as a buffer
    temp_list = []
    

    while(True):
        temp_list = [] # initialise the temp list
        
        # This loop adds the elements to the temp list and break at a certain condition
        for i in range(len(subsets)):
            if(indexes[i] >= len(subsets[i])):
                temp_list.append(999999)
            else:
                temp_list.append(subsets[i][indexes[i]]["Surface Temperature (Celcius)"])  

        smallest = find_min(temp_list)
    
        # break when loop over the indexes and reach the manually set value
        if(temp_list[smallest] == 999999):
            break

        # Continue to the next element
        result.append(subsets[smallest][indexes[smallest]])
        indexes[smallest] +=1
   
    return result
# Test K way merge and check the size of the merged list
len(k_way_merge(rb_partition(fire_data, 10)))
def serial_sorting(dataset, buffer_size):
    if (buffer_size <= 2):
        print("Error: buffer size should be greater than 2")
        return
    
    result = []
   
    # --- Sort Phase ---
    sorted_set = []
    
    # Read buffer_size pages at a time into memory and sort, and write to a subset 
    start_position = 0
    N = len(dataset)
    while True:
        if ((N - start_position) > buffer_size):
            # read records(with the buffer size) from the input list
            subset = dataset[start_position:start_position + buffer_size] 
            # sort the subset
            sorted_subset = qsort(subset) 
            sorted_set.append(sorted_subset)
            start_position += buffer_size
        
        else:
            # read records(with the buffer size) from the input list
            subset = dataset[start_position:] 
            # sort the subset
            sorted_subset = qsort(subset) 
            sorted_set.append(sorted_subset)
            break
            
  

    # --- Merge Phase ---

    merge_buffer_size = buffer_size - 1
    dataset = sorted_set
    
    while True:
        merged_set = []

        N = len(dataset)
        start_position = 0
        while True:
            if ((N - start_position) > merge_buffer_size): 
                # read records(with the buffer size) from the input list
                subsets = dataset[start_position:start_position + merge_buffer_size]
                merged_set.append(k_way_merge(subsets)) # merge lists in subset
                start_position += merge_buffer_size
            else:
                # read records(with the buffer size) from the input list
                subsets = dataset[start_position:]
                merged_set.append(k_way_merge(subsets)) # merge lists in subset
                break

        dataset = merged_set
        if (len(dataset) <= 1): # if the size of merged record set is 1, no list to merge and then break 
            result = merged_set
            break
            
    return result
def parallel_binary_merge_sorting(dataset, n_processor, buffer_size):    
    if (buffer_size <= 2):
        print("Error: buffer size should be greater than 2")
        return
    
    result = []
    
    # Perform data partitioning using round-robin partitioning function
    subsets = rb_partition(dataset, n_processor)
    
    # Enabling parallel processing. 
    pool = Pool(processes = n_processor)

    # ----- Sort phase -----
    sorted_set = []
    for s in subsets:
        # call the serial_sorting method above
        sorted_set.append(*pool.apply(serial_sorting, [s, buffer_size]))
    pool.close()
    
    # ---- Final merge phase ----
    dataset = sorted_set
    while True:
        merged_set = []

        N = len(dataset)
        start_position = 0
        pool = Pool(processes = N//2)

        while True:
            if ((N - start_position) > 2): 
                subset = dataset[start_position:start_position + 2]
                merged_set.append(pool.apply(k_way_merge, [subset]))
                start_position += 2
            else:
                subset = dataset[start_position:]
                merged_set.append(pool.apply(k_way_merge, [subset]))
                break
        
        pool.close()
        dataset = merged_set
        
        if (len(dataset) == 1): #  if the size of merged record set is 1, no list to merge and then break 
            result = dataset[0]
            break
    
    return result
result = parallel_binary_merge_sorting(fire_data, 4, 10)
print("final sorting result:" + str(result))
# Check the output length - Must be same as the dataframe
len(result)

#view fire data
fire_data.head()
# The first step in the merge-all groupby method to count the number of fires
def local_groupby_nfire(dataset):
    #get the number of rows and columns
    nrow, ncol = dataset.shape
    
    #initializing the dict
    dict = {}
    
    for i in range(nrow):
        key = str(dataset.loc[i,"Date"])
        if key not in dict:
            dict[key] = 0
        dict[key] += 1
    
    return dict
def parallel_merge_all_groupby(dataset):
    #initialize the result
    result = {}

    # Define the number of parallel processors: the number of sub-datasets.
    n_processor = len(dataset)

    # enabling parallel processing. 
    pool = Pool(processes = n_processor)

    local_result = []
    for s in dataset:
        # call the local aggregation method to count the number of fire
        local_result.append(pool.apply(local_groupby_nfire, [s]))
    pool.close()

    #calculate the total number of fire for each day   
    for r in local_result:
        for key, val in r.items():
            if key not in result:
                result[key] = 0
            result[key] += val    
    
    return result
parallel_merge_all_groupby([fire_data])
# The first step in the merge-all groupby method
def local_groupby_temp(dataset):
    #Get the number of rows 
    nrow, ncol = dataset.shape
    
    #initialize dict
    dict = {}
    
    #calculate the total temperature for each date in local processors
    for i in range(nrow):
        key = str(dataset.loc[i,"Date"])
        temp = dataset.loc[i,"Surface Temperature (Celcius)"]
        
        if key not in dict:
            dict[key] = [0,0]
        dict[key][0] += temp
        dict[key][1] += 1
    
    return dict
def parallel_merge_all_groupby_temp(dataset):
    #initiailize result and final result
    result = {}
    final_result = {}
   
    # Define the number of parallel processors: the number of sub-datasets.
    n_processor = len(dataset)

    # enabling parallel processing. 
    pool = Pool(processes = n_processor)

    #local aggregation
    local_result = []
    for s in dataset:
        # call the local aggregation method
        local_result.append(pool.apply(local_groupby_temp, [s]))
    pool.close()

    #global aggregation
    for r in local_result:
        for key, val in r.items():
            if key not in result:
                result[key] = [0,0]
            result[key][0] += val[0]
            result[key][1] += val[1]
    
    #calculate the average of temperature
    for key, val in result.items():
        if key not in final_result:
            final_result[key] = 0
        final_result[key] = val[0]/val[1]
   
    return final_result
parallel_merge_all_groupby_temp([fire_data])

#view the unique value of station in climate data
climate_data["Station"].unique()
#set range list of date indices (same as the task 2)
date_list = pd.Series(["2017-3-31","2017-6-30","2017-9-30"])
date_list = pd.to_datetime(date_list).apply(lambda x : x.date())
# Range data partitionining function
def range_partition_join_Groupby(data, range_indices):
    #initialize the result
    result = []
    
    # First, we sort the dataset according their values  
    new_data = data
    new_data.sort_values("Date",inplace = True)
    new_data = new_data.reset_index(drop = True)
    
    #get numbers of rows and columns
    nrow, ncol = new_data.shape
    
    # Calculate the number of bins
    n_bin = len(range_indices) 

    #last_element_index
    last_index = 0
    
    # For each bin, perform the following
    for i in range(n_bin): 
        x = 0
        while x < nrow:
            if new_data.loc[x,"Date"]> range_indices[i]:
                result.append(new_data.iloc[last_index:x,:].reset_index(drop =True))
                last_index = x
                x = nrow
            else:
                x +=1

    # Append the last remaining data frame
    record = new_data.iloc[last_index:,:].reset_index(drop=True)
    result.append(record)
    
    return result
# The first step in the merge-all groupby method
def local_groupby_temp_station(dataset):
    #get the number of rows  
    nrow, ncol = dataset.shape
    
    #initialize dict and result
    dict = {}
    result = []
    
    #calculate the total temperate and number of calculation
    for i in range(nrow):
        key = str(dataset.loc[i,"Station"])
        temp = dataset.loc[i,"Surface Temperature (Celcius)"]
        if key not in dict:
            dict[key] = [0,0]
        dict[key][0] += temp
        dict[key][1] += 1
    
    #convert dict into list
    for key in dict:
        total_temp = dict[key][0]
        count = dict[key][1]
        result.append([key,total_temp,count])
    
    return result
def NL_join_groupby(T1, T2):
    #initialize the dataframe
    result = pd.DataFrame(columns=(list(T1.columns)+(list(T2.columns))))
    
    #get the number of rows of two dataframe
    nrow_T1,ncol_T1 = T1.shape
    nrow_T2,ncol_T2 = T2.shape

    # For each row of T1
    for i in range(nrow_T1):
        # For each row of T2
        for x in range(nrow_T2):
            #If matched Then            
            if T1.loc[i,"Date"] == T2.loc[x,"Date"]:
                # Store the joined records into the result dataframe
                match_record = T1.iloc[i,:].append(T2.iloc[x,:])
                result = result.append(match_record, ignore_index= True)
    
    #Groupby "station" to calculate the total temperatrue for each station
    Groupby_result = local_groupby_temp_station(result)
    
    return Groupby_result
#the stations are partitioned according to the reminder of its id value / number of processors 
def hash_value(value,n):
    return value%n

#Partitioning the station, total temperature and count list
def partition_station(datasets,n):
    #n is the number of processor
    dic = dict()
    
    #Iterating each dataset and aasign different lists into different hash
    for dataset in datasets:
        for list_tmp in dataset:
            s_value = hash_value(int(list_tmp[0]),n)
            if s_value in dic.keys():
                dic[s_value].append(list_tmp)
            else:
                dic[s_value] = [list_tmp]
    return dic
# The first step in the merge-all groupby method
def global_groupby_temp_hash(datasets):
    #initialize dic
    dic = {}
    
    #calculate total temperatue and number of calculation for each station
    for data_list in datasets:
        key = data_list[0]
        if key not in dic:
            dic[key] = [0,0]
        dic[key][0] += data_list[1]
        dic[key][1] += data_list[2]
    
    return dic
def parallel_gourpby_join_temp_station(T1, T2, Join_function, n_processor):
    #initialization
    datasets = []
    final_result = {}
    result = {}
    
    # Partition T1 & T2 into sub-tables using rr_partition().
    # The number of the sub-tables must be the equal to the n_processor
    T1_subsets = range_partition_join_Groupby(T1, date_list)
    T2_subsets = range_partition_join_Groupby(T2, date_list)
    
    # enabling parallel processing. 
    pool = Pool(processes = n_processor)
    
    for i in range(len(T1_subsets)):
        # Apply a join and local groupby on each processor
        datasets.append(pool.apply(Join_function, [T1_subsets[i], T2_subsets[i]]))
      
    #Partition lists according to station value
    station_partition = partition_station(datasets, n_processor)
    
    #groupby each station total temperature
    global_groupby = []
    for x in station_partition:
        global_groupby.append(pool.apply(global_groupby_temp_hash,[station_partition[x]]))
    
    
    for r in global_groupby:
        for key, val in r.items():
            if key not in result:
                result[key] = [0,0]
            result[key][0] += val[0]
            result[key][1] += val[1]
    
    for key, val in result.items():
        if key not in final_result:
            final_result[key] = 0
        final_result[key] = val[0]/val[1]   
      
    return final_result 
average_temp_station = parallel_gourpby_join_temp_station(fire_data,climate_data,NL_join_groupby,4)
average_temp_station
    
average_temp_station_1 = parallel_gourpby_join_temp_station(fire_data,climate_data,NL_join_groupby,7)
average_temp_station_1
