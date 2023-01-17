from csv import reader

#Creating a function which converts csv files to datasets of lists.
def csv_to_list(a_file, head = True):
    opened_file = open(a_file, encoding = 'utf-8')
    read_file = reader(opened_file)
    dataset = list(read_file)
    header = dataset[0]
    dataset = dataset[1:]
    return dataset, header

#Importing Datasets and converting them
gplay, gplay_header = csv_to_list('../input/google-and-apple-store/googleplaystore.csv')
appstore, appstore_header = csv_to_list('../input/google-and-apple-store/AppleStore.csv')
def explore_data(dataset, start, end, rows_and_column = False):
    sliced_data = dataset[start:end]
    for x in sliced_data:
        print(x)
        print('\n')
    if rows_and_column:
        print('Number of rows are - ', len(dataset))
        print('Number of columns are - ', len(dataset[0]))
        
print(gplay_header)
print('\n')
explore_data(gplay, 0, 3, True)
print(appstore_header)
print('\n')
explore_data(appstore, 0, 3, True)
for x in appstore:
    if x[5] == 'USD':
        print(x)
del gplay[10472]
def delete_row(a_list, term, column):
    temp_list = []
    for x in a_list:
        if x[column] != term:
            temp_list.append(x)
    return temp_list
duplicate_apps = []
unique_apps = []

for app in gplay:
    name = app[0]
    if name in unique_apps:
        duplicate_apps.append(name)
    else:
        unique_apps.append(name)
    
print('Number of duplicate apps:', len(duplicate_apps))
print('\n')
print('Examples of duplicate apps:', duplicate_apps[:15])
#Removing Multiple Entries
def remove_multiple(a_list, index):
   
    review_count = {}
    for x in a_list:
        name = x[0]
        if name not in review_count:
            review_count[name] = float(x[index])
        elif float(x[index]) > review_count[name]:
            review_count[name] = float(x[index])
    clean_list = []
    already_added = []
    for x in a_list:
        name = x[0]
        if (float(x[index]) == review_count[name]) and (name not in already_added):
            clean_list.append(x)
            already_added.append(name)
    
    return clean_list


            
gplay = remove_multiple(gplay, 3)
appstore = remove_multiple(appstore, 5)
def is_english(a_list, index):
    temp_list = []
    for x in a_list:
        name = x[index]
        counter = 0
        for y in name:
            if ord(y) > 127:
                counter += 1
        if counter < 3:
            temp_list.append(x)
        counter = 0
    return temp_list

gplay = is_english(gplay, 0)
appstore = is_english(appstore, 1)

print(len(gplay))
print(len(appstore))
def free_apps(a_list, index):
    temp_list = []
    for x in a_list:
        price = x[index]
        if price == '0':
            temp_list.append(x)
        elif price == '0.0':
            temp_list.append(x)
    return temp_list    

gplay = free_apps(gplay, 7)
appstore = free_apps(appstore, 4)

print(len(gplay))
print(len(appstore))
#Making a fucntion which takes in a list and index and returns the percentages of the column.

def column_percentage(a_list, index):
    column_dict = {}
    for x in a_list:
        if x[index] in column_dict:
            column_dict[x[index]] += 1
        else:
            column_dict[x[index]] = 1
    
    #finding total entries
    total = 0
    for x in column_dict:
        total += column_dict[x]
    
    #Updating dictionary to percentage
    for x in column_dict:
        column_dict[x] = float((column_dict[x]/total)*100)
    
    return sort_dict(column_dict)
    
def sort_dict(a_dict): 
    table_display = []
    for x in a_dict:
        key_val = (a_dict[x], x)
        table_display.append(key_val)
    table_sorted = sorted(table_display, reverse = True)
    list_sorted = list(table_sorted)
    return list_sorted
genre_percent_appstore = column_percentage(appstore, -5)
explore_data(genre_percent_appstore, 0, 5)
# Analysing the column "Category"

genre_percent_gplay = column_percentage(gplay, 1)
explore_data(genre_percent_gplay, 0, 5)
# Analysing the column "Genres"

genre_percent_gplay = column_percentage(gplay, 9)
explore_data(genre_percent_gplay, 0, 5)
#Function to calculate the above

def avg_by_ratings(a_list, genre_index, rating_count_index):
    column_dict = {}
    for x in a_list:
        if x[genre_index] in column_dict:
            column_dict[x[genre_index]] += 1
        else:
            column_dict[x[genre_index]] = 1
    
    ratings_sum_dict = {}
    for x in a_list:
        if x[genre_index] in ratings_sum_dict:
            ratings_sum_dict[x[genre_index]] += float(x[rating_count_index])
        else:
            ratings_sum_dict[x[genre_index]] = float(x[rating_count_index])
    
    #finding total entries
    total = 0
    for x in column_dict:
        total += column_dict[x]        
    
    percentage_dict = {}
    for x in column_dict:
        percentage_dict[x] = (ratings_sum_dict[x]/column_dict[x])*100
    
    return sort_dict(percentage_dict)
    
popular_appstore = avg_by_ratings(appstore, -5, 5)
explore_data(popular_appstore, 0, 5)
for x in appstore:
    if x[-5] == "Navigation":
        print(x[1], ':', x[5])
for x in appstore:
    if x[-5] == "Reference":
        print(x[1], ':', x[5])
def find_unique(a_list, index):
    installs_dict = {}
    for x in a_list:
        if x[index] in installs_dict:
            installs_dict[x[index]] += 1
        else:
            installs_dict[x[index]] = 1
            
    return sort_dict(installs_dict)

gplay_installs = find_unique(gplay, 5)
explore_data(gplay_installs, 0, 5)
# Function to remove commas and plus, and find avg
def make_number(a_list, index, index_c):
    temp1_list = []
    for x in a_list:
        temp1_list.append(x)
    temp_list = []
    for x in temp1_list:
        installs = str(x[index])
        installs = installs.replace(',','')
        installs = installs.replace('+','')
        installs = installs.replace(' ','')
        x[index] = float(installs)
        temp_list.append(x)
    
    freq_installs = {}
    genre_count = {}
    total_installs = 0
    for x in temp_list:
        if x[index_c] in freq_installs:
            freq_installs[x[index_c]] += x[index]
            genre_count[x[index_c]] += 1
        else:
            freq_installs[x[index_c]] = x[index]
            genre_count[x[index_c]] = 1
        total_installs += x[index]

    avg_installs = {}
    for x in freq_installs:
        avg_installs[x] = freq_installs[x] / genre_count[x]
    
    return sort_dict(avg_installs)

gplay_avg = make_number(gplay, 5, 1)
explore_data(gplay_avg, 0, 10)
for x in gplay:
    if x[1] == 'COMMUNICATION' and (x[5] == 1000000000 or x[5] == 500000000 or x[5] == 100000000):
        print(x[0], ':', x[5],'+')
for x in gplay:
    if x[1] == 'BOOKS_AND_REFERENCE':
        print(x[0], ':', x[5])
for x in gplay:
    if x[1] == 'BOOKS_AND_REFERENCE' and (x[5] == 1000000
                                            or x[5] == 5000000
                                            or x[5] == 10000000
                                            or x[5] == 50000000):
        print(x[0], ':', x[5],'+')