import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  
from csv import reader 
# import pandas as pd

## The Google Play Store dataset ##
opened_file = open('../input/google-play-store-apps/googleplaystore.csv')
read_file = reader(opened_file)
android = list(read_file)
android_header = android[0]
android = android[1:]

## The Apple Store dataset ##
opened_file = open('../input/app-store-apple-data-set-10k-apps/AppleStore.csv')
read_file = reader(opened_file)
ios = list(read_file)
ios_header = ios[0]
ios = ios[1:]
def explore_data(dataset, start, end, rows_and_columns=False):
    dataset_slice = dataset[start:end]    
    for row in dataset_slice:
        print(row)
        print('\n') # adds a new (empty) line after each row

    if rows_and_columns:
        print('Number of rows:', len(dataset))
        print('Number of columns:', len(dataset[0]))
print(android_header)
print('\n')
explore_data(android,0,1,True)
print('\n')
print(ios_header)
print('\n')
explore_data(ios,0,1,True)
## To find if there are wrong data in Google Play Store dataset
headerlen1 = len(android_header)
for row in android:
    rowlen1 = len(row)
    if rowlen1 != headerlen1:
        print(row)
        print(android.index(row))
        del android[android.index(row)]
## To fing if there are wrong data in App Store dataset
headerlen2 = len(ios_header)
for row in ios:
    rowlen2 = len(row)
    if rowlen2 != headerlen2:
        print(row)
        print(ios.index(row))
## the above code have return the row number is 10472
print(len(android))   
del android[10472]   
print(len(android))  
duplicate_apps = []
unique_apps = []

for app in android:
    name = app[0]
    if name in unique_apps:
        duplicate_apps.append(name)
    else:
        unique_apps.append(name)
    
print('Number of duplicate apps:', len(duplicate_apps))
print('\n')
print('Examples of duplicate apps:', duplicate_apps[:15])
## To find the Google Play Store dataset whether have duplicate apps.
duplicate_android_apps = [] ## store only duplicate apps name
duplicate_android = []      ## store duplicate apps information (all row)
unique_android_apps = []  
unique_android = []
reviews_max_android = {}            ## to make sure the duplicate app, which has max reviews, have been choosed.

for app in android: 
    name = app[0]
    n_reviews = float(app[3])
    
    if name in reviews_max_android and reviews_max_android[name] < n_reviews:
        reviews_max_android[name] = n_reviews
    
    elif name not in reviews_max_android:
        reviews_max_android[name] = n_reviews
        
    if (reviews_max_android[name] == n_reviews) and (name not in unique_android_apps):
        unique_android_apps.append(name)
        unique_android.append(app)
    else:
        duplicate_android_apps.append(name)
        duplicate_android.append(app)

print(len(android))
print(len(unique_android))
print((len(duplicate_android)))
print('\n')
print('Example of duplicate apps:',duplicate_android_apps[:15])
print('\n')
# unique_android_set = pd.DataFrame(columns = android_header,data = unique_android)
# print(len(unique_android_set))
# unique_android_set.to_csv('unique_android_set.csv')
print(len(reviews_max_android))
## To find the App Store dataset whether have duplicate apps.
duplicate_ios_apps = [] 
duplicate_ios = []
unique_ios_apps = []
unique_ios = []
rating_max_ios = {}

for app in ios: 
    name = app[2]
    n_rating = float(app[6])
    
    if name in rating_max_ios and rating_max_ios[name] < n_rating:
        reviews_max_android[name] = n_reviews
    
    elif name not in rating_max_ios:
        rating_max_ios[name] = n_rating
        
    if (rating_max_ios[name] == n_rating) and (name not in unique_ios_apps):
        unique_ios_apps.append(name)
        unique_ios.append(app)
    else:
        duplicate_ios_apps.append(name)
        duplicate_ios.append(app)

print('Number of duplicate apps:',len(duplicate_ios_apps))
print('Number of unique apps:',len(unique_ios_apps))
print('\n')
# unique_ios_set = pd.DataFrame(columns = ios_header,data = unique_ios)
# unique_ios_set.to_csv('unique_ios_set.csv')
def check_name(string):
    for character in string:
        if ord(character) > 127:  ## ord(), a built in function, can get the corresponding number of each character 
            return False
        else:
            return True
print(check_name('爱奇艺'))
print(check_name('dog'))
## english name check function
def check_name_up(string):
    n_ascii = 0
    for character in string:
        if ord(character) > 127: ## English name and relative character can be conver into the ASCII range.
            n_ascii += 1  ## To minimize the impace of data loss, only remove an app if its name have more than three characters 
    if n_ascii > 3:         ## with corresponeding number falling outside the ASCII range.
        return False
    else:
        return True
            
print(check_name_up('Docs To Go™ Free Office Suite'))
print(check_name_up('Instachat 😜'))
print(ord('😜'))
android_english = []
android_non_english = []

for row in unique_android:
    name = row[0]
    if check_name_up(name):
        android_english.append(row)
    else:
        android_non_english.append(row)

# print(explore_data(android_english,0,3,True))
print(len(android_english))

# print(android_non_english)
print(len(android_non_english))
ios_english = []
ios_non_english = []

for row in unique_ios:
    name = row[2]
    if check_name_up(name):
        ios_english.append(row)
    else:
        ios_non_english.append(row)
        
print(len(ios_english))

print(len(ios_non_english))
# android_nonfree = []
android_free = []
# ios_nonfree = []
ios_free = []

for app in android_english:
    price = app[7]
    if price == '0':
        android_free.append(app)
#     else:
#         android_nonfree.append(app)

for app in ios_english:
    price = app[5]
    if price == '0':
        ios_free.append(app)
#     else:
#         ios_nonfree.append(app)


# print(len(android_nonfree))
print(len(android_free))
print('\n')
# print(len(ios_nonfree)) 
print(len(ios_free))
def freq_table(dataset,index):  #calculate the percantage of kinds in dataset.
    table = {}
    total = 0
    
    for app in dataset:
        total += 1
        kind = app[index]
        if kind in table:
            table[kind] += 1
        else: 
            table[kind] = 1
    
    tab_per = {}
    for kind in table:
        table_per = (table[kind]/total)*100
        tab_per[kind] = table_per
    return tab_per

def display_table(dataset,index):# to show the result of freq_table
    table = freq_table(dataset,index)
    table_display = []
    
    for kind in table:
        kind_display = (table[kind],kind)
        table_display.append(kind_display)
    
    table_sorted = sorted(table_display,reverse = True) #sort the result
    for row in table_sorted:
        print(row[1],':',row[0])
        
# display_table(android_nonfree,-4) ## By Genre
# print('\n')
display_table(android_free,-4)
# display_table(ios_nonfree,-5)
# print('\n')
display_table(android_free,5)  # the Install column
categories_android = freq_table(android_free, 1)
x = []
labels = []

for category in categories_android:
    total = 0
    len_category = 0
    for app in android_free:
        category_app = app[1]
        if category_app == category:            
            n_installs = app[5]
            n_installs = n_installs.replace(',', '')
            n_installs = n_installs.replace('+', '')
            total += float(n_installs)
            len_category += 1
    avg_n_installs = total / len_category
    x.append(avg_n_installs)
    labels.append(category)
    print(category, ':', avg_n_installs)
# print (lables)
# print(x)

fig = plt.figure()
plt.pie(x,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("Pie chart")
plt.show() 
for app in android_free:
    if app[1] == 'COMMUNICATION' and (app[5] == '1,000,000,000+'
                                     or app[5] == '500,000,000+'
                                     or app[5] == '100,000,000+'):
        print(app[0],':',app[5])