# 1. 데이터 불러오기

from csv import reader

opened_file = open('../input/googleplaystore.csv')

read_file = reader(opened_file)

android = list(read_file)

android_header = android[0]

android = android[1:]
android_header
# 데이터 읽을 수 있으며 사이즈를 파악할 수 있는 함수 정의

def explore_data(dataset, start, end, rows_and_columns=False):

    dataset_slice = dataset[start:end]

    for row in dataset_slice:

        print(row)

        print('\n')

    if rows_and_columns:

        print('Number of rows', len(dataset))

        print('Number of columns', len(dataset[0]))
print(android[10472])  # incorrect row

print('\n')

print(android_header)  # header

print('\n')

print(android[0])      # correct row
print(len(android))

del android[10472]  # 한 번만 실행하셔야 합니다.

print(len(android))
# 중복된 값도 존재

for app in android:

    name = app[0]

    if name == 'Instagram':

        print(app)
# 중복된 수의 앱 이름이 몇개나 있는지 찾아보자

duplicate_apps = []

unique_apps = []



for app in android:

    name = app[0]

    if name in unique_apps:

        duplicate_apps.append(name)

    unique_apps.append(name)

    

print(len(duplicate_apps))

print(duplicate_apps[:15])
# 앞서 봤듯이 리뷰 수만 차이가 있으므로 리뷰 수가 가장 큰 것만 남기고 나머지는 제거하자.

# 첫번째로, 리뷰 수가 가장 큰 딕셔너리 변수를 만들자.

reviews_max = {}



for app in android:

    name = app[0]

    n_reviews = float(app[3])

    

    if name in reviews_max and reviews_max[name] < n_reviews:

        reviews_max[name] = n_reviews

        

    elif name not in reviews_max:

        reviews_max[name] = n_reviews
# reviews_max 딕셔너리를 이용해 리뷰 수가 가장 큰 행을 뽑아서 android_clean에 넣어줍니다.

android_clean = []

already_added = []



for app in android:

    name = app[0]

    n_reviews = float(app[3])

    

    if (reviews_max[name] == n_reviews) and (name not in already_added):

        android_clean.append(app)

        already_added.append(name)
ord('a'), ord('가') # 예시
# 영어인지 판별해주는 함수 정의

def is_English(string):

    

    for character in string:

        if ord(character) > 127:

            return False

    

    return True

def is_English(string):

    non_ascii = 0

    

    for character in string:

        if ord(character) > 127:

            non_ascii += 1

    

    if non_ascii > 3:

        return False

    else:

        return True


android_english = []



for app in android_clean:

    name = app[0]

    if is_English(name):

        android_english.append(app)

        
android_final = []



for app in android_english:

    price = app[7]

    if price == '0':

        android_final.append(app)

        
# index = columns number



def freq_table(dataset, index):

    table = {}

    total = 0

    

    for row in dataset:

        total += 1

        value = row[index]

        if value in table:

            table[value] += 1

        else:

            table[value] = 1

    

    table_percentages = {}

    for key in table:

        percentage = (table[key] / total) * 100

        table_percentages[key] = percentage 

    

    return table_percentages
def display_table(dataset, index):

    table = freq_table(dataset, index)

    table_display = []

    for key in table:

        key_val_as_tuple = (table[key], key)

        table_display.append(key_val_as_tuple)

        

    table_sorted = sorted(table_display, reverse = True)

    for entry in table_sorted:

        print(entry[1], ':', entry[0])
# 가장 높은 카테고리 범주를 찾아봅시다.

display_table(android_final, 1) 
# 가장 높은 장르(genres)별로 찾아봅시다.

display_table(android_final, -4)
display_table(android_final, 5)
categories_android = freq_table(android_final, 1)

for category in categories_android:

    total = 0

    len_category = 0

    for app in android_final:

        category_app = app[1]

        if category_app == category:

            n_installs = app[5]

            n_installs = n_installs.replace(',', '')

            n_installs = n_installs.replace('+', '')

            total += float(n_installs)

            len_category +=1

    avg_n_installs = total / len_category

    print(category, ':', avg_n_installs)

        
for app in android_final:

    if app[1] == 'COMMUNICATION' and (app[5] == '1,000,000,000+'

                                      or app[5] == '500,000,000+'

                                      or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])
for app in android_final:

    if app[1] == 'VIDEO_PLAYERS' and (app[5] == '1,000,000,000+'

                                      or app[5] == '500,000,000+'

                                      or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])
for app in android_final:

    if app[1] == 'BOOKS_AND_REFERENCE':

        print(app[0], ':', app[5])
for app in android_final:

    if app[1] == 'BOOKS_AND_REFERENCE' and (app[5] == '1,000,000,000+'

                                            or app[5] == '500,000,000+'

                                            or app[5] == '100,000,000+'):

        print(app[0], ':', app[5])
for app in android_final:

    if app[1] == 'BOOKS_AND_REFERENCE' and (app[5] == '1,000,000+'

                                            or app[5] == '5,000,000+'

                                            or app[5] == '10,000,000+'

                                            or app[5] == '50,000,000+'):

        print(app[0], ':', app[5])