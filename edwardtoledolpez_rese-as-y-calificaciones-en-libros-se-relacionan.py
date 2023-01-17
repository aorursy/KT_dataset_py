import scipy.stats

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
books = pd.read_csv('../input/bookscsv/books.csv')
books.head()
books.columns
def data_verifier(file):

    with open(books, mode="r", encoding='utf-8', newline='') as csv_file:

        csv_reader = csv.reader(csv_file)

        

        header = next(csv_reader)

        count = 0

        for row in csv_reader:

            try:

                last_row = row[12]

                if last_row == '':

                    last_row = row[13]

                print(f'{row}')

                print(f'LENs. title: {len(row[1])} authors: {len(row[2])} sup_extra: {len(row[3])} sup_rating: {len(row[4])}\n')

             

            except IndexError:



                pass

        

        csv_file.close()
import csv, os



books = '../input/goodreadsbooks/books.csv'

data_verifier(books)
import csv, os



def load_and_max(file, column_to_find):

    with open(books, mode="r", encoding='utf-8', newline='') as csv_file:

        csv_reader = csv.reader(csv_file)

        

        header = next(csv_reader)

        column_array = []

        max_value = 0



        for row in csv_reader:

            try:

                last_row = row[12]

                if last_row == '':

                    last_row = row[13]



                if column_to_find == 'average_rating': column_data = float(row[4])

                elif column_to_find == 'num_pages': column_data = int(row[8])

                elif column_to_find == 'ratings_count': column_data = int(row[9])

                elif column_to_find == 'text_reviews_count': column_data = int(row[10])

                else:

                    column_array = 'Data not Found. Check input'

                    max_value = 0

                    return column_array, max_value

            except IndexError:

                if column_to_find == 'average_rating': column_data = float(row[3])

                elif column_to_find == 'num_pages': column_data =  int(row[7])

                elif column_to_find == 'ratings_count': column_data = int(row[8])

                elif column_to_find == 'text_reviews_count': column_data = int(row[9])

                else:

                    column_array = 'Data not Found. Check input'

                    max_value = 0

                    return column_array, max_value



            column_array.append(column_data)

            if column_data > max_value:

                max_value = column_data



    csv_file.close()



    return column_array, max_value
def load_array(file, column_to_find):



    with open(books, mode="r", encoding='utf-8', newline='') as csv_file:

        csv_reader = csv.reader(csv_file)

        header = next(csv_reader)

        column_array = []



        for row in csv_reader:

            try:

                last_row = row[12]

                if last_row == '':

                    last_row = row[13]

                #Ordered followinf the fild

                #First Strings

                if column_to_find == 'title': column_data = row[1]

                elif column_to_find == 'authors': column_data = row[2] + ',' + row[3]

                #Float

                elif column_to_find == 'average_rating': column_data = float(row[4])

                #Strings hided as Int

                elif column_to_find == 'isbn': column_data = row[5]

                elif column_to_find == 'isbn13': column_data = row[6]

                #String AF

                elif column_to_find == 'language_code': column_data = row[7]

                elif column_to_find == '  num_pages': column_data = int(row[8])

                elif column_to_find == 'ratings_count': column_data = int(row[9])

                elif column_to_find == 'text_reviews_count': column_data = int(row[10])

                #Reason you import Datetime

                elif column_to_find == 'publication_date': column_data = datetime.strptime(row[11], '%m/%d/%Y')

                #String

                elif column_to_find == 'publisher': column_data = row[12]

                else:

                    column_data = 'Data not Found. Check input'

                    

                    return column_array



                    

            except IndexError:

                #Ordered followinf the fild

                #First Strings

                if column_to_find == 'title': column_data = row[1]

                elif column_to_find == 'authors': column_data = row[2]

                #Float

                elif column_to_find == 'average_rating': column_data = float(row[3])

                #Strings hided as Int

                elif column_to_find == 'isbn': column_data = row[4]

                elif column_to_find == 'isbn13': column_data = row[5]

                #String AF

                elif column_to_find == 'language_code': column_data = row[6]

                elif column_to_find == 'num_pages': column_data =  int(row[7])

                elif column_to_find == 'ratings_count': column_data = int(row[8])

                elif column_to_find == 'text_reviews_count': column_data = int(row[9])

                #Reason you import Datetime

                elif column_to_find == 'publication_date': column_data = datetime.strptime(row[10], '%m/%d/%Y')

                #String

                elif column_to_find == 'publisher': column_data = row[11]

                else:

                    column_data = 'Data not Found. Check input'

                    

                    return column_array



            column_array.append(column_data)

        



    csv_file.close()



    return column_array
books = '../input/goodreadsbooks/books.csv'



titles = load_array(books, 'title')

rating, max_rating = load_and_max(books, 'average_rating')

reviews, max_reviews = load_and_max(books, 'text_reviews_count')
print(f'Información disponible: Titulos: {len(titles)}, rating: {len(rating)}, reviews: {len(reviews)}')
books_dict = {'rating': rating,

             'reviews': reviews,

             'titles': titles}



books_df = pd.DataFrame(books_dict, columns=['rating','reviews','titles'])

books_df
y = books_df['rating']



fig, ax = plt.subplots()

ax.hist(y, bins = 500)

ax.set_xlabel('rating')

ax.set_ylabel('Frecuencia')



plt.axvline(np.mean(y)-np.std(y), c = 'k', linestyle = ':', label = '-1 desv. std.')

plt.axvline(np.mean(y), c = 'r', linestyle = '-', label = 'Promedio')

plt.axvline(np.mean(y)+np.std(y), c = 'k', linestyle = ':', label = '+1 desv. std.')

ax.legend()
y = books_df['rating']



fig, ax = plt.subplots()

ax.hist(y, bins = 100)

ax.set_xlabel('rating')

ax.set_ylabel('Frecuencia')



plt.axvline(np.mean(y)-np.std(y), c = 'k', linestyle = ':', label = '-1 desv. std.')

plt.axvline(np.mean(y), c = 'r', linestyle = '-', label = 'Promedio')

plt.axvline(np.mean(y)+np.std(y), c = 'k', linestyle = ':', label = '+1 desv. std.')

ax.legend()
y = books_df['rating']



fig, ax = plt.subplots()

ax.hist(y, bins = 10)

ax.set_xlabel('rating')

ax.set_ylabel('Frecuencia')



plt.axvline(np.mean(y)-np.std(y), c = 'k', linestyle = ':', label = '-1 desv. std.')

plt.axvline(np.mean(y), c = 'r', linestyle = '-', label = 'Promedio')

plt.axvline(np.mean(y)+np.std(y), c = 'k', linestyle = ':', label = '+1 desv. std.')

ax.legend()
def match_case_condition(array, value):

    match_one_condition = []

    for element_ub in range(len(array)):

        if array[element_ub] == value:

            match_one_condition.append(element_ub)



    return match_one_condition
maxium_books = match_case_condition(rating,5)



verified_rate = []

for ub in maxium_books:

    verified_rate.append(rating[ub])



n_reviews_of_titles = []

for ub in maxium_books:

    n_reviews_of_titles.append(reviews[ub])    

    

best_rated_titles = []

for ub in maxium_books:

    best_rated_titles.append(titles[ub])

    

best_rated_books_dict = {'rating': verified_rate,

             'reviews': n_reviews_of_titles,

             'titles': best_rated_titles}



best_rated_books_df = pd.DataFrame(best_rated_books_dict, columns=['rating','reviews','titles'])

best_rated_books_df
y = books_df['reviews']

fig, ax = plt.subplots()

ax.hist(y, bins = 10)

ax.set_xlabel('reviews')

ax.set_ylabel('Frecuencia')



plt.axvline(np.mean(y)-np.std(y), c = 'k', linestyle = ':', label = '-1 desv. std.')

plt.axvline(np.mean(y), c = 'r', linestyle = '-', label = 'Promedio')

plt.axvline(np.mean(y)+np.std(y), c = 'k', linestyle = ':', label = '+1 desv. std.')

ax.legend()
y = books_df['reviews']

fig, ax = plt.subplots()

ax.hist(y, bins = 100)

ax.set_xlabel('reviews')

ax.set_ylabel('Frecuencia')



plt.axvline(np.mean(y)-np.std(y), c = 'k', linestyle = ':', label = '-1 desv. std.')

plt.axvline(np.mean(y), c = 'r', linestyle = '-', label = 'Promedio')

plt.axvline(np.mean(y)+np.std(y), c = 'k', linestyle = ':', label = '+1 desv. std.')

ax.legend()
def merge_sort(array):

    if len(array) > 1:

        middle = len(array) // 2

        left = array[:middle]

        right = array[middle:]



        merge_sort(left)

        merge_sort(right)

        

        """SubArrays Iterators"""

        i = 0

        j = 0

        """MainArray Iterator"""

        k = 0



        while i < len(left) and j < len(right):

            if left[i] < right[j]:

                array[k] = left[i]

                i += 1

            else:

                array[k] = right[j]

                j += 1

            

            k += 1



        while i < len(left):

            array[k] = left[i]

            i += 1

            k += 1



        while j < len(right):

            array[k] = right[j]

            j += 1

            k += 1



    return array



def binary_search(array, start, end, search_value):

    if start > end:

        return end

    

    middle = (start + end) // 2



    if array[middle] == search_value:

        return middle

    elif array[middle] < search_value:

        return binary_search(array, middle + 1, end, search_value)

    else:

        return binary_search(array, start, middle - 1, search_value)



    

def top_condisioned(array, start_value):

    helper_array = sorted_set(array.copy())

    helpers_end = len(helper_array) - 1

    ubication =  binary_search(helper_array, 0, helpers_end, start_value)

    

    top_condisioned = []

    for i in range(ubication, helpers_end):

        top_condisioned.append(helper_array[i])



    return top_condisioned



def counted_array(elements_to_count, original_array):

    count_array = []

    for value in range(len(elements_to_count)):

        count_array.append(original_array.count(elements_to_count[value]))

    

    return count_array



def sorted_set(array):

    reduced_set = set(array)

    reduced_array = []

    for element in reduced_set:

        reduced_array.append(element)

    reduced_array = merge_sort(reduced_array)



    return reduced_array
topc_reviews = top_condisioned(reviews, 1000)

print(f'Cantidad de libros con más de 1,000 reviews: {len(topc_reviews)}')
topc_rating = top_condisioned(rating, 4.5)

print(f'Cantidad de calificaciones disponibles y obtenidas, que sean mayores o igual a 4.5 = {len(topc_rating)}; las cuales son: \n{topc_rating}')
def match_cases(array1, wanted_array1, array2, wanted_array2):

    """Notice de diference: the original data has to be of the samen lenght,

    BUT the match cases can differ. This means that I can have 3 match cases

    on one side and 50 in the other, and it won't make any trouble"""

    if len(array1)==len(array2):

        match_cases = []



        #Creating Support dictionaries

        wanted_cases_1 = {}

        for data in range(len(wanted_array1)):

            wanted_cases_1[wanted_array1[data]] = 1

        

        wanted_cases_2 = {}

        for data in range(len(wanted_array2)):

            wanted_cases_2[wanted_array2[data]] = 1

        

        for ub in range(len(array1)):

            val1 = array1[ub]

            val2 = array2[ub]

            if val1 in wanted_cases_1 and val2 in wanted_array2:

                match_cases.append(ub)

            

        return match_cases



    else: return 'Arrays must have the samen lenght'
match_cases = match_cases(rating, topc_rating, reviews, topc_reviews)
bv_rate = []

for ub in match_cases:

    bv_rate.append(rating[ub])



bv_reviews = []

for ub in match_cases:

    bv_reviews.append(reviews[ub])    

    

bv_titles = []

for ub in match_cases:

    bv_titles.append(titles[ub])

    

bv_books_dict = {'rating': bv_rate,

             'reviews': bv_reviews,

             'titles': bv_titles}



bv_books_df = pd.DataFrame(bv_books_dict, columns=['rating','reviews','titles'])

bv_books_df
bv_books_df.sort_values(by = 'reviews', ascending = False)
def topx(array, top_size):

    """Considerations.

    1) Array is ORDERED from minor to major.

    2) You can insert Arrays with duplicated values.

    3) You won't insert Sets."""



    sorted_and_reduced_array = sorted_set(array.copy())

    position = len(sorted_and_reduced_array) - 1

    

    topx = []

    value = sorted_and_reduced_array[position]

    topx.append(value)



    position -= 1

    max_value = value

    value = sorted_and_reduced_array[position]



    top_values = 1 



    while top_values < top_size and position >= 0 :

        if value != max_value:

            topx.append(sorted_and_reduced_array[position])

            max_value = value

            top_values += 1

        position -= 1

        value = sorted_and_reduced_array[position]



    return topx
top7_reviews = topx(reviews, 7)
def match_top_cases(array1, wanted_array1):

        match_cases = []



        #Creating Support dictionaries

        wanted_cases_1 = {}

        for data in range(len(wanted_array1)):

            wanted_cases_1[wanted_array1[data]] = 1

        

        for ub in range(len(array1)):

            val1 = array1[ub]

            if val1 in wanted_cases_1:

                match_cases.append(ub)

            

        return match_cases
ub_top7r = match_top_cases(reviews, top7_reviews)
def array_matches(match_array, array1, array2, array3):

    

    match_array1 = []

    for ub in match_array:

        match_array1.append(array1[ub])



    match_array2 = []

    for ub in match_array:

        match_array2.append(array2[ub])    



    match_array3 = []

    for ub in match_array:

        match_array3.append(array3[ub])

    

    return match_array1,match_array2,match_array3
t7rev_rating, t7rev_reviews, t7rev_titles = array_matches(ub_top7r, rating, reviews, titles)

    

top7_books_dict = {'rating': t7rev_rating,

             'reviews': t7rev_reviews,

             'titles': t7rev_titles}



top7_books_df = pd.DataFrame(top7_books_dict, columns=['rating','reviews','titles'])

top7_books_df.sort_values(by = 'reviews', ascending = False)
books_df
fig = books_df.plot(kind="scatter", x='rating', y='reviews', c='green')



x = books_df['rating']

plt.axvline(np.mean(x)-np.std(x), c = 'k', linestyle = ':', label = '-1 desv. std.')

plt.axvline(np.mean(x), c = 'r', linestyle = '-', label = 'Promedio')

plt.axvline(np.mean(x)+np.std(x), c = 'k', linestyle = ':', label = '+1 desv. std.')



y = books_df['reviews']

plt.axhline(np.mean(y)-np.std(y), c = 'c', linestyle = ':', label = 'Y: -1 desv. std.')

plt.axhline(np.mean(y), c = 'b', linestyle = '-', label = 'Promedio y')

plt.axhline(np.mean(y)+np.std(y), c = 'c', linestyle = ':', label = 'Y: +1 desv. std.')



fig.legend()