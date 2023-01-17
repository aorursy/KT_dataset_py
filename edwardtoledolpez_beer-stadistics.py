import scipy.stats

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import csv, os

from datetime import datetime
cerveza_df_or = pd.read_csv('../input/beer-consumption-sao-paulo/Consumo_cerveja.csv')

cerveza_df_or
cerveza_df_or.dtypes
cerveza_df_or.columns
def csv_as_dictionary(file):

    with open(beer_csv, mode="r", encoding='utf-8', newline='') as csv_file:

        csv_reader = csv.reader(csv_file)

        header = next(csv_reader)

        

        column_data = ''

        data_date = []

        av_temp = []

        min_temp = []

        max_temp = []

        precipitation = []

        was_a_weekend = []

        beer_consumed = []

        

        for row in csv_reader:

            #No 'problem'

    

            if row[0]!= '': data_date.append(datetime.strptime(row[0], '%Y-%m-%d'))

                

            if row[5] != '': was_a_weekend.append(bool(int(row[5])))

            

            #Comma problem

            if row[1] != '': av_temp.append(float(row[1].replace(',','.')))

            if row[2] != '': min_temp.append(float(row[2].replace(',','.')))

            if row[3] != '': max_temp.append(float(row[3].replace(',','.')))

            if row[4] != '': precipitation.append(float(row[4].replace(',','.')))

            

            #Period Problem

            if row[6] != '': beer_consumed.append(int(row[6].replace('.','')))

            

    csv_file.close()



    dictionary = {

        'date': data_date,

        'av_temp': av_temp,

        'min_temp': min_temp,

        'max_temp': max_temp,

        'precipitation': precipitation,

        'was_a_weekend': was_a_weekend,

        'beer_consumed': beer_consumed

    }

    return dictionary
beer_csv = '../input/beer-consumption-sao-paulo/Consumo_cerveja.csv'

cerveza_dict = csv_as_dictionary(beer_csv)

cerveza_df = pd.DataFrame(cerveza_dict)

cerveza_df
cerveza_df.dtypes
plt.figure(figsize=(15,3))

plt.plot('date','max_temp', data=cerveza_df, color='r')

plt.show



plt.figure(figsize=(15,3))

plt.plot('date', 'av_temp', data=cerveza_df, color='k')

plt.show



plt.figure(figsize=(15,3))

plt.plot('date', 'min_temp', data=cerveza_df, color='b')

plt.show



plt.figure(figsize=(15,3))

plt.plot('date', 'precipitation', data=cerveza_df, color='darkblue')

plt.show



plt.figure(figsize=(15,3))

plt.plot('date', 'beer_consumed', data=cerveza_df, color='orange')

plt.show
plt.figure(figsize=(15,3))

plt.plot('date','max_temp', data=cerveza_df, color='r')

plt.plot('date', 'av_temp', data=cerveza_df, color='k')

plt.plot('date', 'min_temp', data=cerveza_df, color='b')



plt.axhline(np.mean(cerveza_df['min_temp']), c='cyan', linestyle='--', label = 'mean min_temp')

plt.axhline(np.mean(cerveza_df['av_temp']), c='gray', linestyle='--', label = 'mean av_temp')

plt.axhline(np.mean(cerveza_df['max_temp']), c='m', linestyle='--', label = 'mean max_temp')



plt.legend(loc='best', bbox_to_anchor=(1,1), ncol=1)



plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3), sharey=True, sharex=True)

n_bins = 40



ax1.hist(cerveza_df['min_temp'], align = 'mid', bins = n_bins)

ax1.set_xlabel('Min Temperatures (C)')

ax1.set_ylabel('Count')



ax2.hist(cerveza_df['av_temp'], align = 'mid', color='k', bins = n_bins)

ax2.set_xlabel('Average Temperatures (C)')



ax3.hist(cerveza_df['max_temp'], align = 'mid', color='r', bins = n_bins)

ax3.set_xlabel('Max Temperatures (C)')



plt.show()



kwargs = dict(alpha=0.5, bins = 40, density=True, stacked = True)



plt.figure(figsize=(12,6))



x1=cerveza_df['min_temp']

x2=cerveza_df['av_temp']

x3=cerveza_df['max_temp']



plt.gca().set(title='Histogram of temperatures (C)', ylabel='Count')



plt.axvline(np.mean(x1), c='darkblue', linestyle='--', label = 'mean min temperature')

plt.axvline(np.mean(x2), c='gray', linestyle='--', label = 'mean average temperature')

plt.axvline(np.mean(x3), c='darkred', linestyle='--', label = 'mean max temperature')



sns.distplot(x1, bins=40, color='darkblue')

sns.distplot(x2, bins=40, color='k')

sns.distplot(x3, bins=40, color='darkred')



plt.legend()
x=cerveza_df['beer_consumed']

y=cerveza_df['av_temp']

fig = cerveza_df.plot(kind="scatter", x = 'beer_consumed', y = 'av_temp',c = 'orange')



plt.axvline(np.mean(x)-np.std(x), c = 'r', linestyle = ':', label = '-1 desv. std. sells')

plt.axvline(np.mean(x), c = 'darkgreen', linestyle = '--', label = 'Sells average')

plt.axvline(np.mean(x)+np.std(x), c = 'g', linestyle = ':', label = '+1 desv. std. sells')



plt.axhline(np.mean(y)-np.std(y), c = 'b', linestyle = ':', label = '-1 desv. std. av_temp')

plt.axhline(np.mean(y), c = 'darkred', linestyle = '--', label = 'Mean av_temp')

plt.axhline(np.mean(y)+np.std(y), c = 'r', linestyle = ':', label = '+1 desv. std. av_temp')



plt.legend(loc='best', bbox_to_anchor=(1,1), ncol=2)



plt.show()
x=cerveza_df['beer_consumed']

y=cerveza_df['precipitation']

fig = cerveza_df.plot(kind="scatter", x = 'beer_consumed', y = 'precipitation',c = 'darkblue')



plt.axvline(np.mean(x)-np.std(x), c = 'r', linestyle = (0, (5, 2, 1, 2)), label = '-1 desv. std. sells')

plt.axvline(np.mean(x), c = 'grey', linestyle = '-', label = 'Sells average')

plt.axvline(np.mean(x)+np.std(x), c = 'g', linestyle = (0, (5, 2, 1, 2)), label = '+1 desv. std. sells')



plt.axhline(np.mean(y), c = 'y', linestyle = '--',linewidth=3 , label = 'Mean precipitation')

plt.axhline(np.mean(y)+np.std(y), c = 'b', linestyle = ':', label = '+1 desv. std. precipitation')



plt.legend(loc='best', bbox_to_anchor=(1,1), ncol=2)



plt.show()
def csv_as_arrays(file):

    with open(beer_csv, mode="r", encoding='utf-8', newline='') as csv_file:

        csv_reader = csv.reader(csv_file)

        header = next(csv_reader)

        

        column_data = ''

        data_date = []

        av_temp = []

        min_temp = []

        max_temp = []

        precipitation = []

        was_a_weekend = []

        beer_consumed = []

        

        for row in csv_reader:

            #No 'problem'

    

            if row[0]!= '': data_date.append(datetime.strptime(row[0], '%Y-%m-%d'))

                

            if row[5] != '': was_a_weekend.append(bool(int(row[5])))

            

            #Comma problem

            if row[1] != '': av_temp.append(float(row[1].replace(',','.')))

            if row[2] != '': min_temp.append(float(row[2].replace(',','.')))

            if row[3] != '': max_temp.append(float(row[3].replace(',','.')))

            if row[4] != '': precipitation.append(float(row[4].replace(',','.')))

            

            #Period Problem

            if row[6] != '': beer_consumed.append(int(row[6].replace('.','')))

            

    csv_file.close()



    return data_date, was_a_weekend, av_temp, min_temp, max_temp, precipitation, beer_consumed
data_date, was_a_weekend, av_temp, min_temp, max_temp, precipitation, beer_consumed = csv_as_arrays(beer_csv)
def ub_match_cases(array_filter, array):

  ub_match_cases = []



  #Creating Support dictionaries

  wanted_cases = set(array_filter)

  

  for ub in range(len(array)):

    val = array[ub]

    if val in wanted_cases: ub_match_cases.append(ub)

      

  return ub_match_cases



def top_condisioned(array, start_value):

  helper_array = sorted_set(array.copy())

  helpers_end = len(helper_array) - 1

  ubication =  binary_search(helper_array, 0, helpers_end, start_value)

  

  top_condisioned = []

  for i in range(ubication, helpers_end):

    top_condisioned.append(helper_array[i])



  return top_condisioned



def sorted_set(array):

  reduced_set = set(array)

  reduced_array = []

  for element in reduced_set:

    reduced_array.append(element)

  reduced_array = merge_sort(reduced_array)



  return reduced_array



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



def boolean_clasification(array):

  true_array = []

  false_array = []



  for i in range(len(array)):

    if array[i] == True:

      true_array.append(array[i])

    else: false_array.append(array[i])

  

  return true_array, false_array



def extract_matches(array_filter, array):

    match_cases = []



    #Creating Support dictionaries

    wanted_cases = set(array_filter)



    for ub in range(len(array)):

        val = array[ub]

        if val in wanted_cases: match_cases.append(array[ub])

    return match_cases



def all_major_cases(array, value):

    major_cases = []



    for ub in range(len(array)):

        if array[ub] >= value: major_cases.append(array[ub])



    return major_cases



def all_minor_cases(array, value):

    minor_cases = []



    for ub in range(len(array)):

        if array[ub] <= value: minor_cases.append(array[ub])



    return minor_cases
gta_sells = top_condisioned(beer_consumed,np.mean(beer_consumed))

ub_gta_sells = ub_match_cases(gta_sells, beer_consumed)

gta_av_temp = []

gta_week_day = []

gta_precipitation = []



for i in range(len(ub_gta_sells)):

  match = ub_gta_sells[i]

  gta_av_temp.append(av_temp[match])

  gta_week_day.append(was_a_weekend[match])

  gta_precipitation.append(precipitation[match])



gta_lens = f'''len of gta_av_temp: {len(gta_av_temp)}

len of gta_week_day: {len(gta_week_day)}

len of gta_precipitation: {len(gta_precipitation)}'''



print(gta_lens)
pls_std_best_sells = np.mean(beer_consumed)+np.std(beer_consumed)

best_sells_values = top_condisioned(beer_consumed,pls_std_best_sells)

ub_best_sells = ub_match_cases(best_sells_values, beer_consumed)



best_sells = []

best_av_temp = []

best_week_day = []

best_precipitation = []



for i in range(len(ub_best_sells)):

  match = ub_best_sells[i]

  best_av_temp.append(av_temp[match])

  best_week_day.append(was_a_weekend[match])

  best_precipitation.append(precipitation[match])

  best_sells.append(beer_consumed[match])



best_sells_lens = f'''best_av_temp: {len(best_av_temp)}

best_week_day: {len(best_week_day)}

best_precipitation: {len(best_precipitation)}

best_sells: {len(best_sells)}'''



print(best_sells_lens)
weekend_sells = [0,0]

week_sells = [0,0]



gta_weekend_sells, gta_middle_week_sells = boolean_clasification(gta_week_day)

weekend_sells[0], week_sells[0] = len(gta_weekend_sells), len(gta_middle_week_sells)



best_weekend_sells, best_middle_week_sells = boolean_clasification(best_week_day)

weekend_sells[1], week_sells[1] = len(best_weekend_sells), len(best_middle_week_sells)
labels = ['Mayor al Promedio', 'Mayor a Promedio + 1Desv. std.']

x = np.arange(len(labels))

width = 0.35



fig, ax = plt.subplots(figsize=(7,5))

rects1 = ax.bar(x - width/2, weekend_sells, width, label='Fin de Semana', color = '#43a047')

rects2 = ax.bar(x + width/2, week_sells, width, label='Entre semana', color = '#1f618d')



ax.set_title('Mejores ventas de Cerveza. Fin de Semana vs Entre semana')

ax.set_ylabel('Veces que las ventas superaron el promedio')

ax.set_xlabel('Promedio y mayor a una desviación estándar')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend(loc='best')



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 5 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()
def bool_and_major_matches(bool_array, bool_wanted, numeric_array, start_limit):

    if len(bool_array) == len(numeric_array):

        match_cases = []



        #Creating Support dictionaries

        

        for ub in range(len(bool_array)):

            bool_val = bool_array[ub]

            numeric_val = numeric_array[ub]

            if bool_val == bool_wanted and numeric_val >= start_limit:

                match_cases.append(ub)

            

        return match_cases

    else: pass



def bool_and_minor_matches(bool_array, bool_wanted, numeric_array, end_limit):

    if len(bool_array) == len(numeric_array):

        match_cases = []



        #Creating Support dictionaries

        

        for ub in range(len(bool_array)):

            bool_val = bool_array[ub]

            numeric_val = numeric_array[ub]

            if bool_val == bool_wanted and numeric_val <= end_limit:

                match_cases.append(ub)

            

        return match_cases

    else: pass



def amount_cases_bool_numeric(bool_array, numeric_array , minor_limit, increment):

    '''Returned data:

    1) amount_c1 = number of cases that are True and minor_limit + increment

    1) amount_c2 = number of cases that are True and minor_limit

    1) amount_c3 = number of cases that are False and minor_limit + increment

    1) amount_c4 = number of cases that are False and minor_limit'''

    

    if len(bool_array)==len(numeric_array):

        search_value = minor_limit + increment

        amount_cases = [0,0,0,0]



        amount_cases[0] = len(bool_and_major_matches(bool_array, True, numeric_array, search_value))

        amount_cases[1] = len(bool_and_minor_matches(bool_array, True, numeric_array, minor_limit))

        

        amount_cases[2] = len(bool_and_major_matches(bool_array, False, numeric_array, search_value))

        amount_cases[3] = len(bool_and_minor_matches(bool_array, False, numeric_array, minor_limit))



        return amount_cases

    else: pass
taw_best_sells = amount_cases_bool_numeric(best_week_day, best_av_temp, np.mean(av_temp), 0.01)

hot_bs_days = [0,0]

cold_bs_days = [0,0]



hot_bs_days[0],hot_bs_days[1] = taw_best_sells[0], taw_best_sells[2]

cold_bs_days[0],cold_bs_days[1] = taw_best_sells[1], taw_best_sells[3]
#Making of labels

taw_labels = [0,0,0,0]

taw_labels[0]= 'Fin de semana caluroso: ' + str(taw_best_sells[0])

taw_labels[1]= 'Fin de semana fresco/frío: ' + str(taw_best_sells[1])

taw_labels[2]= 'Entre semana, caluroso: ' + str(taw_best_sells[2])

taw_labels[3]= 'Entre semana, fresco/frío: ' + str(taw_best_sells[3])



#colors

pie_colors = ['#7b241c', '#2e86c1','#c0392b','#85c1e9']



#Creating plot

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(aspect="equal"))



wedges, texts = ax.pie(taw_best_sells, wedgeprops=dict(width=.3), startangle=-40, colors=pie_colors)



bbox_props = dict(boxstyle="square,pad=1.25", fc="w", ec="k", lw=1)

kw = dict(arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    

    ax.annotate(taw_labels[i], xy=(x, y), xytext=(1.2*np.sign(x), 1.1*y),

                horizontalalignment=horizontalalignment, **kw)



ax.set_title("Mejores ventas Considerando Temperatura y si era fin de semana")



plt.show()
category = []



for i in range(len(beer_consumed)):

    if beer_consumed[i] >= (np.mean(beer_consumed) + np.std(beer_consumed)):

        #Hot Cases

        if av_temp[i] >= np.mean(av_temp) and was_a_weekend[i] == True:

            category.append('Hot Weekend')

        elif av_temp[i] >= np.mean(av_temp) and was_a_weekend[i] == False:

            category.append('Hot and Not Weekend')



        #Cold Cases

        elif av_temp[i] <= np.mean(av_temp) and was_a_weekend[i] == True:

            category.append('Cold/Fresh Weekend')

        elif av_temp[i] <= np.mean(av_temp) and was_a_weekend[i] == False:

            category.append('Cold/Fresh and Not Weekend')

    else:

        category.append('N/A')



print(np.mean(av_temp))

cerveza_df.insert(7,'category',category)

cerveza_df.head()
x=cerveza_df['beer_consumed']

y=cerveza_df['av_temp']

groups=cerveza_df.groupby('category')

#[,N/A]

scatter_colors = ['#85c1e9','#2e86c1','#c0392b','#7b241c', 'orange']

from mlxtend.plotting import category_scatter



fig = category_scatter(x = 'beer_consumed', y = 'av_temp', label_col='category',

                       data=cerveza_df, colors=scatter_colors)



plt.axvline(np.mean(x), c = 'k', linestyle = '--', label = 'Sells average')

plt.axvline(np.mean(x)+np.std(x), c = 'g', linestyle = ':', label = '+1 desv. std. sells')



plt.axhline(np.mean(y), c = 'r', linestyle = '--', label = 'Mean av_temp')

plt.axhline(np.mean(y)+np.std(y), c = 'm', linestyle = ':', label = '+1 desv. std. sells')



plt.legend(loc='best', bbox_to_anchor=(1,1), ncol=2)



plt.show()
x=cerveza_df['beer_consumed']

y=cerveza_df['precipitation']

groups=cerveza_df.groupby('category')

#[,N/A]

scatter_colors = ['#85c1e9','#2e86c1','#c0392b','#7b241c', 'grey']

from mlxtend.plotting import category_scatter



fig = category_scatter(x = 'beer_consumed', y = 'precipitation', label_col='category',

                       data=cerveza_df, colors=scatter_colors)



plt.axvline(np.mean(x), c = 'k', linestyle = '--', label = 'Sells average')

plt.axvline(np.mean(x)+np.std(x), c = 'g', linestyle = ':', label = '+1 desv. std. sells')



plt.legend(loc='best', bbox_to_anchor=(1,1), ncol=2)



plt.show()