# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_19_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

open_line_list = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')



#confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

#deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

#recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

#line_list_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
# Format and extract items



indi_date_list, confirmed_list, deaths_list, recovered_list, current_infected_list = [],[],[],[],[]

date_list = []

savedMonth = 0

confirmed, deaths, recovered = 0, 0, 0

for i in range(len(covid_19_data)):

    date = covid_19_data['ObservationDate'][i]

    inner_date_list = date.split('/')

    year = inner_date_list[2]

    month = inner_date_list[0]

    month = int(month)

    day = inner_date_list[1]

    day = int(day)

    confirmed += covid_19_data['Confirmed'][i]

    deaths += covid_19_data['Deaths'][i]

    recovered += covid_19_data['Recovered'][i]

    if savedMonth == month:

        if day % 5 == 0:

            indiDate = str(day)

        else:

            indiDate = ''

    else:

        savedMonth = month

        indiDate = str(month)+' / '+str(day)

    if i == 0 or date_list[-1] != date:

        date_list.append(date) 

        indi_date_list.append(indiDate)

        

    if (i < len(covid_19_data) - 1 and covid_19_data['ObservationDate'][i + 1] != date) or i == len(covid_19_data) - 1:

        current_infected = confirmed - deaths - recovered

        confirmed_list.append(confirmed)

        deaths_list.append(deaths)

        recovered_list.append(recovered)

        current_infected_list.append(current_infected)

        confirmed, deaths, recovered = 0, 0, 0

        

# Display



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7), dpi=100)

plt.title('Global')

plt.xlabel('Date')

plt.ylabel('Number of people')

plt.plot(confirmed_list, label='Confirmed', color='purple')

plt.plot(deaths_list, label='Deaths', color='r')

plt.plot(recovered_list, label='Recovered', color='b')

plt.bar(range(len(indi_date_list)), current_infected_list, label='Active', color='y')

x = np.arange(len(indi_date_list))

plt.xticks(x[:], indi_date_list[:], rotation=-90)

plt.legend()

plt.grid(True)

plt.show()



print('{}/{}/{}'.format(year, month, day))

print('Confirmed:{}'.format(int(confirmed_list[-1])))

print('Deaths:{}'.format(int(deaths_list[-1])))

print('Recovered:{}'.format(int(recovered_list[-1])))

print('Active:{}'.format(int(current_infected_list[-1])))
# max dimension for regression analysis

max_dimension = 7



# predict days

predict_days = 180
# Format data



train_list = []

confirmed, deaths, recovered = 0, 0, 0

for i in range(len(covid_19_data)):

    date = covid_19_data['ObservationDate'][i]

    inner_date_list = date.split('/')

    y = inner_date_list[2]

    m = inner_date_list[0]

    d = inner_date_list[1]

    confirmed += int(covid_19_data['Confirmed'][i])

    deaths += int(covid_19_data['Deaths'][i])

    recovered += int(covid_19_data['Recovered'][i])

    if i == 0:

        saved_date = d

    if i == len(covid_19_data) - 1:

        next_d = ''

    else:

        next_date = covid_19_data['ObservationDate'][i + 1]

        next_date_list = next_date.split('/')

        next_y = next_date_list[2]

        next_m = next_date_list[0]

        next_d = next_date_list[1]

    if next_y != y or next_m != m or next_d != d:

        current_infected = confirmed - deaths - recovered

        inner_dic = {'date':date,

#                     'num_days':i,

                     'confirmed':confirmed,

                     'deaths':deaths,

                     'recovered':recovered,

                     'current_infected':current_infected}

        train_list.append(inner_dic)

        saved_date = date

        confirmed, deaths, recovered = 0, 0, 0



train_list = pd.DataFrame(train_list)

train_list
# Preprocessing



import datetime



last_date = train_list['date'][len(train_list) - 1]

inner_date_list = last_date.split('/')

year = int(inner_date_list[2])

month = int(inner_date_list[0])

day = int(inner_date_list[1])

dt_last = datetime.date(year, month, day)

add_date_list = []

for i in range(predict_days):

    new_date = dt_last + datetime.timedelta(days = i + 1)

    y, m, d = new_date.year, new_date.month, new_date.day

    if m < 10:

        m = '0' + str(m)

    if d < 10:

        d = '0' + str(d)

    new_date = str(m)+'/'+str(d)+'/'+str(y)

    add_date_list.append(new_date)

add_date_list = np.array(add_date_list)



# Nonlinear regression analysis



import matplotlib.dates as mdates



fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['confirmed'])

x1 = np.arange(len(x))



score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)



# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='y') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Confirmed')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))
fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['deaths'])

x1 = np.arange(len(x))





score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)





# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='y') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Deaths')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))
fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['recovered'])

x1 = np.arange(len(x))





score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)





# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='y') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Recovered')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))
#import matplotlib.dates as mdates



fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['current_infected'])

x1 = np.arange(len(x))





score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)





# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='y') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Active')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))
country_name_list = []

inner_name = ''

for i in range(len(covid_19_data)):

    inner_name = covid_19_data['Country/Region'][i]

    flg = 0

    for j in range(len(country_name_list)):

        if country_name_list[j] == inner_name:

            flg = 1

            break

    if flg == 0:

        country_name_list.append(inner_name)

country_name_list.sort()

for i in range(len(country_name_list)):

    print(country_name_list[i])
# Go to "Edit" 



# And Change here (For example... country_name = 'US')

country_name = 'Japan'



# And "Run All" (No accelerator needed)
# Cut out own country's data



covid_19_own = []

for i in range(len(covid_19_data)):

    if covid_19_data['Country/Region'][i] == country_name:

        covid_19_own.append(covid_19_data.loc[i])

covid_19_own = pd.DataFrame(covid_19_own).reset_index()

covid_19_own
# Format and extract items



indi_date_list, confirmed_list, deaths_list, recovered_list, current_infected_list = [],[],[],[],[]

date_list = []

savedMonth = 0

confirmed, deaths, recovered = 0, 0, 0

for i in range(len(covid_19_own)):

    date = covid_19_own['ObservationDate'][i]

    inner_date_list = date.split('/')

    year = inner_date_list[2]

    month = inner_date_list[0]

    month = int(month)

    day = inner_date_list[1]

    day = int(day)

    confirmed += covid_19_own['Confirmed'][i]

    deaths += covid_19_own['Deaths'][i]

    recovered += covid_19_own['Recovered'][i]

    if savedMonth == month:

        if day % 5 == 0:

            indiDate = str(day)

        else:

            indiDate = ''

    else:

        savedMonth = month

        indiDate = str(month)+' / '+str(day)

    if i == 0 or date_list[-1] != date:

        date_list.append(date) 

        indi_date_list.append(indiDate)

        

    if (i < len(covid_19_own) - 1 and covid_19_own['ObservationDate'][i + 1] != date) or i == len(covid_19_own) - 1:

        current_infected = confirmed - deaths - recovered

        confirmed_list.append(confirmed)

        deaths_list.append(deaths)

        recovered_list.append(recovered)

        current_infected_list.append(current_infected)

        confirmed, deaths, recovered = 0, 0, 0

#print(int(confirmed_list[-1]))

#print(int(deaths_list[-1]))

#print(int(recovered_list[-1]))
# Display



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7), dpi=100)

plt.title('Covid 19 infection status in ' + country_name)

plt.xlabel('Date')

plt.ylabel('Number of people')

plt.plot(confirmed_list, label='Confirmed')

plt.plot(deaths_list, label='Deaths')

plt.plot(recovered_list, label='Recovered')

plt.bar(range(len(indi_date_list)), current_infected_list, label='Active', color='pink')

x = np.arange(len(indi_date_list))

plt.xticks(x[:], indi_date_list[:], rotation=-90)

plt.legend()

plt.grid(True)

plt.show()



print('{}/{}/{}'.format(year, month, day))

print('Confirmed:{}'.format(int(confirmed_list[-1])))

print('Deaths:{}'.format(int(deaths_list[-1])))

print('Recovered:{}'.format(int(recovered_list[-1])))

print('Active:{}'.format(int(current_infected_list[-1])))
# Format data



train_list = []

confirmed, deaths, recovered = 0, 0, 0

for i in range(len(covid_19_own)):

    date = covid_19_own['ObservationDate'][i]

    inner_date_list = date.split('/')

    y = inner_date_list[2]

    m = inner_date_list[0]

    d = inner_date_list[1]

    confirmed += int(covid_19_own['Confirmed'][i])

    deaths += int(covid_19_own['Deaths'][i])

    recovered += int(covid_19_own['Recovered'][i])

    if i == 0:

        saved_date = d

    if i == len(covid_19_own) - 1:

        next_d = ''

    else:

        next_date = covid_19_own['ObservationDate'][i + 1]

        next_date_list = next_date.split('/')

        next_y = next_date_list[2]

        next_m = next_date_list[0]

        next_d = next_date_list[1]

    if next_y != y or next_m != m or next_d != d:

        current_infected = confirmed - deaths - recovered

        inner_dic = {'date':date,

#                     'num_days':i,

                     'confirmed':confirmed,

                     'deaths':deaths,

                     'recovered':recovered,

                     'current_infected':current_infected}

        train_list.append(inner_dic)

        saved_date = date

        confirmed, deaths, recovered = 0, 0, 0



train_list = pd.DataFrame(train_list)

train_list
# Preprocessing



last_date = train_list['date'][len(train_list) - 1]

inner_date_list = last_date.split('/')

year = int(inner_date_list[2])

month = int(inner_date_list[0])

day = int(inner_date_list[1])

dt_last = datetime.date(year, month, day)

add_date_list = []

for i in range(predict_days):

    new_date = dt_last + datetime.timedelta(days = i + 1)

    y, m, d = new_date.year, new_date.month, new_date.day

    if m < 10:

        m = '0' + str(m)

    if d < 10:

        d = '0' + str(d)

    new_date = str(m)+'/'+str(d)+'/'+str(y)

    add_date_list.append(new_date)

add_date_list = np.array(add_date_list)



# Nonlinear regression analysis



import matplotlib.dates as mdates



fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['confirmed'])

x1 = np.arange(len(x))



score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)



# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='pink') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Confirmed')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))


fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['deaths'])

x1 = np.arange(len(x))





score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)





# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='pink') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Deaths')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))


fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['recovered'])

x1 = np.arange(len(x))





score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)





# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='pink') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Recovered')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))


fig = plt.figure()

ax = fig.add_subplot(1,1,1)



x = np.array(train_list['date'])

y = np.array(train_list['current_infected'])

x1 = np.arange(len(x))





score_list = []

for j in range(1, max_dimension):

    #**********************************************************

    fit = np.polyfit(x1,y,j)

    #**********************************************************

    y2 = np.poly1d(fit)(x1)

    # r2_score

    from sklearn.metrics import r2_score

    score = r2_score(y,y2)

    score_list.append(score)

max_score = max(score_list)

max_index = score_list.index(max(score_list))

fit = np.polyfit(x1,y,max_index)

y2 = np.poly1d(fit)(x1)





# predict

temp_date = np.append(x, add_date_list)

x2 = x

predict_list = []

for i in range(len(x) - 1, len(temp_date)):

    predict_y = np.poly1d(fit)(i)

    if predict_y >= 0:

        x2 = np.append(x2, temp_date[i])

        predict_list.append(predict_y)

    else:

        break

predict_list = np.array(predict_list)

y3 = np.append(y2, predict_list)



ax.plot(x,y,'bo', color='pink') 

ax.plot(x2,y3,'--k', color='g') 



plt.title('Active')

plt.xlabel("Date")

plt.ylabel("Number of people")

plt.xticks(np.arange(0, len(x2), 10), rotation=-90)

plt.grid(True)

plt.tight_layout()

plt.show()



# Expected convergence date

if i >= len(x) + predict_days - 1:

    conv_date = 'Unknown'

else:

    conv_date = x2[-1]

print('Expected convergence date : {}'.format(conv_date))

print('Score:{:.4f}'.format(max_score))

print('Dimension:{}'.format(max_index))
open_list_own = open_line_list[open_line_list.country == country_name_open]

open_list_own
# Format data



country_name_open = 'Japan'



open_list_own = open_line_list[open_line_list.country == country_name_open]



reset_open_list = open_list_own.reset_index()

new_open_list = reset_open_list[['age', 'sex', 'province', 'outcome', 'date_death_or_discharge', 'additional_information']]



formatted_list = []

for i in range(len(new_open_list)):

    age = new_open_list['age'][i]

    if pd.isnull(age) == False:

        age = str(age)

        age_list = age.split('-')

        age = age_list[0]

        

        age = float(age)

        age = int(age)

        if age % 10 != 0:

            age = int(age / 10) *10

    sex = new_open_list['sex'][i]

    province = new_open_list['province'][i]

    province = str(province)

    if (' Prefecture' in province) == True:

        prefecture = province.strip(' Prefecture')

    else:

        prefecture = province

    outcome = new_open_list['outcome'][i]

    date_death_or_discharge = new_open_list['date_death_or_discharge'][i]

    additional_information = new_open_list['additional_information'][i]

    inner_dic = {'age':age, 'sex':sex, 'prefecture':prefecture, 'outcome':outcome,'outcome_date':date_death_or_discharge, 'additional_information':additional_information}

    formatted_list.append(inner_dic)

formatted_list = pd.DataFrame(formatted_list)

formatted_list
age_max = formatted_list['age'].max(skipna=True)

count = age_max / 10

age_dic = {}

for i in range(int(count + 1)):

    age_dic[(i * 10)] = 0

if pd.isnull(formatted_list['age'].max(skipna=False)):

    age_dic['Unknown'] = 0

for i in range(len(formatted_list)):

    if pd.isnull(formatted_list['age'][i]):

        inner_null_count = age_dic['Unknown']

        inner_null_count += 1

        age_dic['Unknown'] = inner_null_count

    else:

        age = formatted_list['age'][i]

        inner_count = age_dic[age]

        inner_count += 1

        age_dic[age] = inner_count



    

label_list = []

view_list = []

key_list = age_dic.keys()



for key in key_list:

    if type(key) == 'int':

        label_list.append(str(key) + 's')

    else:

        label_list.append(key)

    value = age_dic[key]

    view_list.append(value)

x = np.array(view_list)



plt.figure(figsize=(10, 7), dpi=100)

plt.title('By age')

plt.pie(x, labels=label_list, counterclock=False, startangle=90, shadow=True, labeldistance=1.05,

        wedgeprops={'linewidth': 1, 'edgecolor':"white"}, autopct="%1.1f%%", pctdistance=0.9)

plt.axis('equal')

plt.show()
sex_dic = {'male':0, 'female':0, 'Unknown':0}

for i in range(len(formatted_list)):

    sex = formatted_list['sex'][i]

    if sex == 'male':

        male_count = sex_dic['male']

        male_count += 1

        sex_dic['male'] = male_count

    elif sex == 'female':

        female_count = sex_dic['female']

        female_count += 1

        sex_dic['female'] = female_count

    else:

        Unknown_count = sex_dic['Unknown']

        Unknown_count += 1

        sex_dic['Unknown'] = Unknown_count



label_list = []

view_list = []

key_list = sex_dic.keys()



for key in key_list:

    label_list.append(key)

    value = sex_dic[key]

    view_list.append(value)

x = np.array(view_list)



plt.figure(figsize=(10, 7), dpi=100)

plt.title('By sex')

plt.pie(x, labels=label_list, counterclock=False, startangle=90,shadow=True, labeldistance=1.05,

        wedgeprops={'linewidth': 1, 'edgecolor':"white"}, autopct="%1.1f%%", pctdistance=0.6)

plt.axis('equal')

plt.show()
prefecture_list = []

for i in range(len(formatted_list)):

    prefecture = formatted_list['prefecture'][i]

    flg = 0

    for j in range(len(prefecture_list)):

        if prefecture_list[j]['prefecture'] == prefecture:

            target_pref_dic = prefecture_list[j]

            count = target_pref_dic['count']

            count += 1

            target_pref_dic['count'] = count

            prefecture_list[j] = target_pref_dic

            flg = 1

            break

    if flg == 0:

        target_pref_dic = {'prefecture':prefecture, 'count':1}

        prefecture_list.append(target_pref_dic)



label_list = []

view_list = []

for i in range(len(prefecture_list)):

    target_pref_dic = prefecture_list[i]

    prefecture = target_pref_dic['prefecture']

    count = target_pref_dic['count']

    label_list.append(prefecture)

    view_list.append(count)

x = np.array(view_list)        

plt.figure(figsize=(10, 7), dpi=100)

plt.title('By prefecture')

plt.pie(x, labels=label_list, counterclock=False, startangle=90, shadow=True, labeldistance=1.05,

        wedgeprops={'linewidth': 1, 'edgecolor':"white"}, autopct="%1.1f%%", pctdistance=0.8)

plt.axis('equal')

plt.show()
DP_list = formatted_list[formatted_list['additional_information'].str.contains('Diamond',na=False) == False]

DP_list = DP_list.reset_index()

prefecture_list = []

for i in range(len(DP_list)):

    prefecture = DP_list['prefecture'][i]

    flg = 0

    for j in range(len(prefecture_list)):

        if prefecture_list[j]['prefecture'] == prefecture:

            target_pref_dic = prefecture_list[j]

            count = target_pref_dic['count']

            count += 1

            target_pref_dic['count'] = count

            prefecture_list[j] = target_pref_dic

            flg = 1

            break

    if flg == 0:

        target_pref_dic = {'prefecture':prefecture, 'count':1}

        prefecture_list.append(target_pref_dic)



label_list = []

view_list = []

for i in range(len(prefecture_list)):

    target_pref_dic = prefecture_list[i]

    prefecture = target_pref_dic['prefecture']

    count = target_pref_dic['count']

    label_list.append(prefecture)

    view_list.append(count)

x = np.array(view_list)        

plt.figure(figsize=(10, 7), dpi=100)

plt.title('By prefecture excluding Diamond Princess')

plt.pie(x, labels=label_list, counterclock=False, startangle=90, shadow=True, labeldistance=1.05,

        wedgeprops={'linewidth': 1, 'edgecolor':"white"}, autopct="%1.1f%%", pctdistance=0.8)

plt.axis('equal')

plt.show()