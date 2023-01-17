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
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
df_test
df_train
df_submission
# date_list(from 2020-01-22 untill2020-04-01)

work_list = []

first_date = df_train['Date'][0]

last_date = '2020-04-01'

inner_list = []

data_in_status = 0

for i in range(len(df_train)):

    date = df_train['Date'][i]

    if date == first_date:

        date_list = []

        data_in_status = 1

    if data_in_status == 1:

        province_state = df_train['Province_State'][i]

        country_region = df_train['Country_Region'][i]

        confirmed_cases = df_train['ConfirmedCases'][i]

        fatalities = df_train['Fatalities'][i]

        inner_dic = {'Province_State':province_state,

                     'Country_Region':country_region,

                     'Date':date,

                     'ConfirmedCases':confirmed_cases,

                     'Fatalities':fatalities

                    }

        inner_list.append(inner_dic)

        date_list.append(date)

        if date == last_date:

            work_list.append(inner_list)

            data_in_status = 0

            inner_list = []

np_date_list = np.array(date_list)

np_date_list
df_work_list = pd.DataFrame(work_list)

df_work_list
# Make add_date_list(from 2020-04-02 untill 2020-05-14)

add_date_list = []

for i in range(len(df_test['Date'])):

    date = df_test['Date'][i]

    add_date_list.append(date)

    if date == '2020-05-14':

        break

np_add_date_list = np.array(add_date_list)

np_add_date_list
# Analysys, Visualization, output CSV

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

forecast_id = 0

submission_list = []

test_list = []

for i in range(len(work_list)):

    country_list = work_list[i]

    if pd.isnull(country_list[0]['Province_State']):

        province_state = ''

    else:

        province_state = '(' + country_list[0]['Province_State'] + ')'

    country_region = country_list[0]['Country_Region']

    confirmed_list = []

    fatalities_list = []

    for j in range(len(country_list)):

        confirmed = country_list[j]['ConfirmedCases']

        confirmed_list.append(confirmed)

        fatalities = country_list[j]['Fatalities']

        fatalities_list.append(fatalities)

        

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)



    x = date_list

    y_c = np.array(confirmed_list)

    y_f = np.array(fatalities_list)

    x1 = np.arange(len(x))

    #**********************************************************

    # Determine dimensions

    score_list_c = []

    score_list_f = []

    for dimension in range(1, 7):

        fit_c = np.polyfit(x1, y_c, dimension)

        fit_f = np.polyfit(x1, y_f, dimension)

        y_c2 = np.poly1d(fit_c)(x1)

        y_f2 = np.poly1d(fit_f)(x1)



         # r2_score

        score_c = r2_score(y_c, y_c2)

        score_f = r2_score(y_f, y_f2)

        score_list_c.append(score_c)

        score_list_f.append(score_f)

    max_c = max(score_list_c)

    max_dimension_c = 1

    for k in range(len(score_list_c)):

        if score_list_c[k] == max_c:

            max_dimension_c = k

            break

    max_f = max(score_list_f)

    max_dimension_f = 1

    for k in range(len(score_list_f)):

        if score_list_f[k] == max_f:

            max_dimension_f = k

            break

    fit_c = np.polyfit(x1, y_c, max_dimension_c)

    fit_f = np.polyfit(x1, y_f, max_dimension_f)

    y_c2 = np.poly1d(fit_c)(x1)

    y_f2 = np.poly1d(fit_f)(x1)

    #**********************************************************



    # predict

    temp_date = np.append(x, add_date_list)

    x2 = x

    predict_list_c = []

    predict_list_f = []

    saved_predict_c = 0

    saved_predict_f = 0

    inner_count = 0

    for j in range(len(x), len(temp_date)):

        predict_c = np.poly1d(fit_c)(j)

        predict_f = np.poly1d(fit_f)(j)

        if predict_c < predict_f:

            predict_f = predict_c

        x2 = np.append(x2, temp_date[j])

        if predict_c > saved_predict_c:

            predict_list_c.append(predict_c)

            saved_predict_c = predict_c

        else:

            predict_list_c.append(saved_predict_c)

            

        if predict_f > saved_predict_f:

            predict_list_f.append(predict_f)

            saved_predict_f = predict_f

        else:

            predict_list_f.append(saved_predict_f)

            

        # for submission & display test data

        forecast_id += 1

        submission_dic = {'ForecastId':forecast_id,

                          'ConfirmedCases':saved_predict_c,

                          'Fatalities':saved_predict_f

                         }

        test_dic = {'ForecastId':forecast_id,

                    'ConfirmedCases':saved_predict_c,

                    'Fatalities':saved_predict_f,

                    'Date':np_add_date_list[inner_count],

                    'Province_State':province_state,

                    'Country_Region':country_region

                   }

        

        inner_count += 1

        submission_list.append(submission_dic)

        test_list.append(test_dic)

        

    predict_list_c = np.array(predict_list_c)

    predict_list_f = np.array(predict_list_f)

    y_c3 = np.append(y_c2, predict_list_c)

    y_f3 = np.append(y_f2, predict_list_f)



    ax.plot(x,y_c,'bo', color='y', label='Confirmed')

    ax.plot(x2,y_c3,'--k', color='g', label='Confirmed')

    ax.plot(x,y_f,'bo', color='pink', label='Fatalities')

    ax.plot(x2,y_f3,'--k', color='r', label='Fatalities')



    plt.title(country_region + province_state)

    plt.xlabel("Date")

    plt.ylabel("Number of people")

    plt.xticks(np.arange(0, len(x2), 10), rotation=-45)

    plt.grid(True)

    plt.tight_layout()

    plt.legend()

    plt.show()



   

    print('Score(Confirmed):{:.4f}'.format(score_c))

    print('Score(Fatalities):{:.4f}'.format(score_f))

    print('Dimension(Confirmed):{}'.format(max_dimension_c))

    print('Dimension(Fatalities):{}'.format(max_dimension_f))
my_submission_list = pd.DataFrame(submission_list)

my_submission_list.to_csv(path_or_buf='submission.csv', index=False)

df_test_list = pd.DataFrame(test_list)

df_test_list.to_csv(path_or_buf='test.csv', index=False)

df_test_list