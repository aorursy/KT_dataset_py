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
data_csv = pd.read_csv("/kaggle/input/monthly-temperature-of-aomori-city/monthly_temperature_aomori_city.csv")

df = pd.DataFrame(data_csv)



df.info()
new_df = []

for i in range(len(df)):

    year = df['year'][i]

    month = df['month'][i]

    temp = df['temperature'][i]

    if month == 1:

        temp1 = temp

    elif month == 2:

        temp2 = temp

    elif month == 3:

        temp3 = temp

    elif month == 4:

        temp4 = temp

    elif month == 5:

        temp5 = temp

    elif month == 6:

        temp6 = temp

    elif month == 7:

        temp7 = temp

    elif month == 8:

        temp8 = temp

    elif month == 9:

        temp9 = temp

    elif month == 10:

        temp10 = temp

    elif month == 11:

        temp11 = temp

    elif month == 12:

        inner_dic = {

            '1':temp1,

            '2':temp2,

            '3':temp3,

            '4':temp4,

            '5':temp5,

            '6':temp6,

            '7':temp7,

            '8':temp8,

            '9':temp9,

            '10':temp10,

            '11':temp11,

            '12':temp

        }

        dic_y = {'Year':year, 'Monthly_average_temperature':inner_dic}

        new_df.append(dic_y)



new_df = pd.DataFrame(new_df)

new_df
import matplotlib.pyplot as plt

x = new_df['Year']

# monthly

m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9, m_10, m_11, m_12 = [],[],[],[],[],[],[],[],[],[],[],[]

# Quarter average

temp_Q1, temp_Q2, temp_Q3, temp_Q4 = [],[],[],[]

# yearly

temp_year = []

for i in range(len(x)):

    list_temp = new_df['Monthly_average_temperature']

    m1 = list_temp[i]['1']

    m2 = list_temp[i]['2']

    m3 = list_temp[i]['3']

    m4 = list_temp[i]['4']

    m5 = list_temp[i]['5']

    m6 = list_temp[i]['6']

    m7 = list_temp[i]['7']

    m8 = list_temp[i]['8']

    m9 = list_temp[i]['9']

    m10 = list_temp[i]['10']

    m11 = list_temp[i]['11']

    m12 = list_temp[i]['12']

    

    # Month

    m_1.append(m1)

    m_2.append(m2)

    m_3.append(m3)

    m_4.append(m4)

    m_5.append(m5)

    m_6.append(m6)

    m_7.append(m7)

    m_8.append(m8)

    m_9.append(m9)

    m_10.append(m10)

    m_11.append(m11)

    m_12.append(m12)

    

    # Quarter

    temp_mean_Q1 = (m1+m2+m3)/3

    temp_mean_Q2 = (m4+m5+m6)/3

    temp_mean_Q3 = (m7+m8+m9)/3

    temp_mean_Q4 = (m10+m11+m12)/3

    temp_Q1.append(temp_mean_Q1)

    temp_Q2.append(temp_mean_Q2)

    temp_Q3.append(temp_mean_Q3)

    temp_Q4.append(temp_mean_Q4)

    

    # Year

    temp_mean_year = (m1+m2+m3+m4+m5+m6+m7+m8+m9+m10+m11+m12)/12

    temp_year.append(temp_mean_year)

    

# 10 year moving average

# month

roll10_1 = pd.Series(m_1).rolling(window=10).mean()

roll10_2 = pd.Series(m_2).rolling(window=10).mean()

roll10_3 = pd.Series(m_3).rolling(window=10).mean()

roll10_4 = pd.Series(m_4).rolling(window=10).mean()

roll10_5 = pd.Series(m_5).rolling(window=10).mean()

roll10_6 = pd.Series(m_6).rolling(window=10).mean()

roll10_7 = pd.Series(m_7).rolling(window=10).mean()

roll10_8 = pd.Series(m_8).rolling(window=10).mean()

roll10_9 = pd.Series(m_9).rolling(window=10).mean()

roll10_10 = pd.Series(m_10).rolling(window=10).mean()

roll10_11 = pd.Series(m_11).rolling(window=10).mean()

roll10_12 = pd.Series(m_12).rolling(window=10).mean()

# Quarter

roll10_Q1 = pd.Series(temp_Q1).rolling(window=10).mean()

roll10_Q2 = pd.Series(temp_Q2).rolling(window=10).mean()

roll10_Q3 = pd.Series(temp_Q3).rolling(window=10).mean()

roll10_Q4 = pd.Series(temp_Q4).rolling(window=10).mean()



# Year

roll10_y = pd.Series(temp_year).rolling(window=10).mean()



# 20 year moving average

# Month

roll20_1 = pd.Series(m_1).rolling(window=20).mean()

roll20_2 = pd.Series(m_2).rolling(window=20).mean()

roll20_3 = pd.Series(m_3).rolling(window=20).mean()

roll20_4 = pd.Series(m_4).rolling(window=20).mean()

roll20_5 = pd.Series(m_5).rolling(window=20).mean()

roll20_6 = pd.Series(m_6).rolling(window=20).mean()

roll20_7 = pd.Series(m_7).rolling(window=20).mean()

roll20_8 = pd.Series(m_8).rolling(window=20).mean()

roll20_9 = pd.Series(m_9).rolling(window=20).mean()

roll20_10 = pd.Series(m_10).rolling(window=20).mean()

roll20_11 = pd.Series(m_11).rolling(window=20).mean()

roll20_12 = pd.Series(m_12).rolling(window=20).mean()

# Quarter

roll20_Q1 = pd.Series(temp_Q1).rolling(window=20).mean()

roll20_Q2 = pd.Series(temp_Q2).rolling(window=20).mean()

roll20_Q3 = pd.Series(temp_Q3).rolling(window=20).mean()

roll20_Q4 = pd.Series(temp_Q4).rolling(window=20).mean()

# Year

roll20_y = pd.Series(temp_year).rolling(window=20).mean()



plt.plot(x, m_1, label = 'Monthly')

plt.plot(x, roll10_1, label = '10-years moving average')

plt.plot(x, roll20_1, label = '20-years moving average')

plt.title('January')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_2, label = 'Monthly')

plt.plot(x, roll10_2, label = '10-years moving average')

plt.plot(x, roll20_2, label = '20-years moving average')

plt.title('February')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_3, label = 'Monthly')

plt.plot(x, roll10_3, label = '10-years moving average')

plt.plot(x, roll20_3, label = '20-years moving average')

plt.title('March')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_4, label = 'Monthly')

plt.plot(x, roll10_4, label = '10-years moving average')

plt.plot(x, roll20_4, label = '20-years moving average')

plt.title('April')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_5, label = 'Monthly')

plt.plot(x, roll10_5, label = '10-years moving average')

plt.plot(x, roll20_5, label = '20-years moving average')

plt.title('May')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_6, label = 'Monthly')

plt.plot(x, roll10_6, label = '10-years moving average')

plt.plot(x, roll20_6, label = '20-years moving average')

plt.title('June')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_7, label = 'Monthly')

plt.plot(x, roll10_7, label = '10-years moving average')

plt.plot(x, roll20_7, label = '20-years moving average')

plt.title('July')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_8, label = 'Monthly')

plt.plot(x, roll10_8, label = '10-years moving average')

plt.plot(x, roll20_8, label = '20-years moving average')

plt.title('August')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_9, label = 'Monthly')

plt.plot(x, roll10_9, label = '10-years moving average')

plt.plot(x, roll20_9, label = '20-years moving average')

plt.title('September')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_10, label = 'Monthly')

plt.plot(x, roll10_10, label = '10-years moving average')

plt.plot(x, roll20_10, label = '20-years moving average')

plt.title('October')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_11, label = 'Monthly')

plt.plot(x, roll10_11, label = '10-years moving average')

plt.plot(x, roll20_11, label = '20-years moving average')

plt.title('November')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, m_12, label = 'Monthly')

plt.plot(x, roll10_12, label = '10-years moving average')

plt.plot(x, roll20_12, label = '20-years moving average')

plt.title('December')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, temp_Q1, label = 'Quarter average trends')

plt.plot(x, roll10_Q1, label = '10-years moving average')

plt.plot(x, roll20_Q1, label = '20-years moving average')

plt.title('Q1(Jan,Feb,Mar)')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, temp_Q2, label = 'Quarter average trends')

plt.plot(x, roll10_Q2, label = '10-years moving average')

plt.plot(x, roll20_Q2, label = '20-years moving average')

plt.title('Q2(Apr,May,Jun)')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, temp_Q3, label = 'Quarter average trends')

plt.plot(x, roll10_Q3, label = '10-years moving average')

plt.plot(x, roll20_Q3, label = '20-years moving average')

plt.title('Q3(Jul,Aug,Sep)')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, temp_Q4, label = 'Quarter average trends')

plt.plot(x, roll10_Q4, label = '10-years moving average')

plt.plot(x, roll20_Q4, label = '20-years moving average')

plt.title('Q4(Oct,Nov,Dec)')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()



plt.plot(x, temp_year, label = 'Yearly average')

plt.plot(x, roll10_y, label = '10-years moving average')

plt.plot(x, roll20_y, label = '20-years moving average')

plt.title('Yearly average trends')

plt.ylabel('Temperature')

plt.xlabel('Year')

plt.legend()

plt.show()
