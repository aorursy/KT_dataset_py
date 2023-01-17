!pip install pytrends
# import pandas as pd

# from pytrends.request import TrendReq



# search_period = 'start_date end_date'



# pytrends = TrendReq(hl='en-US', tz=420)

# kw_list = ["depression", "anxiety", "panic attack", "insomnia", "loneliness", "covid"] #list of keywords



# ### Building the dataframe

# df = pd.DataFrame({})

# for id in kw_list:

#     pytrends.build_payload([id], geo='US', timeframe = search_period)

#     df[id] = pytrends.interest_over_time()[id]



# df.to_excel('file_name')
import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

from datetime import date

import datetime

import scipy.stats as st

import plotly.graph_objects as go

import statsmodels.api as sm

import statistics

from scipy import integrate

from pytrends.request import TrendReq
fname = 'https://raw.githubusercontent.com/IronicNinja/covid19api/master/5_year_period.xlsx'

df = pd.read_excel(fname)



df
### Plot the figure



kw_list = ["depression", "anxiety", "panic attack", "insomnia", "loneliness"] #Initialize keywords (without 'covid')

word_color_list = ['red', 'blue', 'orange', 'yellow', 'gray'] #Initialize colors



plt.figure(figsize=(20,10))

for pos in range(len(kw_list)):

    plt.plot(df['date'], df[kw_list[pos]], color = word_color_list[pos], linewidth=2)



plt.plot(df['date'], df['covid'], color = 'black')



plt.xlim(date(2015, 5, 31).toordinal(), date(2020, 5, 31).toordinal())

plt.legend(["depression", "anxiety", "panic attack", "insomnia", "loneliness", "covid"])

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - 5 year period')



plt.show()
df_5 = df.copy() #Make a copy of the dataframe



start_5 = date(2020, 5, 31)



### Converts the indices of the dataframe into dates - helps with the graphing. We will use this alot later

def change_axis_time(df, start):

    temp_list = []

    for time in range(len(df)):

        d0 = start-datetime.timedelta(days=7*time)

        d1 = d0.strftime("%Y-%m-%d")

        temp_list.append(d1)



    temp_list.sort()

    df.index = temp_list



change_axis_time(df_5, start_5)

df_5
plt.figure(figsize=(20,10))

for pos in range(len(kw_list)):

    fig = sm.tsa.seasonal_decompose(df_5[kw_list[pos]], period=26) #Period is every half year

    plt.plot(fig.trend, color=word_color_list[pos], linewidth=3)

    plt.plot(df[kw_list[pos]], color = word_color_list[pos], linewidth=0.5) #blur the original

    

legend_list = []

for keyword in kw_list:

    legend_list.append(keyword + ' trend')

    legend_list.append(keyword + ' org') #Org for original

    

plt.xticks([26*n for n in range(math.ceil(len(df_5)/26))])

plt.xlim(0, 261)

plt.ylim(-5, 105)



plt.legend(legend_list)

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - 5 year period, Trend')



plt.show()
### Getting the average search interest at each date



df['total'] = float(0)



### There will be a warning because we are overwritting a value in the DataFrame. 

pd.set_option('mode.chained_assignment', None) #Turns off warning - if you fork, probably turn this back on



for row in range(len(df)):

  c = 0

  for keyword in kw_list:

    c += df[keyword][row]

  df['total'][row] = (c/5)



df_5 = df.copy()

change_axis_time(df_5, start_5)
plt.figure(figsize=(20,10))



fig = sm.tsa.seasonal_decompose(df_5['total'], period=26)

plt.plot(fig.trend, color='red', linewidth=3)

plt.plot(df_5['total'], color='red', linewidth=0.5)

plt.plot(df_5['covid'], color = 'black', linewidth=0.4)



plt.xticks([26*n for n in range(math.ceil(len(df_5)/26))])



plt.xlim(0, 261)

plt.ylim(-5, 105)



plt.legend(['average trend', 'average org', 'covid'])

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - 5 year period, Average Trend')



plt.show()
### Linear Regression

x5 = [n for n in range(262)] #262 weeks over the 5 year period

y5 = [df_5['total'][row] for row in range(262)]

coef5 = np.polyfit(x5,y5,1)

fn5 = np.poly1d(coef5) 



### Figuring out statistics of linear regression

stat = st.linregress(x5, y5)
print('rvalue - %.4f' % stat[2])

print('pvalue - %.4f' % stat[3])

print('+%.4f relevance per year' % (stat[0]*52))



plt.figure(figsize=(20,10))

fig = sm.tsa.seasonal_decompose(df_5['total'], period=26)

plt.plot(fig.trend, color='red', linewidth=3)

plt.plot(df_5['total'], color='red', linewidth=0.5)

plt.plot(df_5['covid'], color = 'black', linewidth=0.4)

plt.plot(x5, fn5(x5), linestyle='solid', color='blue', linewidth=2)



plt.annotate('+%.4f relevance per week' % stat[0], (100, 85), fontsize=15)



plt.xticks([26*n for n in range(math.ceil(len(df_5)/26))])



plt.xlim(0, 261)

plt.ylim(-5, 105)



plt.legend(['total trend', 'total org', 'covid', 'line of best fit'])

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - 5 year period, Average Trend with Line of Best Fit')



plt.show()
fname = 'https://raw.githubusercontent.com/IronicNinja/covid19api/master/1_year_period.xlsx'

df1 = pd.read_excel(fname)



df1
start_1 = date(2020, 5, 31)



df_1 = df1.copy()

change_axis_time(df_1, start_1)



### Plot the figure



plt.figure(figsize=(20,10))



for pos in range(len(kw_list)):

    plt.plot(df_1[kw_list[pos]], color = word_color_list[pos], linewidth=2)



plt.plot(df_1['covid'], color = 'black')



plt.xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))]) #Every 2 months



plt.xlim(0, 52)

plt.legend(["depression", "anxiety", "panic attack", "insomnia", "loneliness", "covid"])

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - 1 year period')



plt.show()
### Getting the average search interest at each date



df1['total'] = float(0)

for row in range(len(df1)):

  c = 0

  for keyword in kw_list:

    c += df1[keyword][row]

  df1['total'][row] = (c/5)



df_1 = df1.copy()

change_axis_time(df_1, start_1)



### Plotting the figure



plt.figure(figsize=(20,10))

plt.plot(df_1['total'], color = 'red', linewidth=2)

plt.plot(df_1['covid'], color = 'black', linewidth=0.5)



plt.xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

plt.xlim(0, 52)



plt.legend(["total", "covid"])

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US, 1 year period - Average')



plt.show()
begin = date(2019, 6, 2)

plt.figure(figsize=(20,10))

plt.plot(df_1['total'], color = 'red', linewidth=2)

plt.plot(df_1['covid'], color = 'black', linewidth=0.5)



plt.legend(["total", "covid"], loc='upper left')

plt.xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

plt.xlim(0, 52)

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - Average, with events')



plt.annotate("Covid leadup", (13, 15), fontsize=20)

plt.annotate("Before peak", (34.5, 15), fontsize=20)

plt.annotate("After peak", (45, 15), fontsize=20)



plt.plot((date(2020, 2, 29)-begin).days/7, 86.1, 'yo', markersize=10)

plt.plot((date(2020, 3, 20)-begin).days/7, 87.3, 'yo', markersize=10)

plt.plot((date(2020, 5, 25)-begin).days/7, 75.2, 'yo', markersize=10)



plt.axvline(x = (date(2020, 1, 5)-begin).days/7, color = 'green') #WHO publishes a report on covid

plt.axvline(x = (date(2020, 3, 29)-begin).days/7, color = 'green') #Supposed peak of the virus popularity



plt.show()
### Linear Regression using numpy



start_date = date(2019, 6, 2)

point1 = int(((date(2020, 1, 5) - start_date).days)/7)

point2 = int(((date(2020, 3, 29) - start_date).days)/7)

point3 = int(((date(2020, 5, 31) - start_date).days)/7)



### Section 1

x1 = [n for n in range(point1+1)]

y1 = [df1['total'][row] for row in range(point1+1)]

coef1 = np.polyfit(x1,y1,1)

fn1 = np.poly1d(coef1) 



### Section 2

x2 = [n for n in range(point1, point2+1)]

y2 = [df1['total'][row] for row in range(point1, point2+1)]

coef2 = np.polyfit(x2,y2,1)

fn2 = np.poly1d(coef2) 



### Section 3

x3 = [n for n in range(point2, point3+1)]

y3 = [df1['total'][row] for row in range(point2, point3+1)]

coef3 = np.polyfit(x3,y3,1)

fn3 = np.poly1d(coef3) 



df_1 = df1.copy()

change_axis_time(df_1, start_1)
plt.figure(figsize=(20,10))

plt.plot(df_1['total'], color = 'red', linewidth=2)

plt.plot(df_1['covid'], color = 'black', linewidth=0.5)

plt.plot(x1, fn1(x1), linestyle='solid', color='blue', linewidth=1.5)

plt.plot(x2, fn2(x2), linestyle='solid', color='blue', linewidth=1.5)

plt.plot(x3, fn3(x3), linestyle='solid', color='blue', linewidth=1.5)



plt.xticks([8*n for n in range(math.ceil(len(df_1)/8))])

plt.legend(["average", "covid", "line of best fit"], loc='upper left')

plt.xlim(0, 52)

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - Average, with Labels')



plt.annotate("Covid leadup", (13, 15), fontsize=20)

plt.annotate("Before peak", (34.5, 15), fontsize=20)

plt.annotate("After peak", (45, 15), fontsize=20)



plt.axvline(x = point1, color = 'green') #WHO publishes a report on covid

plt.axvline(x = point2, color = 'green') #Supposed peak of the virus popularity



plt.show()
st_list = [st.linregress(x1, y1), st.linregress(x2, y2), st.linregress(x3, y3)]

headers_list = ['', 'slope', 'rvalue', 'pvalue']

values_list = [['Covid leadup', 'Before peak', 'After peak']]



for pos in range(5):

    if pos != 1 and pos != 4:

        tmp_list = []

        for line in st_list:

            tmp_list.append(round(line[pos], 4))

        values_list.append(tmp_list)

        

layout = go.Layout(

    title=go.layout.Title(

        text="Statistical Values for Average Over 1 year period",

        x=0.5

    ),

  margin=go.layout.Margin(

        l=0, #left margin

        r=50, #right margin

        b=0, #bottom margin

        t=40  #top margin

    ), 

  height = 130

)



fig = go.Figure(data=[go.Table(header=dict(values=headers_list), cells=dict(values=values_list))], layout = layout)



#How to add title/caption?



fig.show()
c = 0

fig, ax = plt.subplots(5, figsize=(15,30))

st_list2 = []



for keyword in kw_list:

    ### Section 1

    x1 = [n for n in range(point1+1)]

    y1 = [df_1[keyword][row] for row in range(point1+1)]

    coef1 = np.polyfit(x1,y1,1)

    fn1 = np.poly1d(coef1) 



    ### Section 2

    x2 = [n for n in range(point1, point2+1)]

    y2 = [df_1[keyword][row] for row in range(point1, point2+1)]

    coef2 = np.polyfit(x2,y2,1)

    fn2 = np.poly1d(coef2) 



    ### Section 3

    x3 = [n for n in range(point2, point3+1)]

    y3 = [df_1[keyword][row] for row in range(point2, point3+1)]

    coef3 = np.polyfit(x3,y3,1)

    fn3 = np.poly1d(coef3) 



    df1_tmp = df_1.copy()





    ### Plot Figure



    ax[c].plot(df_1[keyword], color = 'red', linewidth=2)

    ax[c].plot(df_1['covid'], color = 'black', linewidth=0.5)

    ax[c].plot(x1, fn1(x1), color='blue', linestyle='solid', linewidth=1.5)

    ax[c].plot(x2, fn2(x2), color='blue', linestyle='solid', linewidth=1.5)

    ax[c].plot(x3, fn3(x3), color='blue', linestyle='solid', linewidth=1.5)

    ax[c].legend([keyword, "covid"], loc='upper left')

    ax[c].title.set_text('Search Interest of Mental Health in the US, 1 year period with Linear Regressions - ' 

                         + keyword)

    ax[c].axvline(x = point1, color = 'green') #WHO publishes a report on covid

    ax[c].axvline(x = point2, color = 'green') #Supposed peak of the virus popularity

    ax[c].set_xlim([0, 52])

    ax[c].set_xlabel('Time')

    ax[c].set_ylabel('Search Interest')

    ax[c].set_xticks([8*n for n in range(math.ceil(len(df_1)/8))])

    ax[c].annotate("Covid leadup", (13, 15), fontsize=20)

    ax[c].annotate("Before peak", (34.5, 15), fontsize=20)

    ax[c].annotate("After peak", (45, 15), fontsize=20)



    st_list2.append([st.linregress(x1, y1), st.linregress(x2, y2), st.linregress(x3, y3)])

    c += 1



fig.tight_layout(pad=3)

    

fig.show()
headers_list = ['', 'slope', 'rvalue', 'pvalue']

hed_list = ['Covid leadup', 'Before peak', 'After peak']

fig_list = []

data_list = []



for pos in range(3):

    values_list = []

    index_list = []

    for keyword in kw_list:

        index_list.append(hed_list[pos] + ' - ' + keyword)

    values_list.append(index_list)



    for count in range(5):

        if count != 1 and count != 4:

            tmp_list = []

            for nested in st_list2:

                tmp_list.append(round(nested[pos][count], 4))

            values_list.append(tmp_list)

            

    ### table layout

    layout = go.Layout(

        title=go.layout.Title(

            text="Statistical Values of Each Keyword, %s" % hed_list[pos],

            x=0.5

        ),

          margin=go.layout.Margin(

                l=0, #left margin

                r=50, #right margin

                b=0, #bottom margin

                t=40  #top margin

            ), 

          height = 180

        )

    

    data_list.append(values_list)

    fig_list.append(go.Figure(data=[go.Table(header=dict(values=headers_list), cells=dict(values=values_list))], 

                              layout=layout))



for fig in fig_list:

    fig.show()
### r value analysis based off https://www.dummies.com/education/math/statistics/how-to-interpret-a-correlation-coefficient-r/



headers_list = ['', 'Covid leadup', 'Before peak', 'After peak']

values_list = [['depression', 'anxiety', 'panic attack', 'insomnia', 'loneliness']]

color_list = [['rgb(100,149,237)']]



for nested in data_list:

    tmp_list = []

    tmp_color_list = []

    for pos in range(5):

        ### p value, p < 0.05 else null

        if(nested[3][pos] < 0.05):

            ### r values

            correlation = ""

            color = ""

            if(nested[2][pos] >= 0.7):

                correlation = "Strongly Positive"

                color = '(124,252,0)'

            elif(nested[2][pos] >= 0.5):

                correlation = "Moderately Positive"

                color = '(154,205,50)'

            elif(nested[2][pos] >= 0.3):

                correlation = "Weakly Positive"

                color = '(189, 183, 107)'

            elif(nested[2][pos] > -0.3):

                correlation = "No Relationship"

                color = '(238, 232, 170)'

            elif(nested[2][pos] > -0.5):

                correlation = "Weakly Negative"

                color = '(255, 165, 0)'

            elif(nested[2][pos] > -0.7):

                correlation = "Moderately Negative"

                color = '(255, 140, 0)'

            else:

                correlation = "Strongly Negative"

                color = '(255, 69, 0)'

            tmp_list.append(correlation)

            tmp_color_list.append('rgb'+color)

        else:

            tmp_list.append('Null')

            tmp_color_list.append('rgb(47, 79, 79)')

    values_list.append(tmp_list)

    color_list.append(tmp_color_list)

    

layout = go.Layout(

        title = go.layout.Title(

            text = "Statistical Values of Each Keyword, Color Coded",

            x=0.5

        ),

          margin=go.layout.Margin(

                l=0, #left margin

                r=50, #right margin

                b=0, #bottom margin

                t=40  #top margin

            ), 

          height = 180

        )



trace = dict(header=dict(values=headers_list, 

                font = dict(color=['rgb(255,255,255)'], size=12),

             fill=dict(color='rgb(70,130,180)')),

        cells=dict(values=values_list,

                   font = dict(color=['rgb(255,255,255)'], size=12),

                    fill = dict(color=color_list)

                  )

            )





fig = go.Figure(data=[go.Table(trace)], layout=layout)

fig.show()
kw_list_avg = ["depression", "anxiety", "panic attack", "insomnia", "loneliness", "avg"]



fname = "https://raw.githubusercontent.com/IronicNinja/covid19api/master/states_info.xlsx"

df_master = pd.read_excel(fname)



states_list = {

'AL': 0, 'AK': 0, 'AZ': 0, 'AR': 0, 'CA': 0, 'CO': 0, 'CT': 0, 'DE': 0, 'FL': 0, 'GA': 0, 'HI': 0, 'ID': 0, 'IL': 0, 'IN': 0,

'IA': 0, 'KS': 0, 'KY': 0, 'LA': 0, 'ME': 0, 'MD': 0, 'MA': 0, 'MI': 0, 'MN': 0, 'MS': 0, 'MO': 0, 'MT': 0, 'NE': 0, 'NV': 0,

'NH': 0, 'NJ': 0, 'NM': 0, 'NY': 0, 'NC': 0, 'ND': 0, 'OH': 0, 'OK': 0, 'OR': 0, 'PA': 0, 'RI': 0, 'SC': 0, 'SD': 0, 'TN': 0,

'TX': 0, 'UT': 0, 'VT': 0, 'VA': 0, 'WA': 0, 'WV': 0, 'WI': 0, 'WY': 0

}



df_master.index = states_list #The indices of df_master are now state abbreviations instead of numbers
# state_df = {}



# for state in states_list:

#   df0 = pd.DataFrame({})

#   for id in kw_list:

#     pytrends.build_payload([id], geo='US' + '-' + state, timeframe = search_period)

#     df0[id] = pytrends.interest_over_time()[id]

#   state_df[state] = df0



# state_df
state_df = {}

for state in states_list:

    fname = "https://raw.githubusercontent.com/IronicNinja/covid19api/master/states/data_"+state+".xlsx"

    df_tmp = pd.read_excel(fname)

    state_df[state] = df_tmp
### Finding the average and appending it to the dataframe



for state in states_list:

    tmp_list = []

    for row in range(len(state_df[state])):

        c = 0

        for keyword in kw_list:

            c += state_df[state][keyword][row]

        tmp_list.append(c/5)

    state_df[state]['avg'] = tmp_list
df_statetotal = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



for state in states_list:

    for row in range(53):

        for keyword in kw_list:

            value = state_df[state][keyword][row]/50

            df_statetotal[keyword][row] += value

            df_statetotal['avg'][row] += (value/5)
change_axis_time(df_statetotal, start_1)

plt.figure(figsize=(20,10))



for pos in range(len(kw_list)):

    plt.plot(df_statetotal[kw_list[pos]], color=word_color_list[pos], linewidth=2)



plt.plot(df_statetotal['avg'], color='black', linewidth=2.5)



plt.xlim(0, 52)

plt.ylim(-5, 105)

plt.xticks([7.4*n for n in range(math.ceil(len(df_statetotal)/7.4))])

plt.legend(["depression", "anxiety", "panic attack", "insomnia", "loneliness", "average"])

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US, Average of All States')



plt.show()
avg_inc = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

med_inc = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

pov_inc = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



total_inc1 = 0

total_inc2 = 0

poverty_rate = 0

for state in states_list:

    inc1 = df_master['avg income'][state]

    inc2 = df_master['median income'][state]

    poverty = df_master['poverty'][state]

    total_inc1 += inc1

    total_inc2 += inc2

    poverty_rate += poverty

    for row in range(len(state_df[state])):

        for col in kw_list:

            relevance = state_df[state].iloc[row][col]

            v1 = relevance*inc1

            v2 = relevance*inc2

            v3 = relevance*poverty

            avg_inc.iloc[row][col] += v1

            avg_inc.iloc[row]['avg'] += v1/5

            med_inc.iloc[row][col] += v2

            med_inc.iloc[row]['avg'] += v2/5

            pov_inc.iloc[row][col] += v3

            pov_inc.iloc[row]['avg'] += v3/5 



### Normalize

for row in range(len(state_df[state])):

    for col in kw_list_avg:

        avg_inc.iloc[row][col] /= total_inc1

        med_inc.iloc[row][col] /= total_inc2

        pov_inc.iloc[row][col] /= poverty_rate
### Create subplots



fig, ax = plt.subplots(6, 2, figsize=(15,20))



def compareplots(med_inc, title, c):

    for pos in range(len(kw_list_avg)):

        ax[pos][c].plot(med_inc[kw_list_avg[pos]], color = 'red', linewidth=2)

        ax[pos][c].plot(df_statetotal[kw_list_avg[pos]], color = 'black', linewidth=2)

        ax[pos][c].legend(["median income", "average"], loc='upper left')

        ax[pos][c].title.set_text('%s vs Average Search Relevance - %s' % 

                                  (title, kw_list_avg[pos]))

        ax[pos][c].set_xlim([0, 52])

        ax[pos][c].set_xlabel('Time')

        ax[pos][c].set_ylabel('Search Interest')

        ax[pos][c].set_xticks([13*n for n in range(math.ceil(len(df_1)/13))])



compareplots(med_inc, 'Median Household Income', 0)

compareplots(pov_inc, 'Poverty Rate', 1)



fig.tight_layout(pad=3)

fig.show()
inc_list = [df_master['median income'][state] for state in states_list] #Initialize list of median incomes

inc_list.sort()



median_inc = statistics.median(inc_list) #Median of that list



def inc_conv(median_median):

    c = 0

    for states in states_list:

        inc = df_master['median income'][states]

        if inc > median_median:

            c += 1



    print(c) #count number of states above threshold



    high_inc = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

    low_inc = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



    high_rate = 0

    low_rate = 0

    for states in states_list:

        inc = df_master['median income'][states]

        if inc > median_median:

            high_rate += inc

        else:

            low_rate += inc

        for row in range(len(state_df[states])):

            for col in kw_list:

                relevance = state_df[states].iloc[row][col]

                if inc > median_median:

                    v1 = relevance*inc

                    high_inc.iloc[row][col] += v1

                    high_inc.iloc[row]['avg'] += (v1/5)

                else:

                    v2 = relevance*inc

                    low_inc.iloc[row][col] += v2

                    low_inc.iloc[row]['avg'] += (v2/5)



    change_axis_time(low_inc, start_1)

    change_axis_time(high_inc, start_1)

    

    for row in range(len(state_df[states])):

        for col in kw_list_avg:

            high_inc.iloc[row][col] /= high_rate

            low_inc.iloc[row][col] /= low_rate

            

    return high_inc, low_inc
high_inc, low_inc = inc_conv(median_inc)



def plot_diff(low_inc, high_inc, title, legend_list):

    fig, ax = plt.subplots(6, 2, figsize=(15,20))



    for pos in range(len(kw_list_avg)):

        ax[pos][0].plot(low_inc[kw_list_avg[pos]], color = 'red', linewidth=2)

        ax[pos][0].plot(high_inc[kw_list_avg[pos]], color = 'green', linewidth=2)

        ax[pos][0].legend(legend_list, loc='upper left')

        ax[pos][0].title.set_text('%s vs %s looking at %s - %s' % 

                                  (legend_list[0], legend_list[1], title, kw_list_avg[pos]))

        ax[pos][0].set_xlim([0, 52])

        ax[pos][0].set_xlabel('Time')

        ax[pos][0].set_ylabel('Search Interest')

        ax[pos][0].set_xticks([13*n for n in range(math.ceil(len(df_1)/13))])



        ### Percent Differences

        tmp_list = [] 

        avg_diff = 0

        for row in range(53):

            diff = (low_inc[kw_list_avg[pos]][row] - high_inc[kw_list_avg[pos]][row])

            tmp_list.append(diff)

            avg_diff += diff

        

        avg_diff /= 53

        

        ### Change the dates to match so it's graphable

        temp_list = []

        for time in range(len(low_inc)):

            d0 = start_1-datetime.timedelta(days=7*time)

            d1 = d0.strftime("%Y-%m-%d")

            temp_list.append(d1)



        temp_list.sort()



        X = [temp_list[n] for n in range(53)]

        ax[pos][1].bar(X, tmp_list)

        ax[pos][1].title.set_text('Difference between %s and %s - %s' % 

                                  (legend_list[0], legend_list[1], kw_list_avg[pos]))

        ax[pos][1].set_xlim([0, 52])

        ax[pos][1].set_xlabel('Time')

        ax[pos][1].set_ylabel('Difference')

        ax[pos][1].set_xticks([13*n for n in range(math.ceil(len(df_1)/13))])

        

        max_num = max(tmp_list)

        ax[pos][1].annotate('avg diff: %s%.2f' % ('+' if avg_diff>0 else '', avg_diff), (40, max_num-2))



    fig.tight_layout(pad=3)



    fig.show()

      

### Use a previous function to eliminate repeated code

plot_diff(low_inc, high_inc, 'Median Household Income', ["Low Income", "High Income"]) 
pov_list = [df_master['poverty'][state] for state in states_list] #Initialize poverty rates

pov_list.sort()



pov_median = statistics.median(pov_list)



def poverty_conv(pov_rate):

    c = 0

    for states in states_list:

        poverty = df_master['poverty'][states]

        if poverty > pov_rate:

            c += 1



    print(c) #Count number of states above threshold

    

    high_pov = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

    low_pov = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



    high_rate = 0

    low_rate = 0

    for states in states_list:

        poverty = df_master['poverty'][states]

        if poverty > pov_rate:

            high_rate += poverty

        else:

            low_rate += poverty

        for row in range(len(state_df[states])):

            for col in kw_list:

                relevance = state_df[states].iloc[row][col]

                if poverty > pov_rate:

                    v1 = relevance*poverty

                    high_pov.iloc[row][col] += v1

                    high_pov.iloc[row]['avg'] += (v1/5)

                else:

                    v2 = relevance*poverty

                    low_pov.iloc[row][col] += v2

                    low_pov.iloc[row]['avg'] += (v2/5)

    

    change_axis_time(high_pov, start_1)

    change_axis_time(low_pov, start_1)

    

    for row in range(len(state_df[states])):

        for col in kw_list_avg:

            high_pov.iloc[row][col] /= high_rate

            low_pov.iloc[row][col] /= low_rate

   

    return high_pov, low_pov
high_pov, low_pov = poverty_conv(pov_median)



plot_diff(high_pov, low_pov, 'Poverty Rate', ["High Poverty", "Low Poverty"])
white_df = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

other_df = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



white_per = 0

other_per = 0

for state in states_list:

    white = df_master['white'][state]

    hispanic = df_master['hispanic'][state]

    black = df_master['black'][state]

    asian = df_master['asian'][state]

    american_indian = df_master['american indian'][state]

    other = hispanic+black+asian+american_indian

    white_per += white

    other_per += other

    for row in range(len(state_df[state])):

        for col in kw_list:

            relevance = state_df[state].iloc[row][col]

            v1 = relevance*white

            v2 = relevance*other

            white_df.iloc[row][col] += v1

            white_df.iloc[row]['avg'] += v1/5

            other_df.iloc[row][col] += v2

            other_df.iloc[row]['avg'] += v2/5 



### Normalize

for row in range(len(state_df['AL'])):

    for col in kw_list_avg:

        white_df.iloc[row][col] /= white_per

        other_df.iloc[row][col] /= other_per
change_axis_time(white_df, start_1)

change_axis_time(other_df, start_1)



### Create subplots



fig, ax = plt.subplots(6, 2, figsize=(15,20))



for pos in range(len(kw_list_avg)):

    ax[pos][0].plot(white_df[kw_list_avg[pos]], color = 'grey', linewidth=2)

    ax[pos][0].plot(other_df[kw_list_avg[pos]], color = 'black', linewidth=2)

    ax[pos][0].legend(["white", "other"], loc='upper left')

    ax[pos][0].title.set_text('Comparing Mental Health Between Races - %s' % kw_list_avg[pos])

    ax[pos][0].set_xlim([0, 52])

    ax[pos][0].set_xlabel('Time')

    ax[pos][0].set_ylabel('Search Interest')

    ax[pos][0].set_xticks([13*n for n in range(math.ceil(len(df_1)/13))])



    ### Percent Differences

    tmp_list = []

    avg_diff = 0

    for row in range(53):

        diff = (other_df[kw_list_avg[pos]][row] - white_df[kw_list_avg[pos]][row])

        tmp_list.append(diff)

        avg_diff += diff

    

    avg_diff /= 53

    

    temp_list = []

    for time in range(len(low_inc)):

        d0 = start_1-datetime.timedelta(days=7*time)

        d1 = d0.strftime("%Y-%m-%d")

        temp_list.append(d1)



    temp_list.sort()

    

    X = [temp_list[n] for n in range(53)]

    ax[pos][1].bar(X, tmp_list)

    ax[pos][1].title.set_text('Differences between Other Races and White - %s' % kw_list_avg[pos])

    ax[pos][1].set_xlim([0, 52])

    ax[pos][1].set_xlabel('Time')

    ax[pos][1].set_ylabel('Difference')

    ax[pos][1].set_xticks([13*n for n in range(math.ceil(len(df_1)/13))])

    max_num = max(tmp_list)

    ax[pos][1].annotate('avg diff - %s%.2f' % ('+' if avg_diff>0 else '', avg_diff), (10, max_num-0.5))

    

fig.tight_layout(pad=3)



fig.show()
### We will use the two temporary lists from the above code for simplicity sake



plt.figure(figsize=(20,10))



### Section 1

x1 = [n for n in range(point1+1)]

y1 = [tmp_list[n] for n in range(point1+1)]

coef1 = np.polyfit(x1,y1,1)

fn1 = np.poly1d(coef1) 



### Section 2

x2 = [n for n in range(point1, point2+1)]

y2 = [tmp_list[n] for n in range(point1, point2+1)]

coef2 = np.polyfit(x2,y2,1)

fn2 = np.poly1d(coef2) 



### Section 3

x3 = [n for n in range(point2, point3+1)]

y3 = [tmp_list[n] for n in range(point2, point3+1)]

coef3 = np.polyfit(x3,y3,1)

fn3 = np.poly1d(coef3) 



### Plot Figure



plt.plot(temp_list, tmp_list, color = 'red', linewidth=2)

plt.plot(x1, fn1(x1), color='blue', linestyle='solid', linewidth=1.5)

plt.plot(x2, fn2(x2), color='blue', linestyle='solid', linewidth=1.5)

plt.plot(x3, fn3(x3), color='blue', linestyle='solid', linewidth=1.5)

plt.legend(["avg diff between races", "line of best fit"], loc='upper left')

plt.title('Differences between Races - avg')

plt.axvline(x = point1, color = 'green') #WHO publishes a report on covid

plt.axvline(x = point2, color = 'green') #Supposed peak of the virus popularity

plt.xlim(0, 52)

plt.ylim(bottom=0)

plt.xlabel('Time')

plt.ylabel('Difference')

plt.xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

plt.annotate("Covid leadup", (13, 1), fontsize=20)

plt.annotate("Before peak", (34.5, 1), fontsize=20)

plt.annotate("After peak", (45, 1), fontsize=20)



plt.show()
st_list = [st.linregress(x1, y1), st.linregress(x2, y2), st.linregress(x3, y3)]

headers_list = ['', 'slope', 'rvalue', 'pvalue']

values_list = [['Covid leadup', 'Before peak', 'After peak']]



for pos in range(5):

    if pos != 1 and pos != 4:

        tmp_list = []

        for line in st_list:

              tmp_list.append(round(line[pos], 4))

        values_list.append(tmp_list)

        

layout = go.Layout(

  margin=go.layout.Margin(

        l=0, #left margin

        r=50, #right margin

        b=0, #bottom margin

        t=0  #top margin

    ), 

  height = 100

)



fig = go.Figure(data=[go.Table(header=dict(values=headers_list), cells=dict(values=values_list))], layout = layout)



fig.show()
white_df_high = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

other_df_high = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

other_df_low = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

white_df_low = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



white_per_high = 0

white_per_low = 0

other_per_high = 0

other_per_low = 0

med_inc = statistics.median(inc_list)



for state in states_list:

    white = df_master['white'][state]

    hispanic = df_master['hispanic'][state]

    black = df_master['black'][state]

    asian = df_master['asian'][state]

    american_indian = df_master['american indian'][state]

    other = hispanic+black+asian+american_indian

    inc = df_master['median income'][state]

    if inc > med_inc:

        high_pov += poverty

        white_per_high += white

        other_per_high += other

    else:

        low_pov += poverty

        white_per_low += white

        other_per_low += other

        

    for row in range(len(state_df[state])):

        for col in kw_list:

            relevance = state_df[state].iloc[row][col]

            v1 = relevance*white

            v2 = relevance*other

            if inc > med_inc:

                white_df_high.iloc[row][col] += v1

                white_df_high.iloc[row]['avg'] += v1/5

                other_df_high.iloc[row][col] += v2

                other_df_high.iloc[row]['avg'] += v2/5 

            else:

                white_df_low.iloc[row][col] += v1

                white_df_low.iloc[row]['avg'] += v1/5

                other_df_low.iloc[row][col] += v2

                other_df_low.iloc[row]['avg'] += v2/5 

                

### Normalize

for row in range(len(state_df['AL'])):

    for col in kw_list_avg:

        white_df_high.iloc[row][col] /= (white_per_high)

        white_df_low.iloc[row][col] /= (white_per_low)

        other_df_high.iloc[row][col] /= (other_per_high)

        other_df_low.iloc[row][col] /= (other_per_low)
change_axis_time(white_df_high, start_1)

change_axis_time(white_df_low, start_1)

change_axis_time(other_df_high, start_1)

change_axis_time(other_df_low, start_1)



### Create subplots



fig, ax = plt.subplots(6, figsize=(15,30))



for pos in range(len(kw_list_avg)):

    ax[pos].plot(white_df_high[kw_list_avg[pos]], color = 'grey', linewidth=2)

    ax[pos].plot(white_df_low[kw_list_avg[pos]], color = 'green', linewidth=2)

    ax[pos].plot(other_df_high[kw_list_avg[pos]], color = 'black', linewidth=2)

    ax[pos].plot(other_df_low[kw_list_avg[pos]], color = 'yellow', linewidth=2)

    ax[pos].legend(["white, high poverty", "white, low poverty", "other, high poverty", "other, low poverty"], 

                   loc='upper left')

    ax[pos].title.set_text('Search Relevance Looking at both Race and Socio-economic Status - %s' % kw_list_avg[pos])

    ax[pos].set_xlim([0, 52])

    ax[pos].set_xlabel('Time')

    ax[pos].set_ylabel('Search Interest')

    ax[pos].set_xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

    

fig.tight_layout(pad=3)

fig.show()
white_df_high = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

other_df_high = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

other_df_low = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

white_df_low = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



white_per_high = 0

white_per_low = 0

other_per_high = 0

other_per_low = 0

pov_rate = 0.131



for state in states_list:

    white = df_master['white'][state]

    hispanic = df_master['hispanic'][state]

    black = df_master['black'][state]

    asian = df_master['asian'][state]

    american_indian = df_master['american indian'][state]

    other = hispanic+black+asian+american_indian

    poverty = df_master['poverty'][state]

    if poverty > pov_rate:

        white_per_high += white

        other_per_high += other

    else:

        white_per_low += white

        other_per_low += other

        

    for row in range(len(state_df[state])):

        for col in kw_list:

            relevance = state_df[state].iloc[row][col]

            v1 = relevance*white

            v2 = relevance*other

            if poverty > pov_rate:

                white_df_high.iloc[row][col] += v1

                white_df_high.iloc[row]['avg'] += v1/5

                other_df_high.iloc[row][col] += v2

                other_df_high.iloc[row]['avg'] += v2/5 

            else:

                white_df_low.iloc[row][col] += v1

                white_df_low.iloc[row]['avg'] += v1/5

                other_df_low.iloc[row][col] += v2

                other_df_low.iloc[row]['avg'] += v2/5 

                

### Normalize

for row in range(len(state_df['AL'])):

    for col in kw_list_avg:

        white_df_high.iloc[row][col] /= (white_per_high)

        white_df_low.iloc[row][col] /= (white_per_low)

        other_df_high.iloc[row][col] /= (other_per_high)

        other_df_low.iloc[row][col] /= (other_per_low)
change_axis_time(white_df_high, start_1)

change_axis_time(white_df_low, start_1)

change_axis_time(other_df_high, start_1)

change_axis_time(other_df_low, start_1)



### Create subplots



fig, ax = plt.subplots(6, figsize=(15,30))



for pos in range(len(kw_list_avg)):

    ax[pos].plot(white_df_high[kw_list_avg[pos]], color = 'grey', linewidth=2)

    ax[pos].plot(white_df_low[kw_list_avg[pos]], color = 'green', linewidth=2)

    ax[pos].plot(other_df_high[kw_list_avg[pos]], color = 'black', linewidth=2)

    ax[pos].plot(other_df_low[kw_list_avg[pos]], color = 'yellow', linewidth=2)

    ax[pos].legend(["white, high poverty", "white, low poverty", "other, high poverty", "other, low poverty"], 

                   loc='upper left')

    ax[pos].title.set_text('Search Interest of Mental Health in the US, Poverty vs Average - %s' % kw_list_avg[pos])

    ax[pos].set_xlim([0, 52])

    ax[pos].set_xlabel('Time')

    ax[pos].set_ylabel('Search Interest')

    ax[pos].set_xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

    

fig.tight_layout(pad=3)

fig.show()
kw_listextra = ["depression", "anxiety", "panic attack", "insomnia", "loneliness", "covid"]



### Plotting function which we initialize now    



def graph(repub_df, democrat_df):

    change_axis_time(repub_df, start_1)

    change_axis_time(democrat_df, start_1)



    ### republicans

    plt.figure(figsize=(20,10))

    for pos in range(len(kw_list)):

        plt.plot(repub_df[kw_list[pos]], color = word_color_list[pos], linewidth=2)



    plt.plot(repub_df['avg'], color = 'black', linewidth=2)

    

    ### list comprehension where each date is spaced out for every 8 weeks

    plt.xticks([8*n for n in range(math.ceil(len(repub_df)/8))])

    plt.legend(["depression", "anxiety", "panic attack", "insomnia", "loneliness", "average"])

    plt.xlabel('Time')

    plt.ylabel('Search Interest')

    plt.title('Search Interest of Mental Health in the US - Republicans')



    ### democrats

    plt.figure(figsize=(20,10))

    for pos in range(len(kw_list)):

        plt.plot(democrat_df[kw_list[pos]], color = word_color_list[pos], linewidth=2)

        

    plt.plot(democrat_df['avg'], color = 'black')



    plt.xticks([7.4*n for n in range(math.ceil(len(repub_df)/7.4))])

    plt.legend(["depression", "anxiety", "panic attack", "insomnia", "loneliness", "average"])

    plt.xlabel('Time')

    plt.ylabel('Search Interest')

    plt.title('Search Interest of Mental Health in the US - Democrats')
repub_df = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

democrat_df = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



repub_total = 0

democrat_total = 0

for state in states_list:

    population = df_master['population'][state]

    repub_p = df_master['repub'][state]

    democrat_p = df_master['democrat'][state]

    partisan_p = repub_p+democrat_p

    repub_total += population*repub_p

    democrat_total += population*democrat_p

    for row in range(len(state_df[state])):

        for col in kw_list:

            relevance = state_df[state].iloc[row][col]

            v1 = relevance*population*repub_p/partisan_p

            v2 = relevance*population*democrat_p/partisan_p

            repub_df.iloc[row][col] += v1

            repub_df.iloc[row]['avg'] += v1/5

            democrat_df.iloc[row][col] += v2

            democrat_df.iloc[row]['avg'] += v2/5



repub_dfc = repub_df.copy()

democrat_dfc = democrat_df.copy()



### Normalize the data by dividing it by the total population



for row in range(len(state_df['AL'])):

    for col in kw_list_avg:

        repub_dfc.iloc[row][col] *= (democrat_total/(democrat_total+repub_total)/100)

        democrat_dfc.iloc[row][col] *= (repub_total/(democrat_total+repub_total)/100)
def weighted_analysis(repub_dfc, democrat_dfc, algotype):

    c = 0

    fig, ax = plt.subplots(6, figsize=(15,30))



    for keyword in kw_list_avg:

        ax[c].plot(repub_dfc[keyword], color = 'red')

        ax[c].plot(democrat_dfc[keyword], color = 'blue')

        ax[c].legend(["republican", "democrat"], loc='upper left')

        ax[c].title.set_text('Search Interest of Mental Health in the US, Republican vs Democrat, %s - %s' % (algotype, keyword))

        ax[c].set_xlim([0, 52])

        ax[c].set_xlabel('Time')

        ax[c].set_ylabel('Search Interest')

        ax[c].set_xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])



        c += 1



    fig.show()

    

change_axis_time(repub_dfc, start_1)

change_axis_time(democrat_dfc, start_1)

weighted_analysis(repub_dfc, democrat_dfc, 'weighted')
### by political party, weighted by population and repub vs democrat percentage // weighted, black & white



repub_df2 = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

democrat_df2 = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



population_total = 0

repub_population = 0

democrat_population = 0

for state in states_list:

    population = df_master['population'][state]

    repub_p = df_master['repub'][state]

    democrat_p = df_master['democrat'][state]

    if repub_p > democrat_p:

        repub_population += population

    elif democrat_p > repub_p:

        democrat_population += population

    

    for row in range(len(state_df[state])):

        for col in kw_list:

            relevance = state_df[state].iloc[row][col]

            v1 = relevance*population

            ### conditional statement

            if repub_p > democrat_p:

                repub_df2.iloc[row][col] += v1

                repub_df2.iloc[row]['avg'] += v1/5

            elif democrat_p > repub_p:

                democrat_df2.iloc[row][col] += v1

                democrat_df2.iloc[row]['avg'] += v1/5



### Normalizing the data as the population of democrat states is higher than of republican states



for row in range(len(state_df['AL'])):

    for col in kw_list_avg:

        repub_df2.iloc[row][col] /= repub_population

        democrat_df2.iloc[row][col] /= democrat_population



graph(repub_df2, democrat_df2)
weighted_analysis(repub_df2, democrat_df2, 'divided')
### by electoral college



repub_df3 = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

democrat_df3 = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)

ind_df3 = pd.DataFrame(float(0), index=np.arange(len(state_df['AL'])), columns=kw_list_avg)



repub_votes = 0

democrat_votes = 0

ind_votes = 0

population_total = 0

for state in states_list:

    votes = df_master['votes'][state]

    party = df_master['party'][state]

    if party == 'R':

        repub_votes += votes

    elif party == 'D':

        democrat_votes += votes

    else:

        ind_votes += votes

        

    for row in range(len(state_df[state])):

        for col in kw_list:

            relevance = state_df[state].iloc[row][col]

            v1 = relevance*votes

            if party == 'R':

                repub_df3.iloc[row][col] += v1

                repub_df3.iloc[row]['avg'] += v1/5

            elif party == 'D':

                democrat_df3.iloc[row][col] += v1

                democrat_df3.iloc[row]['avg'] += v1/5

            else:

                ind_df3.iloc[row][col] += v1

                ind_df3.iloc[row]['avg'] += v1/5

                    

for row in range(len(state_df['AL'])):

    for col in kw_list_avg:

        repub_df3.iloc[row][col] /= repub_votes

        democrat_df3.iloc[row][col] /= democrat_votes

        ind_df3.iloc[row][col] /= ind_votes



change_axis_time(repub_df3, start_1)

change_axis_time(democrat_df3, start_1)

change_axis_time(ind_df3, start_1)
c = 0

fig, ax = plt.subplots(6, figsize=(15,30))



for keyword in kw_list_avg:

    ax[c].plot(repub_df3[keyword], color = 'red')

    ax[c].plot(democrat_df3[keyword], color = 'blue')

    ax[c].plot(ind_df3[keyword], color = 'green')

    ax[c].legend(["republican", "democrat", "neither"], loc='upper left')

    ax[c].title.set_text('Search Interest of Mental Health in the US, Republican vs Democrat, Electoral - ' + keyword)

    ax[c].set_xlim([0, 52])

    ax[c].set_xlabel('Time')

    ax[c].set_ylabel('Search Interest')

    ax[c].set_xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])



    c += 1



fig.show()
algo_list = []

def sumdiff(repub_dfc, democrat_dfc):

    tmp_list = []

    df_perc = pd.DataFrame(0, index=np.arange(len(state_df['AL'])), columns=kw_list)

    for row in range(len(repub_dfc)):

        c = 0

        for keyword in kw_list:

            diff = (democrat_dfc[keyword][row] - repub_dfc[keyword][row])

            df_perc[keyword][row] += diff

            c += diff

        tmp_list.append(c/5)

    df_perc['total'] = tmp_list

    change_axis_time(df_perc, start_1)

    algo_list.append(df_perc)



sumdiff(repub_dfc, democrat_dfc) #algo 1

sumdiff(repub_df2, democrat_df2) #algo 2

sumdiff(repub_df3, democrat_df3) #algo 3



key_list = [keyword for keyword in kw_list]

key_list.append('total')



c = 0

fig, ax = plt.subplots(5, figsize=(15,30))



for keyword in kw_list:

    comb = [algo_list[0][keyword][n]+algo_list[1][keyword][n]+algo_list[2][keyword][n] for n in range(53)]

    ax[c].plot(algo_list[0][keyword], color = 'orange')

    ax[c].plot(algo_list[1][keyword], color = 'blue')

    ax[c].plot(algo_list[2][keyword], color = 'green')

    ax[c].plot(comb, color='red', linewidth=2)

    ax[c].hlines(0, 0, 53)

    

    ax[c].legend(["weighted", "divided", "electoral", "combined"], loc='upper left')

    ax[c].title.set_text('Search Interest of Mental Health in the US, Difference - ' + keyword)

    ax[c].set_xlim([0, 52])

    ax[c].set_xlabel('Time')

    ax[c].set_ylabel('Difference Between Democrats and Republicans')

    ax[c].set_xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

    

    X = [n for n in range(53)]

    y_int = integrate.cumtrapz(comb, X, initial=0) #Integrate

    

    max_num = max(comb)

    ax[c].annotate('Avg Diff - +%.4f' % (y_int[52]/52), (8.5, max_num-5), fontsize=15)

    

    c += 1



fig.show()
total_list = pd.DataFrame({})



for row in range(53):

    tmp_list = [0, 0, 0, 0] #weighted divided electoral combined

    for keyword in kw_list:

        comb = [algo_list[0][keyword][n]+algo_list[1][keyword][n]+algo_list[2][keyword][n] for n in range(53)]

        for pos in range(3):

            tmp_list[pos] += (algo_list[pos][keyword][row]/5)

        tmp_list[3] += (comb[row]/5)

    total_list[row] = tmp_list



total_list = total_list.T

total_listc = total_list.copy()

change_axis_time(total_listc, start_1)
plt.figure(figsize=(20,10))

plt.plot(total_listc[0], color = 'orange')

plt.plot(total_listc[1], color = 'blue')

plt.plot(total_listc[2], color = 'green')

plt.plot(total_listc[3], color = 'red', linewidth=2)



plt.xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

plt.legend(["weighted", "divided", "electoral", "combined"], loc='upper left')

plt.hlines(0, 0, 53)



plt.xlim([0, 52])

plt.xlabel('Time')

plt.ylabel('Average Percent Difference')

plt.title('Search Interest of Mental Health in the US, Republicans vs Democrats, Total Difference')



    

X = [n for n in range(53)]

y_int = integrate.cumtrapz(total_listc[3], X, initial=0) #Integrate



plt.annotate('Avg Diff - +%.4f' % (y_int[52]/52), (7.5, 23), fontsize=15)



plt.show()
### Linear Regression using numpy



start_date = date(2019, 6, 2)

point1 = int(((date(2020, 1, 5) - start_date).days)/7)

point2 = int(((date(2020, 3, 29) - start_date).days)/7)

point3 = int(((date(2020, 5, 31) - start_date).days)/7)



### Section 1

x1 = [n for n in range(point1+1)]

y1 = [total_list[3][row] for row in range(point1+1)]

coef1 = np.polyfit(x1,y1,1)

fn1 = np.poly1d(coef1) 



### Section 2

x2 = [n for n in range(point1, point2+1)]

y2 = [total_list[3][row] for row in range(point1, point2+1)]

coef2 = np.polyfit(x2,y2,1)

fn2 = np.poly1d(coef2) 



### Section 3

x3 = [n for n in range(point2, point3+1)]

y3 = [total_list[3][row] for row in range(point2, point3+1)]

coef3 = np.polyfit(x3,y3,1)

fn3 = np.poly1d(coef3) 



total_listc2 = total_list.copy()



change_axis_time(total_listc2, start_1)
plt.figure(figsize=(20,10))

plt.plot(total_listc2[3], color = 'red', linewidth=2)

plt.plot(x1, fn1(x1), linestyle='solid', color='blue', linewidth=1.5)

plt.plot(x2, fn2(x2), linestyle='solid', color='blue', linewidth=1.5)

plt.plot(x3, fn3(x3), linestyle='solid', color='blue', linewidth=1.5)



plt.xticks([8*n for n in range(math.ceil(len(df_1)/8))])

plt.legend(["total", "line of best fit"], loc='upper left')

plt.xlim(0, 52)

plt.xlabel('Time')

plt.ylabel('Percentage Difference')

plt.title('Search Interest of Mental Health in the US - Democrat vs Republican, Linear Regressions')



plt.annotate("Covid leadup", (13, 2), fontsize=20)

plt.annotate("Before peak", (34.5, 2), fontsize=20)

plt.annotate("After peak", (45, 2), fontsize=20)



plt.axvline(x = point1, color = 'green') #WHO publishes a report on covid

plt.axvline(x = point2, color = 'green') #Supposed peak of the virus popularity



plt.show()
st_list = [st.linregress(x1, y1), st.linregress(x2, y2), st.linregress(x3, y3)]

headers_list = ['', 'slope', 'rvalue', 'pvalue']

values_list = [['Covid leadup', 'Before peak', 'After peak']]



for pos in range(5):

    if pos != 1 and pos != 4:

        tmp_list = []

        for line in st_list:

            tmp_list.append(round(line[pos], 4))

        values_list.append(tmp_list)

        

layout = go.Layout(

  margin=go.layout.Margin(

        l=0, #left margin

        r=50, #right margin

        b=0, #bottom margin

        t=0  #top margin

    ), 

  height = 100

)



fig = go.Figure(data=[go.Table(header=dict(values=headers_list), cells=dict(values=values_list))], layout = layout)



fig.show()
state_df1 = {}

for state in states_list:

    fname = "https://raw.githubusercontent.com/IronicNinja/covid19api/master/states_covid/data_"+state+".xlsx"

    df_tmp = pd.read_excel(fname)

    state_df1[state] = df_tmp



fname = "https://raw.githubusercontent.com/IronicNinja/covid19api/master/state_covid19.xlsx"

df2 = pd.read_excel(fname)

start_2 = date(2020, 7, 5)





days = 8 #Start on the first sunday, which has an ID of 8

row = 0

c = 0



day_df = pd.DataFrame({})

week_df = pd.DataFrame({})

tmp_list_week = {

    "depression": 0, "anxiety": 0, "panic attack": 0, "insomnia": 0, "loneliness": 0, "avg": 0

}

### Search up to the week of 7/5

while days < 170:

    if (days-1)%7 == 0 and days != 8:

        s = pd.Series(tmp_list_week)

        week_df[(days-8)/7] = s

        for keyword in tmp_list_week:

            tmp_list_week[keyword] = 0

            

    tmp_list = {

        "depression": 0, "anxiety": 0, "panic attack": 0, "insomnia": 0, "loneliness": 0, "avg": 0

    }

    

    total_population = 0

    ### There's a row I added which is 'total', if you delete that row then just use len(df2) for the range

    for row in range(len(df2)-1):

        population_tmp = df2.iloc[row][days]

        total_population += population_tmp

        for keyword in kw_list:

            v1 = (state_df1[df2.iloc[row]['states']][keyword][days-8])*population_tmp

            tmp_list[keyword] += v1

            tmp_list['avg'] += v1/5

    for keyword in tmp_list:

        tmp_list[keyword] /= total_population

        tmp_list_week[keyword] += tmp_list[keyword]/7

        

    s = pd.Series(tmp_list)

    day_df[days-8] = s

    days += 1





### Transpose the week_df dataframe so the dates are the rows



df3 = day_df.T



def change_days(df3, start):

    temp_list = []

    for time in range(len(df3)):

        d0 = start-datetime.timedelta(days=time)

        d1 = d0.strftime("%Y-%m-%d")

        temp_list.append(d1)



    temp_list.sort()

    df3.index = temp_list



change_days(df3, start_2)
df_state_covid = pd.DataFrame(float(0), index=np.arange(len(state_df1['AL'])), columns=kw_list_avg)



for state in states_list:

    for row in range(len(state_df1[state])):

        for keyword in kw_list:

            v1 = (state_df1[state][keyword][row]/50)

            df_state_covid[keyword][row] += v1

            df_state_covid['avg'][row] += v1/5



change_days(df_state_covid, start_2)
fig, ax = plt.subplots(6, 2, figsize=(15,30))



c = 0

for keyword in kw_list_avg:

    fig1 = sm.tsa.seasonal_decompose(df3[keyword], period=7) #Period is weekly

    fig2 = sm.tsa.seasonal_decompose(df_state_covid[keyword], period=7)

    ax[c][0].plot(fig1.trend, color='red', linewidth=3)

    ax[c][0].plot(df3[keyword], color = 'red', linewidth=0.5)

    ax[c][0].plot(fig2.trend, color='black', linewidth=3)

    ax[c][0].plot(df_state_covid[keyword], color = 'black', linewidth=0.5)

    ax[c][0].legend(["weighted with covid cases, trend", "covid org", "average trend", "average org"], loc='upper left')

    

    ax[c][0].title.set_text('Comparing COVID Cases with the Average - ' + keyword)

    ax[c][0].set_xlim([0, 161])

    ax[c][0].set_ylim([0, 105])

    ax[c][0].set_xlabel('Time')

    ax[c][0].set_ylabel('Search Interest')

    ax[c][0].set_xticks([24*n for n in range(math.ceil(len(df3)/24))])

    

    tmp_list = [] 

    avg_diff = 0

    for row in range(len(df3)):

        diff = (df3[keyword][row] - df_state_covid[keyword][row])

        tmp_list.append(diff)

        avg_diff += diff



    avg_diff /= len(df3)

        

    temp_list = []

    for time in range(len(df3)):

        d0 = start_1-datetime.timedelta(days=7*time)

        d1 = d0.strftime("%Y-%m-%d")

        temp_list.append(d1)



    temp_list.sort()



    X = [temp_list[n] for n in range(len(df3))]

    ax[c][1].bar(X, tmp_list)

    ax[c][1].title.set_text('Difference between COVID Cases and Average - %s' % keyword)

    ax[c][1].set_xlim([0, 161])

    ax[c][1].set_xlabel('Time')

    ax[c][1].set_ylabel('Difference')

    ax[c][1].set_xticks([24*n for n in range(math.ceil(len(df3)/24))])



    max_num = max(tmp_list)

    ax[c][1].annotate('avg diff: %s%.2f' % ('+' if avg_diff>0 else '', avg_diff), (120, max_num-2))

    

    c += 1



fig.tight_layout(pad=3)

fig.show()
fname = 'https://raw.githubusercontent.com/IronicNinja/covid19api/master/worldwide.xlsx'

df_world = pd.read_excel(fname)



df_world
df_world['total'] = float(0)

for row in range(len(df_world)):

    c = 0

    for keyword in kw_list:

        c += df_world[keyword][row]

    df_world['total'][row] = (c/5)



change_axis_time(df_world, start_1)
plt.figure(figsize=(20,10))

plt.plot(df1['total'], color = 'red', linewidth=2)

plt.plot(df_world['total'], color = 'black', linewidth=2)



plt.xticks([7.4*n for n in range(math.ceil(len(df_1)/7.4))])

plt.legend(["US", "worldwide"], loc='upper left')

plt.xlim(0, 52)

plt.ylim(-5, 105)

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.title('Search Interest of Mental Health in the US - Average, with Labels')



plt.annotate("Covid leadup", (13, 15), fontsize=20)

plt.annotate("Before peak", (34.5, 15), fontsize=20)

plt.annotate("After peak", (45, 15), fontsize=20)



plt.axvline(x = point1, color = 'green') #WHO publishes a report on covid

plt.axvline(x = point2, color = 'green') #Supposed peak of the virus popularity



plt.show()
c = 0

fig, ax = plt.subplots(5, figsize=(15,30))



for keyword in kw_list:

    ax[c].plot(df1[keyword], color='red', linewidth=2)

    ax[c].plot(df_world[keyword], color='black', linewidth=2)

    ax[c].legend(["US", "worldwide"])

    ax[c].title.set_text("America vs the Rest of the World - %s" % keyword)

    ax[c].set_xlim([0, 52])

    ax[c].set_xlabel('Time')

    ax[c].set_ylabel('Search Interest')

    ax[c].set_xticks([7.4*n for n in range(math.ceil(len(df_world)/7.4))])

    

    c += 1

    

    

fig.show()
fname = 'https://raw.githubusercontent.com/IronicNinja/covid19api/master/treatment.xlsx'

treatment_df = pd.read_excel(fname)
kw_list_treatment = ["rehab", "meditation", "online therapy", "herbs", "online counseling"]

c = 0

fig, ax = plt.subplots(5, figsize=(15,30))

st_list2 = []



for keyword in kw_list_treatment:

    ### Section 1

    x1 = [n for n in range(point1+1)]

    y1 = [treatment_df[keyword][row] for row in range(point1+1)]

    coef1 = np.polyfit(x1,y1,1)

    fn1 = np.poly1d(coef1) 



    ### Section 2

    x2 = [n for n in range(point1, point2+1)]

    y2 = [treatment_df[keyword][row] for row in range(point1, point2+1)]

    coef2 = np.polyfit(x2,y2,1)

    fn2 = np.poly1d(coef2) 



    ### Section 3

    x3 = [n for n in range(point2, point3+1)]

    y3 = [treatment_df[keyword][row] for row in range(point2, point3+1)]

    coef3 = np.polyfit(x3,y3,1)

    fn3 = np.poly1d(coef3) 



    df1_tmp = df_1.copy()





    ### Plot Figure

    ax[c].plot(treatment_df[keyword], color = 'red', linewidth=2)

    ax[c].plot(df_1['covid'], color = 'black', linewidth=0.5)

    ax[c].plot(x1, fn1(x1), color='blue', linestyle='solid', linewidth=1.5)

    ax[c].plot(x2, fn2(x2), color='blue', linestyle='solid', linewidth=1.5)

    ax[c].plot(x3, fn3(x3), color='blue', linestyle='solid', linewidth=1.5)

    ax[c].legend([keyword, "covid"], loc='upper left')

    ax[c].title.set_text('Search Interest of Mental Health in the US, 1 year period - ' + keyword)

    ax[c].axvline(x = point1, color = 'green') #WHO publishes a report on covid

    ax[c].axvline(x = point2, color = 'green') #Supposed peak of the virus popularity

    ax[c].set_xlim([0, 52])

    ax[c].set_xlabel('Time')

    ax[c].set_ylabel('Search Interest')

    ax[c].set_xticks([7.4*n for n in range(math.ceil(len(treatment_df)/7.4))])

    ax[c].annotate("Covid leadup", (13, 15), fontsize=20)

    ax[c].annotate("Before peak", (34.5, 15), fontsize=20)

    ax[c].annotate("After peak", (45, 15), fontsize=20)



    st_list2.append([st.linregress(x1, y1), st.linregress(x2, y2), st.linregress(x3, y3)])

    c += 1



    fig.tight_layout(pad=3)



    fig.show()
headers_list = ['', 'slope', 'rvalue', 'pvalue']

hed_list = ['Covid leadup', 'Before peak', 'After peak']

fig_list = []

data_list = []



for pos in range(3):

    values_list = []

    index_list = []

    for keyword in kw_list:

        index_list.append(hed_list[pos] + ' - ' + keyword)

    values_list.append(index_list)



    for count in range(5):

        if count != 1 and count != 4:

            tmp_list = []

            for nested in st_list2:

                tmp_list.append(round(nested[pos][count], 4))

            values_list.append(tmp_list)

            

    ### table layout

    layout = go.Layout(

        title = go.layout.Title(

            text="Treatment Search Relevance - %s" % hed_list[pos], 

            x=0.5

        ),

          margin=go.layout.Margin(

                l=0, #left margin

                r=50, #right margin

                b=0, #bottom margin

                t=40  #top margin

            ), 

          height = 180

        )

        

    data_list.append(values_list)

    fig_list.append(go.Figure(data=[go.Table(header=dict(values=headers_list), cells=dict(values=values_list))], 

                              layout=layout))



for fig in fig_list:

    fig.show()
headers_list = ['', 'Covid leadup', 'Before peak', 'After peak']

values_list = [kw_list_treatment]

color_list = [['rgb(100,149,237)']]



for nested in data_list:

    tmp_list = []

    tmp_color_list = []

    for pos in range(5):

        ### p value, p < 0.05 else null

        if(nested[3][pos] < 0.05):

            ### r values

            correlation = ""

            color = ""

            if(nested[2][pos] >= 0.7):

                correlation = "Strongly Positive"

                color = '(124,252,0)'

            elif(nested[2][pos] >= 0.5):

                correlation = "Moderately Positive"

                color = '(154,205,50)'

            elif(nested[2][pos] >= 0.3):

                correlation = "Weakly Positive"

                color = '(189, 183, 107)'

            elif(nested[2][pos] > -0.3):

                correlation = "No Relationship"

                color = '(238, 232, 170)'

            elif(nested[2][pos] > -0.5):

                correlation = "Weakly Negative"

                color = '(255, 165, 0)'

            elif(nested[2][pos] > -0.7):

                correlation = "Moderately Negative"

                color = '(255, 140, 0)'

            else:

                correlation = "Strongly Negative"

                color = '(255, 69, 0)'

            tmp_list.append(correlation)

            tmp_color_list.append('rgb'+color)

        else:

            tmp_list.append('Null')

            tmp_color_list.append('rgb(47, 79, 79)')

    values_list.append(tmp_list)

    color_list.append(tmp_color_list)

    

layout = go.Layout(

        title=go.layout.Title(

            text="Treatment Relevance Color Coded",

            x=0.5

        ),

          margin=go.layout.Margin(

                l=0, #left margin

                r=50, #right margin

                b=0, #bottom margin

                t=0  #top margin

            ), 

          height = 150

        )



trace = dict(header=dict(values=headers_list, font = dict(color=['rgb(255,255,255)'], size=12),

             fill=dict(color='rgb(70,130,180)')),

        cells=dict(values=values_list,

                   font = dict(color=['rgb(255,255,255)'], size=12),

                    fill = dict(color=color_list)

                  )

            )





fig = go.Figure(data=[go.Table(trace)], layout=layout)

fig.show()
pytrends = TrendReq(hl='en-US', tz=420)

pytrends.build_payload(kw_list, geo='US', timeframe = '2019-6-2 2020-5-31')

df1 = pytrends.interest_over_time()

df2 = df1.drop(columns = ['isPartial'])



y = []



for keyword in kw_list:

  tmp_list = []

  for n in range(53): #Analyzing 53 weeks

    tmp_list.append(df2.iloc[n][keyword]/2.16) #March 29 relevance sum divided by 100

  y.append(tmp_list)



x = [date(2019, 6, 2)+datetime.timedelta(days=7*n) for n in range(53)]



plt.figure(figsize=(20,10))

plt.stackplot(x,y, labels=kw_list)

plt.plot(x, df_1['covid'], color='black')

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.legend(loc='upper left')

plt.xlim(date(2019, 6, 2), date(2020, 5, 31))

plt.title('Combined Search Interest of Mental Health in the US')



plt.show()
pytrends = TrendReq(hl='en-US', tz=420)

pytrends.build_payload(kw_list_treatment, geo='US', timeframe = '2019-6-2 2020-5-31')

df1 = pytrends.interest_over_time()

df2 = df1.drop(columns = ['isPartial'])



y = []



for keyword in kw_list_treatment:

    tmp_list = []

    for n in range(53): #Analyzing 53 weeks

        tmp_list.append(df2.iloc[n][keyword]/2.16) #March 29 relevance sum divided by 100

    y.append(tmp_list)



x = [date(2019, 6, 2)+datetime.timedelta(days=7*n) for n in range(53)]



plt.figure(figsize=(20,10))

plt.stackplot(x,y, labels=kw_list_treatment)

plt.plot(x, df_1['covid'], color='black')

plt.xlabel('Time')

plt.ylabel('Search Interest')

plt.legend(loc='upper left')

plt.xlim(date(2019, 6, 2), date(2020, 5, 31))

plt.title('Combined Search Interest of Mental Health Treatment in the US')



plt.show()