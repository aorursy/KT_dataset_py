import pandas as pd

testing=pd.read_csv('../input/covid-testing/covid-testing-all-observations.csv')

testing.columns

testing['Daily change in cumulative total per thousand']=testing['Daily change in cumulative total per thousand'].fillna(testing['Daily change in cumulative total per thousand'].mean())

testing.describe

import matplotlib.pyplot as plt

import matplotlib as mp

list=[]

tot=[]

test=[]

for x in range(len(testing['ISO code'])):

    if(testing['ISO code'][x]=='USA'):

        list.append(testing['Date'][x])

        test.append(testing['Daily change in cumulative total'][x])

        tot.append(testing['Cumulative total'][x])

for iter_num in range(len(list)-1,0,-1):

        for idx in range(iter_num):

            if list[idx]>list[idx+1]:

                temp1 = list[idx]

                list[idx] = list[idx+1]

                list[idx+1] = temp1

                temp2=test[idx]

                test[idx]=test[idx+1]

                test[idx+1]=temp2



usa=max(tot)

print(usa)

fig, ax = plt.subplots(figsize=(40,12))

ax.bar(list,test)

plt.title("Daily testing in USA")

plt.xticks(rotation=90)

plt.show()
import matplotlib.pyplot as plt

import matplotlib as mp

list=[]

test=[]

tot=[]

for x in range(len(testing['ISO code'])):

    if(testing['ISO code'][x]=='ITA'):

        list.append(testing['Date'][x])

        test.append(testing['Daily change in cumulative total'][x])

        tot.append(testing['Cumulative total'][x])



for iter_num in range(len(list)-1,0,-1):

        for idx in range(iter_num):

            if list[idx]>list[idx+1]:

                temp1 = list[idx]

                list[idx] = list[idx+1]

                list[idx+1] = temp1

                temp2=test[idx]

                test[idx]=test[idx+1]

                test[idx+1]=temp2

italy=max(tot)

print(italy)



fig, ax = plt.subplots(figsize=(25,12))

ax.bar(list,test,color='Red')

plt.title("Daily testing in ITALY")

plt.xticks(rotation=90)

plt.show()
import matplotlib.pyplot as plt

import matplotlib as mp

list=[]

test=[]

tot=[]

for x in range(len(testing['ISO code'])):

    if(testing['ISO code'][x]=='RUS'):

        list.append(testing['Date'][x])

        test.append(testing['Daily change in cumulative total'][x])

        tot.append(testing['Cumulative total'][x])



for iter_num in range(len(list)-1,0,-1):

        for idx in range(iter_num):

            if list[idx]>list[idx+1]:

                temp1 = list[idx]

                list[idx] = list[idx+1]

                list[idx+1] = temp1

                temp2=test[idx]

                test[idx]=test[idx+1]

                test[idx+1]=temp2

russia=max(tot)

print(russia)

fig, ax = plt.subplots(figsize=(25,12))

ax.bar(list,test,color='Green')

plt.title("Daily testing in RUSSIA")

plt.xticks(rotation=90)

plt.show()
import matplotlib.pyplot as plt

import matplotlib as mp

list=[]

test=[]

tot=[]

for x in range(len(testing['ISO code'])):

    if(testing['ISO code'][x]=='GBR'):

        list.append(testing['Date'][x])

        test.append(testing['Daily change in cumulative total'][x])

        tot.append(testing['Cumulative total'][x])



for iter_num in range(len(list)-1,0,-1):

        for idx in range(iter_num):

            if list[idx]>list[idx+1]:

                temp1 = list[idx]

                list[idx] = list[idx+1]

                list[idx+1] = temp1

                temp2=test[idx]

                test[idx]=test[idx+1]

                test[idx+1]=temp2

uk=max(tot)

fig, ax = plt.subplots(figsize=(25,12))

ax.bar(list,test,color='Gray')

plt.title("Daily testing in UK")

plt.xticks(rotation=90)

plt.show()
import matplotlib.pyplot as plt

import matplotlib as mp

list=[]

tot=[]

test=[]

for x in range(len(testing['ISO code'])):

    if(testing['ISO code'][x]=='AUS'):

        list.append(testing['Date'][x])

        test.append(testing['Daily change in cumulative total'][x])

        tot.append(testing['Cumulative total'][x])



for iter_num in range(len(list)-1,0,-1):

        for idx in range(iter_num):

            if list[idx]>list[idx+1]:

                temp1 = list[idx]

                list[idx] = list[idx+1]

                list[idx+1] = temp1

                temp2=test[idx]

                test[idx]=test[idx+1]

                test[idx+1]=temp2

aus=max(tot)

fig, ax = plt.subplots(figsize=(25,12))

ax.bar(list,test,color='Yellow')

plt.title("Daily testing in AUSTRALIS")

plt.xticks(rotation=90)

plt.show()
import matplotlib.pyplot as plt

import matplotlib as mp

list=[]

tot=[]

test=[]

for x in range(len(testing['ISO code'])):

    if(testing['ISO code'][x]=='IND'):

        list.append(testing['Date'][x])

        test.append(testing['Daily change in cumulative total'][x])

        tot.append(testing['Cumulative total'][x])

     

for iter_num in range(len(list)-1,0,-1):

        for idx in range(iter_num):

            if list[idx]>list[idx+1]:

                temp1 = list[idx]

                list[idx] = list[idx+1]

                list[idx+1] = temp1

                temp2=test[idx]

                test[idx]=test[idx+1]

                test[idx+1]=temp2

ind=max(tot)



fig, ax = plt.subplots(figsize=(25,12))

ax.bar(list,test,color='Orange')

plt.title("Daily testing in INDIA")

plt.xticks(rotation=90)

plt.show()
fig1, ax1 = plt.subplots(figsize=(20,10))

tested=[ind,russia,usa,uk,italy,aus]

names=['INDIA','RUSSIA','USA','UK','ITALY','AUSTRALIA']

ax1.pie([ind,russia,usa,uk,italy,aus], labels=['INDIA','RUSSIA','USA','UK','ITALY','AUSTRALIA'], autopct='%1.1f%%',

        shadow=True, startangle=90,frame=True,colors=["orange","Gray","blue","pink","red","yellow"])

ax1.set_title("COUNTRY WISE ANALYSIS OF COVID TESTING")



ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

code=[]

for x in testing['Entity']:

    if(code.count(x)>0):

        continue

    code.append(x)

resn=[]

rest=[]

for x in code:

    grouped = testing.groupby(testing.Entity)

    ya = grouped.get_group(x)

    resn.append(x)

    rest.append(max(ya['Cumulative total']))

    print((x+"\t"+(str)(max(ya['Cumulative total']))))



    



    

    

    
import plotly.express as px

fig = px.bar(x=resn, y=rest, color=rest, height=400,title="Testing done by various countries")



fig.show()
tested[2]=27317035.0

tested[4]=3057902.0

tested[3]=2144626.0

tested[0]=525667.0

print("Numebr of people tested"+(str)(tested))

print("Name    of     the country"+(str)(names))

import plotly.express as px

fig = px.bar(x=tested, y=names, color=tested, height=400,title="People tested by various countries")



fig.show()











population=pd.read_csv('../input/population/population-figures-by-country-csv_csv.csv')

population.columns

popl=[]

nas=[]

for x in range(len(population['Country'])):

    popl.append(population['Country'][x])

    nas.append((population['Year_2016'][x]))

    print(population['Country'][x]+"\t"+(str)(population['Year_2016'][x]))

    
cntry=[]

pcnt=[]

for x in range(len(population['Country_Code'])):

    chk=population['Country_Code'][x]

    if(chk=='IND'):

        cntry.append(chk)

        pcnt.append(population['Year_2016'][x])

    if(chk=='USA'):

        cntry.append(chk)

        pcnt.append(population['Year_2016'][x])

    if(chk=='ITA'):

        cntry.append(chk)

        pcnt.append(population['Year_2016'][x])

    if(chk=='RUS'):

        cntry.append(chk)

        pcnt.append(population['Year_2016'][x])

    if(chk=='GBR'):

        cntry.append(chk)

        pcnt.append(population['Year_2016'][x])

    if(chk=='AUS'):

        cntry.append(chk)

        pcnt.append(population['Year_2016'][x])





poplcnt=['INDIA', 'RUSSIA', 'USA', 'UK', 'ITALY', 'AUSTRALIA']

poplnum=[pcnt[2],pcnt[4],pcnt[5],pcnt[1],pcnt[3],pcnt[0]]

print(poplcnt)

print(poplnum)





        

        

   
import plotly.express as px

fig = px.bar(x=poplcnt, y=poplnum, color=poplnum, height=400,title="Population of Countires under our research")



fig.show()
import plotly.graph_objects as go







fig = go.Figure()

fig.add_trace(go.Bar(

    x=poplcnt,

    y=poplnum,

    name='Population Of the country',

    marker_color='yellow'

))

fig.add_trace(go.Bar(

    x=poplcnt,

    y=tested,

    name='Covid-19 People tested',

    marker_color='dark blue'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45,height=1000)

fig.show()

perc=[]

for x in range(len(poplcnt)):

    perc.append((100*(float)(tested[x])/(float)(poplnum[x])))

print(perc)

print(poplcnt)
import matplotlib.pyplot as plt

import squarify    

 

# If you have 2 lists

squarify.plot(sizes=tested, label=poplcnt, alpha=.7 )

plt.axis('off')

plt.show()
fig1, ax1 = plt.subplots(figsize=(20,10))



ax1.pie(perc, labels=poplcnt, autopct='%1.1f%%',

        shadow=True, startangle=90,frame=True,colors=["orange","Gray","blue","pink","red","yellow"])

print("% of population tested is as follows")



ax1.axis('equal')



plt.show()
import plotly.graph_objects as go





fig = go.Figure(data=[

    go.Bar(name='Tested', x=poplcnt, y=tested,marker_color='rgb(255,69,0)'),

    go.Bar(name='Total Population', x=poplcnt, y=poplnum,marker_color='rgb(255,255,0)')])



# Change the bar mode



fig.update_layout(barmode='stack',title="Relative Study OF Population vs Testing")

fig.show()
import pandas as pd

final=pd.read_csv('../input/complete-corona-details/covid_19_clean_complete.csv')

final.columns
conf=[]

conff=[]

['INDIA', 'RUSSIA', 'USA', 'UK', 'ITALY', 'AUSTRALIA']

for x in range(len(final['Country/Region'])):

    if(final['Country/Region'][x]=='India'):

        conf.append(final['Confirmed'][x])

conff.append(max(conf))

conf=[]

for x in range(len(final['Country/Region'])):

    if(final['Country/Region'][x]=='Russia'):

        conf.append(final['Confirmed'][x])

conff.append(max(conf))

conf=[]



for x in range(len(final['Country/Region'])):

    if(final['Country/Region'][x]=='US' ):

        conf.append(final['Confirmed'][x])

conff.append(max(conf))

conf=[]

for x in range(len(final['Country/Region'])):

    if(final['Country/Region'][x]=='United Kingdom' ):

        conf.append(final['Confirmed'][x])

conff.append(max(conf))

conf=[]

for x in range(len(final['Country/Region'])):

    if(final['Country/Region'][x]=='Italy' ):

        conf.append(final['Confirmed'][x])

conff.append(max(conf))

conf=[]

for x in range(len(final['Country/Region'])):

    if(final['Country/Region'][x]=='Australia' ):

        conf.append(final['Confirmed'][x])

conff.append(max(conf))

print("Covid 19 positive cases as on 23rdJune")

print(conff)

print(poplcnt)

        
import plotly.graph_objects as go





fig = go.Figure(data=[

    go.Bar(name='CONFIRMED CASES', x=poplcnt, y=conff,marker_color='rgb(255,0,0)'),

    go.Bar(name='TOTAL  TESTS DONE TILL 23rd JUNE', x=poplcnt, y=tested,marker_color='rgb(0,255,0)')])



# Change the bar mode



fig.update_layout(barmode='stack',title="Relative Study OF Testing vs Confirmation",height=1000)

fig.show()
ar=['India','Russia','Australia','Italy','US','United Kingdom']

for j in ar:

    if(j=='United Kingdom'):

        break

    for i in range(len(final['Country/Region'])):

        if(final['Country/Region'][i]==j):

            deat=final['Deaths'][i]

            recov=final['Recovered'][i]

            act=final['Active'][i]



    my_circle=plt.Circle( (0,0), 0.7, color='pink')

    size=[deat,act,recov]

    fig = plt.figure()

    fig.patch.set_facecolor('pink')

    names=["Deaths","Active","Recovered"]

    plt.pie(size,  colors=['black','red','green'],radius=1,startangle=200)

    p=plt.gcf()

    patches, texts = plt.pie(size, colors=['black','red','green'], shadow=True, startangle=90)

    plt.legend(patches, labels=names, loc="best")

    p.gca().add_artist(my_circle)

    plt.axis('equal')

    plt.tight_layout()

    plt.title("NUMBER OF COVID-19 PATIENTS IN "+(str)(j)+" AS ON 23rd JUNE 2020")

    plt.show
