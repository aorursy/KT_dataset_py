#I am new to both Python and Plotly so my codes might not be as concise and efficient. 

# While practicing with Plotly, I created the following plots and appreciate your feedback.





#import required libraries

import pandas as pd

import numpy as np

from scipy import stats

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import pycountry

import plotly.express as px



#read the csv file

df = pd.read_csv("../input/NBA_list_new.csv",encoding='latin1')
#create country codes (alpha_3) from the names to be used for the geolocation

d=list()

l=list()

for name in df.COUNTRY.unique():

    #if name in l: continue

    #else: 

        j=df.loc[df.COUNTRY==name]['COUNTRY'].count()

        #print(j)

        d.append(name)

        l.append(j)



#print(d)

type(j)

q={"Country":d,"Count":l}

counts=pd.DataFrame(q)

#counts.columns=["COUNTRY", "COUNT"]

#counts





code=list()

for i in counts.Country:

    if i=='USA':code.append(i)

    else:

        aa=pycountry.countries.get(name=i)

        code.append(aa.alpha_3)

    

C={'Code':code}   



counts['Code']=code

counts['Count']=l

len(counts.Country)







#Plot the number of players per country

fig2=go.Figure()



fig2.add_trace(go.Scattergeo(locations=counts.Country,

        locationmode = 'country names',

        hovertext=counts.Count,

        visible=True,

         marker = dict(

            size =10* counts.Count,

            color = 'rgb(225, 0, 255)', 

            opacity=0.5,

            line_color='rgb(105, 0, 252)',

            line_width=0.5,

            sizemode = 'area'

        ),

           ))

fig2.update_layout(showlegend=False, title_text="NBA Stats_2018/2019 Season: Players' Country",

                  

                  )

fig2.show()
#Create figure

fig = go.Figure()



# Add Height histogram trace

fig.add_trace(go.Histogram(histfunc="count", y=None, x=df.HEIGHT, name="Height(in)",

                           visible=True, hoverinfo="all", marker_color='rgb(207, 34, 4)',

                           opacity=1))





# Add AGE histogram trace

fig.add_trace(go.Histogram(histfunc="count", y=None, x=df.AGE, name="Age(Year)",

                           hoverinfo="all", visible=False, marker_color='rgb(2, 219, 227)', 

                           opacity=1))



# Add WEIGHT histogram trace

fig.add_trace(go.Histogram(histfunc="count", y=None, x=df.WEIGHT, name="Weight(lb)",

                           hoverinfo="all", visible=False, marker_color='rgb(142, 0, 161)', 

                           opacity=1))





# Add dropdown

fig.update_layout(

    updatemenus=[

        go.layout.Updatemenu(

            buttons=list([

                          

                dict(

                    args=[{"visible":[True, False, False]},

                          {"title":"NBA_2018/2019 Season: Players' Height (in)"}],

                    label="Height (in)",

                    method="update"

                    ),

                

                dict(

                    args=[{"visible":[False,True, False]},

                          {"title":"NBA_2018/2019 Season: Players' Age (Year)"}],

                    label="Age (Year)",

                    method="update", 

                ),

                

                dict(

                    args=[{"visible":[False,False, True]},

                          {"title":"NBA_2018/2019 Season: Players' Weight (lb)"}],

                    label="Weight (lb)",

                    method="update",

                                   )

            ]),

            direction="down",

            pad={"r": 10, "t": 10},

            showactive=True,

            x=-.25,

            xanchor="left",

            y=1.13,

            yanchor="top"

        ),

    ]

)



#Update the layout and template

fig.update_layout(

    xaxis_title_text='Value', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.02, # gap between bars of adjacent location coordinates

    bargroupgap=0.02, # gap between bars of the same location coordinates

    autosize=False,

    showlegend=False,

    template="plotly_white"

)





fig.show()
#Group the colleges with less than five players in "other" list

x=list()

other=0

college_name=list()

college_count=list()

dic={"Other":other}

for n in df['COLLEGE']:

    x.append(n)



for j in x:

    if x.count(j)<5: 

        other=other+1

        #if j not in college_name: college_name.append(j)

        dic.update({"Other":other})

        continue

    else:

       

        if j not in college_name: 

            college_count.append(x.count(j))

            college_name.append(j)

            dic.update({j:x.count(j)})



#Group the countries with less than three players in "other" list

y=list()

other1=0

country_name=list()

country_count=list()

dic1={"other":other1}



for n in df['COUNTRY']:

    y.append(n)



for j in y:

    if y.count(j)<3: 

        other1=other1+1

        #if j not in college_name: college_name.append(j)

        dic1.update({"other":other1})

        continue

    else:

       

        if j not in college_name: 

            country_count.append(y.count(j))

            country_name.append(j)

            dic1.update({j:y.count(j)})



list1=list()

list2=list()

for n,c in dic.items():

    list1.append(n)

    list2.append(c)



list3=list()

list4=list()

for n,c in dic1.items():

    list3.append(n)

    list4.append(c)





#Create subplot for two histogram traces in a same row

from plotly.subplots import make_subplots



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],

                   subplot_titles=['Players per Country', 'Players per College'])



fig.add_trace(go.Pie(labels=list3, values=list4, textinfo="none", hole=.5, pull=0.05,

                              title="Players' Country", name="Country",

                              marker=dict(colors=px.colors.sequential.Plasma[4::]) ),row=1, col=1)

              

fig.add_trace(go.Pie(labels=list1, values=list2, textinfo="none", hole=.5, pull=0.05,

                            title="Players' College", name="College",

                            marker=dict(colors=px.colors.sequential.Redor[3::1])),row=1, col=2) 

                             



fig.update_layout(showlegend=False,template="plotly_dark")              
#Convert USG% column saved as string to float

l=list()

for i in df["USG%"]:

    l.append(round(float(i[:-1])/100,2))

    

#print(l)

d={"USG":l}

df["USG"]=pd.DataFrame(d)



#Create Scatter plot

fig=px.scatter(df, x="PTS", y="REB", marginal_x="box", marginal_y="box", color="AST",

               size="USG", size_max=10,hover_name="PLAYER", #trendline="ols", trendline_color_override="red"

              color_continuous_scale=px.colors.diverging.Portland[1::1],

              labels={"AST": "Assists", "PTS":"Points", "REB":"Rebounds"})



#Update the template and title

fig.update_layout(showlegend=False, title_text='NBA Stats_2018/2019 Season: Performance', template="plotly_dark")

fig.show()