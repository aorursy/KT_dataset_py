import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, plot

# word cloud library

from wordcloud import WordCloud

# matplotlib

import matplotlib.pyplot as plt



Data=pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")

Data.info()

Data.head(10)
June=Data[:2].copy()

sum_of_June_for_open=sum([June.Open])

sum_of_June_for_High=sum([June.High])

sum_of_June_for_Low=sum([June.Low])

sum_of_June_for_Close=sum([June.Close])

sum_of_June_for_Volume=sum([June.Volume])



July=Data[2:23].copy()

sum_of_July_for_open=sum([July.Open])

sum_of_July_for_High=sum([July.High])

sum_of_July_for_Low=sum([July.Low])

sum_of_July_for_Close=sum([July.Close])

sum_of_July_for_Volume=sum([July.Volume])





August=Data[24:45].copy()

sum_of_August_for_open=sum([August.Open])

sum_of_August_for_High=sum([August.High])

sum_of_August_for_Low=sum([August.Low])

sum_of_August_for_Close=sum([August.Close])

sum_of_August_for_Volume=sum([August.Volume])





September=Data[46:66].copy()

sum_of_September_for_open=sum([September.Open])

sum_of_September_for_High=sum([September.High])

sum_of_September_for_Low=sum([September.Low])

sum_of_September_for_Close=sum([September.Close])

sum_of_September_for_Volume=sum([September.Volume])



October = Data[66:87].copy()

sum_of_October_for_open=sum([October.Open])

sum_of_October_for_High=sum([October.High])

sum_of_October_for_Low=sum([October.Low])

sum_of_October_for_Close=sum([October.Close])

sum_of_October_for_Volume=sum([October.Volume])



November = Data[87:100].copy()

sum_of_November_for_open=sum([November.Open])

sum_of_November_for_High=sum([November.High])

sum_of_November_for_Low=sum([November.Low])

sum_of_November_for_Close=sum([November.Close])

sum_of_November_for_Volume=sum([November.Volume])







    

    



   


JUNE_OPEN=go.Scatter(

    x=June.Date,

    y=sum_of_June_for_open,

    mode="lines",

    name="OPENING PRİCE ",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=June["Date"]

    

)



JUNE_HİGH=go.Scatter(

    x=June.Date,

    y=sum_of_June_for_High,

    mode="lines",

    name="HIGH PRİCE ",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=June["Date"]

    

)



JUNE_LOW=go.Scatter(



    x=June.Date,

    y=sum_of_June_for_Low,

    mode='lines',

    name='LOW PRICE' ,

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=June["Date"]





)



JUNE_CLOSE=go.Scatter(

    

    x=June.Date,

    y=sum_of_June_for_Close,

    mode='lines',

    name='CLOSE PRICE ',

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=June["Date"]





)

JUNE_VOLUME=go.Scatter(

    

    x=June.Date,

    y=sum_of_June_for_Volume,

    mode='lines',

    name='VOLUME PRICE ',

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=June["Date"]





)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



JULY_OPEN=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=July["Date"]

)





JULY_HIGH=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=July["Date"]

)



JULY_LOW=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=July["Date"]

)



JULY_CLOSE=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=July["Date"]

    )

JULY_VOLUME=go.Scatter(

    x=July.Date,

    y=sum_of_July_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=July["Date"]





)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------





AUGUST_OPEN=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=August["Date"]

)



AUGUST_HIGH=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=August["Date"]

)



AUGUST_LOW=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=August["Date"]

)



AUGUST_CLOSE=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=August["Date"]

)

AUGUST_VOLUME = go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=August["Date"]

)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



SEPTEMBER_OPEN=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=September["Date"]

)

SEPTEMBER_HIGH=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=September["Date"]

)



SEPTEMBER_LOW=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=September["Date"]

)



SEPTEMBER_CLOSE=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=September["Date"]

)



SEPTEMBER_VOLUME=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=September["Date"]

)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------





OCTOBER_OPEN=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)



OCTOBER_HIGH=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_High,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=October["Date"]

)

OCTOBER_LOW=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Low,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=October["Date"]

)



OCTOBER_CLOSE=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Close,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=October["Date"]

)



OCTOBER_VOLUME=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Volume,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=October["Date"]

)





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------





NOVEMBER_OPEN=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_HIGH=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_LOW=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=October["Date"]

)

NOVEMBER_CLOSE=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_VOLUME=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=October["Date"]

)





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------









data=[JUNE_OPEN ,JUNE_LOW , JUNE_HİGH ,JUNE_CLOSE,

      JULY_OPEN ,JULY_HIGH,JULY_LOW,JULY_CLOSE,

      AUGUST_OPEN,AUGUST_HIGH,AUGUST_CLOSE,

      SEPTEMBER_OPEN,SEPTEMBER_HIGH,SEPTEMBER_LOW,

      OCTOBER_OPEN,OCTOBER_HIGH,OCTOBER_LOW,OCTOBER_CLOSE,

      NOVEMBER_OPEN,NOVEMBER_HIGH,NOVEMBER_LOW,NOVEMBER_CLOSE]



layout=dict(title =" REWİEV ON PRİCE FOR MOUNTH FROM TESLA  ",

           xaxis=dict(title="date" , ticklen=5  , zeroline=False)

            

           )

fig=dict(data=data, layout=layout)



iplot(fig)
JUNE_OPEN=go.Scatter(

    x=June.Date,

    y=sum_of_June_for_open,

    mode="markers",

    name="OPENING PRİCE ",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=June["Date"]

    

)





#-------------------------------------------------------------------------------------------------------------------------------------------------

JULY_OPEN=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=July["Date"]

)









#-------------------------------------------------------------------------------------------------------------------------------------------------

AUGUST_OPEN=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=August["Date"]

)







#-------------------------------------------------------------------------------------------------------------------------------------------------

SEPTEMBER_OPEN=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=September["Date"]

)



#-------------------------------------------------------------------------------------------------------------------------------------------------

OCTOBER_OPEN=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)











#-------------------------------------------------------------------------------------------------------------------------------------------------



NOVEMBER_OPEN=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)







#-------------------------------------------------------------------------------------------------------------------------------------------------

data=[JUNE_OPEN,

     JULY_OPEN,

      AUGUST_OPEN,

      SEPTEMBER_OPEN,

      OCTOBER_OPEN ,

      NOVEMBER_OPEN , 

     ]

layout=dict(title="PRICE 'S SPLITTED UP FOR MOUNTH SUCH AS BELOW (OPENING,HIGH , LOW ,CLOSE),",

           xaxis=dict(title="MONTHS",ticklen=5 ,zeroline=False),

            yaxis=dict(title="AMOUNT OF PRICE(K)" , ticklen=5 , zeroline=False)

           )

fig=dict(data=data ,  layout=layout)

iplot(fig)





JUNE_HİGH=go.Bar(

    x=June.Date,

    y=sum_of_June_for_High,

    name="OPENING PRİCE ",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)),

    text=June["Date"]

    

)



#----------------------------------------------------------------------------------------------------------------------------







JULY_HIGH=go.Bar(



    x=July.Date,

    y=sum_of_July_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=July["Date"]

)







#----------------------------------------------------------------------------------------------------------------------------





AUGUST_HIGH=go.Bar(



    x=August.Date,

    y=sum_of_August_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=August["Date"]

)





#----------------------------------------------------------------------------------------------------------------------------









SEPTEMBER_HIGH=go.Bar(



    x=September.Date,

    y=sum_of_September_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=September["Date"]

)













#----------------------------------------------------------------------------------------------------------------------------









OCTOBER_HIGH=go.Bar(



    x=October.Date,

    y=sum_of_October_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=October["Date"]

)









#----------------------------------------------------------------------------------------------------------------------------







NOVEMBER_HIGH=go.Bar(



    x=November.Date,

    y=sum_of_November_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=November["Date"]

)





#----------------------------------------------------------------------------------------------------------------------------



data = [ JUNE_HİGH,

        JULY_HIGH ,

       AUGUST_HIGH ,

       SEPTEMBER_HIGH ,

       OCTOBER_HIGH ,

       NOVEMBER_HIGH ,

        

        

        

       

       ]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)







data_list = [i for i in Data.Low]



label = ["June" , "July","August" , "September","October" , "November"]



data1=[sum_of_June_for_Low, 

      sum_of_July_for_Low,

     sum_of_August_for_Low,

     sum_of_September_for_Low,

     sum_of_October_for_Low,

     sum_of_November_for_Low]





#Figure



fig = {

  "data": [

    {

      "values": data_list,

      "labels": label,

      "domain": {"x": [0, .5]},

      "name": "OPENING STOCK RATIO",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"OPENING STOCK",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)



trace0 = go.Box(

    y=sum_of_June_for_Close,

    name = 'Close stock of June in 2010',

    marker = dict(

        color = 'rgb(0, 255, 0)',

    )

)

trace1 = go.Box(

    y=sum_of_July_for_Close,

    name = 'Close stock of July in 2010',

    marker = dict(

        color = 'rgb(255, 0, 0)',

    )

)

trace2 = go.Box(

    y=sum_of_August_for_Close,

    name = 'Close stock of Augyst in 2010',

    marker = dict(

        color = 'rgb(0, 0, 255)',

    )

)

trace3 = go.Box(

    y=sum_of_September_for_Close,

    name = 'Close stock of September in 2010',

    marker = dict(

        color = 'rgb(255, 255, 0)',

    )

)

trace4 = go.Box(

    y=sum_of_October_for_Close,

    name = 'Close stock of October in 2010',

    marker = dict(

        color = 'rgb(255, 0, 255)',

    )

)

trace5 = go.Box(

    y=sum_of_November_for_Close,

    name = 'Close stock of November in 2010',

    marker = dict(

        color = 'rgb(0, 255, 255)',

    )

)

data = [trace0, trace1,trace2 , trace3 , trace4 , trace5]

iplot(data)
# create trace 1 that is 3d scatter



june_data= ([i  for i in June["Volume"]])

july_data=([i  for i in July["Volume"]])

data = [

    {

        'y': Data.Volume,

        'x': Data.Date,

        'mode': 'markers',

        'marker': {

            'color': 'rgb(0, 255, 255)',

           

            'showscale': True

        },

        "text" :  Data.Date  

    }

   ]



iplot(data)
from sklearn.preprocessing import StandardScaler



X= Data.iloc[:,1:5]

Y=Data.iloc[: ,5:]

sc1 = StandardScaler()

x_scalled = sc1.fit_transform(X)

sc2 = StandardScaler()

y_scalled = sc2.fit_transform(Y)



Data.columns=['X1','X2','X3','X4','X5','X6','Y']

X=Data.iloc[:,1:6].values

Y=Data.iloc[:,6].values

print(Y)

from sklearn.ensemble import RandomForestRegressor



rf_reg=RandomForestRegressor(n_estimators=10 , random_state=0) 

rf_reg.fit(X,Y)

from sklearn.metrics import r2_score



print(r2_score(Y,rf_reg.predict(X)))
