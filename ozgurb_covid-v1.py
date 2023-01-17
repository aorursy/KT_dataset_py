import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline



import plotly.offline as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

py.init_notebook_mode(connected=True)



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999

pd.options.display.float_format = '{:.2f}'.format



#FOLDER_ROOT = '/../'

#FOLDER_INPUT = FOLDER_ROOT + '/input'









# Any results you write to the current directory are saved as output.
df_cvd1 = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv",index_col=0)
df_cvd1.info()
df_cvd1=df_cvd1[pd.isna(df_cvd1.index)]
df_cvd1 = df_cvd1.rename(columns = {"Country/Region": "Con_Reg"})
def get_slope(x1,y1,x2,y2):

    a = (y2 - y1) / (x2 - x1)  

    return a



def  calc_country_slope(c_list,interval,list_f):

    

    for value in  c_list:

        #print(value)

        filter_param2="Con_Reg=='"+str(value)+"'"

        df_cvd_country=df_cvd1.query(filter_param2)



        counter=[]

        for count,d in enumerate(df_cvd_country.Date):

            counter.append(count)



        df_cvd_country['date_num'] = counter



        #print(df_germany.columns)



        max_time=df_cvd_country.date_num.max()

        filter_param="date_num=="+str(max_time)



        length_1 = df_cvd_country.query("date_num=="+str(df_cvd_country.date_num.max()-interval)).Confirmed

        time_1 = df_cvd_country.query("date_num=="+str(df_cvd_country.date_num.max()-interval)).date_num

        length_2 = df_cvd_country.query(filter_param).Confirmed

        time_2 = df_cvd_country.query(filter_param ).date_num



        list_f.append(  get_slope(time_1,length_1,time_2,length_2) )



country_list=["Germany","US","Brazil","Japan","Italy","Turkey","South Korea","Russia","United Kingdom","France","South Africa","Spain","Mexico",

              "Sweden","India","Belarus","Netherlands","Nigeria","Norway","Malaysia","Saudi Arabia"]        

df_cvd_final = pd.DataFrame(columns=['Country',"Slope_30","Slope_15","Slope_7","Slope_1"])

df_cvd_final["Country"]=country_list











lister1=[]

lister2=[]

calc_country_slope(country_list,30,lister1)

for  d in lister1:

    for c in d:

        #print(c)

        lister2.append(c)

df_cvd_final["Slope_30"]=lister2





lister1=[]

lister2=[]

calc_country_slope(country_list,15,lister1)

for  d in lister1:

    for c in d:

        #print(c)

        lister2.append(c)

df_cvd_final["Slope_15"]=lister2





lister1=[]

lister2=[]

calc_country_slope(country_list,7,lister1)

for  d in lister1:

    for c in d:

        #print(c)

        lister2.append(c)

df_cvd_final["Slope_7"]=lister2





lister1=[]

lister2=[]

calc_country_slope(country_list,1,lister1)

for  d in lister1:

    for c in d:

        #print(c)

        lister2.append(c)

df_cvd_final["Slope_1"]=lister2







df_cvd_final["Chg_Slope_1_30"]=df_cvd_final["Slope_1"]-df_cvd_final["Slope_30"] # compare current with 30days before







def color_negative(value):

  if value > 0:

    color = 'red'

  elif value < 0:

    color = 'green'

  else:

    color = 'black'



  return 'color: %s' % color





df_cvd_final=df_cvd_final.sort_values(by=['Chg_Slope_1_30'],ascending=False)

df_cvd_final.style.applymap(color_negative, subset=['Chg_Slope_1_30'])

#df_cvd1.query('Con_Reg=="Japan"')
import plotly.graph_objects as go



co=df_cvd_final



fig = go.Figure(data=[

    go.Bar(name='Slope_30', x=co["Country"], y=co.Slope_30),

    go.Bar(name='Slope_15', x=co["Country"], y=co.Slope_15),

    go.Bar(name='Slope_7', x=co["Country"], y=co.Slope_7),

    go.Bar(name='Slope_1', x=co["Country"], y=co.Slope_1)

])



fig.update_layout(title_text='Country vs Slope',xaxis={'categoryorder':'category ascending'})



fig.show()
#df_cvd1.query("Con_Reg=='Japan'")