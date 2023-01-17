import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
zone,year="NW",'2016'

Filename='/kaggle/input/meteonet/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations_'+year+".csv"

df = pd.read_csv(Filename,parse_dates=[4],infer_datetime_format=True)

date_selection='2016-01-01T06:00:00'

date_sub=df[ df['date'] == date_selection ] #Sub_tableau avec la date selectionne.
# SHOW AVAIABLE STATION

plt.scatter(date_sub['lon'], date_sub['lat'], c=date_sub['psl'], cmap='jet')

plt.show() # Many NAN on the pressure data

plt.scatter(date_sub['lon'], date_sub['lat'], c=date_sub['t'], cmap='jet')

plt.show()
selec_station= 14066001

station_sub= df[df["number_sta"]==selec_station]

station_sub.head()
import folium

m=folium.Map(location=[df["lat"][0],df["lon"][0]],

           tiles='Stamen Terrain',

          zoom_start=8)

tooltip="station"

folium.Marker([df["lat"][0], df["lon"][0]],tooltip=tooltip).add_to(m)

m
Pt_jour=240 # One acquisition every 6 minits, so 240 acquisition for one day

Station_sub_j0=station_sub.iloc[:Pt_jour]
tab_variable={"t":10,"dd":5,"ff":6,"precip":7,"hu":8,"td":9,",psl":11}

tab_variable2=[11,10,6,5,8,9,7]

nb_row=7

Number_val=10

colorscale=[[0, 'rgb(0,0,255)'], [1,'rgb(255,0,0)']]



Hour_loop= np.linspace(0, Number_val*10, Number_val+1 ,dtype=int)

Station_sub_j0=Station_sub_j0.iloc[Hour_loop,:]
import plotly.figure_factory as ff

import plotly.graph_objects as go



from plotly.colors import n_colors

# COLOR SELECTION FOR THE HEATMAP

colors_pmer=[[0, 'rgb(255, 255, 255)'],[1,'rgb(255,255,255)']]

colors_hu = n_colors('rgb(255, 255, 255)', 'rgb(0, 0, 255)', 5, colortype='rgb')

colors_t =[[0, 'rgb(255, 255, 0)'],[1, 'rgb(231,61,1)']]

colors_precip= n_colors('rgb(255, 255, 255)', 'rgb(0, 0, 255)', 10, colortype='rgb')

color_wind=[[0, 'rgb(255, 255, 255)'],[1, 'rgb(135, 233, 144)']]



# SUBPLOT ORGANISATION

from plotly.subplots import make_subplots

fig=make_subplots(rows=nb_row,cols=Number_val,vertical_spacing=0.02,

                  specs=[[{"colspan":Number_val,"type":"heatmap"},None,None,None,None,None,None,None,None,None],

                         [{"colspan":Number_val,"type":"heatmap"},None,None,None,None,None,None,None,None,None],

                         [{"colspan":Number_val,"type":"heatmap"},None,None,None,None,None,None,None,None,None],

                         [{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"},{"colspan":1,"type":"polar"}],

                         [{"colspan":Number_val,"type":"heatmap"},None,None,None,None,None,None,None,None,None],

                         [{"colspan":Number_val,"type":"heatmap"},None,None,None,None,None,None,None,None,None],

                         [{"colspan":Number_val,"type":"heatmap"},None,None,None,None,None,None,None,None,None]])



# HEAT MAP FOR THE 3 FIRST ROW

fig.add_trace(go.Heatmap(y=["Pressure (Pa)"],colorscale=colors_pmer,hoverongaps = False,z=[Station_sub_j0['psl'][:Number_val].values],xgap=2,colorbar={"thickness":10,"len":0.15,'y':-1,'ticktext':['coucou']}),row=1,col=1)

fig.add_trace(go.Heatmap(y=["Temperature (°C)"],colorscale=colors_t,z=[Station_sub_j0['t'][:Number_val].values],xgap=2,colorbar={"thickness":10,"len":0.15,'yanchor':'bottom','y':0.72,'ticktext':['coucou']}),row=2,col=1)

fig.add_trace(go.Heatmap(y=["Wind Speed (ms-1)"],colorscale=color_wind,z=[Station_sub_j0['ff'][:Number_val].values],xgap=2,colorbar={"thickness":10,"len":0.15,'yanchor':'bottom','y':0.58,'ticktext':['coucou']}),row=3,col=1)



# POLAR BAR FOR WIND DIRECTION

for data in range(Number_val):

    fig.add_trace(go.Barpolar(showlegend=False,r=[float(Station_sub_j0.iloc[data,6])],theta=[float(Station_sub_j0.iloc[data,5])],width=[40],marker=dict(color="black",colorscale=color_wind,cmin=2,cmax=3)),row=4,col=data+1)



# HEAT MAP FOR THE 3 LAST ROW

fig.add_trace(go.Heatmap(y=["Humidity (%)"],zmin=80,zmax=100,colorscale=colors_hu,z=[Station_sub_j0['hu'][:Number_val].values],text=[Station_sub_j0['hu'][:Number_val].values] ,hoverlabel={"bgcolor":"blue"},colorbar={"thickness":10,"len":0.15,'y':0.28,'yanchor':'bottom','ticktext':['coucou']},opacity=1,hovertext=Station_sub_j0['hu'][:5],xgap=2,ids=Station_sub_j0['hu'][:5].values),row=5,col=1)

fig.add_trace(go.Heatmap(y=["Dew point (°C)"],colorscale=colors_t,z=[Station_sub_j0['td'][:Number_val].values],xgap=2,colorbar={"thickness":10,"len":0.15,'yanchor':'bottom','y':0.13,'ticktext':['coucou']}),row=6,col=1)

fig.add_trace(go.Heatmap(y=["Precipitation"],zmin=0,zmax=100,colorscale=colors_precip,x=Station_sub_j0['date'][:Number_val].dt.time.values,z=[Station_sub_j0['precip'][:Number_val].values],xgap=2,colorbar={"thickness":10,"len":0.15,'yanchor':'bottom','y':-0.02,'ticktext':['coucou']}),row=7,col=1)



#REMOVE Xaxis FOR THE 5 FIRST HEATMAP

fig.update_layout(xaxis_visible=False)

fig.update_layout(xaxis2_visible=False)

fig.update_layout(xaxis3_visible=False)

fig.update_layout(xaxis4_visible=False)

fig.update_layout(xaxis5_visible=False,xaxis5_title= "Horaires")



# UPDATE LAYOUT OF POLAR BAR WIND DIRECTION

fig.update_layout(polar=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar2=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar3=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar4=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar5=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar6=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar7=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar8=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar9=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))

fig.update_layout(polar10=dict(radialaxis=dict(range=[0, 3],type= "linear",showticklabels=False,ticks=''), angularaxis=dict(showticklabels=False, gridcolor = "white",ticks='')))



# UPDATE ANNOTATION BECAUSE THOSE HEATMAP ARE NOT BUILT-IN ANNOTED HEATMAP

for row in range(nb_row-1):

    for data in range(Number_val):

        if row ==0:

            fig.add_annotation(xref="x",yref="y",x=data,y=0,text=str(Station_sub_j0.iloc[data,tab_variable2[row]]))

        elif row < 3:

            fig.add_annotation(xref="x"+str(row+1),yref="y"+str(row+1),x=data,y=0,text=str(Station_sub_j0.iloc[data,tab_variable2[row]]))   

        elif row >=3 :

            fig.add_annotation(xref="x"+str(row+1),yref="y"+str(row+1),x=data,y=0,text=str(Station_sub_j0.iloc[data,tab_variable2[row+1]]))   

                    

fig.update_annotations(dict(showarrow=False,\

                            font_color='black',\

                            font_size=18,\

                            font_family="Droid Sans"))





# FIGURE LAYOUT

fig.update_layout(height=600, width=1300,showlegend=False, title_text="Observations station "+str(Station_sub_j0.iloc[0,0]),title_xref="paper",title_font_size=30)





fig.show()