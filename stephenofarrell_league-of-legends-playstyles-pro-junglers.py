import numpy as np 

from numpy.linalg import norm as norm

from ast import literal_eval

import pandas as pd 

import plotly.express as px

import os

from PIL import Image



h = 153

w = 155



radius = int(w/2.5)
df = pd.read_csv("/kaggle/input/na-lcs-summer-2020-player-locations/nalcssummer2020.csv")

df.head()
df_c9_jgl_blue = df[(df.team == 'c9') & (df.roles == "jgl") & (df.side == "blue")]

df_c9_jgl_blue
minimap = Image.open("/kaggle/input/na-lcs-summer-2020-player-locations/lcs.png")

px.imshow(minimap)
def classify_jgl(points):

    reds = [0]*9

    for point in points:

        try:

            point = literal_eval(point)

            if(norm(point - np.array([0,0])) < radius): # Toplane

                reds[5]+=1

            elif(norm(point - np.array([149,0])) < radius): # Red base

                reds[6]+=1

            elif(norm(point - np.array([149,149])) < radius): # Botlane

                reds[8]+=1

            elif(norm(point - np.array([0,149])) < radius): # Blue base

                reds[7]+=1

            elif(point[0] < h - point[1] - h/10): # Mid lane upper border

                if(point[0] < (5/4)*point[1]): # Blue side

                    reds[1]+=1

                else: # Red side

                    reds[0]+=1

            elif(point[0] < h - point[1] + h/10): # Mid lane lower border

                reds[4]+=1 

            elif(point[0] > h - point[1] + h/10): # Below lower border

                if(point[0] < (5/4)*point[1]): # Blue side

                    reds[2]+=1

                elif(point[0] > (5/4)*point[1]): # Red side

                    reds[3]+=1

        except:

            pass

    return(reds)
colour = "blue"

col="Cloud 9"



timesplits =  {480:"0-8", 840:"8-14", 1200:"14-20"}

timesplits2 = {480:0, 840:480, 1200:840}



for times in timesplits.keys():



    times_floor = timesplits2[times]

    

    all_games = df_c9_jgl_blue.iloc[0][6+times_floor:6+times]

    for i in range(1,df_c9_jgl_blue.shape[0]):

        all_games = all_games.append(df_c9_jgl_blue.iloc[i][6+times_floor:6+times])

    

    reds = classify_jgl(all_games)

    reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))



    fig = px.scatter(

            x = [1], 

            y = [1],

            range_x = [0,w],

            range_y = [h, 0],

            width = 800,

            height = 800)





    fig.update_layout(

            template = "plotly_white",

            xaxis_showgrid = False,

            yaxis_showgrid = False

            )



    fig.update_xaxes(showticklabels = False, title_text = "")

    fig.update_yaxes(showticklabels = False, title_text = "")



    # Different colours for each team

    fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"

    fig.update_layout(

        shapes=[

        dict(

                type="path",

                path = "M 0,0 L %d,%d L %d,0 Z" % (w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),

            ),

        dict(

                type="path",

                path = "M 0,0 L %d,%d L 0,%d Z" % (w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),

            ),



        dict(

                type="path",

                path = "M %d,%d L %d,%d L 0,%d Z" % (w,h,w/2, h/2,h),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),

            ),

        dict(

                type="path",

                path = "M %d,%d L %d,%d L %d,0 Z" % (w,h,w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),

            ),

        dict(

                type="path",

                path = "M %d,%d L %d,%d L %d,0 L 0,%d Z" % (w/10,h, w,h/10,w-w/10,h-h/10),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),

            ),



        dict(

                type="circle",

                xref="x",

                yref="y",

                x0=-radius,

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),

                y0=radius,

                x1=radius,

                y1=-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),

                x0=w-radius,

                y0=radius,

                x1=w+radius,

                y1=-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),

                x0=-radius,

                y0=h+radius,

                x1=radius,

                y1=h-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[8],reds[8]),

                x0=w-radius,

                y0=h+radius,

                x1=w+radius,

                y1=h-radius,

                line_color="white",

            )])

    fig.update_layout(

        title = "%s: %smins" % (col.capitalize(), timesplits[times]),

        template = "plotly_white",

        xaxis_showgrid = False,

        yaxis_showgrid = False

        )



    fig.show()
c9_olaf = df_c9_jgl_blue[df_c9_jgl_blue['champ']=="olaf"]

c9_other = df_c9_jgl_blue[df_c9_jgl_blue['champ']!="olaf"]



titles = ['Cloud 9: Olaf (0-8mins)', 'Cloud 9: Others (0-8mins)']



times = 480

times_floor = timesplits2[times]



for index, dataf in enumerate([c9_olaf, c9_other]):

    

    all_games = dataf.iloc[0][6+times_floor:6+times]

    for i in range(1,dataf.shape[0]):

        all_games = all_games.append(dataf.iloc[i][6+times_floor:6+times])

    

    reds = classify_jgl(all_games)

    reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))



    fig = px.scatter(

            x = [1], 

            y = [1],

            range_x = [0,w],

            range_y = [h, 0],

            width = 800,

            height = 800)





    fig.update_layout(

            template = "plotly_white",

            xaxis_showgrid = False,

            yaxis_showgrid = False

            )



    fig.update_xaxes(showticklabels = False, title_text = "")

    fig.update_yaxes(showticklabels = False, title_text = "")



    # Different colours for each team

    fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"

    fig.update_layout(

        shapes=[

        dict(

                type="path",

                path = "M 0,0 L %d,%d L %d,0 Z" % (w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),

            ),

        dict(

                type="path",

                path = "M 0,0 L %d,%d L 0,%d Z" % (w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),

            ),



        dict(

                type="path",

                path = "M %d,%d L %d,%d L 0,%d Z" % (w,h,w/2, h/2,h),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),

            ),

        dict(

                type="path",

                path = "M %d,%d L %d,%d L %d,0 Z" % (w,h,w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),

            ),

        dict(

                type="path",

                path = "M %d,%d L %d,%d L %d,0 L 0,%d Z" % (w/10,h, w,h/10,w-w/10,h-h/10),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),

            ),



        dict(

                type="circle",

                xref="x",

                yref="y",

                x0=-radius,

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),

                y0=radius,

                x1=radius,

                y1=-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),

                x0=w-radius,

                y0=radius,

                x1=w+radius,

                y1=-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),

                x0=-radius,

                y0=h+radius,

                x1=radius,

                y1=h-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[8],reds[8]),

                x0=w-radius,

                y0=h+radius,

                x1=w+radius,

                y1=h-radius,

                line_color="white",

            )])

    fig.update_layout(

        title = titles[index],

        template = "plotly_white",

        xaxis_showgrid = False,

        yaxis_showgrid = False

        )



    fig.show()
colour = "red"

df_c9_jgl_red = df[(df.team == 'c9') & (df.roles == "jgl") & (df.side == "red")]



c9_lee = df_c9_jgl_red[df_c9_jgl_red['champ']=="leesin"]

c9_other = df_c9_jgl_red[df_c9_jgl_red['champ']!="leesin"]



titles = ['Cloud 9: Lee Sin (0-8mins)', 'Cloud 9: Others (0-8mins)']



times = 480

times_floor = timesplits2[times]



for index, dataf in enumerate([c9_lee, c9_other]):

    

    all_games = dataf.iloc[0][6+times_floor:6+times]

    for i in range(1,dataf.shape[0]):

        all_games = all_games.append(dataf.iloc[i][6+times_floor:6+times])

    

    reds = classify_jgl(all_games)

    reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))



    fig = px.scatter(

            x = [1], 

            y = [1],

            range_x = [0,w],

            range_y = [h, 0],

            width = 800,

            height = 800)





    fig.update_layout(

            template = "plotly_white",

            xaxis_showgrid = False,

            yaxis_showgrid = False

            )



    fig.update_xaxes(showticklabels = False, title_text = "")

    fig.update_yaxes(showticklabels = False, title_text = "")



    # Different colours for each team

    fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"

    fig.update_layout(

        shapes=[

        dict(

                type="path",

                path = "M 0,0 L %d,%d L %d,0 Z" % (w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),

            ),

        dict(

                type="path",

                path = "M 0,0 L %d,%d L 0,%d Z" % (w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),

            ),



        dict(

                type="path",

                path = "M %d,%d L %d,%d L 0,%d Z" % (w,h,w/2, h/2,h),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),

            ),

        dict(

                type="path",

                path = "M %d,%d L %d,%d L %d,0 Z" % (w,h,w/2,h/2,w),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),

            ),

        dict(

                type="path",

                path = "M %d,%d L %d,%d L %d,0 L 0,%d Z" % (w/10,h, w,h/10,w-w/10,h-h/10),

                line=dict(

                    color="white",

                    width=2,

                ),

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),

            ),



        dict(

                type="circle",

                xref="x",

                yref="y",

                x0=-radius,

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),

                y0=radius,

                x1=radius,

                y1=-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),

                x0=w-radius,

                y0=radius,

                x1=w+radius,

                y1=-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),

                x0=-radius,

                y0=h+radius,

                x1=radius,

                y1=h-radius,

                line_color="white",

            ),

        dict(

                type="circle",

                xref="x",

                yref="y",

                fillcolor=('rgba(%s,1)' % fill_team) % (reds[8],reds[8]),

                x0=w-radius,

                y0=h+radius,

                x1=w+radius,

                y1=h-radius,

                line_color="white",

            )])

    fig.update_layout(

        title = titles[index],

        template = "plotly_white",

        xaxis_showgrid = False,

        yaxis_showgrid = False

        )



    fig.show()