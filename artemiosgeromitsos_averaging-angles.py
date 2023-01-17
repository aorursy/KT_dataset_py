for name in dir():

    if not name.startswith('_'):

        del globals()[name]
import numpy as np 

import pandas as pd

import math as mt

from cmath import rect, phase

from math import radians, degrees

import matplotlib.pyplot as plt

import numpy.linalg as la
#create function that creates average angle

#https://rosettacode.org/wiki/Averages/Mean_angle#Python

def mean_angle(deg):

      return degrees(phase(sum(rect(1, radians(d)) for d in deg)/len(deg)))

 

#creating a function based on what I understood

def mean_angle2(deg):

    #print("sum(cos):",sum(np.cos(np.deg2rad(deg))))

    #print("sum(sin):",sum(np.sin(np.deg2rad(deg))))

    return np.rad2deg(np.arctan2(sum(np.sin(np.deg2rad(deg))),sum(np.cos(np.deg2rad(deg))))) % 360



def plot_angles(angle=45):    

     plt.figure()

     ax = plt.gca()

     #plot the various vectors:

     ax.quiver(0, 0,np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), angles='xy', scale_units='xy', scale=2,label="vector angles")

     #plot the average vector

     ax.quiver(0, 0,np.cos(np.deg2rad(mean_angle2(angle))), np.sin(np.deg2rad(mean_angle2(angle))),color='green', angles='xy', scale_units='xy', scale=2,label="correct avg angle")

     #if moufa:

     ax.quiver(0, 0,np.cos(np.deg2rad(np.mean(angle))), np.sin(np.deg2rad(np.mean(angle))),color='red', angles='xy', scale_units='xy', scale=2,label="false avg angle")

     ax.set_xlim([-1, 1])

     ax.set_ylim([-1, 1])

     ax.legend()

     plt.draw()

     plt.show()
#testing function



for angles in [

               [90, 180, 270, 360],

               [90, 180, 270, 0],

               [0, 90, 180, 270],

               [50, 95, 120+55, 250+55]    



              ]:

    print('The correct mean angle of', angles, 'is:Trig. Function', round(mean_angle2(angles), 12),'Euler Function [',round(mean_angle(angles), 12) ,'] degrees')

    #print('The false mean angle of', angles, 'is:', round(np.mean(angles), 12), 'degrees')

    plot_angles(angles)

#######################################################################################

#split the circle in slices:

def sliceme(N,blah=False,plot=False,end_angle=360):

    #the reason for end_angle and not some const is because at 360 degs things are really spooky

    pizza_slice = np.arange(0,end_angle,end_angle/N)

    if blah:

        print("Angle slice:",pizza_slice,"\nmean_angle:",mean_angle2(pizza_slice))

        print('The false mean angle of', angles, 'is:', round(np.mean(pizza_slice), 2), 'degrees')

    if plot:

        plot_angles(pizza_slice)

    return mean_angle2(pizza_slice), round(np.mean(pizza_slice), 2)

#you can call the 

#sliceme(10)

#or

N=100

splits=np.arange(2,N+1,1)

slice_angle=360/splits

ang_df = pd.DataFrame({'splits':splits,'slice_angle':slice_angle})

f = lambda x,index:tuple( i[index] for i in x) #func that splits the cells (was the return of the above function)



ang_df['correct_avg_angle'] = f([sliceme(angs) for angs in splits],0);ang_df['correct_avg_angle']=round(ang_df['correct_avg_angle'],2)

ang_df['false_avg_angle']   = f([sliceme(angs) for angs in splits],1)

ang_df['diff_angle']  = ang_df['correct_avg_angle'] - ang_df['false_avg_angle']

ang_df['ratio_angle'] = ang_df['correct_avg_angle'] / ang_df['false_avg_angle']





ang_df.plot(y='correct_avg_angle',x='slice_angle',marker='.')

ang_df.plot(y='false_avg_angle',x='slice_angle',marker='.')

ang_df.plot(y='diff_angle',x='slice_angle',marker='.')

ang_df.plot(y='ratio_angle',x='slice_angle',marker='.')



display(ang_df)

#create equidistant splits based on degress rather than number of splits :) (the latter are not equidistant "ohfoso")



N=360



ang_df2 = pd.DataFrame(columns = ['splits','slice_angle','correct_avg_angle','false_avg_angle','diff_angle','ratio_angle'])#preparing dataframe



for incr in np.arange(2,N,1):

    #print("incr:",incr)

    splits = np.arange(0,incr,1)

    #print("splits: ",splits)

    slice_angle = 360/incr

    #print("slice_angle:",slice_angle)

    angs = splits*slice_angle

    print("angs:",angs)

    tempo_key = {'splits':incr,'slice_angle':slice_angle,'correct_avg_angle':mean_angle2(angs),'false_avg_angle':np.mean(angs)}

    ang_df2 = ang_df2.append(tempo_key,ignore_index=True)   

    

#ang_df2 = pd.DataFrame({'splits':splits,'slice_angle':slice_angle})

#f = lambda x,index:tuple( i[index] for i in x ) #func that splits the cells (was the return of the above function)



ang_df2['diff_angle']  = ang_df2['correct_avg_angle'] - ang_df2['false_avg_angle']

ang_df2['ratio_angle'] = ang_df2['correct_avg_angle'] / ang_df2['false_avg_angle']

display(ang_df2)





ang_df2.plot(y='correct_avg_angle',x='slice_angle',marker='.')

ang_df2.plot(y='false_avg_angle',x='slice_angle',marker='.')

ang_df2.plot(y='diff_angle',x='slice_angle',marker='.')

ang_df2.plot(y='ratio_angle',x='slice_angle',marker='.')











ang_df2