import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv(r'../input/SEN oct-2016.csv')

df.head()
groups = df.groupby(['Fuente','Dia', 'Hora'])['Produccion (MWh)'].sum()

sources = groups.index.levels[0]

days = groups.index.levels[1]

hours = groups.index.levels[2]
fig = plt.figure(figsize=(9,70))



ax = plt.subplot(611)  # num rows, num columns, figure number

bx = plt.subplot(612)

cx = plt.subplot(613)

dx = plt.subplot(614)

ex = plt.subplot(615)





for i in sources:

    for j in days:

        k = float(j.split('/')[1])             #the day of the month of the profile

        color = (0, k / 31.1, 0, 1)

        label = i + ', ' + j

        

        if i == "Biomasa":

            ax.plot(hours, groups[i][j], label=label, color=color)

        if i == "Eolica":  #Wind power

            bx.plot(hours, groups[i][j], label=label, color=color) 

        if i == "Geotermica":

            cx.plot(hours, groups[i][j], label=label, color=color) 

        if i == "Hidroelectrica":

            dx.plot(hours, groups[i][j], label=label, color=color) 

        if i == "Solar":

            ex.plot(hours, groups[i][j], label=label, color=color)



def edit_plot(zx):

    plt.xlabel('Hour')

    plt.ylabel('Production (MWh)')

    plt.legend()



    subplot1 = zx.get_position()

    zx.set_position([subplot1.x0, subplot1.y0, subplot1.width * 1, subplot1.height])



    # Place legend

    zx.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    



plt.subplot(611)

plt.title('Biomass')

edit_plot(ax)





plt.subplot(612)

plt.title('Wind Power')

edit_plot(bx)





plt.subplot(613)

plt.title('Geothermal')

edit_plot(cx)



plt.subplot(614)

plt.title('Hydroelectric')

edit_plot(dx)



plt.subplot(615)

plt.title('Solar')

edit_plot(ex)

   

#plt.subplots_adjust(hspace=0)

plt.show()
groups2 = df.groupby(['Dia','Hora'])['Produccion (MWh)'].sum()

days = groups2.index.levels[0]

hours = groups2.index.levels[1]



fig2 = plt.figure(figsize=(10,9))

fx = plt.subplot(111)



for i in days:

   k = float(i.split('/')[1])             #the day of the month of the profile

   color = (0, k / 31.1, 0, 1)

   fx.plot(hours, groups2[i], label=i, color=color)



plt.subplot(111)

plt.title('Total Generation')

edit_plot(fx)

plt.show()