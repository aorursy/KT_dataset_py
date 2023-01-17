# data analysis on my name

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from matplotlib import style



style.use('ggplot')



name = "Jaime"



#Read the data into DataFrame df

df = pd.read_csv( "../input/NationalNames.csv")



# set name DataFrame

df_name = df[df['Name'] == name]

# Gender

df_gender = pd.pivot_table(data=df_name[["Gender","Count"]],index="Gender",aggfunc=np.sum)



# Plot Gender Results in a pie chart

plt.pie(df_gender,labels=['F','M'],startangle=90,colors=['lightcoral','steelblue'])

plt.axis('equal')

plt.title('Jaimes of Each Gender 1880-2014')

plt.show()



print('I always assumed there were more females than males with the name Jaime!'

     'Here we see that\'s not the case.')
# Plot Year Counts by Gender

df_name_M = df_name[df_name['Gender'] == 'M']

df_year_M = pd.pivot_table(data=df_name_M[["Year","Count"]],index="Year",aggfunc=np.sum)

df_name_F = df_name[df_name['Gender'] == 'F']

df_year_F = pd.pivot_table(data=df_name_F[["Year","Count"]],index="Year",aggfunc=np.sum)



fig = plt.figure()

ax = plt.subplot2grid((1,1),(0,0))

plt.ylabel('Count')

plt.xlabel('Year')

ax.plot(df_year_M,'steelblue',label="M")

ax.plot(df_year_F,'lightcoral',label="F")

plt.xlim(1880,2014)

plt.legend()

plt.show()
# Was the spike in the 1970s caused by the popularity of Jamie Lee Curtis?



jlc_firstmovie = 1977; #Halloween - her big break!



fig = plt.figure()

ax = plt.subplot2grid((1,1),(0,0))

plt.ylabel('Count')

plt.xlabel('Year')

ax.plot(df_year_M,'steelblue',label="M")

ax.plot(df_year_F,'lightcoral',label="F")

plt.xlim(1970,1990) #Zoom in a little

plt.legend()

ax.add_patch(patches.Rectangle(

    (jlc_firstmovie,0),

    1, #width

    8000, #height

    alpha=0.5) #transparency

    )

plt.show()



name2 = "Jamie"



#Read the data into DataFrame df

df2 = pd.read_csv( "../input/NationalNames.csv")



# set name DataFrame

df_name2 = df[df['Name'] == name2]

# Gender

df_gender2 = pd.pivot_table(data=df_name2[["Gender","Count"]],index="Gender",aggfunc=np.sum)

# Plot Gender Results in a pie chart

plt.pie(df_gender2,labels=['F','M'],startangle=90,colors=['lightcoral','steelblue'])

plt.axis('equal')

plt.title('Jamies of Each Gender 1880-2014')

plt.show()

print('Interestingly, there are significantly more Female Jamies compared to Male than with Jaime.')

print('Let\'s compare Female Jaime vs Jamie:')

df_name_F2 = df_name2[df_name2['Gender'] == 'F']

df_year_F2 = pd.pivot_table(data=df_name_F2[["Year","Count"]],index="Year",aggfunc=np.sum)



fig = plt.figure()

ax = plt.subplot2grid((1,1),(0,0))

plt.ylabel('Count')

plt.xlabel('Year')

ax.plot(df_year_F,'lightcoral',label="Jaime")

ax.plot(df_year_F2,'maroon',label="Jamie")

plt.xlim(1920,2014)

plt.legend()

plt.show()