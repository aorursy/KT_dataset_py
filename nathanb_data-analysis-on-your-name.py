# data analysis on my name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

my_name = "Nathan"


df_me = pd.read_csv( "../input/StateNames.csv").drop("Id" , axis = 1)
df_me = df_me[df_me["Name"] == my_name]


df_me_gender = pd.pivot_table(data = df_me[["Gender" , "Count"]] , index = "Gender" , aggfunc = np.sum ) #interesting some girls have my name
df_me_gender.plot(kind = "pie" , subplots = True , autopct = "%.2f")
#year analysis
df_me_year = pd.pivot_table(data = df_me[["Year" , "Count"]] , index = "Year" , aggfunc = np.sum ) 
df_me_year.plot()
print ( " -----  top years ------- \n\n"  )
print ( df_me_year.sort_values(by = "Count" , ascending = False)[:5] )
#state analysis
df_me_state = pd.pivot_table(data = df_me[["State" , "Count"]] , index = "State" , aggfunc = np.sum ).sort_values(by = "Count" , ascending = False) 
df_me_state[:10].plot(kind = "barh" )

#top 5 most frequent
print (  df_me.sort_values(by = "Count"  , ascending = False)[:10]  )


#for the top 5 states, distribution by year
top_states = list(df_me_state[:5].index)
colors = ['r' ,'b' , 'm' , 'g' ,'y']
plt.figure()
for i , state in enumerate(top_states):   
    dff = df_me[df_me["State"] == state]
    dff_year = pd.pivot_table(data = dff[["Year" , "Count"]] , index = "Year" , aggfunc = np.sum )
    values = [x[0] for x in dff_year.values]
    index = list(dff_year.index )
    #graph = plt.scatter(x= index, y = values , color = colors[i] , label = state)
    plt.plot( index,  values , color = colors[i] , label = str(state))
    legend = plt.legend(loc='upper center', shadow=True)

