#load libraries and modules



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import numpy as np 

import pandas as pd 

#initial load of data



pokemon = pd.read_csv('../input/Pokemon.csv')
#create a function that compares the character of your choice to the others

def calculations(pokemon_char):

    global pokemon

    

    pokemon = pokemon.set_index('Name')

    

    #total ratio

    pokemon['Vs '+pokemon_char]=pokemon['Total']/pokemon.loc[pokemon_char]['Total'] 

    

    #HP ratio

    pokemon['HP '+pokemon_char] = pokemon['HP']/pokemon.loc[pokemon_char]['HP']

    

    #Attack ratio

    pokemon['Attack '+pokemon_char] = pokemon['Attack']/pokemon.loc[pokemon_char]['Attack'] 

    

    #Defense ratio

    pokemon['Defense '+pokemon_char] = pokemon['Defense']/pokemon.loc[pokemon_char]['Defense']

    

    #Special attack ratio

    pokemon['Sp Atk '+pokemon_char] = pokemon['Sp. Atk']/pokemon.loc[pokemon_char]['Sp. Atk']

    

    #Special defense ratio

    pokemon['Sp Def '+pokemon_char] = pokemon['Sp. Def']/pokemon.loc[pokemon_char]['Sp. Def']

    

    #Speed ratio

    pokemon['Speed '+pokemon_char] = pokemon['Speed']/pokemon.loc[pokemon_char]['Speed']

    

    #sort by total ratio

    pokemon = pokemon.sort_values(by='Vs '+pokemon_char) 

    pokemon = pokemon.reset_index(drop=False)

    

    #rank total power

    pokemon['Rank']= pokemon['Total'].rank(ascending=False).astype('int') #rank total power

    return pokemon

   

    

#functions for plot set-up (ticks, labels, legend, etc.)



def set_up():

    

    #set the overall title

    fig.suptitle(ttl,fontsize = 18, alpha = .9, ha = 'center',

                 va = 'bottom', style = 'normal', weight = 'bold',color = 'black')

    fig.subplots_adjust(hspace=.5) #set height between subplots

    plt.subplots_adjust(top=.9) #set top of subplots



    

def spine_ticks():

    

    ax.grid(linestyle='-',zorder=1,alpha = .1,color='#7570b3') #set grid colors

    ax.yaxis.set_ticks_position('none') #remove y ticks

    ax.xaxis.set_ticks_position('none') #remove x ticks



    #remove spines 

    ax.spines["top"].set_visible(False)  

    ax.spines["right"].set_visible(False)  

    ax.spines["left"].set_visible(False)  

    ax.spines["bottom"].set_visible(False)  







def labels():

    #set label position

    ax.xaxis.set_label_position('bottom')

    #set labels and style

    ax.set_xlabel('Pokemon\n(In Order of Total Power)', ha = 'center',fontsize=12, color='grey')

    ax.set_ylabel(ylabel = 'Ratio to '+pokemon_char, ha = 'center',fontsize=12, color='grey')

    #set label positions

    ax.xaxis.set_label_coords(x=.5,y=-.05)

    ax.yaxis.set_label_coords(x=-.15,y=.5)  



def ticks ():

    #remove tick labels from plots

    ax.set_yticklabels('')

    ax.set_xticklabels('')

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1)) #set the Y-axis major tick values

    

def ticks2 ():

    """

    Set tick labels for first plot 

    

    Note: this code is set-up so that each plot has the same Y scale for easier comparison.

    If you want different scales, you will need to alter tick styles through the rcParams

    

    """

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1)) #set the Y-axis major tick values

    ylabels = ['']+([str(i) for i in range(max_range+1)])

    ax.set_yticklabels(ylabels,size = 10, alpha=.8,horizontalalignment = 'center',color = 'grey', 

                       position=(-.06,.5))

    ax.set_xticklabels('')



def patch():

    

    #set patches underneath subplots

    

    left, width = 0, 1

    bottom, height = 0,1

    right = left + width

    top = bottom + height



    p = patches.Rectangle(

    (left, bottom), width, height,

    transform=ax.transAxes, color = '#ebebeb', clip_on=False, alpha = .1,zorder=1

    )



    ax.add_patch(p)

    

def legend():

    

    #set legend properties

    

    legendframe = 0

    legendcolor = '#ffffff' # set value legend color

    boxx = .5

    boxy = -.2

    boxwidthoffset = 0

    boxheightoffset = 0

    loc = 10

    columns = 2

    legendfontsize = 9

    leg = plt.legend(bbox_to_anchor=(boxx,boxy,boxwidthoffset,boxheightoffset), loc=loc,

                     scatterpoints=2,

           ncol=columns,fontsize=legendfontsize,mode="Expand")



    leg.get_frame().set_linewidth(legendframe)

    leg.get_frame().set_facecolor(legendcolor)

    
#Let's use my favorite monster, Doduo, as an example



pokemon_char = 'Doduo'



#run the calulations function and re-set the pokemon variable to now equal it ..

#note, each time this is re-run, columns will be added to the Pokemon DF



pokemon = calculations(pokemon_char) 



#specify the columns that you will plot



columns_to_use = ['HP '+pokemon_char,

                  'Attack '+pokemon_char,

                  'Defense '+pokemon_char,

                  'Sp Atk '+pokemon_char,

                  'Sp Def '+pokemon_char,

                  'Speed '+pokemon_char]



label_subplots = ['Vs HP','Vs Attack','Vs Defense','Vs Special Attack','Vs Special Defense','Vs Speed']



s = 2 #size of scatter points

a= .6 #global alpha



ttl = pokemon_char+ ' vs the Others' #title for the chart

fig = plt.figure(1,figsize=(11,8),frameon=True)



#loop to create six comparison plots based on columns defined above



for item in range(6):

    set_up()

    max_range = np.ceil((pokemon[columns_to_use].max()).max()).astype('int')

    ax = fig.add_subplot(2,3,item+1,xlim=(-1,801),ylim=(0,max_range),title=label_subplots[item])

 

    spine_ticks()

    plt.plot(pokemon.index,pokemon['Vs '+pokemon_char],

        color='black',zorder=1,linewidth=2,alpha=.5,label = 'Vs '+pokemon_char+' Total')

    

    plt.scatter(x=pokemon[pokemon['Vs '+pokemon_char]<1].index,

                y=pokemon[pokemon['Vs '+pokemon_char]<1][columns_to_use[item]],

                color='#377eb8',label = 'Total - Less Powerful Than '+pokemon_char,

                s=s,marker='o',alpha=a)

    

    plt.scatter(x=pokemon[pokemon['Vs '+pokemon_char]==1].index,

                y=pokemon[pokemon['Vs '+pokemon_char]==1][columns_to_use[item]],

                color='#e41a1c',label = 'Total - Equal to '+pokemon_char,s=s,marker='o',alpha=1)

    

    plt.scatter(x=pokemon[pokemon['Vs '+pokemon_char]>1].index,

                y=pokemon[pokemon['Vs '+pokemon_char]>1][columns_to_use[item]],

                color='#33a02c'  ,label = 'Total - More Powerful Than '+pokemon_char,

                s=s,marker='o',alpha=a)

    

    if item == 0:

        labels ()

        ticks2 ()

        

    else:

        ticks()

        

    if item == 4:

        legend()

    else:

        pass

    

    if item == 2:

        ax.text(0,1.26,pokemon_char+' total power: '+\

                str(pokemon[pokemon['Name']==pokemon_char]['Total'].values[0]),

                transform=ax.transAxes, weight='bold',size=12,alpha=.6)

        

        ax.text(0,1.19,'Rank = '+str(pokemon[pokemon['Name']==pokemon_char]['Rank'].values[0])+\

                '/'+str(len(pokemon)),

                

        transform=ax.transAxes, weight='bold',size=10,alpha=.6)

        

    else:

        pass

    

    

plt.show()