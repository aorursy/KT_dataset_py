import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
TrainDF = pd.read_csv("../input/train.csv")

TestDF = pd.read_csv("../input/test.csv")



print("Columns")

print("\t".join(TrainDF.columns.tolist()))
def cleanupDF(dataframe, train = True):

    # Remove the ID column

    ID = dataframe["id"]

    df1 = dataframe.drop('id', axis=1)

    

    # Split the colors

    df1 = pd.get_dummies(df1, columns=['color'], prefix = "", prefix_sep="")

    

    if(train == True):

        # Remove the type

        y = df1['type']

        df1 = df1.drop('type', axis = 1)

        return (ID, df1, y)

    else:

        return (ID, df1)

    
ID, trainDFC, y = cleanupDF(TrainDF) 

ID, testDFC = cleanupDF(TrainDF, train= False)

print(trainDFC)
# Make the plots for each feature

plots = ["bone_length",  "rotting_flesh",  "hair_length",  "has_soul"]

fig, ax = plt.subplots(nrows=len(plots), figsize= (12, 15))



# Control the width and placements

width = 0.2

num = 11

types = list(y.unique())

color = ['r', 'g', 'b']

indx = np.arange(num-1)



# Iterate through the plots

for i, plot in enumerate(plots):

    

    cuts, bins = pd.cut(trainDFC[plot], np.linspace(0.0, 1.0, num), labels = False, retbins = True)

    

    # Iterate through the types plotting their data

    for v, tp in enumerate(types):

        mask = (y == tp)

        bars = [] 

        for n in range(0,num-1):

            # Count how many are of each cut

            maskBin = (cuts == n)

            

            # Make this sum the number of true statements.

            t = ((mask) & (maskBin)).sum()

            bars.append(t)

            

        ax[i].bar(indx + v*width, bars, color=color[v],

                  width= width, align= "center", label = tp)

    ax[i].set_title(plot)

    ax[i].set_xticks(range(0, num-1))

    ax[i].set_xlim([-1, num-1])

    ax[i].legend(loc="upper right")

    
colours = ["black", "blood", "blue", "clear", "green", "white"]



fig, ax = plt.subplots(nrows= len(colours), figsize= (12, 15))



for i, colour in enumerate(colours):

    # Start the enumeration at 1 to 

    for v, mon in enumerate(types, 1):

        mask1 = (y== mon)

        mask2 = (trainDFC[colour] == 1)

        ax[i].bar(v, (mask1 & mask2).sum(), align= "center", color=color[v-1], label = mon)

        

    ax[i].set_xticks(np.arange(1,len(types) + 1))

    ax[i].set_xticklabels(types)

    ax[i].set_title(colour)



    # Make each y-axis on the same scale.

    ax[i].set_ylim([0, 60])

    ax[i].set_xlim([0.4, 4.0])





    ax[i].set_ylabel("Count")

    ax[i].legend(loc= "upper right")

fig.subplots_adjust(top= 0.96, bottom= 0.03, hspace= 0.3)


for colour in colours:

    fig, ax = plt.subplots(nrows = len(plots),figsize= (10,15))

    maskcol = (trainDFC[colour] == 1)

    

    # Iterate through the plots    

    for i, plot in enumerate(plots):

        cuts, bins = pd.cut(trainDFC[plot], np.linspace(0.0, 1.0, num), labels = False, retbins = True)

        

        # Iterate through the species of monsters

        for v, tp in enumerate(types):

            mask = (y == tp)

            

            bars = [] 

            for n in range(0,num-1):

                # Count how many are of each type

                maskBin = (cuts == n)



                # Make this sum the number of true statements.

                t = ((mask) & (maskBin) & (maskcol)).sum()

                bars.append(t)



            ax[i].bar(indx + v*width, bars, color=color[v],

                      width= width, align= "center", label = tp)

        

        ax[i].set_title(plot.capitalize())

        ax[i].set_xticks(range(0, num-1))

        ax[i].set_xlim([-0.1, num-1])

        ax[i].set_ylim([0, 16])

        ax[i].legend(loc="upper right")

    fig.suptitle(colour.capitalize(), fontsize=20, fontweight='bold')