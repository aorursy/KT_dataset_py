import matplotlib.pyplot as plt
TrainDF = pd.read_csv("train.csv")

TestDF = pd.read_csv("test.csv")



print("Columns")

print("\t".join(TrainDF.columns.tolist()))
%matplotlib notebook
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



# Iterate through the plots

for i, plot in enumerate(plots):

    # Number of bins required

    num = 11

    indx = np.arange(num-1)

    

    cuts, bins = pd.cut(trainDFC[plot], np.linspace(0.0, 1.0, num), labels = False, retbins = True)

    types = list(y.unique())

    

    color = ['r', 'g', 'b']

    # Plot the bar chart

    for v, tp in enumerate(types):

        mask = (y == tp)

        bars = [] 

        for n in range(0,num-1):

            # Count how many are of each type

            maskBin = (cuts == n)

            

            # Make this sum the number of true statements.

            t = ((mask) & (maskBin)).sum()

            bars.append(t)

            

        ax[i].bar(indx + v*width, bars, color=color[v],

                  width= width, align= "center", label = tp)

        plt.legend

    ax[i].set_title(plot)

    ax[i].set_xticks(range(0, num-1))

    ax[i].set_xlim([-1, num-1])

    ax[i].legend(loc="upper right")

    