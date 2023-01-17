import numpy as np

import pandas as pd

import matplotlib
diamonds = pd.read_csv("../input/diamonds/diamonds.csv")

diamonds = diamonds.drop("Unnamed: 0", axis=1)



print(diamonds.shape)        # Check data shape



diamonds.head(5)
diamonds.hist(column="carat",        # Column to plot

              figsize=(8,8),         # Plot size

              color="blue");         # Plot color
diamonds.hist(column="carat",        # Column to plot

              figsize=(8,8),         # Plot size

              color="blue",          # Plot color

              bins=50,               # Use 50 bins

              range= (0,3.5));       # Limit x-axis range
diamonds[diamonds["carat"] > 3.5]
diamonds.boxplot(column="carat");
diamonds.boxplot(column="price",        # Column to plot

                 by= "clarity",         # Column to split upon

                 figsize= (8,8));       # Figure size
diamonds.boxplot(column="carat",        # Column to plot

                 by= "clarity",         # Column to split upon

                 figsize= (8,8));       # Figure size
diamonds["carat"].plot(kind="density",  # Create density plot

                      figsize=(8,8),    # Set figure size

                      xlim= (0,5));     # Limit x axis values
carat_table = pd.crosstab(index=diamonds["clarity"], columns="count")

carat_table
carat_table.plot(kind="bar",

                 figsize=(8,8));
carat_table = pd.crosstab(index=diamonds["clarity"], 

                          columns=diamonds["color"])



carat_table
carat_table.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True);
carat_table.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=False);
diamonds.plot(kind="scatter",     # Create a scatterplot

              x="carat",          # Put carat on the x axis

              y="price",          # Put price on the y axis

              figsize=(10,10),

              ylim=(0,20000));
# Create some data

years = [y for y in range(1950,2016)]



readings = [(y+np.random.uniform(0,20)-1900) for y in years]



time_df = pd.DataFrame({"year":years,

                        "readings":readings})



# Plot the data

time_df.plot(x="year",

             y="readings",

             figsize=(9,9));
my_plot = time_df.plot(x="year",     # Create the plot and save to a variable

             y="readings",

             figsize=(9,9))



my_fig = my_plot.get_figure()            # Get the figure



my_fig.savefig("line_plot_example.png")  # Save to file