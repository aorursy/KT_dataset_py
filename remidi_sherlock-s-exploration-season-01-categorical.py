# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import seaborn as sns # plotting the charts

import matplotlib.pyplot as plt 



%matplotlib inline



color = sns.color_palette()

plt.style.use('seaborn-notebook')



pd.options.display.max_columns = 1000  # displaying all the columns on the screen

pd.options.mode.chained_assignment = None



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



print("train data shape {}".format(train_df.shape))

print("test data shape {}".format(test_df.shape))
train_df.head()
ids = pd.DataFrame({"train_ids": train_df.ID, "test_ids": test_df.ID})

ids.head(10)
# describing the 'y' variable

train_df["y"].describe()
# created a reusable function for plotting stacked bar charts



def plot_stack(col, train_color="green", test_color="#0000A3", sortby="total_val", ascending=False, title="categorical X"):

    test_x = dict(test_df[col].value_counts())

    train_x = dict(train_df[col].value_counts())

    test_xd = pd.DataFrame({"cols": list(test_x.keys()), "v_test": list(test_x.values())})

    train_xd = pd.DataFrame({"cols": list(train_x.keys()), "v_train": list(train_x.values())})



    

    total_xd = pd.merge(test_xd, train_xd, how="outer", on="cols")

    total_xd.fillna(0, inplace=True)

    total_xd["total_val"] = total_xd["v_test"] + total_xd["v_train"]

    

    total_xd.sort_values(by=sortby, inplace=True, ascending=ascending)

    

    # plotting the graph

    sns.set_style("darkgrid")

    sns.set_context({"figure.figsize": (20, 9)})

    

    sns.barplot(x = total_xd.cols, y = total_xd.total_val, color = train_color)

    bottom_plot = sns.barplot(x = total_xd.cols, y = total_xd.v_train, color = test_color)

    

    

    # adding legends

    topbar = plt.Rectangle((0,0),1,1,fc=train_color, edgecolor = 'none')

    bottombar = plt.Rectangle((0,0),1,1,fc=test_color,  edgecolor = 'none')

    l = plt.legend([bottombar, topbar], ['train', 'test'], loc=1, ncol = 2, prop={'size':16})

    l.draw_frame(False)

    

    sns.despine(left=True)

    bottom_plot.set_ylabel("frequency")

    bottom_plot.set_xlabel("category")

    bottom_plot.set_title(title, fontsize=15)

    

    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +

             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):

        item.set_fontsize(16)  

        

    plt.show()    

    print("  # of categories : {}".format(len(total_xd)))
# exploring variable X0 

plot_stack("X0", train_color="green", test_color="#0000A3", 

           title="variable 'X0' chart")
# exploring variable X1

plot_stack("X1", train_color="red", test_color="#1220A6", 

           sortby="total_val", 

           ascending=False, 

           title="variable 'X1' chart")
# exploring variable X2

plot_stack("X2", train_color="yellow", test_color="purple", 

           title="variable 'X2' chart")
# exploring variable X3

plot_stack("X3", train_color="aqua", test_color="olive", 

           sortby="cols", 

           title="variable 'X3' chart")
# exploring variable X4

plot_stack("X4", train_color="white", test_color="black", 

           sortby="cols", 

           ascending=False, 

           title="variable 'X4' chart")
# exploring variable X5

plot_stack("X5", train_color="lightgreen", test_color="blue", 

           sortby="cols", 

           ascending=True, 

           title="variable 'X5' chart")
# exploring variable X6

plot_stack("X6", train_color="orange", test_color="darkblue", 

           sortby="cols", 

           ascending=True, 

           title="variable 'X6' chart")
# exploring variable X8, there exists no X7 just incase you missed

plot_stack("X8", train_color="hotpink", test_color="magenta", 

           sortby="cols", 

           ascending=True, 

           title="variable 'X8' chart")