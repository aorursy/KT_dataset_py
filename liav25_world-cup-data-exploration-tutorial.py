# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option("display.max_rows",8)

from plotnine import *



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data

PATH = "/kaggle/input/fifa-world-cup/"



players = pd.read_csv(PATH+"WorldCupPlayers.csv")

cups = pd.read_csv(PATH+"WorldCups.csv")

matches = pd.read_csv(PATH+"WorldCupMatches.csv").dropna()
cups.head()
#ggplot(dataframe) is creating the graph objectm and geom_line() is a line plot layer.

# note: when we use columns from the dataframe, we must put them inside the aesthetics (aes=()...)

p = ggplot(cups) + geom_line(aes(x="Year", y="GoalsScored")) 

#show the object. When using Pycharm, print it.

p
# Nice trick: if you wrap your ggplot object with round brackets, you can drop down to a new row.

# This way, you can make your code more readable and add comments



#add labels

p = (p + labs(title = "Fifa World Cup Goals Scored per Year",x="Tournament Year", y='Goals Scored') + 

    #add more xticks

    scale_x_continuous(breaks = range(1930,2014,8)) + 

    #add theme.

    theme_bw())



# you can use draw() function and add ; if you want avoid the <ggplot...> object tag.

p.draw();
#create our barplot and plot it

(ggplot(cups) + geom_bar(aes(x="Winner"))).draw();
#create a sorted list of national teams, by number of wins

sorted_champions = cups['Winner'].value_counts().index.tolist()



print("Sorted list by winnings:\n",sorted_champions)



#create the plot object

(ggplot(cups)    

     #add bars colored by values

    + geom_bar(aes(x='Winner', fill='Winner')) 

     #sort by sorted list of values

    + scale_x_discrete(limits=sorted_champions) 

     #add xticks angle

    + theme(axis_text_x = element_text(angle = 45)) 

     #add title and axis labels

    + labs(title="World Cup Wins", x='National Team',y='Titles')).draw();
(ggplot(cups)    

     #add bars colored by values

    + geom_bar(aes(x='Winner', fill='blue')));
(ggplot(cups)    

     # add bars colored by plain color

    + geom_bar(aes(x='Winner'), fill='orangered')

     # add angle to xticks

    + theme(axis_text_x = element_text(angle = 45))

     # add titles

    + labs(title="World Cup Wins", x='National Team',y='Titles')).draw();
(ggplot(cups)    

     # add winners bars colored in red, width smaller width

    + geom_bar(aes(x='Winner'), fill='orangered', width=.3)

     # add runner up thin bars colored in blue, nuged to the right

    + geom_bar(aes(x="Runners-Up"), fill='lightblue', position = position_nudge(x = 0.3), width=.3)

     # rotate xticks

     + theme(axis_text_x = element_text(angle = 45))

     # add title

    + labs(title="World Cup Wins", x='National Team',y='Titles')).draw();

cups[['Winner','Runners-Up']].melt()
#create ggplot object over melted data

(ggplot(cups[['Winner','Runners-Up']].melt()) + 

     # add bars, color splitted by variable (winner or runner up)

     # position = 'dodge' is for side-by-side bars. for stacked bars use position="stack"

     geom_bar(aes(x='value', fill='variable'),position = "dodge", width=0.9)

     # rotate xticks and  set figure size

     + theme(axis_text_x = element_text(angle = 45), figure_size=[10,4])

     # add title

    + labs(title="World Cup Wins", x='National Team',y='Titles')

     # change legend title and set new legend labels

    + scale_fill_discrete(name = "Final Place", labels=["2nd Place","1st Place"])).draw();
matches.head()

#create ggplot object for matches data

(ggplot(matches) 

     # add boxplots. We need to use the Year.astype() since year is a float

     + geom_boxplot(aes(x='Year.astype("str")', y='Attendance'))

     # rotate xticks and  set figure size

     + theme(axis_text_x = element_text(angle = 45), figure_size=[8,4])

     # add title

    + labs(title="World Cups Attendance by Year", x='Year')

     # change legend title and set new legend labels

    ).draw();
matches['Total Goals'] = matches['Home Team Goals'] + matches['Away Team Goals']



p3 = (ggplot(matches) + geom_boxplot(aes(y="Total Goals", x='Year.astype(str)', fill='Year'))

      + theme(axis_text_x = element_text(angle = 90), figure_size=[8,4])

     # add title

    + labs(title="World Cups Goals by Year", x='Year'))

     # change legend title and set new legend labels



p3.draw();
matches.Stage[matches['Stage'].str.contains("Group")] = "Group Stage"

matches.Stage[~matches['Stage'].str.contains("Group")] = "Knockout"



p3 + facet_wrap("~Stage")