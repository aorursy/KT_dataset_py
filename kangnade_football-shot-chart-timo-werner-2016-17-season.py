from IPython.display import Image

Image("../input/timo_werner_shot_chart.png")
from IPython.display import Image

Image("../input/rb_leipzig_shots_chart.png")
%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd
# Standard football pitch has a width from x,y coordinate (-34, 0) to (34, 0), the other side is from (0, 105) to (34, 105)

# and a length from x,y coordinate range(-34, 0) to (-34, 105) and (34,0) to (34, 105), these are the two length lines.

# The half-way line is from  (-34, 105/2) and to (34, 105/2)

# Penalty area starts from ((7.32/2 + 5.5 + 11), 0) to ((7.32/2 + 5.5 + 11), 16.5), so it has a total width of 40.3 and 

# a length of 16.5

# Penalty point is at (0, 11)

# Goal Area starts from (7.32/2+5.5, 0) and has a width of (7.32 + 5.5 * 2) and height of 5.5

# Half-way center circle has x,y coord (0, 105/2) and a radius of 9.15

# The penalty area arc has a x,y coord of (0, 16.5), width = 18.3, height = 3.65

# Corner kick arc is no more than radius 1

# ALL PARAMETERS ARE MULTIPLIED BY SCALE



from matplotlib.patches import Circle, Rectangle, Arc, Ellipse



def plot_pitch_full_vertical(ax=None, color='black', lw=2, scale = 15):

    # get the current ax if ax is None

    if ax is None:

        ax = plt.gca()



    # Let's draw the standard football pitch (soccer pitch)

    

    # First, draw the overall pitch rectangle

    pitch_box = Rectangle((-34 * scale, 0), width = 68 * scale, height = 105 * scale, 

                          linewidth=lw, color=color, fill=False)

    

    # Plot the lower part's panelty area

    # Eliminate fill attribute, and set facecolor (fc) to white color

    # Below, the penalty_arc_low has an attribute zorder = 0, which means it gets plotted behind

    penalty_area_lower =  Rectangle((-(7.32 * scale / 2+ 5.5 * scale +11 * scale),0), 

                                 width = (5.5 * scale * 2 + 11 * scale * 2 + 7.32 * scale), height = 16.5 * scale,

                                 linewidth = lw, color = color, fc = "white")

    

    # Plot the lower goal area

    goal_area_lower = Rectangle((-(7.32 * scale/ 2 + 5.5 * scale), 0), width = 7.32 * scale + 5.5 * scale * 2, 

                              height = 5.5 * scale, linewidth = lw, color = color, fill = False)

    

    # Plot the lower penalty kick point

    penalty_point_lower = Circle((0, 11 * scale), radius = 3, color = color)

    

    # Plot the arc out side the lower penalty area

    # The zorder = 0 makes the circle plottted behind, and so the arc is shown

    penalty_arc_lower = Circle((0, 11 * scale), radius = 9.15 * scale, color = color, lw = lw, fill = False,

                            zorder = 0)

    

    # Lower part's goal

    goal_lower = Rectangle((-3.66 * scale, -20), width = 7.32 * scale, height = 20, lw = lw, color = color,

                          fill = False)

    

    # Half-way center line starting from -34 * scale and has a length across the field, but set the height to

    # zero so it is plotted as a line

    center_line = Rectangle((-34 * scale, 105 * scale / 2), width = 68 * scale, height = 0, lw = lw, color = color)

    

    # Half-way center circle has a radius of 9.15 meters and must multiply scale on our plot, the center of the

    # circle is located at (0, 105/2 * scale)

    center_circle = Circle((0, 105 * scale / 2), radius = 9.15 * scale, lw = lw, color = color, fill = False)

    

    # Half-way center point

    center_point = Circle((0, 105 * scale / 2), radius = 3, lw = lw, color = color)

    

    # So Half of the picture is finished now, it's time to plot the upper part of the whole pitch

    

    # First let's plot the upper part penalty area

    penalty_area_upper =  Rectangle((-(7.32 * scale / 2+ 5.5 * scale +11 * scale),(105 - 16.5) * scale), 

                                    width = (5.5 * scale * 2 + 11 * scale * 2 + 7.32 * scale), height = 16.5 * scale,

                                    linewidth = lw, color = color, fc = "white")

    

    # The upper goal area

    goal_area_upper = Rectangle((-(7.32 * scale/ 2 + 5.5 * scale), (105 - 5.5) * scale), width = 7.32 * scale + 5.5 * scale * 2, 

                              height = 5.5 * scale, linewidth = lw, color = color, fill = False)

    

    # The upper part's penalty point

    penalty_point_upper = Circle((0, (105 - 11) * scale), radius = 3, color = color)

    

    # The arc outside the penalty area in the upper part

    penalty_arc_upper = Circle((0, (105 - 11) * scale), radius = 9.15 * scale, color = color, lw = lw, fill = False,

                            zorder = 0)

    

    # Upper goal

    goal_upper = Rectangle((-3.66 * scale, 105 * scale), width = 7.32 * scale, height = 20, lw = lw, color = color,

                          fill = False)

    

    # Finally we plot the 4 corner areas left

    corner_left_lower = Arc((-34 * scale , 0), width = 2 * scale, height = 2 * scale, theta1 = 0, theta2 = 90, lw = lw, fill = False)

    

    # Lower right corner area

    corner_right_lower = Arc((34 * scale , 0), width = 2 * scale, height = 2 * scale, theta1 = 90, theta2 = 180, lw = lw, fill = False)

    

    # Upper left corner area

    corner_left_upper = Arc((-34 * scale , 105 * scale), width = 2 * scale, height = 2 * scale, 

                            theta1 = -90, theta2 = 0, lw = lw, fill = False)

    

    # Upper right corner area

    corner_right_upper = Arc((34 * scale , 105 * scale), width = 2 * scale, height = 2 * scale, 

                            theta1 = 180, theta2 = -90, lw = lw, fill = False)

    

    # List of the football pitch elements to be plotted onto the axes

    court_elements = [pitch_box, penalty_area_lower, goal_area_lower, penalty_point_lower,

                     penalty_arc_lower, goal_lower, center_line, center_circle, center_point,

                     penalty_area_upper, goal_area_upper, penalty_point_upper, penalty_arc_upper,

                     goal_upper, corner_left_lower, corner_right_lower, corner_left_upper,

                     corner_right_upper]





    # Add the court elements onto the axes

    for element in court_elements:

        ax.add_patch(element)



    return ax
plt.figure(figsize=(14, 20))

plt.xlim(-600,600)

plt.ylim(-100,1700)

plot_pitch_full_vertical()

plt.show()
def plot_pitch_half_vertical(ax=None, color='black', lw=2, scale = 15):

    # get the current ax if ax is None

    if ax is None:

        ax = plt.gca()





    # Plot the overall pitch but with only the lower part

    pitch_box =  Rectangle((-34 * scale, 0), width = 68 * scale, height = 105 / 2 * scale, linewidth=lw, color=color, fill=False)

    

    

    # Below, the penalty_arc_low has an attribute zorder = 0, which means it gets plotted behind

    penalty_area_lower =  Rectangle((-(7.32 * scale / 2+ 5.5 * scale +11 * scale),0), 

                                 width = (5.5 * scale * 2 + 11 * scale * 2 + 7.32 * scale), height = 16.5 * scale,

                                 linewidth = lw, color = color, fc = "white")

    

    # Plot the lower goal area

    goal_area_lower = Rectangle((-(7.32 * scale/ 2 + 5.5 * scale), 0), width = 7.32 * scale + 5.5 * scale * 2, 

                              height = 5.5 * scale, linewidth = lw, color = color, fill = False)

    

    # Plot the lower penalty kick point

    penalty_point_lower = Circle((0, 11 * scale), radius = 3, color = color)

    

    # Plot the arc out side the lower penalty area

    # The zorder = 0 makes the circle plottted behind, and so the arc is shown

    penalty_arc_lower = Circle((0, 11 * scale), radius = 9.15 * scale, color = color, lw = lw, fill = False,

                            zorder = 0)

    

    # new_arc_lower = Arc((0, 16.5 * scale), width = 15 * scale, height = 10 * scale, theta1 = 0, theta2 = 180,

                        # color = color, lw = lw, fill = False)

    

    # Lower part's goal

    goal_lower = Rectangle((-3.66 * scale, -20), width = 7.32 * scale, height = 20, lw = lw, color = color,

                          fill = False)

    

    # Finally we plot the 4 corner areas left

    corner_left_lower = Arc((-34 * scale , 0), width = 2 * scale, height = 2 * scale, theta1 = 0, theta2 = 90, lw = lw, fill = False)

    

    # Lower right corner area

    corner_right_lower = Arc((34 * scale , 0), width = 2 * scale, height = 2 * scale, theta1 = 90, theta2 = 180, lw = lw, fill = False)

    

    # Center half-way half circle, using circle

    # through 180 to 360 in theta1 and theta2

    center_circle = Arc((0, 105 * scale /2), width = 9.15 * scale * 2, height = 9.15 * scale * 2, 

                  theta1 = 180, theta2 = 360, lw = lw, color = color, fill = False)

    

    # Center point

    center_point = Circle((0, 105 * scale / 2), radius = 3, lw = lw, color = color)

    

    # List of elements to be plotted

    pic_elements = [pitch_box, penalty_area_lower, penalty_point_lower, goal_area_lower, penalty_arc_lower,

                   goal_lower, corner_left_lower, corner_right_lower, center_circle, center_point]





    # Add the elements onto the axes

    for element in pic_elements:

        ax.add_patch(element)



    return ax
plt.figure(figsize=(16, 16))

plt.xlim(-600,600)

plt.ylim(-100,1000)

plot_pitch_half_vertical()

plt.show()
# Import the libraries that we need for this part:

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib.offsetbox import  OffsetImage

from matplotlib.patches import Circle, Rectangle, Arc, Ellipse, FancyArrow
# After loading the packages that we need for plotting the shot chart for Timo Werner,

# we upload the scraped data

werner = pd.read_excel("../input/timowerner_shots.xlsx")

werner.head(10)
# The issue here is that this dataset is used for FourFourTwo's (442) graph, which has

# the goal on the top. My pitch plot has goal on the bottom. The 442's left side is on my

# chart's right side, and the 442's right side should be on my chart's left side.

# In addition, the scale on my chart is different from the scale on 442.

# As a result, we need to flip the x-axis of the data entries, and also adjust the y-axis

# entries to fit the data on my own pitch chart.



# The first thing to do is to use 442 to check certain shots' positions, and compare it with

# the standard football (soccer) pitch dimensions. Here I use 105 x 68 pitch dimensions.

# From analysis and examination, I concluded the following:



# 442's chart has a height of 497 and width of 713.

# On x-axis, the flip point of x=358.16 on 442's chart is the x = 0 on my chart.

# Each standard football pitch's 1 meter equals 14 units on 442's chart. So real : 442 is 1 : 14

# On y-axis, the total length is 497, my plot's y-axis starts from y = 0, while 442 starts from

# y = 140 to 145

# So to adjust y-axis, simply use the y entries to minus 145 or 140, (y-145) or (y-140) to adjust y values

# The ratio of 1 meter on real football pitch has a ratio to 442's chart of 12.01, so roughly real : 442 is 1 : 12



# We adjust Shot_x1 and Shot_x2 by using the following equation:

# -(Shot_x-axis - 358.16) / 14

# The minus sign helps to flip the right side on 442 to left side on my chart, and vice versa.



# For Shot_y1 and Shot_y2 we use the following equation:

# (Shot_y-axis - 140)/12



# I'll define the following function for our purpose

def transformXY (df, scale = 15):

    df['act_Shot_x1'] = -((df['Shot_x1'] - 358.16)/14) * scale

    df['act_Shot_y1'] = ((df['Shot_y1'] - 145)/12) *scale

    df['act_Shot_x2'] = -((df['Shot_x2'] - 358.16)/14) * scale

    df['act_Shot_y2'] = ((df['Shot_y2'] - 145)/12) * scale

    return df
# Apply this to the werner dataset

werner = werner.apply(transformXY, axis = 1)

werner.head(10)
# To get a color column indicating the color of the specific shots, we need to apply the following function

def addColorCol(df):

    df['shotTypeColor'] = None

    for i in range(df.shape[0]):

        # df.columns.get_loc("") returns the index of the "" column

        if df.iloc[i,df.columns.get_loc("Shot Type")] == 'On Target Saved':

            df.iloc[i, df.shape[1] -1] = 'b'

        if df.iloc[i,df.columns.get_loc("Shot Type")] == 'Off Target':

            df.iloc[i, df.shape[1] -1] = 'r'

        if df.iloc[i,df.columns.get_loc("Shot Type")] == 'Blocked':

            df.iloc[i, df.shape[1] -1] = 'k'

        if df.iloc[i,df.columns.get_loc("Shot Type")] == 'Goal':

            df.iloc[i, df.shape[1] -1] = 'y'

    return df
# Apply the function on the dataset

addColorCol(werner)
# We can then separate the whole werner dataset into four subsets, each based

# Goal

goals = werner.loc[werner['Shot Type'] == 'Goal']

goals.shape[0]
# On Target Saved

onTargetSaved = werner.loc[werner['Shot Type'] == 'On Target Saved']

onTargetSaved.shape[0]
# Off Target

offTarget =  werner.loc[werner['Shot Type'] == 'Off Target']

offTarget.shape[0]
# Blocked

blocked = werner.loc[werner['Shot Type'] == 'Blocked']

blocked.shape[0]
# Sum up the total shots

goals.shape[0] + onTargetSaved.shape[0] + offTarget.shape[0] + blocked.shape[0]
# Check if everyting matches the original legnth of the data

werner.shape[0]
# Nope, why? Because there are two matches where Timo didn't achieve a single shot

# We can check if this is true by asking where are the two null entries

# as long as x1 is null, y1, x2, y2 is null

werner[werner.Shot_x1.isnull()]
# So we see that here, Timo unfortunately couldn't achieve a single shot in these two games

# While these won't influence the subsets Goal, offTarget, onTargetSaved and Blocked, because these

# will automatically drop the Shot Type of NaN, but when we are plotting the graph, we do not want to

# be influenced by these NaN, I am actually not sure if these will influence the graphs. So I will create a 

# new dataset called werner_dropna, where we drop the NaN

werner_dropna = werner.dropna()
def addZone(df):

    # First, define the area within the goal box

    if (df['act_Shot_y1'] <= 6 * 15) and (df['act_Shot_x1'] >= -9 * 15 and df['act_Shot_x1'] <= 9 * 15):

        value = 'Goal Box'

    # The first boolean categorizes the penalty box area on the right side of the goal box

    # The second boolean categorizes the penalty box area on the left side of the goal box

    # The third boolean categorizes the penalty box area in the middle between the goal box and outskirt of penalty box

    elif ((df['act_Shot_y1'] <= 16 * 15) and ((df['act_Shot_x1'] > 9 * 15 and 

                                                 df['act_Shot_x1'] <= 20 * 15))):

        value = 'Penalty Area'

    elif ((df['act_Shot_y1'] <= 16 * 15) and ((df['act_Shot_x1'] < -9 * 15 and 

                                                 df['act_Shot_x1'] >= -20 * 15))):

        value = 'Penalty Area'

    elif ((df['act_Shot_y1'] <= 16 * 15 and df['act_Shot_y1'] > 6 * 15) and ((df['act_Shot_x1'] >= -9 * 15 and 

                                                 df['act_Shot_x1'] <= 9 * 15))):

        value = 'Penalty Area'

    # Categorize all the rest as outside penalty area shots

    else:

        value = 'Outside Penalty'

    return value
# I also add a detailed shot zone, indicating the left or right wing

def addDetailedZone(df):

    # First, define the area within the goal box

    if (df['act_Shot_y1'] <= 6 * 15) and (df['act_Shot_x1'] >= -9 * 15 and df['act_Shot_x1'] <= 9 * 15):

        value = 'Goal Box'

    # The first boolean categorizes the penalty box area on the right side of the goal box

    # The second boolean categorizes the penalty box area on the left side of the goal box

    # The third boolean categorizes the penalty box area in the middle between the goal box and outskirt of penalty box

    elif ((df['act_Shot_y1'] <= 16 * 15) and ((df['act_Shot_x1'] > 9 * 15 and 

                                                 df['act_Shot_x1'] <= 20 * 15))):

        value = 'Penalty Area Left Wing'

    elif ((df['act_Shot_y1'] <= 16 * 15) and ((df['act_Shot_x1'] < -9 * 15 and 

                                                 df['act_Shot_x1'] >= -20 * 15))):

        value = 'Penalty Area Right Wing'

    elif ((df['act_Shot_y1'] <= 16 * 15 and df['act_Shot_y1'] > 6 * 15) and ((df['act_Shot_x1'] >= -9 * 15 and 

                                                 df['act_Shot_x1'] <= 9 * 15))):

        value = 'Penalty Area Middle'

    # Categorize all the rest as outside penalty area shots

    else:

        value = 'Outside Penalty'

    return value
werner['shotZone'] = werner.apply(addZone, axis=1)

werner['shotZoneDetailed'] = werner.apply(addDetailedZone, axis = 1)

werner.head(20)
# After adding the zones, we want to calculate the scoring rate based on each zone:

werner.groupby('shotZone')['Shot Type'].apply(lambda s: '{0:.2f}%'.format((s[s=='Goal']).size/(s.size) * 100))
# We can also calculate the same for the amount of shot attempts in each zone

werner.groupby('shotZone')['Shot Type'].apply(lambda s: '{0:d}'.format(s.size))
# The same can also be applied to detailed shot zone

werner.groupby('shotZoneDetailed')['Shot Type'].apply(lambda s: '{0:.2f}%'.format((s[s=='Goal']).size/(s.size) * 100))
# Calculate the number of attempts of shot with detailed shot zone

werner.groupby('shotZoneDetailed')['Shot Type'].apply(lambda s: '{0:d}'.format(s.size))
# Total scoring rate of Timo Werner

'Scoring Rate: {0:.2f}%'.format(werner[werner['Shot Type'] == 'Goal'].shape[0] / werner['Shot Type'].count() * 100)
# Using both goal and on target saved, calculate the scoring rate of on target shot

'Scoring Rate: {0:.2f}%'.format((werner[werner['Shot Type'] == 'Goal'].shape[0] + 

                                 werner[werner['Shot Type'] == 'On Target Saved'].shape[0]) / werner['Shot Type'].count() * 100)
joint_shot_chart = sns.jointplot(werner.act_Shot_x1, werner.act_Shot_y1, stat_func=None,

                                 kind='reg', space=0, color='orange')



# Set figure ratio

joint_shot_chart.fig.set_size_inches(16,10)



# Assign ax with the joinplot's ax_joint

ax = joint_shot_chart.ax_joint



# Clear scatter plot and reg plot

joint_shot_chart.ax_joint.cla() # This allows us the cover new layers of scatter plots with different color on it



# Plot the football pitch half vertical background

plot_pitch_half_vertical(ax)



# Set the ax's limits

ax.set_xlim(-34 * 15,34 * 15)

ax.set_ylim(-50, 600)



# Get rid of the labels and axis ticks

ax.set_xlabel('')

ax.set_ylabel('')

ax.set_xticks([])

ax.set_yticks([])



# Add chart title

ax.set_title('Timo Werner Shot Chart, 2016/2017 Season, Bundesliga', y=1.2, fontsize=18,

            family = 'fantasy', color = 'r')



# Add footnote

ax.text(-34*15, -100,'Data Source: www.fourfourtwo.com/statszone'

        '\nAuthor: Nade Kang (kangnade@gmail.com)',

        fontsize=12)



# Add in elements



# Cover the jointplot's points with scatter plot of points in goals subset

ax.scatter(goals.act_Shot_x1, goals.act_Shot_y1, label = 'Goal', color = 'white', edgecolors='purple', zorder = 10, s = 60,

          linewidths = 4)



# Cover the jointplot's points with scatter plot of points in onTargetSaved subset

ax.scatter(onTargetSaved.act_Shot_x1, onTargetSaved.act_Shot_y1, label = 'On Target Saved', color = 'white', edgecolors = 'blue',

           zorder = 10, s = 60, alpha = 0.4, linewidths = 4)



# Cover the jointplot's points with scatter plot of points in offTarget subset

ax.scatter(offTarget.act_Shot_x1, offTarget.act_Shot_y1, label = 'Off Target', color = 'red', zorder = 10 , alpha = 0.4)



# Cover the jointplot's points with scatter plot of points in Blocked subset

ax.scatter(blocked.act_Shot_x1, blocked.act_Shot_y1, label = 'Blocked', color = 'black', zorder = 10, alpha = 0.4)





# Add in the additional boxes for stats

# Add the specific shot zones in the graph with the scoring rate

# Add goal box area rectangle:

goalBoxZone = Rectangle((-9.5 * 15, 0), width = 19 * 15, height = 6 * 15, color = "orange", lw = 2, ls = '--',

                            fill = False, zorder = 10, alpha = 0.5)

ax.add_artist(goalBoxZone)

# Add text for the goal box area rectangle:

ax.text(-16 * 15, -1.5 * 15, 

        "Goal Box Scoring Rate: ",

       fontsize = 10,

       color = "orange",

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)

ax.text(-16 * 15, -3 * 15,

       '33.33%',

       fontsize = 20,

       color = 'orange',

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)

# Add annotation arrow

arrowBoxZone = FancyArrow(-5 * 15, -1 * 15, 3 * 15, 3 * 15, width=2, length_includes_head=True, 

                          head_width=None, head_length=None, shape='full', zorder = 10,

                         ec = 'orange', fc = 'orange', overhang = 1)

ax.add_artist(arrowBoxZone)





# Add goalbox left wing area rectangle:

PBoxZoneLeftWing = Rectangle((9.5 * 15, 0), width = 11 * 15, height = 16.5 * 15, color = "b", lw = 2.5, ls = '-.',

                            fill = False, zorder = 10, alpha = 0.5)

ax.add_artist(PBoxZoneLeftWing)

# Add text for left wing rectangle

ax.text(12.3 * 15, 3 * 15, 

        "Left Wing \nScoring Rate: ",

       fontsize = 10,

       color = "b",

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)

ax.text(12.3 * 15, 1.6 * 15,

       '20.00%',

       fontsize = 20,

       color = 'b',

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)



# Add goal box right wing area rectangle:

# The rectangle starts at x = -20.5 * 15, y = 0

PBoxZoneRightWing = Rectangle((-20.5 * 15, 0), width = 11 * 15, height = 16.5 * 15, color = "b", lw = 2.5, ls = '-.',

                            fill = False, zorder = 10, alpha = 0.5)

ax.add_artist(PBoxZoneRightWing)

# Add text for left wing rectangle

ax.text(-18.1 * 15, 3 * 15, 

        "Right Wing \nScoring Rate: ",

       fontsize = 10,

       color = "b",

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)

ax.text(-18.1 * 15, 1.6 * 15,

       '11.11%',

       fontsize = 20,

       color = 'b',

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)



# Add circle to central area within the penalty area

# centralCircle = Circle((0, 13 * 15), 8 * 15, ls = ':', lw = 1, ec = 'b', fill = False, zorder = 10)

# ax.add_artist(centralCircle)



# Add central penalty area text

ax.text(1 * 15, 8.5 * 15, 

        "Central Area \nScoring Rate: ",

       fontsize = 10,

       color = "r",

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)

ax.text(1 * 15, 7 * 15,

       '55.00%',

       fontsize = 20,

       color = 'r',

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)



# Add outside penalty scoring rate

ax.text(-18.1 * 15, 24 * 15, 

        "Outside Penalty Area \nScoring Rate: ",

       fontsize = 10,

       color = "purple",

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)

ax.text(-18.1 * 15, 22 * 15,

       '20.00%',

       fontsize = 20,

       color = 'purple',

       fontweight = 'bold',

       family = 'monospace',

       zorder = 10)



# Add other major stats

# Add total number of shots, total season scoring rate, and total scoring rate for on target goal

# ax.text(-33 * 15, 30 * 15, "Total Seasonal \nShot Attempts:",

#       fontsize = 15, color = 'r',

#       fontweight = 'bold', family = 'monospace', zorder = 10)

ax.text(-33 * 15, 38 * 15, "74 Shots 21 Goals",

       fontsize = 22, color = 'r',

       fontweight = 'bold', family = 'fantasy', zorder = 10)

ax.text(-33 * 15, 36 * 15, "28.38% Scoring Rate",

       fontsize = 22, color = 'r',

       fontweight = 'bold', family = 'fantasy', zorder = 10)

ax.text(-33 * 15, 34 * 15, "48.65% On Target Goal Scoring Rate",

       fontsize = 22, color = 'r',

       fontweight = 'bold', family = 'fantasy', zorder = 10)

ax.text(10 * 15, 22 * 15, 'Timo is a predator \nin penalty area!',

       fontsize = 22, color = 'k', family = 'fantasy', fontstyle = 'italic',

       fontweight = 'bold', zorder = 10)



ax.legend(fontsize = 15, frameon=True)



# Add in pictures

# Add additional pictures to the graph

logo=mpimg.imread('../input/rbl_logo.png')

addLogo = OffsetImage(logo, zoom=0.4, zorder = 10)

addLogo.set_offset((46 * 15, 1.5 * 15)) # pass the position in a tuple

addLogo.set_zorder(10)

ax.add_artist(addLogo)



# Add Timo Werner's Image

timo= mpimg.imread('../input/timo.png')

addTimo = OffsetImage(timo, zoom=0.6)

addTimo.set_offset((70,60)) # pass the position in a tuple

addTimo.set_zorder(10)

ax.add_artist(addTimo)