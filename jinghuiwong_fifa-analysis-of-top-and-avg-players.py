# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



os.listdir("../input")
dat=pd.read_csv('/kaggle/input/fifa19/data.csv',encoding='utf-8')

dat = dat.drop(dat.columns[0], axis = 1)

dat.head()

# Removing spaces in the column names to enable easy column reference  

dat.columns = dat.columns.str.replace(' ','')

# Removing special characters (e.g. euro dollar sign)

euro_sign = dat['ReleaseClause'][0][:3]

dat['ReleaseClause'] = dat['ReleaseClause'].str.replace(euro_sign,'')

dat['Value'] = dat['Value'].str.replace(euro_sign,'')

dat['Wage'] = dat['Wage'].str.replace(euro_sign,'')

dat['ReleaseClause'] = dat['ReleaseClause'].str.replace('M','') # Removing Millions in the field

dat['Value'] = dat['Value'].str.replace('M','') # Removing Millions in the field

dat['Wage'] = dat['Wage'].str.replace('K','') # Removing Thousands in the field



# Converting wages,release clause, value from string into integers/float

dat.ReleaseClause = pd.to_numeric(dat.ReleaseClause, errors='coerce')

dat.Value = pd.to_numeric(dat.Value, errors='coerce')

dat.Wage = pd.to_numeric(dat.Wage, errors='coerce')



# Converting Weight from string into integers/float

dat['Weight'] = dat['Weight'].str.replace('lbs','') # Removing lbs in the field

dat.Weight = pd.to_numeric(dat.Weight, errors='coerce')

# Use Regular Expression to convert Height from feet + inches into cm. Convert from text into integer

r = re.compile(r"([0-9]+)'([0-9]+)")    # to set the pattern e.g. 5'7 , 4'12 etc

def get_cm(height):

    height = str(height)

    m = r.match(height)

    if m == None:

        return float('NaN')

    else:

        return float(m.group(1))*30.48 + float(m.group(2))*2.54

dat["Height"] = dat["Height"].apply(lambda x:get_cm(x))



# check

# dat['ReleaseClause'].head()

# dat['Wage'].head()

# dat['Value'].head()

# dat['Weight'].head()

# dat['Height'].head()



# Drop unnecessary columns that are not used for analyzing a player's performance

# This inclues: 'Real Face', 'Jersey Number', 'Loaned From' LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM

# CDM, RDM, RWB LB, LCB, CB, RCB, RB

dat = dat.drop(dat.columns[27:53], axis = 1)

dat = dat.drop(labels = ['RealFace','JerseyNumber','LoanedFrom'], axis = 1)

#dat.head()



# Check for missing values

# Check if there any null values in the dataset 

dat.isnull().values.any() # There are indeed missing values in some rows in the dataset

dat.isnull().columns

#cols = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning']

# set mean values to all empty entries

dat['Crossing'].fillna(dat['Crossing'].mean(),inplace=True)

dat['Finishing'].fillna(dat['Finishing'].mean(),inplace=True)

dat['HeadingAccuracy'].fillna(dat['HeadingAccuracy'].mean(),inplace=True)

dat['ShortPassing'].fillna(dat['ShortPassing'].mean(),inplace=True)

dat['Volleys'].fillna(dat['Volleys'].mean(),inplace=True)

dat['Dribbling'].fillna(dat['Dribbling'].mean(), inplace=True)

dat['Curve'].fillna(dat['Curve'].mean(),inplace=True)

dat['FKAccuracy'].fillna(dat['FKAccuracy'].mean(),inplace=True)

dat['LongPassing'].fillna(dat['LongPassing'].mean(),inplace=True)

dat['BallControl'].fillna(dat['BallControl'].mean(),inplace=True)

dat['Acceleration'].fillna(dat['Acceleration'].mean(),inplace=True)

dat['SprintSpeed'].fillna(dat['SprintSpeed'].mean(),inplace=True)

dat['Agility'].fillna(dat['Agility'].mean(),inplace=True)

dat['Reactions'].fillna(dat['Reactions'].mean(),inplace=True)

dat['Balance'].fillna(dat['Balance'].mean(),inplace=True)

dat['ShotPower'].fillna(dat['ShotPower'].mean(),inplace=True)

dat['Jumping'].fillna(dat['Jumping'].mean(),inplace=True)

dat['Stamina'].fillna(dat['Stamina'].mean(),inplace=True)

dat['Strength'].fillna(dat['Strength'].mean(),inplace=True)

dat['LongShots'].fillna(dat['LongShots'].mean(),inplace=True)

dat['Aggression'].fillna(dat['Aggression'].mean(),inplace=True)

dat['Interceptions'].fillna(dat['Interceptions'].mean(),inplace=True)

dat['Positioning'].fillna(dat['Positioning'].mean(),inplace=True)

dat['Vision'].fillna(dat['Vision'].mean(),inplace=True)

dat['Penalties'].fillna(dat['Penalties'].mean(),inplace=True)

dat['Composure'].fillna(dat['Composure'].mean(),inplace=True)

dat['Marking'].fillna(dat['Marking'].mean(),inplace=True)

dat['StandingTackle'].fillna(dat['StandingTackle'].mean(),inplace=True)

dat['SlidingTackle'].fillna(dat['SlidingTackle'].mean(),inplace=True)

dat['GKDiving'].fillna(dat['GKDiving'].mean(),inplace=True)

dat['GKHandling'].fillna(dat['GKHandling'].mean(),inplace=True)

dat['GKKicking'].fillna(dat['GKKicking'].mean(),inplace=True)

dat['GKPositioning'].fillna(dat['GKPositioning'].mean(),inplace=True)



# Develop the FIFA attributes: Pace, Shooting, Passing, Dribbling, Defending, Physical

# Each of these attributes are made up of the following skills (https://www.fifauteam.com/fifa-18-attributes-guide/)

# Pace: Sprint Speed, Acceleration

# Shooting: FINISHING, LONG SHOTS, PENALTIES, POSITIONING, SHOT POWER, VOLLEYS

# PASSING: CROSSING, CURVE, FREE KICK, LONG PASSING, SHORT PASSING, VISION

# DRIBBLING: AGILITY, BALANCE, BALL CONTROL, COMPOSURE, DRIBBLING, REACTIONS

# DEFENDING: HEADING, INTERCEPTIONS, MARKING, SLIDING TACKLE, STANDING TACKLE

# PHYSICAL: AGGRESSION, JUMPING, STAMINA, STRENGTH

# GOALKEEPING: DIVING, HANDLING, KICKING, POSITIONING

dat['Pace'] = ( dat['SprintSpeed'] + dat['Acceleration'] ) /2 

dat['Shooting'] = ( dat['Finishing'] + dat['LongShots'] + dat['Penalties'] + dat['Positioning'] + dat['ShotPower'] + dat['Volleys'] ) / 6

dat['Passing'] = ( dat['Crossing'] + dat['Curve'] + dat['FKAccuracy'] + dat['LongPassing'] + dat['ShortPassing'] + dat['Vision'] ) / 6

dat['Dribbling Skill'] = ( dat['HeadingAccuracy'] + dat['Interceptions'] + dat['Marking'] + dat['StandingTackle'] + dat['SlidingTackle'] ) / 5

dat['Physical'] = ( dat['Aggression'] + dat['Jumping'] + dat['Stamina'] + dat['Strength'] ) / 4

dat['Goal Keeping'] = ( dat['GKDiving'] + dat['GKHandling'] + dat['GKKicking'] + dat['GKPositioning'] ) / 4

dat.head()
# The top 10 players based on the overall score are:

top_10 = dat.nlargest(10, 'Overall')

top_10[['Name','Overall','Club','Position']]
# The best players in each position are (based on the overall score):

top_3_position = dat.groupby(['Position']).apply(lambda x: x.sort_values(['Overall'],ascending = False) )

top_3_position.groupby(level=0).head(1).sort_values(['Overall'],ascending = False)
# The best players in each position are (based on the overall score):

top_3_position = dat.groupby(['Position']).apply(lambda x: x.sort_values(['Overall'],ascending = False) )

top_3_position.groupby(level=0).head(3)

# We would like to understand the characteristics of the best players
# Function for plotting a radar plot. Taken from (https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python)



import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon

from matplotlib.path import Path

from matplotlib.projections.polar import PolarAxes

from matplotlib.projections import register_projection

from matplotlib.spines import Spine

from matplotlib.transforms import Affine2D



def radar_factory(num_vars, frame='circle'):

    """Create a radar chart with `num_vars` axes.



    This function creates a RadarAxes projection and registers it.



    Parameters

    ----------

    num_vars : int

        Number of variables for radar chart.

    frame : {'circle' | 'polygon'}

        Shape of frame surrounding axes.



    """

    # calculate evenly-spaced axis angles

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)



    class RadarAxes(PolarAxes):



        name = 'radar'



        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)

            # rotate plot such that the first axis is at the top

            self.set_theta_zero_location('N')



        def fill(self, *args, closed=True, **kwargs):

            """Override fill so that line is closed by default"""

            return super().fill(closed=closed, *args, **kwargs)



        def plot(self, *args, **kwargs):

            """Override plot so that line is closed by default"""

            lines = super().plot(*args, **kwargs)

            for line in lines:

                self._close_line(line)



        def _close_line(self, line):

            x, y = line.get_data()

            # FIXME: markers at x[0], y[0] get doubled-up

            if x[0] != x[-1]:

                x = np.concatenate((x, [x[0]]))

                y = np.concatenate((y, [y[0]]))

                line.set_data(x, y)



        def set_varlabels(self, labels):

            self.set_thetagrids(np.degrees(theta), labels)



        def _gen_axes_patch(self):

            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5

            # in axes coordinates.

            if frame == 'circle':

                return Circle((0.5, 0.5), 0.5)

            elif frame == 'polygon':

                return RegularPolygon((0.5, 0.5), num_vars,

                                      radius=.5, edgecolor="k")

            else:

                raise ValueError("unknown value for 'frame': %s" % frame)



        def draw(self, renderer):

            """ Draw. If frame is polygon, make gridlines polygon-shaped """

            if frame == 'polygon':

                gridlines = self.yaxis.get_gridlines()

                for gl in gridlines:

                    gl.get_path()._interpolation_steps = num_vars

            super().draw(renderer)





        def _gen_axes_spines(self):

            if frame == 'circle':

                return super()._gen_axes_spines()

            elif frame == 'polygon':

                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.

                spine = Spine(axes=self,

                              spine_type='circle',

                              path=Path.unit_regular_polygon(num_vars))

                # unit_regular_polygon gives a polygon of radius 1 centered at

                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,

                # 0.5) in axes coordinates.

                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)

                                    + self.transAxes)





                return {'polar': spine}

            else:

                raise ValueError("unknown value for 'frame': %s" % frame)



    register_projection(RadarAxes)

    return theta



top_10_plot = top_10[['Name','Age','Nationality','Club','Position','Pace','Shooting',

                     'Passing','Dribbling Skill','Physical','Goal Keeping']]

radar_plot_data = top_10[['Pace','Shooting','Passing','Dribbling Skill','Physical','Goal Keeping']]

radar_plot_data  = radar_plot_data.values.tolist()



theta = radar_factory(6, frame='polygon') # Since there are 6 attributes: Pace, Shooting,...,Goal Keeping

spoke_labels = ['Pace','Shooting','Passing','Dribbling Skill','Physical','Goal Keeping']







count = 0

for d in radar_plot_data:

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))

    fig.subplots_adjust(top=0.85, bottom=0.05)

    ax.set_rgrids([20,40, 60, 80])

    ax.set_title(top_10['Name'][count] + " Overall:" + str(top_10['Overall'][count]) + " Pos:"+ top_10['Position'][count],  position=(0.5, 1.1), ha='center')

    count += 1

    line = ax.plot(theta, d, linewidth=1, linestyle='solid')

    ax.fill(theta, d,  alpha=0.25)

    ax.set_varlabels(spoke_labels)



plt.show()
# The skill distribution of top 20 players in each position are:



top_20_position = dat.groupby(['Position']).apply(lambda x: x.sort_values(['Overall'],ascending = False) )

top20_data = top_20_position.groupby(level=0).head(20)

top20_data = top20_data[['Name','Age','Nationality','Overall','Club','Position','Pace','Shooting',

                     'Passing','Dribbling Skill','Physical','Goal Keeping']]

Positions = ['GK','LB','CB','RB','CM','CAM','RW','LW','ST','RF']

for position in Positions: 

    dat1 = top20_data[top20_data['Position']==position]

    radar_plot_data = dat1[['Pace','Shooting','Passing','Dribbling Skill','Physical','Goal Keeping']]

    radar_plot_data = radar_plot_data.values.tolist()

    theta = radar_factory(6, frame='polygon') # Since there are 6 attributes: Pace, Shooting,...,Goal Keeping

    spoke_labels = ['Pace','Shooting','Passing','Dribbling Skill','Physical','Goal Keeping']

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))

    fig.subplots_adjust(top=0.85, bottom=0.05)

    ax.set_rgrids([20,40, 60, 80])

    ax.set_title("Position: " + position + " Top 20 Score: "+ str(int(dat1['Overall'].mean())),  position=(0.5, 1.1), ha='center')

    for d in radar_plot_data:

        line = ax.plot(theta, d, linewidth=1, linestyle='solid')

        ax.fill(theta, d,  alpha=0.25)

        ax.set_varlabels(spoke_labels)

    plt.show()
from matplotlib import rcParams



# figure size in inches

rcParams['figure.figsize'] = 11.7,8.27



top_20_position = dat.groupby(['Position']).apply(lambda x: x.sort_values(['Overall'],ascending = False) )

top20_data = top_20_position.groupby(level=0).head(20)





# Distribution of their ages

p1=sns.kdeplot(top20_data['Age'], shade=True, color="r", label='Top 20 Players')

p2=sns.kdeplot(dat['Age'], shade=True, color="b", label='Average Players')

p1.axvline(top20_data['Age'].mean(), color='r', linestyle='--')

p2.axvline(dat['Age'].mean(), color='b', linestyle='--')

plt.xlabel('Age')

plt.yticks([])

plt.title('Top players continue to peak after 25 while average players drop off after 25')

plt.show()



# Distribution of their nationality

p1 = sns.countplot(x="Nationality", data=top20_data)

plt.xticks([])

plt.title('Top players come from: Brazil, Spain, Germany, France, Argentina')

plt.show()



# Distribution of their height

p1=sns.kdeplot(top20_data['Height'], shade=True, color="r", label='Top 20 Players')

p2=sns.kdeplot(dat['Height'], shade=True, color="b", label='Average Players')

p1.axvline(top20_data['Height'].mean(), color='r', linestyle='--')

p2.axvline(dat['Height'].mean(), color='b', linestyle='--')

plt.xlabel('Height')

plt.yticks([])

plt.title('Height does not play a crucial role in determining a top player')

plt.show()



# Distribution of their weight

p1=sns.kdeplot(top20_data['Weight'], shade=True, color="r", label='Top 20 Players')

p2=sns.kdeplot(dat['Weight'], shade=True, color="b", label='Average Players')

p1.axvline(top20_data['Weight'].mean(), color='r', linestyle='--')

p2.axvline(dat['Weight'].mean(), color='b', linestyle='--')

plt.xlabel('Weight')

plt.yticks([])

plt.title('Weight does not play a crucial role in determining a top player')

plt.show()





# Distribution of their reputation and ratings
# Distribution of their work rate

plt, ax =plt.subplots(1,2)

plt.suptitle("Top Players do not necessarily work harder than Average Players", fontsize=14)



g = sns.countplot(x="WorkRate", data=top20_data, color="r", label='Top 20 Players', ax=ax[0])

g.set_xticklabels(g.get_xticklabels(),rotation=30)

g.set(xlabel = 'Work Rate for Top Players')





g2 = sns.countplot(x="WorkRate", data=dat, color="b", label='Average Players',ax=ax[1])

g2.set_xticklabels(g.get_xticklabels(),rotation=30)

g2.set(xlabel='Work Rate for Average Players')

plt.show()
# Distribution of their skill moves

p1 = sns.countplot(x="SkillMoves", data=top20_data, color="r")

p1.set(xlabel='Number of Skill Moves for Top Players')

#plt.title('Top Players have more Skill Moves than Average Players')

plt.show()
g2 = sns.countplot(x="SkillMoves", data=dat, color="b")

g2.set(xlabel='Number of Skill Moves for Average Players')

plt.show()
# Distribution of their reputation of top players

p1 = sns.countplot(x="InternationalReputation", data=top20_data, color="r")

p1.set(xlabel='International Reputation for Top Players')

plt.show()
# Distribution of their reputation of average players

p2 = sns.countplot(x="InternationalReputation", data=dat, color="b")

p2.set(xlabel='International Reputation for Average Players')

plt.show()
g = sns.lineplot(top20_data['Age'], top20_data['Overall'], palette = 'Wistia')

g.set(xlabel = 'Age vs Overall Top Players')

plt.show()
g = sns.lineplot(dat['Age'], dat['Overall'], palette = 'Wistia')

g.set(xlabel = 'Age vs Overall Average Players')



plt.show()