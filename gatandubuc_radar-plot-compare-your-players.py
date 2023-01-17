# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
data = pd.read_csv(r'/kaggle/input/top-women-chess-players/top_women_chess_players_aug_2020.csv')
print(data.info())
import seaborn as sns

data_sorted = data.sort_values(by='Standard_Rating', ascending=False)

data_sorted.Title = data_sorted.Title.fillna('No title')

df = data_sorted.loc[:,['Title','Year_of_birth','Standard_Rating','Rapid_rating','Blitz_rating']]

sns.pairplot(df, hue="Title")
from matplotlib import patches



f, ax = plt.subplots(figsize=(10,10))

sns.scatterplot(x="Year_of_birth", y="Rapid_rating",

                hue="Title",

                sizes=(1, 8),

                data=data_sorted, ax=ax)



ax.add_artist(

    patches.Rectangle((1995, 1200),15,400, color = 'r', zorder = 1, alpha=0.1))



ax.annotate('Young area', (1995,1400),(1980,1405), arrowprops=dict(facecolor='black'))

plt.title('Rapid_rating')
import scipy.stats as stat
def best_group_proximity(dataset, Title, target):

    """

    dataset: DataFrame

    Title: group you want to analyse (str)

    target: On which data (str)

    

    return: Name of the closest group to 'Title' (str)

    """

    results = pd.Series(dtype='float64')

    for group in dataset.Title.loc[dataset.Title!=Title].unique():

        print(group)

        w, p = stat.mannwhitneyu(dataset[dataset.Title==Title][target], dataset[dataset.Title==group][target])

        print(stat.mannwhitneyu(dataset[dataset.Title==Title][target], dataset[dataset.Title==group][target]))

        results.loc[group] = p

    print('---end---')

    return results.sort_values(ascending=False).index[0]

        

best_group_proximity(data_sorted, 'No title', 'Rapid_rating')

best_group_proximity(data_sorted, 'No title', 'Blitz_rating')

best_group_proximity(data_sorted, 'No title', 'Standard_Rating')
# Fillna

data_sorted[['Standard_Rating','Rapid_rating','Blitz_rating']] = data_sorted[['Standard_Rating','Rapid_rating','Blitz_rating']].fillna(0)

data_sorted
import numpy as np



import matplotlib.pyplot as plt

from matplotlib.path import Path

from matplotlib.spines import Spine

from matplotlib.projections.polar import PolarAxes

from matplotlib.projections import register_projection





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



    def draw_poly_patch(self):

        # rotate theta such that the first axis is at the top

        verts = unit_poly_verts(theta + np.pi / 2)

        return plt.Polygon(verts, closed=True, edgecolor='k')



    def draw_circle_patch(self):

        # unit circle centered on (0.5, 0.5)

        return plt.Circle((0.5, 0.5), 0.5)



    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}

    if frame not in patch_dict:

        raise ValueError('unknown value for `frame`: %s' % frame)



    class RadarAxes(PolarAxes):



        name = 'radar'

        # use 1 line segment to connect specified points

        RESOLUTION = 1

        # define draw_frame method

        draw_patch = patch_dict[frame]



        def __init__(self, *args, **kwargs):

            super(RadarAxes, self).__init__(*args, **kwargs)

            # rotate plot such that the first axis is at the top

            self.set_theta_zero_location('N')

            self.angle = theta



        def fill(self, *args, **kwargs):

            """Override fill so that line is closed by default"""

            closed = kwargs.pop('closed', True)

            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)



        def plot(self, *args, **kwargs):

            """Override plot so that line is closed by default"""

            lines = super(RadarAxes, self).plot(*args, **kwargs)

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

            

        def label_pos(self):

            for label, angle_rad in zip(self.get_xticklabels(), self.angle):

                if angle_rad == 0 or angle_rad == np.pi:

                    ha = 'center'

                elif angle_rad == np.pi/2:

                    ha = 'right'

                elif angle_rad == 3*np.pi/2:

                    ha = 'left'

                label.set_horizontalalignment(ha)             



        def _gen_axes_patch(self):

            return self.draw_patch()



        def _gen_axes_spines(self):

            if frame == 'circle':

                return PolarAxes._gen_axes_spines(self)

            # The following is a hack to get the spines (i.e. the axes frame)

            # to draw correctly for a polygon frame.



            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.

            spine_type = 'circle'

            verts = unit_poly_verts(theta + np.pi / 2)

            # close off polygon by repeating first vertex

            verts.append(verts[0])

            path = Path(verts)



            spine = Spine(self, spine_type, path)

            spine.set_transform(self.transAxes)

            return {'polar': spine}



    register_projection(RadarAxes)

    return theta

def unit_poly_verts(theta):

    """Return vertices of polygon for subplot axes.



    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)

    """

    x0, y0, r = [0.5] * 3

    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]

    return verts





class Players():

    """Create players object with their stats like attributes.

    

    Params: str

            players name (like it's written in the dataset)

    """

    def __init__(self, *args):

        self.data, self.titles, self.federation = self.chess_players(*args)

        self.players = [p+' in '+t[0]+' from '+f[0] for p,t,f in zip(args, self.titles, self.federation)]

        

    def chess_players(self, *args):

        names, stats, titles, federation = self.get_chess_stats(*args)

        data = [

            ['Age', 'Standard_Rating', 'Rapid_rating', 'Blitz_rating'],

            (names, stats)

        ]

        return data, titles, federation



    def get_chess_stats(self, *args, data=data_sorted.copy()):

        stat_default = [0,0,0,0]

        try:

            return (' vs '.join(args),

                data.loc[data.Name.isin(args),['Year_of_birth', 'Standard_Rating', 'Rapid_rating', 'Blitz_rating']].fillna(0).values + stat_default,

                data.loc[data.Name.isin(args),['Title']].values,

                data.loc[data.Name.isin(args),['Federation']].values)

        except ValueError:

            return (stat_default,'No title')

        



def _scale_data(data, ranges):

    """

    Need to scale the data to have different scale on a same radar plot

    Params: 

    

    data: list

          The values of yours players

    ranges: tuple

            Limits of each axes

    """

    x1, x2 = ranges[0]

    d = data[0]

    sdata = [d]



    for d, (y1, y2) in zip(data[1:], ranges[1:]):

        sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)



    return sdata



def append_label(label, value, ranges):

    """

    Append value in label and remove closest values

    Params: 

    label: list

           Grid and players values

    value: numpy.array

           Values to append

    ranges: tuple

            Limits of each axes

    """

    for val in value:

        x = label - val

        perc = (max(ranges) - min(ranges)) * 10 / 100

        label = [label[i] for i,e in enumerate(x) if abs(e)>perc] 

    label = label + value.astype('int').tolist()

    return label

# Get players

N = 4

theta = np.degrees(radar_factory(N, frame='circle'))



# Add players to compare them

players = Players('Vorpahl, Sina Fleur','Gochoshvili, Anetta')

spoke_labels = players.data.pop(0)

 

#Create figure

fig = plt.figure(figsize=(10,10))

    

fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

   

#Create 4 differents axes for the 4 differents scales

axes = [fig.add_axes([0.05, 0.05, 0.95, 0.95], projection="radar", label="axes%d" % i) 

                     for i in range(len(spoke_labels))]



#Main axes where the data will be plot

axe1 = axes[0]

axe1.set_thetagrids(theta, labels=spoke_labels, fontsize=12, weight="bold", color="black")

axe1.yaxis.grid(False)

axe1.label_pos()

    

title, case_data = players.data[0]

   

#Set invisible the other axes but not the grid

for ax in axes[1:]:

    ax.patch.set_visible(False)

    ax.grid("off")

    ax.xaxis.set_visible(False)

    ax.yaxis.grid(False)



# Define limits

ranges = [[1950, 2010], [1500, 2900],[1500, 2900], [1500, 2900]] 

ranges_grid = [list(range(1960,2000, 10)),

                   list(range(1600,2800,200)),

                   list(range(1600,2800,200)),

                   list(range(1600,2800,200))]



# For each axe we are defining the grid.

i=0

for ax, angle, lim, label in zip(axes, theta, ranges, ranges_grid):

    label = append_label(label, case_data[:,i], lim)

    ax.set_rgrids(label, labels=label, angle=angle)

    ax.spines["polar"].set_visible(False)

    ax.set_ylim(*lim)  

    ax.xaxis.grid(True,color='black',linestyle='-')

    i += 1

        

axe1.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),

                     horizontalalignment='center', verticalalignment='center')



#Plot

color = ['b', 'r', 'g','y','black']

for d, c in zip(case_data, color):

    angle = np.deg2rad(np.r_[theta])

    sdata = _scale_data(d, ranges)

    axe1.plot(angle, np.r_[sdata], color=c)

    axe1.fill(angle, np.r_[sdata], facecolor=c, alpha=0.25)

        

labels = players.players

legend = fig.legend(labels, loc=(0.7, .85),

                       labelspacing=0.1, fontsize='small')



plt.show()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multioutput import MultiOutputClassifier

import xgboost

from sklearn.model_selection import ParameterGrid

import sklearn

import eli5

from eli5.lime import TextExplainer
encoder = LabelEncoder()

Y = encoder.fit_transform(data_sorted.Federation)

X_train, X_test, y_train, y_test = train_test_split(data_sorted.Name.str.replace(',',''), 

                                                    Y, random_state=0, test_size=0.2)
classes = [MultinomialNB(),

           SGDClassifier(),

           KNeighborsClassifier(),

           RandomForestClassifier(), 

           AdaBoostClassifier(),

           LogisticRegression(),

           xgboost.XGBClassifier()

]



for cls in classes:

    

    txt_cls = Pipeline([

        ('vect', CountVectorizer(analyzer='char', ngram_range=(1,4))),

        ('model', cls)

    ])

    

    txt_cls.fit(X_train, y_train)

    print('accuracy: '+str(txt_cls.score(X_test, y_test)))

txt_cls = Pipeline([

        ('vect', CountVectorizer(analyzer='char')),

        ('model', LogisticRegression())

    ])

    

parameters = {

    'vect__ngram_range':[(1,2),(1,3),(1,4)],

    'model__solver':['lbfgs'],

    'model__C': [1, 1.5, 2],

 }



txt_cls = GridSearchCV(txt_cls, parameters)

txt_cls.fit(X_train, y_train)

for param_name in sorted(parameters.keys()):

    print("%s: %r" % (param_name, txt_cls.best_params_[param_name]))

print('accuracy: '+str(txt_cls.score(X_test, y_test)))