# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime # date and time

import matplotlib.pyplot as plt # drawing plot

# select best model

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LogisticRegression # Logistic regression

from sklearn.preprocessing import StandardScaler # scaler



# some imports for Radar plot

from matplotlib.path import Path

from matplotlib.spines import Spine

from matplotlib.projections.polar import PolarAxes

from matplotlib.projections import register_projection





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Part 1. Choose best goalkeeper by Rating into Four Age Groups

## Age groups:

##- Up to 20 (incl. 20)

##- 21-25 

##- 26-30

##- 30+
# import data

FullData = pd.read_csv('../input/FullData.csv')

Names = pd.read_csv('../input/PlayerNames.csv')



FullData.assign(Index=np.nan)

FullData['Index'] = [v.split('/')[2] for v in Names['url']]



FullData.head()
# Add column with actual age

# Dataset has column with age but it is static column. Let's add actual for current date age

def age(dobStr):

    dob = datetime.datetime.strptime(dobStr, '%m/%d/%Y')

    today = datetime.date.today()

    years = today.year - dob.year

    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):

        years -= 1

    return years



FullData.assign(AgeReal=np.nan)

FullData['AgeReal'] = [age(v) for v in FullData['Birth_Date']]
#let's have a look on features names

FullData.columns.values
# Get all goalkeepers from DataFrame

# Select by Preffered_Position as Club_Position is not filled for every player

Goalies = FullData.loc[FullData['Preffered_Position'] == 'GK']

Goalies.head()
# Select goalkeeper's skills

gk_skills = ['Name', 'Rating', 'Nationality', 'Club', 'Height', 'Weight', 'Vision',

             'Short_Pass', 'Long_Pass', 'GK_Positioning', 'GK_Diving', 'GK_Kicking',

               'GK_Handling', 'GK_Reflexes', 'Index', 'AgeReal']

drop_skills = [s for s in FullData.columns.values if s not in gk_skills]



Goalies = Goalies.drop(drop_skills, axis=1)
# For Radar plot drawing 

def _radar_factory(num_vars):

    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)

    theta += np.pi/2



    def unit_poly_verts(theta):

        x0, y0, r = [0.5] * 3

        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]

        return verts



    class RadarAxes(PolarAxes):

        name = 'radar'

        RESOLUTION = 1



        def fill(self, *args, **kwargs):

            closed = kwargs.pop('closed', True)

            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)



        def plot(self, *args, **kwargs):

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

            self.set_thetagrids(theta * 180/np.pi, labels)



        def _gen_axes_patch(self):

            verts = unit_poly_verts(theta)

            return plt.Polygon(verts, closed=True, edgecolor='k')



        def _gen_axes_spines(self):

            spine_type = 'circle'

            verts = unit_poly_verts(theta)

            verts.append(verts[0])

            path = Path(verts)

            spine = Spine(self, spine_type, path)

            spine.set_transform(self.transAxes)

            return {'polar': spine}



    register_projection(RadarAxes)

    return theta



# Draw Radar plot

def radar_graph(labels = [], values = [], label = ''):

    N = len(labels) 

    theta = _radar_factory(N)

    max_val = max(values)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='radar')

    ax.plot(theta, values, color='r', label=label)

    ax.set_varlabels(labels)

    plt.show()



# Choose top in every age group and plot it

def grouptopfigure(gk, skills):

    top = gk.groupby('Rating').head(10).reset_index(drop=True).head(10)

    radar_graph(top['Name'], top['Rating'], 'Top-10 Rating of Goalkeepers')

    print(top[['Name', 'Rating', 'AgeReal', 'Nationality', 'Club']])



#divide Goalkeepers for 4 Age groups

gk_upto20 = Goalies.loc[(Goalies['AgeReal'] <= 20)]

gk_20to25 = Goalies.loc[(Goalies['AgeReal'] > 20) & (Goalies['AgeReal'] <= 25)]

gk_25to30 = Goalies.loc[(Goalies['AgeReal'] > 25) & (Goalies['AgeReal'] <= 30)]

gk_from30 = Goalies.loc[(Goalies['AgeReal'] > 30)] 



skills = [s for s in gk_skills if s not in ['Name', 'Rating', 'Height', 'Weight', 'Index', 'AgeReal']]

grouptopfigure(gk_upto20, skills)

grouptopfigure(gk_20to25, skills)

grouptopfigure(gk_25to30, skills)

grouptopfigure(gk_from30, skills)

grouptopfigure(Goalies, skills)
# Part 2. Preffered position prediction.
## Let's try to find linear dependence between rankings and Rating feature
# Select all possible club positions 

Positions = pd.unique(FullData['Club_Position']).tolist()

index = Positions.index('Sub')

Positions = np.delete(Positions, index).tolist()

index = Positions.index('Res')

Positions = np.delete(Positions, index).tolist()

index = Positions.index('nan')

Positions = np.delete(Positions, index).tolist()

Positions
# Divide features to several groups



# Main features

Main_features = ['Name', 'Nationality', 'Club', 'Index']



# Numerical = skills + Height&Weight + Actual Age

Numerical_features = [ 'Height', 'Weight', 'Skill_Moves',

                       'Ball_Control', 'Dribbling', 'Marking', 'Sliding_Tackle',

                       'Standing_Tackle', 'Aggression', 'Reactions', 'Attacking_Position',

                       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass',

                       'Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength',

                       'Balance', 'Agility', 'Jumping', 'Heading', 'Shot_Power',

                       'Finishing', 'Long_Shots', 'Curve', 'Freekick_Accuracy',

                       'Penalties', 'Volleys', 'GK_Positioning', 'GK_Diving', 'GK_Kicking',

                       'GK_Handling', 'GK_Reflexes', 'AgeReal']



# It will be used below

Result = ['Result']



# Drop features

# These features are possibly not necessary for Position prediction

Drop_features = ['National_Position', 'National_Kit', 'Club_Kit', 'Club_Joining', 'Contract_Expiry', 

                 'Birth_Date', 'Age', 'Preffered_Position', 'Work_Rate', 'Weak_foot']
# prepare data, delete 'cm' from Height and 'kg' from Weight to make features numerical

# We will loop through possible positions. So we add column Result to store 

# 1 - if player has predicted position in his list of preffered positions

# 0 - if he has not.

# Also return count of examples for certain position

def prepare_data(pos):

    #prepare data for logistic regression

    Players = FullData[Numerical_features]

    Players.loc[:,('Height')] = [int(s[:3]) for s in Players['Height']]

    Players.loc[:,('Weight')] = [int(s[:3]) for s in Players['Weight']]

    Players['Result'] = 0

    mask = FullData.index[FullData['Preffered_Position'].str.find(pos) != -1]

    Players.loc[mask,('Result')] = 1

    return Players, mask



# make cross validation to choose best parameter C (regularisation parameter)

# Use best C for prediction and return score

def make_prediction(data):

    #divide data to X and y

    X = data[Numerical_features]

    y = data[Result]

    

    #get train and test sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=241)

    y_train = y_train.values.ravel()

    y_test = y_test.values.ravel()

    

    #build for folds

    cv = KFold(n_splits=5, shuffle=True, random_state=241)

    

    #find best C

    best_C = 0

    best_score = 0

    for C in np.power(10.0, np.arange(-5, 6)):

        start_time = datetime.datetime.now()

        clf = LogisticRegression(penalty='l2', C=C, n_jobs=-1)

        score = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=cv, scoring='roc_auc').mean()

        if score > best_score:

            best_score = score

            best_C = C

        #print('  C = {}, Score = {:.7f}'.format(C, score))

        #print('  <- Time elapsed: {}'.format(datetime.datetime.now() - start_time))

    

    print('    Best C is {}'.format(C))

    #make prediction

    clf = LogisticRegression(penalty='l2', C=best_C, n_jobs=-1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    score = clf.score(X_test, y_test)

    

    return score
# Loop thruogh possible position.

# Check if there are several examples for every position

# print results

for pos in Positions:

    Players, mask = prepare_data(pos)

    print('-> Position = {}, Number of examples = {}'.format(pos, len(mask)))

    if len(mask) > 0:

        score = make_prediction(Players)

        print('      Score = {}'.format(score))

    else:

        print('<- Ignore it')
# We can see that linear model predict GK position with a score = 1. 
# Thank you for attention.