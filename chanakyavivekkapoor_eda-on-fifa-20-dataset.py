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
dataset = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')
dataset.describe()
dataset.info()
print("The number of rows in the dataset are:", dataset.shape[0])

print("The number of columns in the dataset are:", dataset.shape[1])
dataset.head()
dataset.drop(["sofifa_id", "player_url", "long_name", "dob"], axis = 1, inplace = True)
dataset.columns
dataset1 = dataset[['short_name', 'age', 'height_cm', 'weight_kg', 'nationality', 'club',

       'overall', 'potential', 'value_eur', 'wage_eur', 'player_positions',

       'preferred_foot', 'international_reputation', 'weak_foot',

       'skill_moves', 'work_rate', 'body_type', 'real_face',

       'release_clause_eur']]
from bokeh.io import output_notebook

from bokeh.io import show

from bokeh.plotting import figure

from bokeh.transform import cumsum

from bokeh.palettes import Spectral6

from bokeh.models import ColumnDataSource

from bokeh.layouts import gridplot
output_notebook()
dataset1['age'].isnull().sum()
hist, edges = np.histogram(dataset1['age'], density=True, bins = 20)

Age = figure(

    x_axis_label = 'Age of Players',

    title = 'Distribution of Age of Players'

)



Age.quad(

    bottom = 0,

    top = hist,

    left = edges[:-1],

    right = edges[1:],

    line_color = 'white'

)





show(Age)

print("Skewness of age is", dataset1['age'].skew())
print("The age of the youngest player is", dataset1['age'].min())

print("The age of the oldest player is", dataset1['age'].max())
dataset.loc[dataset['age'] == dataset1['age'].min()]
dataset.loc[dataset['age'] == dataset1['age'].max()]
dataset.loc[dataset['overall'] == dataset1['overall'].max()][['short_name', 'age', 'overall']]
dataset.loc[dataset['value_eur'] == dataset1['value_eur'].max()][['short_name', 'age', 'value_eur']]
dataset1['height_cm'].isnull().sum()
hist, edges = np.histogram(dataset1['height_cm'], density=True, bins = 20)

Height = figure(

    x_axis_label = 'Height of Players',

    title = 'Distribution of Height of Players',

)



Height.quad(

    bottom = 0,

    top = hist,

    left = edges[:-1],

    right = edges[1:],

    line_color = 'white'

)





show(Height)

print("Skewness of height is", dataset1['height_cm'].skew())
print("The height of the shortest player is {} cm and his name is {}".format(dataset1['height_cm'].min(),

            list(dataset.loc[dataset['height_cm'] == dataset1['height_cm'].min()]['short_name'])[0]))



print("The height of the tallest player is {} cm and his name is {}".format(dataset1['height_cm'].max(),

            list(dataset.loc[dataset['height_cm'] == dataset1['height_cm'].max()]['short_name'])[0]))
dataset1['weight_kg'].isnull().sum()
hist, edges = np.histogram(dataset1['weight_kg'], density=True, bins = 20)

Weight = figure(

    x_axis_label = 'Weight of Players',

    title = 'Distribution of Weight of Players'

)



Weight.quad(

    bottom = 0,

    top = hist,

    left = edges[:-1],

    right = edges[1:],

    line_color = 'white'

)





show(Weight)

print("Skewness of weight is", dataset1['weight_kg'].skew())
print("The weight of the lightest player is {} kg and his name is {}".format(dataset1['weight_kg'].min(),

            list(dataset1.loc[dataset1['weight_kg'] == dataset1['weight_kg'].min()]['short_name'])[0]))



print("The weight of the heaviest player is {} kg and his name is {}".format(dataset1['weight_kg'].max(),

            list(dataset1.loc[dataset1['weight_kg'] == dataset1['weight_kg'].max()]['short_name'])[0]))
dataset1.loc[dataset1['weight_kg'] == dataset1['weight_kg'].min()]
dataset1.loc[dataset1['weight_kg'] == dataset1['weight_kg'].max()]
dataset1['nationality'].isnull().sum()
countries = list(dataset1['nationality'].value_counts()[:40].index)

count = list(dataset1['nationality'].value_counts()[:40].values)

source = ColumnDataSource(data = dict(Country = countries, counts = count, color = ['teal'] * 40))



p = figure(x_range = countries, plot_height = 600, plot_width = 1000, title = "Each Nationality Player Count", tools = "hover", tooltips = "@Country: @counts")

p.vbar(x = 'Country', top = 'counts', width = 0.9, source = source, color = 'color')



p.xgrid.grid_line_color = None

p.y_range.start = 0

p.xaxis.major_label_orientation = "vertical"

show(p)
dataset1.loc[dataset1['nationality'] == 'India'].head(5)
names = dataset1.loc[dataset1['nationality'] == 'India']['short_name'].to_list()

score = dataset1.loc[dataset1['nationality'] == 'India']['overall'].to_list()



source = ColumnDataSource(data = dict(Names = names, Score = score, color = ['salmon'] * 40))



p1 = figure(x_range = names, plot_height = 400, plot_width = 600, title = "Overall Score of Each Indian Player ", 

            tools = "hover", tooltips = "@Names: @Score")



p1.vbar(x = 'Names', top = 'Score', width = 0.9, source = source, color = 'color')



p1.xgrid.grid_line_color = None

p1.y_range.start = 0

p1.xaxis.major_label_orientation = "vertical"

show(p1)
def avg_score(country_name):

    """

    A function which is used to return the average score of players belonging to country given as argument.

    """

    

    return np.mean(dataset1.loc[dataset1['nationality'] == country_name]['overall'])



top_20_countries = countries[:20] # Taking the 20 country names(to which players belong) from the 'countries' variable defined above.

avg_scores = []



for i in top_20_countries:

    avg_scores.append(round(avg_score(i), 2))

    

print(avg_scores)

source = ColumnDataSource(data = dict(Countries = top_20_countries, Score = avg_scores, color = ['tomato'] * 20))



p2 = figure(x_range = top_20_countries, plot_height = 400, plot_width = 600, title = "Average Score of Players Belonging to Different Countries ",

            tools = "hover", tooltips = "@Countries: @Score")



p2.vbar(x = 'Countries', top = 'Score', width = 0.9, source = source, color = 'color')



p2.xgrid.grid_line_color = None

p2.y_range.start = 0



p2.xaxis.major_label_orientation = "vertical"



show(p2)
dataset1.loc[dataset1['overall'] == dataset1['overall'].max()]
dataset1['club'].isnull().sum()
clubs = list(dataset1['club'].value_counts()[:50].index)

count = list(dataset1['club'].value_counts()[:50].values)
source = ColumnDataSource(data = dict(Clubs = clubs, count = count, color = ['lightskyblue'] * 50))



p3 = figure(x_range = clubs, plot_height = 400, plot_width = 1000, title = "Count of Players in Each Club", 

            tools = "hover", tooltips = "@Clubs: @count")



p3.vbar(x = 'Clubs', top = 'count', width = 0.9, source = source, color = 'color')



p3.xgrid.grid_line_color = None

p3.y_range.start = 0

p3.xaxis.major_label_orientation = "vertical"

show(p3)
def avg_score(club_name):

    """

    A function which is used to return the average score of players belonging to club given as argument.

    """

    

    return np.mean(dataset1.loc[dataset1['club'] == club_name]['overall'])



clubs_40 = clubs[:40] # Taking the 20 country names(to which players belong) from the 'countries' variable defined above.

avg_scores_clubs = []



for i in clubs_40:

    avg_scores_clubs.append(round(avg_score(i), 2))

    

print(avg_scores_clubs)



source = ColumnDataSource(data = dict(Clubs = clubs_40, Score = avg_scores_clubs, color = ['tomato'] * 40))



p4 = figure(x_range = clubs_40, plot_height = 400, plot_width = 1000, title = "Average Score of Players Belonging to Different Clubs ",

            tools = "hover", tooltips = "@Clubs: @Score")



p4.vbar(x = 'Clubs', top = 'Score', width = 0.9, source = source, color = 'color')



p4.xgrid.grid_line_color = None

p4.y_range.start = 0



p4.xaxis.major_label_orientation = "vertical"



show(p4)
np.mean(dataset1.loc[dataset1['club'] == 'FC Barcelona']['overall'])
dataset1['overall'].isnull().sum()
print("The average overall score of all players is ", round(np.mean(dataset1['overall']), 2))
print("The heighest rated player is {} and his rating is {}".format(dataset1.loc[dataset1['overall'] == 

                                                                                 dataset1['overall'].max()]['short_name'][0],

                                                                    dataset1.loc[dataset1['overall'] == 

                                                                                 dataset1['overall'].max()]['overall'][0]))
dataset1.loc[dataset1['overall'] == dataset1['overall'].min()][['short_name', 'overall', 'nationality']]
hist, edges = np.histogram(dataset1['overall'], density=True, bins = 20)

overall = figure(

    x_axis_label = 'Overall ratings of Players',

    title = 'Distribution of Overall Score of Players'

)



overall.quad(

    bottom = 0,

    top = hist,

    left = edges[:-1],

    right = edges[1:],

    line_color = 'white'

)





show(Weight)

print("Skewness of overall score is", dataset1['overall'].skew())
def plot_overall_accto_club(club_name):

    player_names = dataset1.loc[dataset1['club'] == club_name]['short_name'].to_list()

    player_overall_scores = dataset1.loc[dataset1['club'] == club_name]['overall'].to_list()

    

    source = ColumnDataSource(data = dict(names = player_names, overall = player_overall_scores, color = ['cornflowerblue'] * len(player_overall_scores)))



    p5 = figure(x_range = player_names, plot_height = 400, plot_width = 800, title = "Overall Score of Players of " + club_name,

                tools = "hover", tooltips = "@names: @overall")



    p5.vbar(x = 'names', top = 'overall', width = 0.9, source = source, color = 'color')



    p5.xgrid.grid_line_color = None

    p5.y_range.start = 0



    p5.xaxis.major_label_orientation = "vertical"



    show(p5)

    

    print("The average of overall rating of players belonging to {} is {}".format(club_name, round(np.mean(dataset1.loc[dataset1['club'] == club_name]['overall']), 2)))
plot_overall_accto_club('FC Barcelona')
plot_overall_accto_club('Juventus')
plot_overall_accto_club('Manchester City')
plot_overall_accto_club('Manchester United')
dataset1['potential'].isnull().sum()
dataset1.loc[dataset1['potential'] == dataset1['potential'].max()]
dataset1.loc[dataset1['potential'] == dataset1['potential'].min()]
dataset1.sort_values("potential", axis = 0, ascending = False)[['short_name', 'club', 'overall', 'potential', 'value_eur']].head(20)
from scipy.stats import pearsonr

# Calculating correlation coefficient of Overall Score and Potential of players

corr, _ = pearsonr(dataset1['overall'], dataset1['potential'])

corr1, _ = pearsonr(dataset1['value_eur'], dataset1['potential'])

print('Pearsons correlation for overall score and potential is %.3f' % corr)

print('Pearsons correlation for value_eur and potential is %.3f' % corr1)
dataset1['preferred_foot'].isnull().sum()
dataset1['preferred_foot'].value_counts()
from math import pi

foot = dataset1['preferred_foot'].value_counts().index.to_list()

count = dataset1['preferred_foot'].value_counts().values

count1 = count / sum(count) * 100

angle = count / sum(count) * 2 * pi
source = ColumnDataSource(data = dict(foot = foot, count = count, color = ['skyblue', 'salmon'], angle = angle, percentage = count1))



p6 = figure(x_range = foot, plot_height = 300, plot_width = 500, title = "Count of left and right foot players ",

                tools = "hover", tooltips = "@foot: @count")



p6.vbar(x = 'foot', top = 'count', width = 0.9, source = source, color = 'color')



p6.xgrid.grid_line_color = None

p6.y_range.start = 0



p6.xaxis.major_label_orientation = "horizontal"
p7 = figure(plot_height = 300, plot_width = 300, title="Pie Chart",

           tools = "hover", tooltips = "@foot: @percentage", x_range=(-0.5, 1.0))  ## Pie chart for right and left foot players



p7.wedge(x = 0, y = 1, radius = 0.4,

        start_angle = cumsum('angle', include_zero=True), end_angle = cumsum('angle'),

        line_color = "white", fill_color = 'color', legend_field = 'foot', source = source)

p7.legend.location = "top_right"

p7.legend.label_text_font_size = '5pt'
show(gridplot([[p7, p6]]))
print("The average of overall scores of players who prefer Right foot is", round(dataset1.loc[dataset1['preferred_foot'] == 'Right']['overall'].mean(), 2))



print("The average of overall scores of players who prefer Left foot is", round(dataset1.loc[dataset1['preferred_foot'] == 'Left']['overall'].mean(), 2))
dataset1['international_reputation'].isnull().sum()
dataset1['international_reputation'].value_counts()
reputation = dataset1['international_reputation'].value_counts().index.to_list()

count = dataset1['international_reputation'].value_counts().values

count1 = count / sum(count) * 100

angle = count / sum(count) * 2 * pi



source = ColumnDataSource(data = dict(reputation = reputation, count = count, color = ['skyblue', 'salmon', 'brown', 'cyan', 'red'], angle = angle, percentage = count1))
p8 = figure(plot_height = 400, plot_width = 400, title="Pie Chart",

           tools = "hover", tooltips = "@reputation: @percentage", x_range=(-1.0, 1.0))  ## Pie chart for reputations



p8.wedge(x = 0, y = 1, radius = 0.8,

        start_angle = cumsum('angle', include_zero=True), end_angle = cumsum('angle'),

        line_color = "white", fill_color = 'color', legend_field = 'reputation', source = source)



p8.legend.location = "top_right"



p8.legend.label_text_font_size = '5pt'
count1
p9 = figure(plot_height = 300, plot_width = 500, title = "International Reputation ",

                tools = "hover", tooltips = "@reputation: @count")



p9.vbar(x = 'reputation', top = 'count', width = 0.9, source = source, color = 'color')



p9.xgrid.grid_line_color = None

p9.y_range.start = 0



p9.xaxis.major_label_orientation = "horizontal"
show(gridplot([[p8, p9]]))
dataset1.loc[dataset['international_reputation'] == 1].head(5)
dataset1.loc[dataset['international_reputation'] == 2].head(5)
dataset1.loc[dataset['international_reputation'] == 5]
corr, _ = pearsonr(dataset1['overall'], dataset1['international_reputation'])

corr1, _ = pearsonr(dataset1['value_eur'], dataset1['international_reputation'])

print('Pearsons correlation for overall score and international reputations of players is %.3f' % corr)

print('Pearsons correlation for value_eur and internationalreputations of players is %.3f' % corr1)
dataset1['skill_moves'].isnull().sum()
dataset1['skill_moves'].value_counts()
no_of_skills = dataset1['skill_moves'].value_counts().index.to_list()

count = dataset1['skill_moves'].value_counts().values

count1 = count / sum(count) * 100

angle = count / sum(count) * 2 * pi



source = ColumnDataSource(data = dict(no_of_skills = no_of_skills, count = count, color = ['skyblue', 'salmon', 'turquoise', 'cyan', 'red'], angle = angle, percentage = count1))
p10 = figure(plot_height = 400, plot_width = 400, title="Pie Chart",

           tools = "hover", tooltips = "@no_of_skills: @percentage", x_range=(-1.0, 1.0))  ## Pie chart for skill moves



p10.wedge(x = 0, y = 1, radius = 0.8,

        start_angle = cumsum('angle', include_zero=True), end_angle = cumsum('angle'),

        line_color = "white", fill_color = 'color', legend_field = 'no_of_skills', source = source)



p10.legend.location = "top_right"



p10.legend.label_text_font_size = '5pt'

p11 = figure(plot_height = 300, plot_width = 500, title = "Number of Skill Moves",

                tools = "hover", tooltips = "@no_of_skills: @count")



p11.vbar(x = 'no_of_skills', top = 'count', width = 0.9, source = source, color = 'color')



p11.xgrid.grid_line_color = None

p11.y_range.start = 0



p11.xaxis.major_label_orientation = "horizontal"
show(gridplot([[p10, p11]]))
dataset1.loc[dataset['skill_moves'] == 5].head(5)
dataset1.loc[dataset['skill_moves'] == 4].head(5)
dataset1.loc[dataset['skill_moves'] == 2].head(5)
dataset1['work_rate'].isnull().sum()
dataset1['work_rate'].value_counts()
different_work_rates = dataset1['work_rate'].value_counts().keys().to_list()

count = dataset1['work_rate'].value_counts().values

count1 = count / sum(count) * 100

angle = count / sum(count) * 2 * pi



source = ColumnDataSource(data = dict(work_rates = different_work_rates, count = count, color = ['skyblue', 'salmon', 'turquoise', 'cyan', 'red', 'lightseagreen', 'teal', 'mediumaquamarine', 'yellowgreen'], angle = angle, percentage = count1))
p12 = figure(plot_height = 600, plot_width = 600, title="Pie Chart",

           tools = "hover", tooltips = "@work_rates: @percentage", x_range=(-1.0, 1.0))  ## Pie chart for work rates



p12.wedge(x = 0, y = 1, radius = 0.8,

        start_angle = cumsum('angle', include_zero=True), end_angle = cumsum('angle'),

        line_color = "white", fill_color = 'color', legend_field = 'work_rates', source = source)



p12.legend.location = "top_right"



p12.legend.label_text_font_size = '5pt'



show(p12)

p13 = figure(x_range = different_work_rates, plot_height = 300, plot_width = 500, title = "Different Work Rates",

                tools = "hover", tooltips = "@work_rates: @count")



p13.vbar(x = 'work_rates', top = 'count', width = 0.9, source = source, color = 'color')



p13.xgrid.grid_line_color = None

p13.y_range.start = 0



p13.xaxis.major_label_orientation = "vertical"



show(p13)
dataset1.loc[dataset1['work_rate'] == 'High/High'].head(5)