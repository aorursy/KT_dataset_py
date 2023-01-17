import pandas as pd



all_profs = pd.read_csv('../input/all_profs.csv')
all_profs['overall_rating'] = all_profs['overall_rating'].apply(lambda x: float(x)) # transform

all_profs = all_profs[['school', 'name'] + 

                      [col for col in all_profs if col not in ['school', 'name']]] # move things around
all_profs[all_profs.school == 'Oregon State University']
all_profs[all_profs.school == 'Oregon State University'].shape
all_profs[all_profs.school == 'Portland State University']
all_profs[all_profs.school == 'Portland State University'].shape
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Oregon State University'].overall_rating,

              name = 'Oregon State University'))

fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Portland State University'].overall_rating,

              name = 'Portland State University'))

fig.update_layout(title_text = 'Figure 1: Box Plots of Overall Ratings for Statistics Professors/Instructors at OSU and PSU',

                  yaxis_title_text = 'Overall Rating',

                  showlegend = False)

fig.show()
#histogram

fig = go.Figure()

fig.add_trace(go.Histogram(x = all_profs[all_profs.school == 'Oregon State University'].overall_rating,

              name = 'Oregon State University',

                           xbins = dict(size = 0.25)))

fig.add_trace(go.Histogram(x = all_profs[all_profs.school == 'Portland State University'].overall_rating,

              name = 'Portland State University',

                           xbins = dict(size = 0.25)))

fig.update_traces(opacity=0.75)

fig.update_layout(title_text = 'Figure 2: Histogram of Overall Ratings', # title of plot

                  xaxis_title_text = 'Overall Rating', # xaxis label

                  yaxis_title_text = 'Count', # yaxis label

                  bargap = 0.1, # gap between bars of adjacent location coordinates

                  bargroupgap = 0.05 # gap between bars of the same location coordinates

)

fig.show()
# total number of OSU ratings

all_profs[all_profs.school == 'Oregon State University'].number_of_ratings.sum()
# total number of PSU ratings

all_profs[all_profs.school == 'Portland State University'].number_of_ratings.sum()
#number of ratings plots

#box plot

fig = go.Figure()

fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Oregon State University'].number_of_ratings,

              name = 'Oregon State University',

              marker = dict(size = 10)))

fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Portland State University'].number_of_ratings,

              name = 'Portland State University',

              marker = dict(size = 10)))

fig.update_layout(title_text = 'Figure 3: Box Plots of Number of Ratings',

                  yaxis_title_text = 'Number of Ratings',

                  showlegend = False)

fig.show()
#histogram

fig = go.Figure()

fig.add_trace(go.Histogram(x =  all_profs[all_profs.school == 'Oregon State University'].number_of_ratings,

              name = 'Oregon State University',

                           xbins = dict(size = 5)))

fig.add_trace(go.Histogram(x = all_profs[all_profs.school == 'Portland State University'].number_of_ratings,

              name = 'Portland State University',

                           xbins = dict(size = 5)))

fig.update_traces(opacity=0.75)

fig.update_layout(title_text = 'Figure 4: Histogram of Number of Ratings', # title of plot

                  xaxis_title_text = 'Number of Ratings', # xaxis label

                  yaxis_title_text = 'Count', # yaxis label

                  #bargap = 1, # gap between bars of adjacent location coordinates

                  bargroupgap = 0.1 # gap between bars of the same location coordinates

)

fig.show()
# scatter overall

fig = go.Figure()

fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Oregon State University'].highest_level_rated,

                         y = all_profs[all_profs.school == 'Oregon State University'].overall_rating,

                         name = 'Oregon State University',

                         mode = 'markers',

                         marker = dict(size = 20),

                         hovertext = all_profs[all_profs.school == 'Oregon State University'].name))

fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Portland State University'].highest_level_rated,

                         y = all_profs[all_profs.school == 'Portland State University'].overall_rating,

                         name = 'Portland State University',

                         mode = 'markers',

                         marker = dict(size = 20),

                         hovertext = all_profs[all_profs.school == 'Portland State University'].name))

fig.update_traces(opacity=0.75)

fig.update_layout(title_text = 'Figure 5: Scatter Plot of Overall Ratings by Class Level', # title of plot

                  yaxis_title_text = 'Overall Rating', # yaxis label

                  bargroupgap = 0.1, # gap between bars of the same location coordinates

                  xaxis = go.layout.XAxis(tickvals = [200, 300, 400, 500]),

                  xaxis_title_text = 'Class Level')

fig.show()
# scatter number

fig = go.Figure()

fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Oregon State University'].highest_level_rated,

                         y = all_profs[all_profs.school == 'Oregon State University'].number_of_ratings,

                         name = 'Oregon State University',

                         mode = 'markers',

                         marker = dict(size = 20),

                         hovertext = all_profs[all_profs.school == 'Oregon State University'].name))

fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Portland State University'].highest_level_rated,

                         y = all_profs[all_profs.school == 'Portland State University'].number_of_ratings,

                         name = 'Portland State University',

                         mode = 'markers',

                         marker = dict(size = 20),

                         hovertext = all_profs[all_profs.school == 'Portland State University'].name))

fig.update_traces(opacity=0.75)

fig.update_layout(title_text = 'Figure 6: Scatter Plot of Number of Ratings by Class Level', # title of plot

                  yaxis_title_text = 'Number of Ratings', # yaxis label

                  bargroupgap = 0.1, # gap between bars of the same location coordinates

                  xaxis = go.layout.XAxis(tickvals = [200, 300, 400, 500]),

                  xaxis_title_text = 'Class Level')

fig.show()
all_profs.groupby(['school', 'highest_level_rated']).overall_rating.describe()
from scipy import stats

# 2/300 level

osu_300 = all_profs[(all_profs.school == 'Oregon State University') & (all_profs.highest_level_rated == 300)].overall_rating

psu_200 = all_profs[(all_profs.school == 'Portland State University') & (all_profs.highest_level_rated == 200)].overall_rating

stats.mannwhitneyu(osu_300, psu_200, alternative = 'greater')
# 400 level

osu_400 = all_profs[(all_profs.school == 'Oregon State University') & (all_profs.highest_level_rated == 400)].overall_rating

psu_400 = all_profs[(all_profs.school == 'Portland State University') & (all_profs.highest_level_rated == 400)].overall_rating

stats.mannwhitneyu(osu_400, psu_400, alternative = 'less')
# 500 level

osu_500 = all_profs[(all_profs.school == 'Oregon State University') & (all_profs.highest_level_rated == 500)].overall_rating

psu_500 = all_profs[(all_profs.school == 'Portland State University') & (all_profs.highest_level_rated == 500)].overall_rating

stats.mannwhitneyu(osu_500, psu_500, alternative = 'less')
all_profs.groupby(['school', 'highest_level_rated']).number_of_ratings.describe()
all_ratings = pd.read_csv('../input/all_ratings.csv') # load in individual ratings

all_ratings.head()
all_ratings.shape[0] ==  sum(all_profs.number_of_ratings)
all_merged = pd.merge(all_profs, all_ratings, on = 'id')

all_merged.head()
# 2/300 level

stats.ttest_ind(all_merged[(all_merged.school == 'Oregon State University') & (all_merged.highest_level_rated == 300)].ratings,

                all_merged[(all_merged.school == 'Portland State University') & (all_merged.highest_level_rated == 200)].ratings,

                equal_var = False)

# 400 level

stats.ttest_ind(all_merged[(all_merged.school == 'Oregon State University') & (all_merged.highest_level_rated == 400)].ratings,

                all_merged[(all_merged.school == 'Portland State University') & (all_merged.highest_level_rated == 400)].ratings,

                equal_var = False)

# 500 level

stats.ttest_ind(all_merged[(all_merged.school == 'Oregon State University') & (all_merged.highest_level_rated == 500)].ratings,

                all_merged[(all_merged.school == 'Portland State University') & (all_merged.highest_level_rated == 500)].ratings,

                equal_var = False)