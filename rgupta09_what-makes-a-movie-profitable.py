# Import libraries & read the data
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import seaborn as sns
import statsmodels.formula.api as smf
import os
import matplotlib.cm as cm
%matplotlib inline

df = pd.read_csv('../input/imdb-dataset-updated/movie_metadata.csv')
df.head()
df.drop(df.columns[[0,2,3,4,5,7,13,15,16,17,18,19,24,26,27]], axis=1, inplace=True)
df.columns = ['Director','Actor_1','Gross','Genre','Actor_2','Movie_Name','Number_of_Votes','Actor_3','Country', 'Rating', 'Budget', 'Year', 'IMDB_Score']
df = df.drop_duplicates('Movie_Name')
df = df[pd.notnull(df['Gross']) & pd.notnull(df['Budget'])]
df = df.set_index('Movie_Name')
df['Year'] = df['Year'].astype(int)
df.head()
df_bybudget = df.sort_values('Budget', ascending=False)
df_bybudget['Country'].head(5)

vlist = ['USA', 'UK']
df = df[df['Country'].isin(vlist)]


# Build a function to adjust dollar value in millions
def millions(number):
    '''show dollar value in millions'''
    return number*(1/1000000)

#Use this millions function to create new columns 
#   - "Budget_inmillion" and 
#   - "Gross_inmillions"
budgetmillions = millions(df['Budget'])
grossmillions = millions(df['Gross'])
df = df.assign(Budget_inmillions=budgetmillions)
df = df.assign(Gross_inmillions=grossmillions)

# Now we can get rid of the old columns "Budget" and "Gross"
df = df.drop(['Budget','Gross'], 1)

# Rename the new values in millions as:
#  "Budget_inmillions"  ---> "Budget"
#  "Gross_inmillions"   ---> "Gross"
df = df.rename(columns={'Budget_inmillions': 'Budget', 'Gross_inmillions': 'Gross'})
# Let's look and our list of variables and their types
#print('Variable dtypes:\n', df.dtypes, sep='')
# Create a new data frame called df_budget that has only
# - Year
# - Budget, and
# - Gross
# Drop every other column

df_budget = df.drop(['Director','Actor_1','Actor_2','Actor_3','Genre','Country','Rating','IMDB_Score','Number_of_Votes'], 1)
df_budget.head()
# Plot of Budget vs Gross
fig, ax = plt.subplots(figsize=(15, 8))

ax.stem(df_budget.Budget, df_budget.Gross, '#707B7C', markerfmt='C0o', basefmt='#707B7C')
ax.set_title("Budget vs Gross", fontsize=15)
ax.set_xlabel("Budget", fontsize=15)
ax.set_ylabel("Gross", fontsize=15)



f, ax = plt.subplots(figsize=(8, 5))
sns.regplot(x = 'Gross', y = 'Budget', data=df_budget, color='c')
ax.set_title("Budget vs Gross", fontsize=15)
ax.set_xlabel("Gross", fontsize=12)
ax.set_ylabel("Budget", fontsize=12)

regression = smf.ols('Budget ~ Gross ', data=df_budget).fit()

regression.rsquared
# Start with sorting the df_budget table by year
df_budget = df_budget.sort_values('Year', ascending=False)
df_budget.head()

# Select only the movies from 2000 to 2017
yearlist = range(2000,2017)
df_newbudget = df_budget[df_budget['Year'].isin(yearlist)]


# Look at the scatter plot
f, ax = plt.subplots(figsize=(8, 5))
sns.regplot(x = 'Gross', y = 'Budget', data=df_newbudget, color='#AF7AC5')
ax.set_title("Budget vs Gross", fontsize=15)
ax.set_xlabel("Budget", fontsize=12)
ax.set_ylabel("Gross", fontsize=12)
regression = smf.ols('Budget ~ Gross', data=df_newbudget).fit()
regression.rsquared

# To better visualize this relationship, let's bin budget ranges into buckets 
# (0-1, 1-10, 10-50, 50-100, 100-150, and 150-300) and 
# add a new column called "Avg Budget" with these new values

bins = [0,1,10,50,100,150,301]
group_names = ['0-1','1-10','10-50','50-100','100-150','150-300']

df_budget = df_budget.sort_values('Budget', ascending=False)
categories = pd.cut(df_budget['Budget'], bins, labels=group_names)
df_budget['Avg_Budget'] = pd.cut(df_budget['Budget'], bins, labels=group_names)

# lets look at how many movies are in each budget range
pd.value_counts(df_budget['Avg_Budget'])

# Generate a bar chart of gross revenues.
f, ax = plt.subplots(figsize=(10, 5))
Budget = sns.barplot(x="Avg_Budget", y="Gross", data=df_budget, palette="magma")
Budget.axes.set_title('Average Budget v. Gross Revenue', fontsize=15)
# Generate a bar chart of gross revenues.
f, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x="Avg_Budget", y="Gross", data=df_budget, palette="cubehelix")
ax.set_title('Average Budget v. Gross Revenue', fontsize=15)

# Take a subset of columns from df for IMDB score analysis
df_imdb = df.drop(['Director','Actor_1','Actor_2','Actor_3','Budget','Country','Rating','Genre','Year'], 1)

# Remove nulls from IMDB score dataset
df_imdb = df_imdb[pd.notnull(df_imdb['IMDB_Score']) & pd.notnull(df_imdb['Number_of_Votes'])]

df_imdb.head()
# To better visualize the distribution of IMDB scores,, let's bin budget ranges into buckets 
# (0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9 and 9-10) and 
# add a new column called "Scores_groups" with these new values

bins = [0,1,2,3,4,5,6,7,8,9,10]
group_names = ['1','2','3','4','5','6','7','8','9','10']
df_imdb['Score_groups'] = pd.cut(df_imdb['IMDB_Score'], bins, labels=group_names)

# View the IMDB Score table with the newly added column
df_imdb.head()
# Generate a bar chart of IMDB Score Groups & Gross
f, ax = plt.subplots(figsize=(10, 5))
scores = sns.barplot(x="Score_groups", y="Gross", data=df_imdb, palette="Oranges")
scores.axes.set_title('IMDB Score Groups v. Gross Revenue', fontsize=15)
scores.axes.set_xlabel('IMDB Score Groups', fontsize=12)
scores.axes.set_ylabel('IMDB Score Groups', fontsize=12)
# View the correlation between IMDB Score & Gross
c = df_imdb['IMDB_Score'].corr(df_imdb['Gross'])
print ('Correlation between IMDB Score & Gross :\n', c, sep='')
# Plot  IMDB Score vs Gross
sns.jointplot("IMDB_Score", "Gross", data=df_imdb, kind='reg', color='#E67E22');
# Build a function to convert a number on a 10000's scale
def thou(number):
    '''show dollar value in 10,000s'''
    return number*(1/10000)


#Use this function to create a new column
# "Number_of_Votes" in 10.1000's
# For better visualization
voters_thou = thou(df_imdb['Number_of_Votes'])

df_imdb = df_imdb.assign(Votes_in_Thou=voters_thou)
df_imdb = df_imdb.drop(['Number_of_Votes'], 1)
df_imdb = df_imdb.rename(columns={'Votes_in_Thou' : 'Number_of_Votes'})

# Correlation of Number of Votes by Gross
c = df_imdb['Number_of_Votes'].corr(df_imdb['Gross'])
print ('Correlation between Number of Votes & Gross :\n', c, sep='')

# This shows a higher correlation than between IMDB score & Gross
# Plot Number of Votes vs Gross
sns.jointplot("Number_of_Votes", "Gross", data=df_imdb, kind='reg', color='#F5B041');

# Following plot  shows positive association
# with a higher r value (0.63) than Gross vs IMDB score (0.25)
# Define the bins
bins1 = [0,50,100,150,200]
group_names1 = ['Low','Average','Good','High']

# Add column IMDB_Score_cat to break IMDB scores into these groups
df_imdb['Votes_Groups'] = pd.cut(df_imdb['Number_of_Votes'], bins1, labels=group_names1)

#View the newly create dataset
df_imdb.head()
# View distribution of the 3 variables
fig, ax = plt.subplots(1, 3, figsize=(18, 6))


# Plot distribution of IMDB scores
df_imdb['IMDB_Score'].plot(kind='hist', color = '#D35400', histtype='step', ax=ax[0], linewidth=2)
ax[0].set_title("IMDB Score", fontsize=18, color = '#D35400')
ax[0].set_xlabel("Scores", fontsize=15, color = '#D35400')
ax[0].set_ylabel("Counts", fontsize=15, color = '#D35400')
ax[0].grid(color="slateblue", which="both", linestyle=':', linewidth=0.5)

# Plot distribution of Number of Votes
df_imdb['Number_of_Votes'].plot(kind='hist', color = '#D35400', histtype='step', ax=ax[1], linewidth=2)
ax[1].set_title("Number of Votes", fontsize=18, color = '#D35400')
ax[1].set_xlabel("Votes", fontsize=15, color = '#D35400')
ax[1].set_ylabel("Counts", fontsize=15, color = '#D35400')
ax[1].grid(color="slateblue", which="both", linestyle=':', linewidth=0.5)

# Plot distribution of Gross
df_imdb['Gross'].plot(kind='hist', color = '#D35400', histtype='step', ax=ax[2], linewidth=2)
ax[2].set_title("Gross", fontsize=18, color = '#D35400')
ax[2].set_xlabel("Gross", fontsize=15, color = '#D35400')
ax[2].set_ylabel("Values", fontsize=15, color = '#D35400')
ax[2].grid(color="slateblue", which="both", linestyle=':', linewidth=0.5)

fig.tight_layout()
colors = cm.YlOrRd(np.linspace(0,1,len(df_imdb)))

f, ax = plt.subplots(figsize=(14, 7))
plt.scatter(df_imdb['IMDB_Score'], df_imdb['Gross'], s=df_imdb['Number_of_Votes'], color=colors, alpha=0.5)
ax.set_title("IMDB Score vs Gross based on Num of Votes", fontsize=18)
ax.set_xlabel("Scores", fontsize=18)
ax.set_ylabel("Gross", fontsize=18)
plt.show()
# Drop the Score groups & Number of Votes columns
df_imdb = df_imdb.drop(['Score_groups', 'Number_of_Votes'], 1)

# Plot of IMDB_Score vs Gross, with reference on Groups per Number of Votes
sns.pairplot(df_imdb, hue = 'Votes_Groups', palette = 'tab10', size=3, aspect=2)
# Take a subset of columns from df for IMDB score analysis
df_mpaa = df[['Rating', 'Gross']].copy()
df_mpaa = df_mpaa.fillna('Unrated')

# View the distinct values for Ratings
df_mpaa['Rating'].unique()
# Remove Movie Name index from mpaa dataset
df_mpaa = df_mpaa.reset_index()
df_mpaa = df_mpaa.drop(['Movie_Name'], 1)
df_mpaa.head()
# Plot of Distribution of MPAA Ratings' counts 
fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(x='Rating', data=df_mpaa, palette='Pastel2')
ax.set_title("Distribution of Ratings", fontsize=15)
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Counts", fontsize=12)
plt.tight_layout()
# Plot of MPAA Ratings vs Gross
fig, ax = plt.subplots(figsize=(12,5))

sns.stripplot(x="Rating", y="Gross", data=df_mpaa, ax = ax)
sns.swarmplot(x="Rating", y="Gross", data=df_mpaa, ax = ax)

ax.set_title("Ratings vs Gross", fontsize=15)
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Gross", fontsize=12)

plt.tight_layout()
# Another plot of MPAA Ratings vs Gross
fig, ax = plt.subplots(figsize=(12,5))

sns.boxplot(x="Rating", y="Gross", data=df_mpaa, ax = ax)
sns.violinplot(x="Rating", y="Gross", data=df_mpaa, ax = ax)

ax.set_title("Ratings vs Gross", fontsize=15)
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Gross", fontsize=12)

plt.tight_layout()
# Create a new data frame called df_genre that has only
# - Year
# - Gross
# - Genre
# Drop every other column

df_genre = df[['Genre','Year','Gross']].copy()

df_genre.head()

# Import lib for word cloud visualization
from wordcloud import WordCloud

# Define function for counting words
def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

# From df_genre, generate a list of only genres
genre_list = list()
for g in df_genre['Genre'].str.split('|').values:
    genre_list.append(g)
    

# Convert genres list to str for wordc cloud
genre = ' '.join(str(r) for v in genre_list for r in v)

# Count occurence of each genre
cnt = word_count(genre)

# Generate wordcloud for the count of each Genre
wordcloud = WordCloud(colormap="copper").generate_from_frequencies(cnt)

# Display word cloud
fig = plt.figure(figsize=(15, 25))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# Plotting the distribution of genres using histograms
fig, ax = plt.subplots(figsize=(17, 6))

plt.bar(cnt.keys(), cnt.values(), color='#CA6F1E')

ax.set_title("Distribution of Genres", fontsize=12)
ax.set_ylabel("Counts")
ax.set_xlabel("Genres")

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_rotation(45)

plt.show()
# Since there are multiple values of Genre per movie
# Let's first split it into individual Genres

# Extract the Year column & split the Genres into individual genres
# into a new dataset - genre_year
genre_year = pd.concat([pd.Series(row['Year'], row['Genre'].split('|'))              
                    for _, row in df_genre.iterrows()]).reset_index()

genre_year.columns = ['Genre', 'Year']

# Then we can group by the different genres
grouped = genre_year.groupby('Genre')

# In order to visualize how genres have evolved over the years
# Let's first bin the years into different groups

bins = [1920,1990,2000,2010,2020]
group_names = ['1920-1990','1990-2000','2000-2010','2010-Current']

# Then we can add a new column to the dataset
# "Year_bin" - containing the group of year per the bins
genre_year_bin = genre_year[['Genre', 'Year']].copy()
genre_year_bin['Year_bin'] = pd.cut(genre_year_bin['Year'], bins, labels=group_names)

# We can now drop the Year column from the dataset, keeping only the year bins
genre_year_bin = genre_year_bin.drop(['Year'], 1)

# And view the updated dataset
genre_year_bin.head()
# Create grouping of count of each genre per the Year bins
x = genre_year_bin.groupby(['Year_bin', 'Genre'])

# Save the counts & view as a dataframe
y = x.size()
z = y.unstack()
z.fillna(0)

# Display the data of count of genres per year group
z
# Plot the evolution of genres over time
fig, ax = plt.subplots(figsize=(27,11))
z.plot(kind='bar',ax=ax, width=0.9)
ax.set_title("Evolution of genres over time", fontsize=25)
ax.set_xlabel("Genre Groups in Year Groups", fontsize=20)
ax.set_ylabel("Count of each Genre per Year Groups", fontsize=20)
ax.legend(fontsize=13)
# Extract  & split the Genres into individual genres
# into a new dataset - genres
genres = df_genre.Genre.str.split('|', expand=True).stack().reset_index(level=1, drop=True)

# Convert to a Dataframe
genres = pd.DataFrame({'Movie_Name':genres.index, 'Genre':genres.values})
genres = genres.set_index('Movie_Name')

# View the genres dataset
genres.head()
# Now let's merge the genres list with original genres dataset
genre_year_gross = df_genre.drop(['Genre'], axis=1).join(genres).reset_index(drop=True)

# View the new dataset
genre_year_gross.head()

# Now we will again group the years into the same bins that were defined earlier
genre_year_gross['Year_bin'] = pd.cut(genre_year_gross['Year'], bins, labels=group_names)
genre_year_gross = genre_year_gross.drop(['Year'], 1)


# Create grouping of Total Gross for each genre per the Year bins
x1 = genre_year_gross.groupby(['Year_bin', 'Genre'])

# Save the data & view as a dataframe
y1 = x1.sum()
z1 = y1.unstack()
z1.fillna(0)

# Display the data for Total Gross of each genre per year group
z1
# Plot the Impact of genres on Gross
fig, ax = plt.subplots(figsize=(27,11))
z1.plot(kind='bar',ax=ax, width=0.9)
ax.set_title("Gross of Genre groups per Year Groups", fontsize=25)
ax.set_xlabel("Genre groups in Year groups", fontsize=20)
ax.set_ylabel("Gross", fontsize=20)
ax.legend(fontsize=13)
fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(25,35))

# Plot the evolution of genres over time
z.plot(ax=ax1, linewidth=3)
ax1.set_title("Evolution of genres over time", fontsize=25)
ax1.set_xlabel("Genre Groups in Year Groups", fontsize=20)
ax1.set_ylabel("Count of each Genre per Year Groups", fontsize=20)
ax1.legend(fontsize=13)

# Plot the Impact of genres on Gross
z1.plot(ax=ax2, linewidth=3)
ax2.set_title("Impact of Genres on Gross", fontsize=25)
ax2.set_xlabel("Genre groups in Year groups", fontsize=20)
ax2.set_ylabel("Gross", fontsize=20)
ax2.legend(fontsize=13)
# First we will create a sub-dataset of Director data
df_director = df.drop(['Actor_1','Actor_2','Actor_3','Number_of_Votes','Genre','Country','Rating','IMDB_Score','Budget','Year'], 1)

# Add column Value_Counts with the Count of Movies for each Director
df_director['Movie_Counts'] = df_director.groupby('Director')['Director'].transform('count')

df_director.head()
# We will bin the Counts into 2 categories
# Less than 12 Movies, & More than 12 Movies
# add a new column called "Popularity_of_Director" with these groups
# Define the bins
bins = [0,12,30]
group_names = ['Less than 12','More than 12']

# Add the column
df_director['Popularity_Movie_Counts'] = pd.cut(df_director['Movie_Counts'], bins, labels=group_names)

# View the dataset
df_director.head()
# Box-Plot of Gross vs Director Popularity in terms of Number of Movies
fig, ax = plt.subplots(figsize=(14,5))
sns.boxplot(x="Movie_Counts", y="Gross", data=df_director, palette="Set3",ax=ax)

ax.set_title("Director Popularity in Number of Movies vs Gross", fontsize=15)
ax.set_xlabel("Movie Counts", fontsize=12)
ax.set_ylabel("Gross", fontsize=12)

plt.tight_layout()
c = pd.value_counts(df_director['Popularity_Movie_Counts'])
print ('How Many directors with less/more than 12 movies: \n', c, sep = '')
# Another way to determine Director popularity is 
# check in terms of how long the director has been making movies for
# Copy the relevant columns for Director in a new dataset
df_dir = df[['Director','Year','Gross']].copy()

df_dir = df_dir.reset_index()
df_dir = df_dir.drop(['Movie_Name'],1)

# View the data
df_dir.head()
# Create a pivot table with the Number of Years worked for each Director
p1 = pd.pivot_table(df_dir, index = ['Director'], values=['Year'],
               aggfunc= [np.max, np.min])

p1['Years'] = p1.amax['Year'] - p1.amin['Year']

p1.columns = p1.columns.droplevel(level = 1)

p1.reset_index('Director', inplace=True)

p1 = p1[['Director','Years']].copy()

p1 = p1.sort_values('Years',ascending = False)

# Merge this data with earlier dataset for directors, to include Gross
df_director = pd.merge(df_director, p1, on='Director', how='left')



# Box-Plot of Gross vs Director Popularity in terms of Number of Movies
fig, ax = plt.subplots(figsize=(14,5))
sns.boxplot(x="Years", y="Gross", data=df_director, palette="Set3",ax=ax)

ax.set_title("Director Popularity in Number of Years Worked vs Gross", fontsize=15)
ax.set_xlabel("Years Worked", fontsize=12)
ax.set_ylabel("Gross", fontsize=12)


plt.tight_layout()
# Next we will bin the Number of Years worked for each Director into 2 categories
    # - Less than 30 Years
    # - More than 30 Years
bins1 = [0,30,60]
group_names1 = ['Less than 30','More than 30']
df_director['Popularity_Years_Counts'] = pd.cut(df_director['Years'], bins1, labels=group_names1)

c1 = pd.value_counts(df_director['Popularity_Years_Counts'])
print ('How Many directors with less/more than 30 years: \n', c1, sep = '')
f, ax = plt.subplots(1,2, figsize=(14, 7))
Years = sns.barplot(x="Popularity_Years_Counts", y="Gross", data=df_director, palette="PuBu", ax=ax[0])
Years.axes.set_title('Popularity of Director in Count of Years Worked v. Gross', fontsize=15)
Years.axes.set_xlabel("Years Worked", fontsize=12)
Years.axes.set_ylabel("Gross", fontsize=12)


Fame = sns.barplot(x="Popularity_Movie_Counts", y="Gross", data=df_director, palette="PuBu_r", ax=ax[1])
Fame.axes.set_title('Popularity of Director per Count of Movies v. Gross', fontsize=15)
Fame.axes.set_xlabel("Movie Counts", fontsize=12)
Fame.axes.set_ylabel("Gross", fontsize=12)

df_director['Gross'].corr(df_director['Movie_Counts'])
# Plot of Gross vs Director Popularity in terms of Number of Movies
fig, ax = plt.subplots(figsize=(14,5))
sns.swarmplot(x="Movie_Counts", y="Gross", data=df_director, palette="winter_r",ax=ax)

ax.set_title("Director Popularity in Number of Movies vs Gross", fontsize=15)
ax.set_xlabel("Movie Counts", fontsize=12)
ax.set_ylabel("Gross", fontsize=12)

plt.tight_layout()
df_director['Years'].corr(df_director['Gross'])
# Plot of Gross vs Director Popularity in terms of Number of Years
fig, ax = plt.subplots(figsize=(14,5))
sns.stripplot(x="Years", y="Gross", data=df_director, palette="terrain",ax=ax)

ax.set_title("Director Popularity in Number of Years Worked vs Gross", fontsize=15)
ax.set_xlabel("Years Worked", fontsize=12)
ax.set_ylabel("Gross", fontsize=12)

plt.tight_layout()
df_director['Years'].corr(df_director['Movie_Counts'])
# Plot of Director Popularity in terms of Number of Movies vs Number of Years
fig, ax = plt.subplots(figsize=(14,5))
sns.barplot(x="Years", y="Movie_Counts", data=df_director, palette="ocean_r",ax=ax)

ax.set_title("Comparison of Director Popularity in terms of Number of Movie & Years Worked", fontsize=15)
ax.set_xlabel("Years Worked", fontsize=12)
ax.set_ylabel("Number of Movies", fontsize=12)

ax.patch.set_facecolor('w')
fig.set_facecolor('w')

plt.tight_layout()
# We will first make a list of Top 100 actors by their individual Gross
# Then we will analyse whether a higher grossing actor being in a movie
# results in higher grossing movies or not

# Take a subset of columns from df for IMDB score analysis
df_actor = df[['Actor_1', 'Actor_2', 'Actor_3', 'Gross']].copy()

#Replace the nulls
df_actor = df_actor.fillna('Unknown')

#View the dataset
df_actor.head()
# Take a subset of the rows with the top 100 Grossing movies from thisdataset
gross_top100 = df_actor.nlargest(100, 'Gross')

# View the dataset
# This is the list of Top 100 Grossing movies 
print ('Top 100 Grossing movies : \n ' )
gross_top100.head()
# Create a list of all actor names
actors_list = df_actor['Actor_1'].tolist() + df_actor['Actor_2'].tolist() + df_actor['Actor_3'].tolist()

# Distinct list of actor names
actors_uni = set(actors_list)

# Convert back to list
actors = list(actors_uni)

# Number of distinct actors
l = len(actors)

# Create a new dataset with the distinct list of actor names
# And their individual gross
# Here we will consider the gross for an actor as the Sum of all the movies' gross they have starred in

cols = ['Name','Total']
actor_sum = pd.DataFrame(index=range(1,l), columns=cols)

for i in range(1, l):
    name = actors[i]
    total = df_actor['Gross'][(df_actor['Actor_1'] == name) | (df_actor['Actor_2'] == name) | (df_actor['Actor_3'] == name)].sum()
    
    actor_sum['Name'][i] = name
    actor_sum['Total'][i] = total
    
# Data types of the dataset
actor_sum.dtypes

# Convert Gross column to numeric
actor_sum['Total'] = actor_sum['Total'].apply(pd.to_numeric)

# Now create a new dataset with the top 100 from this list
actor_s_top100 = actor_sum.nlargest(100, 'Total')

actor_s_top100 = actor_s_top100.reset_index(drop=True)
# Add a column to Top 100 Actor's list to include whether they were part of the Top 100 Grossing movies or not
val = 'Yes'

for i, row1 in actor_s_top100.iterrows():
    for j, row2 in gross_top100.iterrows():
        if ( (row1.Name == row2.Actor_1) | (row1.Name == row2.Actor_2) | (row1.Name == row2.Actor_3) ):
            actor_s_top100.set_value(i, 'Presence', val)
            
actor_s_top100[['Presence']] = actor_s_top100[['Presence']].fillna('No')
# This is the list of Top 100 actors 
# per the sum of grosses of all their movies
print ('Top 100 Grossing actors : \n ')
actor_s_top100.head()
print ('Count of Top Grossing Actors present in Top Grossing Movies or not : \n ')
actor_s_top100.Presence.value_counts()
f, ax = plt.subplots(1, 2, figsize=(11, 7))

sns.violinplot(x="Presence", y="Total", data=actor_s_top100, color='#EBDEF0', inner="stick", ax=ax[0])
ax[0].set_title('Popularity of Actors (in Gross) v. Movies Gross', fontsize=15,color='#7D3C98', alpha = 0.8)

sns.barplot(x="Presence", y="Total", data=actor_s_top100, color='#EBDEF0', ax=ax[1])
ax[1].set_title('Popularity of Actors (in Gross) v. Movies Gross', fontsize=15,color='#7D3C98', alpha = 0.8)

f.tight_layout()
bins = [0,1000,1500,2000,2500,3000,3500,4000]
group_names = ['0-1k','1-1.5k','1.5-2k','2-2.5k','2.5-3k','3-3.5k','3.5-4k']
actor_s_top100['Total_groups'] = pd.cut(actor_s_top100['Total'], bins, labels=group_names)

# View the dataset with the newly added column
actor_s_top100[:20]
f, ax = plt.subplots(figsize=(13, 7))
sns.barplot(x="Total_groups", y="Total", hue="Presence", data=actor_s_top100, palette = "autumn")
ax.set_title('Popularity of Actors (in Gross) v. Movies Gross', fontsize=15,color='k', alpha = 0.8)
ax.set_xlabel("Popularity", fontsize=12)
ax.set_ylabel("Total", fontsize=12)
# Create a new dataset with the distinct list of actor names
# And their individual gross
# Here we will consider the gross for an actor as the Mean of all the movies' gross they have starred in

cols = ['Name','Total']
actor_mean = pd.DataFrame(index=range(1,l), columns=cols)

for i in range(1, l):
    name = actors[i]
    total = df_actor['Gross'][(df_actor['Actor_1'] == name) | (df_actor['Actor_2'] == name) | (df_actor['Actor_3'] == name)].mean()
    
    actor_mean['Name'][i] = name
    actor_mean['Total'][i] = total
    
# Data types of the dataset
actor_mean.dtypes

# Convert Gross column to numeric
actor_mean['Total'] = actor_mean['Total'].apply(pd.to_numeric)
# Now, we will use the mean of gross for each actor, from this list
# To add 2 columns in the original dataset, to include
# - Star (Highest grossing actor out of the 3 actors for each movie)
# - Star_Gross (their individual mean gross)

# Function to find the larget of 3 numbers
def largest(a, b, c):
    if (a > b) :
        if (a > c) :
            return a
        else :
            return c
    elif (b > c):
        return b
    else :
        return c
for i, row1 in df_actor.iterrows():
    actor_1_gross = 0
    actor_2_gross = 0
    actor_3_gross = 0
    
    actor_1_name = ""
    actor_2_name = ""
    actor_3_name = ""
    
    for j, row2 in actor_mean.iterrows():
        if (row1.Actor_1 == row2.Name):
            actor_1_gross = row2.Total
            actor_1_name = row2.Name
        if (row1.Actor_2 == row2.Name):
            actor_2_gross = row2.Total
            actor_2_name = row2.Name
        if (row1.Actor_3 == row2.Name):
            actor_3_gross = row2.Total
            actor_3_name = row2.Name
      
    top = largest(actor_1_gross, actor_2_gross, actor_3_gross)
    if (top == actor_1_gross):
        star = actor_1_name
    if (top == actor_2_gross):
        star = actor_2_name
    if (top == actor_3_gross):
        star = actor_3_name 
            
    df_actor.set_value(i, 'Star', star)
    df_actor.set_value(i, 'Star_Gross', top)       
# View the updated dataset
df_actor.head()
# See the correlation
df_actor['Gross'].corr(df_actor['Star_Gross'])
# Find out the regression value
regression = smf.ols('Gross ~ Star_Gross ', data=df_actor).fit()
regression.rsquared
# lets also view the correlation for the top 1000 movies
top_1000 = df_actor.nlargest(1000, 'Gross')
top_1000['Gross'].corr(top_1000['Star_Gross'])
# View Gross vs Star Gross
f, ax = plt.subplots(figsize=(8, 5))
sns.regplot(x = 'Gross', y = 'Star_Gross', data=df_actor, color = '#7FB3D5')
ax.set_title("Star Power vs Gross", fontsize=15)
ax.set_xlabel("Movie Gross", fontsize=12)
ax.set_ylabel("Star Gross", fontsize=12)
f, ax = plt.subplots(figsize=(8, 5))
sns.regplot(x = 'Gross', y = 'Star_Gross', data=top_1000, color = 'c')
ax.set_title("Star Power vs Gross (Top 1000)", fontsize=15)
ax.set_xlabel("Movie Gross", fontsize=12)
ax.set_ylabel("Star Gross", fontsize=12)