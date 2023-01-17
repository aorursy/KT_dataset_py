import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
filename= "/kaggle/input/fandango-data-set/fandango_score_comparison.csv"
before = pd.read_csv(filename)
pd.options.display.max_columns = 100             # to get all the columns

before.head(2)
before_f = before[['FILM', 'Fandango_Stars', 'Fandango_Ratingvalue', 'Fandango_votes', 'Fandango_Difference']].copy()
before_f.head(2)
before_f['year'] = before_f['FILM'].apply(lambda x: x[-5:-1])      # seperating the year from film column. 
before_f.head(2)
filename = "/kaggle/input/movies-rating-in-20162017/movie_ratings_16_17.csv"
after = pd.read_csv(filename)
after.head(2)
after_f = after[['movie', 'year', 'fandango']].copy()
after_f.head(2)
print('Data before email' + '\n' + '-------------------')
print(before_f.year.value_counts())
print('\n\n' + 'Data after email' + '\n' + '----------------------')
print(after_f.year.value_counts())
before_f_in_2015 = before_f[before_f.year == '2015']
before_f_in_2015.year.value_counts()
after_f_in_2016 = after_f[after_f.year == 2016]
after_f_in_2016.year.value_counts()
# figure style and size
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,4), dpi= 90, linewidth= 3)

# plotting the graph
before_f_in_2015.Fandango_Stars.plot.kde(linewidth= 3, color= 'red', label= '2015', legend= True)
after_f_in_2016.fandango.plot.kde(linewidth= 3, color= 'green', label= '2016', legend= True)

# figure ornaments
plt.title("Change in distribution shape of Fandago's rating in 2015 and 2016")
plt.xlabel('Rating')
plt.xlim(0, 5)
plt.xticks(np.arange(0,5.5,.5))
plt.show()
import matplotlib.style as style
style.available
print('Data before email' + '\n' + '-------------------')
before_f_in_2015.Fandango_Stars.value_counts(normalize= True).sort_index() * 100
print('Data after email' + '\n' + '-------------------')
after_f_in_2016.fandango.value_counts(normalize= True).sort_index() * 100
# mean of movie rating
mean_in_2015 = before_f_in_2015.Fandango_Stars.mean()
mean_in_2016 = after_f_in_2016.fandango.mean()

# median of movie rating
median_in_2015 = before_f_in_2015.Fandango_Stars.median()
median_in_2016 = after_f_in_2016.fandango.median()

# mdoe of movie rating
mode_in_2015 = before_f_in_2015.Fandango_Stars.mode()[0]           # mode comes with series
mode_in_2016 = after_f_in_2016.fandango.mode()[0]
# making a dataframe with those mean, median and mode
summary = pd.DataFrame(index= ['mean', 'median', 'mode'])

# assigning column
summary['2015'] = [mean_in_2015, median_in_2015, mode_in_2015]
summary['2016'] = [mean_in_2016, median_in_2016, mode_in_2016]
summary
# figure style
plt.style.use('ggplot')
plt.figure(figsize=(6,4),dpi=95, linewidth= 3)

# plotting mean, median and mode
summary['2015'].plot.bar(color= 'red',label= '2015', legend= True, width= .25)
summary['2016'].plot.bar(color='green',label= '2016', legend= True, position= .9, rot= 0, width= .25, fontsize= 15)

# setting ornaments
plt.yticks(np.arange(0,5,.5))
plt.title("Mean, median and mode of Fandago's rating in 2015 and 2016", fontsize= 18)
plt.legend(bbox_to_anchor= (1,1), fontsize= 'x-large')         # legend position. 
plt.show()
before.head()
before['year'] = before['FILM'].apply(lambda x: x[-5:-1])      # seperating the year from film column. 
before.head()
# Truncating the data set and taking rotten tomato and imdb user rating

before_others_in_2015 = before[before.year == '2015'][['FILM','year','RT_user_norm_round','IMDB_norm_round']].copy()
before_others_in_2015.head()
after.head()
after_others_in_2016 = after[after.year == 2016][['year','nr_imdb','nr_audience']].copy()
after_others_in_2016.head()
# figure style and subplot
plt.style.use('ggplot')
fig, ax = plt.subplots(2, 2, figsize=(10, 6), dpi= 95)
fig.suptitle('Comparison of distribution in 2015 and 2016', y=1.05, color= 'blue')

# plotting rotten tomatoes distribution
before_others_in_2015.RT_user_norm_round.plot.kde(linewidth= 3, color= 'blue', label= 'RT 2015', legend= True, ax= ax[0][0])
after_others_in_2016.nr_audience.plot.kde(linewidth= 3, color= 'green', label= 'RT 2016', legend= True, ax= ax[0][0])

# plotting IMDB distribution
before_others_in_2015.IMDB_norm_round.plot.kde(linewidth= 3, color= 'blue', label= 'IMDB 2015', legend= True, ax= ax[0][1])
after_others_in_2016.nr_imdb.plot.kde(linewidth= 3, color= 'green', label= 'IMDB 2016', legend= True, ax= ax[0][1])

# plotting fandango distribution
before_f_in_2015.Fandango_Stars.plot.kde(linewidth= 3, color= 'blue', label= 'Fandango 2015', legend= True, ax= ax[1][0])
after_f_in_2016.fandango.plot.kde(linewidth= 3, color= 'green', label= 'Fandango 2016', legend= True, ax= ax[1][0])

# setting figure ornaments
ax[0][0].set_title("Rotten Tomatoes")
ax[0][1].set_title("IMDB")
ax[1][0].set_title('Fandango')
ax[0][0].set_xlim(0, 5.1)
ax[0][1].set_xlim(0, 5.1)
ax[1][0].set_xlim(0, 5.1)

plt.tight_layout()
# mean and mdoe of Rotten tomato user for before and after dataset
# mean of movie rating
mean_others_2015 = before_others_in_2015.RT_user_norm_round.mean()
mean_others_2016 = after_others_in_2016.nr_audience.mean()

# mdoe of movie rating
mode_others_2015 = before_others_in_2015.RT_user_norm_round.mode()[0]           # mode comes with series
mode_others_2016 = after_others_in_2016.nr_audience.mode()[0]
# making a dataframe with those mean and mode
summary_others_RT = pd.DataFrame(index= ['mean', 'mode'])

# assigning column
summary_others_RT['2015'] = [mean_others_2015, mode_others_2015]
summary_others_RT['2016'] = [mean_others_2016, mode_others_2016]
summary_others_RT
# mean and mdoe of IMDB user for before and after dataset
# mean of movie rating
mean_others_2015 = before_others_in_2015.IMDB_norm_round.mean()
mean_others_2016 = after_others_in_2016.nr_imdb.mean()

# mdoe of movie rating
mode_others_2015 = before_others_in_2015.IMDB_norm_round.mode()[0]           # mode comes with series
mode_others_2016 = after_others_in_2016.nr_imdb.mode()[0]
# making a dataframe with those mean, median and mode
summary_others_IMDB = pd.DataFrame(index= ['mean', 'mode'])

# assigning column
summary_others_IMDB['2015'] = [mean_others_2015, mode_others_2015]
summary_others_IMDB['2016'] = [mean_others_2016, mode_others_2016]
summary_others_IMDB
# figure style and subplot
plt.style.use('ggplot')
fig, ax = plt.subplots(2, 2, figsize=(10, 7), dpi= 95)
fig.suptitle('Comparison mean and median in 2015 and 2016', y=1.03, color= 'blue')

# plotting rotten tomatoes 
summary_others_RT['2015'].plot.bar(color= 'blue', label= '2015', legend= True, width= .25, ax= ax[0][0])
summary_others_RT['2016'].plot.bar(color= 'green', label= '2016',position= .9,rot=0, legend= True, width= .25, ax= ax[0][0])

# plotting IMDB
summary_others_IMDB['2015'].plot.bar(color= 'blue', label= '2015', legend= True, width= .25, ax= ax[0][1])
summary_others_IMDB['2016'].plot.bar(color= 'green', label= '2016', legend= True, position= .9, rot= 0, width= .25, ax= ax[0][1])

# plotting Fandango
summary.drop('median')['2015'].plot.bar(color= 'blue',label= '2015', legend= True, width= .25, ax= ax[1][0])
summary.drop('median')['2016'].plot.bar(color='green',label= '2016', legend= True, position= .9, rot= 0, width= .25, ax= ax[1][0])

# figure ornaments
ax[0][0].set_title("Rotten Tomatoes")
ax[0][1].set_title("IMDB")
ax[1][0].set_title('Fandango')
ax[1][0].set_yticks(np.arange(0,5,.5))

plt.tight_layout()
