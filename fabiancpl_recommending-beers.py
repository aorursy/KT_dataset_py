import math
import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
beers_df = pd.read_csv( '../input/beer_reviews.csv' )
beers_df.shape
beers_df[ 'review_time' ] = pd.to_datetime( beers_df[ 'review_time' ], unit = 's' )
beers_df.head()
beers_df.dtypes
#pandas_profiling.ProfileReport( beers_df )
# I consider reviews from 2002 because for previous years there are no much information.
beers_df = beers_df.loc[ beers_df[ 'review_time' ].dt.year >= 2002 ]
beers_df.shape
group_by_date = beers_df[ [ 'review_time' ] ].groupby( beers_df[ 'review_time' ].dt.date ).agg( [ 'count' ] )
plt.figure( figsize = ( 20, 5 ) )
plt.plot( group_by_date )
plt.xlabel( 'Date' )
plt.ylabel( 'Number of reviews' )
plt.title( 'Number of Reviews per Day' )
plt.show()
# Count of unique breweries => Integrity issues evidenced => Id is not considered for subsequent analysis
print( 'Unique breweries' )
print( 'By id:', beers_df[ 'brewery_id' ].nunique() )
print( 'By name:', beers_df[ 'brewery_name' ].nunique() )
# Count of unique beers => Integrity issues evidenced => Id is not considered for subsequent analysis
print( 'Unique beers' )
print( 'By id:', beers_df[ 'beer_beerid' ].nunique() )
print( 'By name:', beers_df[ 'beer_name' ].nunique() )
# Count of unique users
print( 'Unique users:', beers_df[ 'review_profilename' ].nunique() )
print( 'Unique users with more than 1 review:', beers_df[ 'review_profilename' ].value_counts()[ beers_df[ 'review_profilename' ].value_counts() > 1 ].shape[ 0 ], '-' , str( round( beers_df[ 'review_profilename' ].value_counts()[ beers_df[ 'review_profilename' ].value_counts() > 1 ].shape[ 0 ] / beers_df[ 'review_profilename' ].nunique(), 2 ) * 100 ) + '%' )
# Reviews by user
beers_df[ 'review_profilename' ].value_counts().head()
# A beer subset removing review information is created 
grouped_beers_df = beers_df[ [ 'beer_name', 'brewery_name', 'beer_style', 'beer_abv' ] ].drop_duplicates()
# Count of unique beers in grouped dataset => Integrity issues evidenced with respect to previous analysis => For beer identification, I will use these 4 keys
grouped_beers_df.shape
# Count of beers with the same name but different brewery, style or AVB%
grouped_beers_df.loc[ grouped_beers_df.duplicated( subset = [ 'beer_name' ], keep = False ) ].sort_values( by = 'beer_name'  ).shape
# Beers by brewery
grouped_beers_df[ 'brewery_name' ].value_counts( dropna = False ).head()
# Beers by style
grouped_beers_df[ 'beer_style' ].value_counts( dropna = False ).head()
plt.figure()
plt.hist( grouped_beers_df[ 'beer_abv' ], bins = 50 )
plt.xlabel( 'ABV%' )
plt.ylabel( 'Frecuency' )
plt.title( 'Histogram by ABV%' )
#plt.yscale( 'log' )
plt.show()
# Pearson correlation
sns.heatmap( beers_df[ [ 'review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv' ] ].corr(), center = 0,  vmin = -1, vmax = 1 )
plt.title( 'Pearson Correlation' )
# Spearman correlation
sns.heatmap( beers_df[ [ 'review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv' ] ].corr( method = 'spearman' ), center = 0,  vmin = -1, vmax = 1 )
plt.title( 'Spearman Correlation' )
# An new meassure is created by averaging review by factor
beers_df[ 'review_average' ] = round( ( ( beers_df[ 'review_overall' ] + beers_df[ 'review_aroma' ] + beers_df[ 'review_appearance' ] + beers_df[ 'review_palate' ] + beers_df[ 'review_taste' ] ) / 5 ) * 2 ) / 2
# Groupping by different review factors for visualization purposes
group_by_review_overall = beers_df[ 'review_overall' ].value_counts( dropna = False ).reset_index().rename( columns = { 'index' : 'review', 'review_overall' : 'overall' } ).sort_values( by = 'review' )
group_by_review_aroma = beers_df[ 'review_aroma' ].value_counts( dropna = False ).reset_index().rename( columns = { 'index' : 'review', 'review_aroma' : 'aroma' } ).sort_values( by = 'review' )
group_by_review_appearance = beers_df[ 'review_appearance' ].value_counts( dropna = False ).reset_index().rename( columns = { 'index' : 'review', 'review_appearance' : 'appearance' } ).sort_values( by = 'review' )
group_by_review_palate = beers_df[ 'review_palate' ].value_counts( dropna = False ).reset_index().rename( columns = { 'index' : 'review', 'review_palate' : 'palate' } ).sort_values( by = 'review' )
group_by_review_taste = beers_df[ 'review_taste' ].value_counts( dropna = False ).reset_index().rename( columns = { 'index' : 'review', 'review_taste' : 'taste' } ).sort_values( by = 'review' )
group_by_review_average = beers_df[ 'review_average' ].value_counts( dropna = False ).reset_index().rename( columns = { 'index' : 'review', 'review_average' : 'average' } ).sort_values( by = 'review' )

group_by_review_overall[ 'review' ] = group_by_review_overall[ 'review' ].astype( str )
group_by_review_aroma[ 'review' ] = group_by_review_aroma[ 'review' ].astype( str )
group_by_review_appearance[ 'review' ] = group_by_review_appearance[ 'review' ].astype( str )
group_by_review_palate[ 'review' ] = group_by_review_palate[ 'review' ].astype( str )
group_by_review_taste[ 'review' ] = group_by_review_taste[ 'review' ].astype( str )
group_by_review_average[ 'review' ] = group_by_review_average[ 'review' ].astype( str )

group_by_review = group_by_review_overall.merge( group_by_review_aroma, how = 'outer', on = [ 'review' ] )
group_by_review = group_by_review.merge( group_by_review_appearance, how = 'outer', on = [ 'review' ] )
group_by_review = group_by_review.merge( group_by_review_palate, how = 'outer', on = [ 'review' ] )
group_by_review = group_by_review.merge( group_by_review_taste, how = 'outer', on = [ 'review' ] )
group_by_review = group_by_review.merge( group_by_review_average, how = 'outer', on = [ 'review' ] )
group_by_review = group_by_review.fillna( 0 )
cm = plt.cm.get_cmap( 'tab10' ).colors
f, ( ( ax1, ax2, ax3 ), ( ax4, ax5, ax6 ) ) = plt.subplots( 2, 3, sharex = 'col', sharey = 'row', figsize = ( 17, 10 ) )
ax1.barh( group_by_review[ 'review' ], group_by_review[ 'overall' ], color = cm )
ax1.set_title( 'Review Overall' )
ax2.barh( group_by_review[ 'review' ], group_by_review[ 'aroma' ], color = cm )
ax2.set_title( 'Review Aroma' )
ax3.barh( group_by_review[ 'review' ], group_by_review[ 'appearance' ], color = cm )
ax3.set_title( 'Review Appearance' )
ax4.barh( group_by_review[ 'review' ], group_by_review[ 'palate' ], color = cm )
ax4.set_title( 'Review Palate' )
ax5.barh( group_by_review[ 'review' ], group_by_review[ 'taste' ], color = cm )
ax5.set_title( 'Review Taste' )
ax6.barh( group_by_review[ 'review' ], group_by_review[ 'average' ], color = cm )
ax6.set_title( 'Review Average' )
f.suptitle( 'Distribution of Reviews by Value' )
# This is a python implementarion of the Lower bound of Wilson score confidence interval for a Bernoulli parameter
# Implementation details: http://www.evanmiller.org/how-not-to-sort-by-average-rating.html?fbclid=IwAR2RNIB8geL9V0V9ereqidgRMasdytDOoqlGfCKWOcrRHKsUHFzMb7Xkemw

# pos: number of positive ratings
# n: total number of ratings
def ci_lower_bound( pos, n ):
    if n == 0:
        return 0
    z = 1.96 # For a IC of 0.95
    phat = 1.0 * pos / n
    return ( phat + ( z ** 2 ) / ( 2 * n ) - z * math.sqrt( ( phat * ( 1 - phat ) + ( z ** 2 ) / ( 4 * n ) ) / n ) ) / ( 1 + ( z ** 2 ) / n )
# Aggregation function for reviews
# Positive reviews are defined as a constant fraction of their real value
# Aggregation is performed using the Lower bound of Wilson score confidence interval for a Bernoulli parameter
def agg_reviews( reviews ):
    pos = 0
    for index, review in reviews[ reviews >= 3 ].iteritems():
        pos += review / 5
    #pos = ratings[ ratings >= 3 ].shape[ 0 ]
    return ci_lower_bound( pos, reviews.shape[ 0 ] ) * 5
# Grouping beers and aggregating reviews
grouped_beers_df = beers_df.groupby( [ 'beer_name', 'brewery_name', 'beer_style', 'beer_abv' ] ) \
    .agg( { 'review_overall' : agg_reviews, 'review_aroma' : agg_reviews, 'review_appearance' : agg_reviews, 'review_palate' : agg_reviews, 'review_taste' : agg_reviews, 'review_average' : agg_reviews, 'review_profilename' : 'count' } ).reset_index() \
    .rename( columns = { 'review_profilename' : 'number_of_reviews' } )
# Count of unique beers
grouped_beers_df[ 'beer_name' ].nunique()
# TOP 5 beers by number of reviews
grouped_beers_df.sort_values( by = 'number_of_reviews', ascending = False ).head()
# Beers with ABV% higher than 30
grouped_beers_df.loc[ grouped_beers_df[ 'beer_abv' ] > 30 ] \
    .sort_values( by = [ 'beer_abv' ], ascending = False )[ [ 'brewery_name', 'beer_name', 'beer_abv' ] ]
plt.figure( figsize = ( 7, 5 ) )
plt.scatter( grouped_beers_df[ 'number_of_reviews' ], grouped_beers_df[ 'review_average' ], marker ='.', alpha = .5 )
plt.xlabel( 'Number od Reviews' )
plt.ylabel( 'Review Average' )
plt.show()
grouped_beers_df.sort_values( by = 'review_average', ascending = False ).head( 3 )
# Defining the linear model
linear_model = LinearRegression( normalize = True )
# Training and generating predictions for the model
linear_model.fit( X = beers_df[ [ 'review_aroma', 'review_appearance', 'review_palate', 'review_taste' ] ], y = beers_df[ 'review_overall' ] )
preds = linear_model.predict( beers_df[ [ 'review_aroma', 'review_appearance', 'review_palate', 'review_taste' ] ] )
# Coeffifients for each feature (aroma, appearance, palate, taste)
linear_model.coef_
# Validating the error in the model
# Apparently, a linear model is enough to represent the phenomenon evidencing a global error of 0.42 when the range of possible values for the target is betwenn 0 and 5
# Most sophisitcated validation schemas must be developed
np.sqrt( mean_squared_error( beers_df[ 'review_overall' ], preds ) )
grouped_beers_df.sort_values( by = [ 'review_aroma', 'review_appearance' ], ascending = False ).head( 10 ) \
    [ 'beer_style' ].unique().tolist()
