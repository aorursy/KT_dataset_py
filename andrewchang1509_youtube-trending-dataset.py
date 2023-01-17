import pandas as pd

import numpy as np

import json

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import math

from PIL import Image


original = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

new = original.copy()

with open("/kaggle/input/youtube-new/US_category_id.json","r") as category:

    category = json.load(category)

### Extract the category information from the JSON file

vid_cat = []

cat_id = []

for i in category['items']:

    vid_cat.append(i['snippet']['title'])

    cat_id.append(int(i['id']))

    

### Mapping the category_id

new.category_id = original.category_id.map(dict(zip(cat_id,vid_cat)))

new.category_id.isnull().sum() # we have no nan values.



### Prepare date type columns

new['trending_date'] = pd.to_datetime(new['trending_date'], format='%y.%d.%m')

new['publish_time'] = pd.to_datetime(new['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')



### Add column for publish time

new['publish_date'] = new['publish_time'].dt.date

new['publish_wd'] = new['publish_time'].dt.weekday

new['publish_hr'] = new['publish_time'].dt.hour

new['publish_time'] = new['publish_time'].dt.time



new.head()

#,'comments_disabled','ratings_disabled'

#For the purpose of this analysis, some columns are irrelevant, we should remove them

new = new.drop(['tags','video_error_or_removed','description'],axis = 1)

# Remove duplicates in the data

new = new.drop_duplicates(keep = 'first')
new.info() # We do not have any nan.
df = new[['category_id','views']].groupby('category_id').aggregate(np.sum).reset_index().sort_values(by ='views', ascending = False)

df.views = df.views/10**6

plt.figure(figsize = (20,10))

view_box = sns.barplot(y = 'category_id', x = 'views', data = df, orient = 'h')

plt.title('Barplot of number of views in each category (Unit: milliions)')

plt.ylabel('Category')

plt.xlabel('Views')

#view_box.set_xticklabels(view_box.get_xticklabels(), rotation=45, horizontalalignment='right')
print(new[['views','likes']].corr())

print(new[['views','dislikes']].corr())

data_bar = new['publish_wd'].map(dict(zip(range(7),

        ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']))).value_counts()

# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x=data_bar.index.values, y=data_bar,

            textposition='auto',

        )])

fig.update_layout(title = "Number of Videos Published in Each Weekday",yaxis=dict(

            title='Videos'))

fig.show()
lastdate = max(new.publish_date)-dt.timedelta(days = 7)

print(lastdate)
# Load data, define hover text and bubble size, only look at videos with 10M views or above.

data = new[['title','channel_title','category_id',

            'views','publish_wd','publish_hr','likes','dislikes','publish_date']].loc[new.views > 10**6].reset_index()

data.publish_wd = data.publish_wd.map(dict(zip(range(7),

        ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'])))

def bubble_plot(target, plot_title, target_title, data):

    hover_text = []

    bubble_size = []

    for index, row in data.iterrows():

        hover_text.append(('Title: {title}<br>'+

                      'Channel: {channel_title}<br>'+

                      'Category: {category_id}<br>'+

                      'Views: {views}<br>'+

                      'Likes: {likes} <br>'+

                       'Dislikes: {dislikes}<br>'

                      ).format(title=row['title'],

                                            channel_title=row['channel_title'],

                                            category_id=row['category_id'],

                                            views = row['views'],

                                            likes = row['likes'],

                                            dislikes = row['dislikes']))

        bubble_size.append(row[target]/row['views'])

    data['text'] = hover_text

    data['size'] = bubble_size

    fig = go.Figure()

    # Dictionary with dataframes for each weekday

    weekday = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']

    wd_data = {wd:data.query("publish_wd == '%s'" %wd)

                              for wd in weekday}

    # Create figure



    for key, values in wd_data.items():

        fig.add_trace(go.Scatter(

            x=values['views'], y=values[target]/values['views'],

            name=key, text=values['text'],

            marker_size=values['size'],

            ))

    # The following formula is recommended by https://plotly.com/python/bubble-charts/

    sizeref = 2.*max(data['size'])/(1000)





    # Tune marker appearance and layout

    fig.update_traces(mode='markers', marker=dict(sizemode='area',

                                              sizeref=sizeref, line_width=2))

    fig.update_layout(

        title=plot_title,

        xaxis=dict(

            title='Number of views in millions',

            gridcolor='white',

            type='log',

            gridwidth=2,

        ),

        yaxis=dict(

            title=target_title,

            gridcolor='white',

            gridwidth=2,

        ),

        paper_bgcolor='rgb(243, 243, 243)',

        plot_bgcolor='rgb(243, 243, 243)',

        legend= {'itemsizing': 'constant'}

    )

    

    fig.show()

pd.options.mode.chained_assignment = None 

bubble_plot('likes', "Like/View Ratio vs. Number of Views", "Like/View Ratio", data.loc[data.publish_date >= lastdate])
data.loc[data.publish_date >= lastdate,'publish_wd'].value_counts()
bubble_plot('dislikes', "Dislike/View Ratio vs. Number of Views", "Dislike/View Ratio",data)
# Create a dataframe for modeling

learn_data = new.loc[(new.comments_disabled) &

                   (~new.ratings_disabled)].copy()

# Create a column for number of days a video takes to get on the trending list

learn_data['day_to_trend'] = abs(np.subtract(learn_data.trending_date.dt.date,learn_data.publish_date,dtype=np.float32).apply(lambda x: x.days))

rel_vars = ['views','likes','dislikes','comment_count','publish_wd', 'publish_hr', 'day_to_trend','title']

learn_data = learn_data[rel_vars]

learn_data.reset_index(inplace=True)

learn_data.head()
from pandas.plotting import scatter_matrix

scatter_matrix(learn_data[['publish_wd', 'publish_hr', 'day_to_trend']])

plt.show()

plt.hist(learn_data['day_to_trend'])

plt.title("Histogram of Original Days to Trend ")

plt.show()



learn_data = learn_data.loc[learn_data.day_to_trend <= 14]

plt.hist(learn_data['day_to_trend'])

plt.title("Histogram of Days to Trend After Removing Values > 7")

plt.show()
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

learn_data.day_to_trend = learn_data.day_to_trend <=7
def rfr_model(X, y, my_param_grid = None ):

# Perform Grid-Search

    if my_param_grid is None:

        #the followings are hyperparameter to optimize: max depth of a tree and number of trees in the forest

        my_param_grid = {

                'max_depth': range(6,10),

                'n_estimators': range(155,170),

                }

    gsc = GridSearchCV(

        estimator=RandomForestClassifier(),    

        param_grid= my_param_grid,

        cv=5, scoring='accuracy', verbose=0, n_jobs=-1)

    

    grid_result = gsc.fit(X, y)



    return grid_result.best_params_,grid_result.best_score_
X = learn_data[['views','likes','dislikes','publish_wd', 'publish_hr']]

y = learn_data['day_to_trend']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=4, test_size = .3)
print(rfr_model(X_train,y_train)) # ({'max_depth': 9, 'n_estimators': 160}, 0.889456740442656)
from sklearn.metrics import classification_report

my_forest = RandomForestClassifier(max_depth = 9,n_estimators = 160,oob_score = True,warm_start = True )

my_forest.fit(X_train,y_train)

print(my_forest.oob_score_)# 0.8696883852691218

print(my_forest.score( X_test, y_test))# 0.9276315789473685

print(my_forest.feature_importances_)

#print(pd.crosstab(pd.Series(y_train, name='Actual'), pd.Series(my_forest.predict(X_train), name='Predicted')))

print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(my_forest.predict(X_test), name='Predicted')))

pred = my_forest.predict(X_test)

print(classification_report(y_test, pred))

import scikitplot as skplt

from sklearn.metrics import average_precision_score, plot_precision_recall_curve

prob = my_forest.predict_proba(X_test)

myplot = skplt.metrics.plot_roc(y_test, prob)

average_precision = average_precision_score(y_test, prob[:,1]) # prob[:,1] is the estimated probability of positive outcome

disp = plot_precision_recall_curve(my_forest, X_test,y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))

score = metrics.f1_score(np.array(y_test),pred)

print('The f1 score for this model is {}'.format(score))
from xgboost import XGBClassifier

parameters = [{'n_estimators': range(100,150,1)},

              {'learning_rate': np.arange(0.01,1.0, 0.01)}]

gbm=XGBClassifier(max_features='sqrt', subsample=0.8, random_state=10)

grid_search = GridSearchCV(estimator = gbm, param_grid = parameters, scoring='accuracy', cv = 4, n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)

#grid_search.cv_results_

#grid_search.best_params_, grid_search.best_score_

grid_search.best_estimator_
gbm = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.24, max_delta_step=0, max_depth=6,

              max_features='sqrt', min_child_weight=1, missing=None,

              monotone_constraints=None, n_estimators=100, n_jobs=0,

              num_parallel_tree=1, objective='binary:logistic', random_state=10,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.8,

              tree_method=None, validate_parameters=False, verbosity=None)

gbm.fit(X_train,y_train)

y_pred = gbm.predict(X_test)

print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

print(classification_report(y_test, y_pred))
prob = gbm.predict_proba(X_test)

myplot = skplt.metrics.plot_roc(y_test, prob)

average_precision = average_precision_score(y_test, prob[:,1])

disp = plot_precision_recall_curve(gbm, X_test,y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

title = learn_data.loc[learn_data.day_to_trend <= 7].title.copy()

title = " ".join(x for x in title)

stopwords = set(STOPWORDS)

stopwords.update(["Official", "Trailer"])



mask = np.array(Image.open("/kaggle/input/logoyou/logo.jpg"))

wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=500, mask=mask).generate(title)

image_colors = ImageColorGenerator(mask)

plt.figure(figsize=[10,14])

plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")

plt.show()

wordcloud = WordCloud(max_words=100, background_color="white", stopwords = stopwords, colormap = 'Reds').generate(title)

plt.figure(figsize=[10,14])

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()