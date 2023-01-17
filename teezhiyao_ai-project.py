### Importing Relevant Library and reading dataset

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")



### Preliminary Data Cleaning to remove data which are not relevant to us

USvideos = USvideos[USvideos['video_error_or_removed'] == False] ###Remove video with error

USvideos = USvideos[USvideos['comments_disabled'] == False] ###Remove video that does not allow comment/ratings(likes/dislikes)

USvideos = USvideos[USvideos['ratings_disabled'] == False]



USvideos_id_date = USvideos.loc[:,['video_id','tags']].drop_duplicates()

### Creating a new column which tells us the number of tags that the video has. 

USvideos_id_date['tag_counts'] = USvideos_id_date['tags'].str.split("|").apply(len)

USvideos_id_date['tag_counts'].plot.box()

plt.show()

USvideos_id_date['tag_counts'].hist(bins=80)

plt.xlabel('tag count')

plt.ylabel('number of videos')

plt.show()  

###Min-Max normalization

USvideos['views_mmnorm'] = (USvideos['views'] - USvideos['views'].min()) / (USvideos['views'].max() - USvideos['views'].min())

USvideos['likes_mmnorm'] = (USvideos['likes'] - USvideos['likes'].min()) / (USvideos['likes'].max() - USvideos['likes'].min())

USvideos['dislikes_mmnorm'] = (USvideos['dislikes'] - USvideos['dislikes'].min()) / (USvideos['dislikes'].max() - USvideos['dislikes'].min())

USvideos['comment_count_mmnorm'] = (USvideos['comment_count'] - USvideos['comment_count'].min()) / (USvideos['comment_count'].max() - USvideos['comment_count'].min())

mmnorm_col = ['video_id', 'views_mmnorm','likes_mmnorm','dislikes_mmnorm','comment_count_mmnorm','tag_counts']



###Mean normalization

USvideos['views_mnorm'] = (USvideos['views'] - USvideos['views'].min()) / (USvideos['views'].max() - USvideos['views'].min())

USvideos['likes_mnorm'] = (USvideos['likes'] - USvideos['likes'].min()) / (USvideos['likes'].max() - USvideos['likes'].min())

USvideos['dislikes_mnorm'] = (USvideos['dislikes'] - USvideos['dislikes'].min()) / (USvideos['dislikes'].max() - USvideos['dislikes'].min())

USvideos['comment_count_mnorm'] = (USvideos['comment_count'] - USvideos['comment_count'].min()) / (USvideos['comment_count'].max() - USvideos['comment_count'].min())

mnorm_col = ['video_id', 'views_mnorm','likes_mnorm','dislikes_mnorm','comment_count_mnorm','tag_counts']



###Standardizing data

USvideos['views_std'] = (USvideos['views'] - USvideos['views'].mean()) / (USvideos['views'].std())

USvideos['likes_std'] = (USvideos['likes'] - USvideos['likes'].mean()) / (USvideos['likes'].std())

USvideos['dislikes_std'] = (USvideos['dislikes'] - USvideos['dislikes'].mean()) / (USvideos['dislikes'].std())

USvideos['comment_count_std'] = (USvideos['comment_count'] - USvideos['comment_count'].mean()) / (USvideos['comment_count'].std())

std_col = ['video_id', 'views_std','likes_std','dislikes_std','comment_count_std','tag_counts']



USvideos['tag_counts'] = USvideos['tags'].str.split("|").apply(len)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures



df_mmnorm = USvideos[mmnorm_col].groupby(['video_id'])

df_mnorm = USvideos[mnorm_col].groupby(['video_id'])

df_std = USvideos[std_col].groupby(['video_id'])

###Testing with Linear Regression

def lin_reg_helper(df,norm_name):

    data = []

    for vid_id,rows in df:

    #     print("vid_id: " + vid_id)

        rel_rows = rows.drop(['video_id'],axis=1).drop(['tag_counts'],axis=1)

        days_trended = len(rel_rows)



        if days_trended >= 5:

            Xs = rel_rows

            y = rows['tag_counts'].values.reshape(-1,1)

            lin_reg = LinearRegression()

            lin_reg.fit(Xs, y)

    #         print('Intercept: \n', lin_reg.intercept_)

    #         print('Coefficients: \n', lin_reg.coef_)

            view_coef = lin_reg.coef_[0][0]

            likes_coef = lin_reg.coef_[0][1]

            dislikes_coef = lin_reg.coef_[0][2]

            comment_count_coef = lin_reg.coef_[0][3]

            data.append([vid_id, days_trended, view_coef,likes_coef,dislikes_coef,comment_count_coef])

    data_df = pd.DataFrame(data, columns = ['Video ID', 'Trending Duration','view_coef','likes_coef','dislikes_coef','comment_count_coef']) 



    fig = plt.figure()

    plt.title(norm_name + ' Linear')

    # Divide the figure into a 2x1 grid, and give me the first section

    ax1 = fig.add_subplot(221)

    ax2 = fig.add_subplot(222)

    ax3 = fig.add_subplot(223)

    ax4 = fig.add_subplot(224)

    data_df.plot.scatter(x ='Trending Duration', y= 'view_coef', c= 'blue', ax=ax1)

    data_df.plot.scatter(x ='Trending Duration', y= 'likes_coef', c= 'orange', ax=ax2)

    data_df.plot.scatter(x ='Trending Duration', y= 'dislikes_coef', c= 'orange', ax=ax3)

    data_df.plot.scatter(x ='Trending Duration', y= 'comment_count_coef', c= 'orange', ax=ax4)

    fig.savefig(norm_name + '_lin_reg')

    

# df_lin_mmnorm = lin_reg_helper(df_mmnorm,'lin_mmnorm')

# df_lin_mnorm = lin_reg_helper(df_mnorm,'lin_mnorm')

# df_lin_std = lin_reg_helper(df_std,'lin_std')
from sklearn.preprocessing import PolynomialFeatures



###Testing with Polynomial Regression

def poly_reg_helper(df,norm_name,deg):

    data = []

    for vid_id,rows in df:

    #     print("vid_id: " + vid_id)

        rel_rows = rows.drop(['video_id'],axis=1).drop(['tag_counts'],axis=1)

        days_trended = len(rel_rows)



        if days_trended >= 5:

            Xs = rel_rows

            y = rows['tag_counts'].values.reshape(-1,1)

            

            polynomial_features= PolynomialFeatures(degree=deg)

            x_poly = polynomial_features.fit_transform(Xs)



            lin_reg = LinearRegression()

            lin_reg.fit(x_poly, y)

            view_coef = lin_reg.coef_[0][0]

            likes_coef = lin_reg.coef_[0][1]

            dislikes_coef = lin_reg.coef_[0][2]

            comment_count_coef = lin_reg.coef_[0][3]

            data.append([vid_id, days_trended, view_coef,likes_coef,dislikes_coef,comment_count_coef])

    data_df = pd.DataFrame(data, columns = ['Video ID', 'Trending Duration','view_coef','likes_coef','dislikes_coef','comment_count_coef']) 



    fig = plt.figure()

    plt.title(norm_name + ' Polynomial - ' + str(deg))

    # Divide the figure into a 2x1 grid, and give me the first section

    ax1 = fig.add_subplot(221)

    ax2 = fig.add_subplot(222)

    ax3 = fig.add_subplot(223)

    ax4 = fig.add_subplot(224)

    data_df.plot.scatter(x ='Trending Duration', y= 'view_coef', c= 'blue', ax=ax1)

    data_df.plot.scatter(x ='Trending Duration', y= 'likes_coef', c= 'orange', ax=ax2)

    data_df.plot.scatter(x ='Trending Duration', y= 'dislikes_coef', c= 'orange', ax=ax3)

    data_df.plot.scatter(x ='Trending Duration', y= 'comment_count_coef', c= 'orange', ax=ax4)

    fig.savefig(norm_name + '_poly_reg' + str(deg))

    return fig

    

# df_lin_mmnorm = poly_reg_helper(df_mmnorm,'mmnorm',2)

# df_lin_mnorm = poly_reg_helper(df_mnorm,'mnorm',2)

# df_lin_std = poly_reg_helper(df_std,'std',2)

# df_lin_mmnorm = poly_reg_helper(df_mmnorm,'mmnorm',3)

# df_lin_mnorm = poly_reg_helper(df_mnorm,'mnorm',3)

# df_lin_std = poly_reg_helper(df_std,'std',3)

# df_lin_mmnorm = poly_reg_helper(df_mmnorm,'mmnorm',4)

# df_lin_mnorm = poly_reg_helper(df_mnorm,'mnorm',4)

# df_lin_std = poly_reg_helper(df_std,'std',4)
import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings(action='ignore',category=DeprecationWarning)

warnings.filterwarnings(action='ignore',category=FutureWarning)



###Testing with Ridge Regression

def ridge_helper(df,norm_name):

    data = []

    for vid_id,rows in df:

        rel_rows = rows.drop(['video_id'],axis=1).drop(['tag_counts'],axis=1)

        days_trended = len(rows)

        if days_trended >= 5:

            Xs = rel_rows

            y = rows['tag_counts'].values.reshape(-1,1)

            alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

            ridge = Ridge()

            parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3]}

            ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=2)

            ridge_regressor.fit(Xs, y)

            print(ridge_regressor.coef_)

            data.append([vid_id, days_trended, ridge_regressor.best_params_['alpha'], ridge_regressor.best_score_])

    data_df = pd.DataFrame(data, columns = ['Video ID', 'Trending Duration','intercept','coef']) 



    fig = plt.figure()

    plt.title(norm_name + ' Ridge Regression')

    # Divide the figure into a 2x1 grid, and give me the first section

    ax1 = fig.add_subplot(221)

    ax2 = fig.add_subplot(222)

    data_df.plot.scatter(x = 'Trending Duration', y = 'intercept', c= 'blue', ax = ax1)

    data_df.plot.scatter(x = 'Trending Duration', y = 'coef', c= 'orange',ax = ax2)

    fig.savefig(norm_name + '_ridge_reg')



# df_rid_mmnorm = ridge_helper(df_mmnorm,'lin_mmnorm')

# df_rid_mnorm = ridge_helper(df_mnorm,'lin_mnorm')

# df_rid_std = ridge_helper(df_std,'lin_std')
#In the code below, we create a regression model based on Tags & Trending Duration



USvideos['tag_counts'] = USvideos['tags'].str.split("|").apply(len)

rel_col = ['video_id','views','tag_counts','tags','title']

df = USvideos[rel_col].groupby(['video_id'])

data = [] 



# Editing the data such that we are able to loop through each tag in each column

for vid_id,rows in df:

    days_trended = len(rows)

    data.append([vid_id,rows['title'].values[0], days_trended,rows['tags'].values[0].split("|"),rows['tag_counts'].values[0]])



# Creating new dataframe with the following columns

df = pd.DataFrame(data, columns = ['Video ID', 'Title','Trending Duration','Tags','Tag Counts'])



dic_freq = {}

dic_trendsum = {}

dic_trendsum_weighted = {}

# Looping through the rows

for index, row in df.iterrows():

    # Looping through each tag

    for item in row['Tags']:

        if item in dic_freq:

            dic_freq[item] = dic_freq[item] + 1

        else:

            dic_freq[item] = 1



        if item in dic_trendsum:

            dic_trendsum[item] = dic_trendsum[item] + (row['Trending Duration'] / row['Tag Counts'])

#             dic_trendsum[item] = dic_trendsum[item] + (row['Trending Duration'])

        else:

#             dic_trendsum[item] = (row['Trending Duration'] / row['Tag Counts'])

            dic_trendsum[item] = (row['Trending Duration'])

    

        if item in dic_trendsum_weighted:

            dic_trendsum_weighted[item] = dic_trendsum[item] + (row['Trending Duration'])

        else:

            dic_trendsum_weighted[item] = (row['Trending Duration'] / row['Tag Counts'])

            

tag_df = pd.DataFrame([dic_freq]).T

tag_df.columns = ['word_freq']



tag_trendsum = pd.DataFrame([dic_trendsum]).T

tag_trendsum.columns = ['trendsum']

tag_df = tag_df.join(tag_trendsum)

tag_df['trendmean'] = tag_df['trendsum'] / tag_df['word_freq']



tag_trendsum_weighted = pd.DataFrame([dic_trendsum_weighted]).T

tag_trendsum_weighted.columns = ['trendsum_weight']

tag_df = tag_df.join(tag_trendsum_weighted)

tag_df['trendmean_weight'] = tag_df['trendsum_weight'] / tag_df['word_freq']

tag_df = tag_df[tag_df['word_freq'] > 5]

# # tag_df = tag_df.sort_values(by=['trendsum'],ascending=False)

tag_df.plot.scatter(x ='word_freq', y= 'trendmean', c= 'blue')

# ax.set_xlim(0,50)

plt.show()
def cal_tag_val(tag_lst,metric):

    dicc = tag_df[metric].T.to_dict()

    return sum(list(filter(None,map(dicc.get,tag_lst))))

    #     return sum(list(map(dicc,tag_lst)))



df['tag_trend_val1'] = df['Tags'].apply(cal_tag_val, metric = 'trendsum')

df['tag_trend_val2'] = df['Tags'].apply(cal_tag_val, metric = 'trendmean')

df['tag_trend_val3'] = df['Tags'].apply(cal_tag_val, metric = 'trendsum_weight')

df['tag_trend_val4'] = df['Tags'].apply(cal_tag_val, metric = 'trendmean_weight')

print(df.sort_values(by=['tag_trend_val1'],ascending=False).head(10))

print(df.sort_values(by=['tag_trend_val2'],ascending=False).head(10))

print(df.sort_values(by=['tag_trend_val3'],ascending=False).head(10))

print(df.sort_values(by=['tag_trend_val4'],ascending=False).head(10))
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



df = pd.DataFrame(df)

def metric_testing(lst,dura):

    score_lst = []

    for metric in lst:

        X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(df[metric]), pd.DataFrame(df['Trending Duration'] >= dura), test_size = 0.2)

        knn = KNeighborsClassifier(n_neighbors=5)

        knn.fit(X_train,y_train)

        # Print the accuracy

        score_lst.append(knn.score(X_test, y_test))

    return score_lst



print(metric_testing(['tag_trend_val1','tag_trend_val2','tag_trend_val3','tag_trend_val4'],2))

print(metric_testing(['tag_trend_val1','tag_trend_val2','tag_trend_val3','tag_trend_val4'],3))

print(metric_testing(['tag_trend_val1','tag_trend_val2','tag_trend_val3','tag_trend_val4'],4))

print(metric_testing(['tag_trend_val1','tag_trend_val2','tag_trend_val3','tag_trend_val4'],5))

print(metric_testing(['tag_trend_val1','tag_trend_val2','tag_trend_val3','tag_trend_val4'],6))

print(metric_testing(['tag_trend_val1','tag_trend_val2','tag_trend_val3','tag_trend_val4'],7))

print(metric_testing(['tag_trend_val1','tag_trend_val2','tag_trend_val3','tag_trend_val4'],8))