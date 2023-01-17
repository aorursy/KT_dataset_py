import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import pandas_profiling 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_color_codes()

sns.set(style="whitegrid")

%matplotlib inline

from scipy.stats import zscore

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from math import sqrt



import requests

from io import StringIO



#setting up for customized printing

from IPython.display import Markdown, display

from IPython.display import HTML

def printmd(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))

    

#function to display dataframes side by side    

from IPython.display import display_html

def display_side_by_side(args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)
def distplot(figRows,figCols,xSize, ySize, data, features, colors, kde=True, bins=None):

    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    

    features = np.array(features).reshape(figRows, figCols)

    colors = np.array(colors).reshape(figRows, figCols)

    

    for row in range(figRows):

        for col in range(figCols):

            if (figRows == 1 and figCols == 1) :

                axesplt = axes

            elif (figRows == 1 and figCols > 1) :

                axesplt = axes[col]

            elif (figRows > 1 and figCols == 1) :

                axesplt = axes[row]

            else:

                axesplt = axes[row][col]

            plot = sns.distplot(data[features[row][col]], color=colors[row][col], bins=bins, ax=axesplt, kde=kde, hist_kws={"edgecolor":"k"})

            plot.set_xlabel(features[row][col],fontsize=20)
def scatterplot(rowFeature, colFeature, data):

    f, axes = plt.subplots(1, 1, figsize=(10, 8))

        

    plot=sns.scatterplot(x=rowFeature, y=colFeature, data=data, ax=axes)

    plot.set_xlabel(rowFeature,fontsize=20)

    plot.set_ylabel(colFeature,fontsize=20)            
# orig_url='https://drive.google.com/file/d/1ClBptsK3V5KgKXtK2GSRzFNAW7GnTPDW/view?usp=sharing'

# file_id = orig_url.split('/')[-2]

# file_id
# dwn_url='http://drive.google.com/uc?export=download&confirm=pAkV&id=' + file_id

# url = requests.get(dwn_url)

# url.text
electronics = pd.read_csv('../input/ratings_Electronics.csv', names=['userId', 'productId', 'Rating','timestamp'], header=None)
electronics.head()
electronics.drop('timestamp', axis=1, inplace=True)
electronics_groupby_users_Ratings = electronics.groupby('userId')['Rating']

electronics_groupby_users_Ratings = pd.DataFrame(electronics_groupby_users_Ratings.count())
user_list_min50_ratings = electronics_groupby_users_Ratings[electronics_groupby_users_Ratings['Rating'] >= 50].index

electronics =  electronics[electronics['userId'].isin(user_list_min50_ratings)]
print('The total number of rows :', electronics.shape[0])

print('The total number of columns :', electronics.shape[1])
electronics.info()
display(electronics.isna().sum().sort_values())

print('===================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"No Missing"** values in the data', color="blue")
display(electronics[['Rating']].describe().transpose())
print('Total unique users in the dataset', electronics['userId'].nunique())

print('Total unique products in the dataset', electronics['productId'].nunique())
pal = sns.color_palette(palette='Set1', n_colors=16)
distplot(1, 1, 10, 7, data=electronics, features=['Rating'], colors=['blue'])
distplot(1, 1, 10, 7, data=electronics, features=['Rating'], colors=['red'], kde=False)
electronics_groupby_products_Ratings = electronics.groupby('productId')['Rating']
electronics_groupby_products_Ratings.count().clip(upper=30).unique()
ratings_products = pd.DataFrame(electronics_groupby_products_Ratings.count().clip(upper=30))

ratings_products.rename(columns={"Rating": "Rating_Count"}, inplace=True)
distplot(1, 1, 10, 7, data=ratings_products, features=['Rating_Count'], colors=['green'], kde=False)
electronics_groupby_users_Ratings = electronics.groupby('userId')['Rating']
electronics_groupby_users_Ratings.count().clip(lower=50).unique()
rating_users = pd.DataFrame(electronics_groupby_users_Ratings.count().clip(lower=50, upper=300))

rating_users.rename(columns={"Rating": "Rating_Count"}, inplace=True)
distplot(1, 1, 10, 7, data=rating_users, features=['Rating_Count'], colors=['orange'], kde=False, bins=50)
ratings = pd.DataFrame(electronics.groupby('productId')['Rating'].mean())

ratings.rename(columns={"Rating": "Rating_Mean"}, inplace=True)
distplot(1, 1, 10, 7, data=ratings, features=['Rating_Mean'], colors=['brown'], kde=False, bins=50)
ratings['Rating_Count'] = electronics.groupby('productId')['Rating'].count()
scatterplot('Rating_Mean', 'Rating_Count', data=ratings)
ratings = pd.DataFrame(electronics.groupby('userId')['Rating'].mean())

ratings.rename(columns={"Rating": "Rating_Mean"}, inplace=True)

distplot(1, 1, 10, 7, data=ratings, features=['Rating_Mean'], colors=['brown'], kde=False, bins=50)
ratings['Rating_Count'] = electronics.groupby('userId')['Rating'].count()

scatterplot('Rating_Mean', 'Rating_Count', data=ratings)
train_data, test_data = train_test_split(electronics, test_size =.30, random_state=10)

printmd('**Training and Testing Set Distribution**', color='brown')



print(f'Training set has {train_data.shape[0]} rows and {train_data.shape[1]} columns')

print(f'Testing set has {test_data.shape[0]} rows and {test_data.shape[1]} columns')

class popularity_based_recommender_model():

    def __init__(self, train_data, test_data, user_id, item_id):

        self.train_data = train_data

        self.test_data = test_data

        self.user_id = user_id

        self.item_id = item_id

        self.popularity_recommendations = None

        

    #Create the popularity based recommender system model

    def fit(self):

        #Get a count of user_ids for each unique product as recommendation score

        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()

        train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)

    

        #Sort the products based upon recommendation score

        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])

    

        #Generate a recommendation rank based upon score

        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        

        #Get the top 10 recommendations

        self.popularity_recommendations = train_data_sort.head(20)



    #Use the popularity based recommender system model to make recommendations

    def recommend(self, user_id, n=5):    

        user_recommendations = self.popularity_recommendations

        

        #Filter products that are not rated by the user

        products_already_rated_by_user = self.train_data[self.train_data[self.user_id] == user_id][self.item_id]        

        user_recommendations = user_recommendations[~user_recommendations[self.item_id].isin(products_already_rated_by_user)]

        

        #Add user_id column for which the recommendations are being generated

        user_recommendations['user_id'] = user_id

    

        #Bring user_id column to the front

        cols = user_recommendations.columns.tolist()

        cols = cols[-1:] + cols[:-1]

        user_recommendations = user_recommendations[cols].head(n)     

        self.plot(user_recommendations)

        return user_recommendations

    

    def plot(self, user_recommendations):

        f, axes = plt.subplots(1, 2, figsize=(20, 8))

        cplot1 = sns.barplot(x='Rank', y='score', data=user_recommendations, hue='Rank', ax=axes[0])

        cplot1.set_xlabel('Rank',fontsize=20)

        cplot1.set_ylabel('score',fontsize=20)            

        cplot2 = sns.pointplot(x='Rank', y='score', data=user_recommendations, hue='Rank', ax=axes[1])

        cplot2.set_xlabel('Rank',fontsize=20)

        cplot2.set_ylabel('score',fontsize=20)

        

    def predict_evaluate(self):        

        ratings = pd.DataFrame(self.train_data.groupby(self.item_id)['Rating'].mean())

        

        pred_ratings = [];            

        for data in self.test_data.values:

            if(data[1] in (ratings.index)):

                pred_ratings.append(ratings.loc[data[1]])

            else:

                pred_ratings.append(0)

        

        mse = mean_squared_error(self.test_data['Rating'], pred_ratings)

        rmse = sqrt(mse)

        return rmse

        

        
pr = popularity_based_recommender_model(train_data=train_data, test_data=test_data, user_id='userId', item_id='productId')
pr.fit()
result_pop_user1 = pr.recommend('ANTN61S4L7WG9')

result_pop_user1
result_pop_user2 = pr.recommend('AYNAH993VDECT')

result_pop_user2
result_pop_user3 = pr.recommend('A18YMFFJW974QS')

result_pop_user3
display_side_by_side([result_pop_user1, result_pop_user2, result_pop_user3])
pr.predict_evaluate()
from surprise import accuracy

from surprise.model_selection.validation import cross_validate

from surprise.dataset import Dataset

from surprise.reader import Reader

from surprise import SVD

from surprise import KNNBasic

from surprise import KNNWithMeans

reader = Reader()

surprise_data = Dataset.load_from_df(electronics, reader)
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(surprise_data, test_size=.3, random_state=10)
from collections import defaultdict



def get_top_n(predictions, n=10):

    # First map the predictions to each user.

    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:

        top_n[uid].append((iid, est))



    # Then sort the predictions for each user and retrieve the k highest ones.

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key=lambda x: x[1], reverse=True)

        top_n[uid] = user_ratings[:n]



    return top_n
class collab_filtering_based_recommender_model():

    def __init__(self, model, trainset, testset, data):

        self.model = model

        self.trainset = trainset

        self.testset = testset

        self.data = data

        self.pred_test = None

        self.recommendations = None

        self.top_n = None

        self.recommenddf = None



    def fit_and_predict(self):        

        printmd('**Fitting the train data...**', color='brown')

        self.model.fit(self.trainset)       



        printmd('**Predicting the test data...**', color='brown')

        self.pred_test = self.model.test(self.testset)        

        rmse = round(accuracy.rmse(self.pred_test), 3)

        printmd('**RMSE for the predicted result is ' + str(rmse) + '**', color='brown')   

        

        self.top_n = get_top_n(self.pred_test)

        self.recommenddf = pd.DataFrame(columns=['userId', 'productId', 'Rating'])

        for item in self.top_n:

            subdf = pd.DataFrame(self.top_n[item], columns=['productId', 'Rating'])

            subdf['userId'] = item

            cols = subdf.columns.tolist()

            cols = cols[-1:] + cols[:-1]

            subdf = subdf[cols]        

            self.recommenddf = pd.concat([self.recommenddf, subdf], axis = 0)        

        return rmse

        

    def cross_validate(self):

        printmd('**Cross Validating the data...**', color='brown')

        cv_result = cross_validate(self.model, self.data, n_jobs=-1)

        cv_result = round(cv_result['test_rmse'].mean(),3)

        printmd('**Mean CV RMSE is ' + str(cv_result)  + '**', color='brown')

        return cv_result



    def recommend(self, user_id, n=5):

        printmd('**Recommending top ' + str(n)+ ' products for userid : ' + user_id + ' ...**', color='brown')

        

        #df = pd.DataFrame(self.top_n[user_id], columns=['productId', 'Rating'])

        #df['UserId'] = user_id

        #cols = df.columns.tolist()

        #cols = cols[-1:] + cols[:-1]

        #df = df[cols].head(n)

        df = self.recommenddf[self.recommenddf['userId'] == user_id].head(n)

        display(df)

        return df

        
from surprise.model_selection import RandomizedSearchCV



def find_best_model(model, parameters,data):

    clf = RandomizedSearchCV(model, parameters, n_jobs=-1, measures=['rmse'])

    clf.fit(data)             

    print(clf.best_score)

    print(clf.best_params)

    print(clf.best_estimator)

    return clf
sim_options = {

    "name": ["msd", "cosine", "pearson", "pearson_baseline"],

    "min_support": [3, 4, 5],

    "user_based": [True],

}

params = { 'k': range(30,50,1), 'sim_options': sim_options}

clf = find_best_model(KNNWithMeans, params, surprise_data)
knnwithmeans = clf.best_estimator['rmse']

col_fil_knnwithmeans = collab_filtering_based_recommender_model(knnwithmeans, trainset, testset, surprise_data)
knnwithmeans_rmse = col_fil_knnwithmeans.fit_and_predict()
knnwithmeans_cv_rmse = col_fil_knnwithmeans.cross_validate()
result_knn_user1 = col_fil_knnwithmeans.recommend(user_id='ANTN61S4L7WG9', n=5)

result_knn_user2 = col_fil_knnwithmeans.recommend(user_id='AYNAH993VDECT', n=5)

result_knn_user3 = col_fil_knnwithmeans.recommend(user_id='A18YMFFJW974QS', n=5)
params= {

    "n_epochs": [5, 10, 15, 20],

    "lr_all": [0.002, 0.005],

    "reg_all": [0.4, 0.6]

}

clf = find_best_model(SVD, params, surprise_data)
svd = clf.best_estimator['rmse']

col_fil_svd = collab_filtering_based_recommender_model(svd, trainset, testset, surprise_data)
svd_rmse = col_fil_svd.fit_and_predict()
svd_cv_rmse = col_fil_svd.cross_validate()
result_svd_user1 = col_fil_svd.recommend(user_id='ANTN61S4L7WG9', n=5)

result_svd_user2 = col_fil_svd.recommend(user_id='AYNAH993VDECT', n=5)

result_svd_user3 = col_fil_svd.recommend(user_id='A18YMFFJW974QS', n=5)
from surprise import NMF

from surprise import KNNBaseline

from surprise import KNNBasic

from surprise import KNNWithZScore

from surprise import BaselineOnly

from surprise import CoClustering

benchmark = []

# Iterate over all algorithms

for algorithm in [NMF(), KNNBaseline(), KNNBasic(), KNNWithZScore(), BaselineOnly(), CoClustering()]:

    # Perform cross validation

    results = cross_validate(algorithm, surprise_data, measures=['RMSE'], cv=5, verbose=False)

    

    # Get results & append algorithm name

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    benchmark.append(tmp)

    

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 
display_side_by_side([result_pop_user1, result_pop_user2, result_pop_user3])
display_side_by_side([result_knn_user1, result_knn_user2, result_knn_user3])
display_side_by_side([result_svd_user1, result_svd_user2, result_svd_user3])