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
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor,KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline,SVD,BaselineOnly,SVDpp,NMF,SlopeOne,CoClustering
from surprise.accuracy import rmse
from surprise import accuracy

users = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep='\";\"', names=['User-ID', 'Location', 'Age'], encoding='latin-1', skiprows=1)
books = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep='\";\"', names=['ISBN', 'Book-Title' ,'Book-Author','Year-Of-Publication', 'Publisher', 'Image-Url-S', 'Image-Url-M', 'Image-Url-L'], encoding='latin-1', skiprows=1)
ratings = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep='\";\"', names=['User-ID', 'ISBN', 'Book-Rating'], encoding='latin-1', skiprows=1)
users['User-ID'] = users['User-ID'].str.replace("\"","")
users['Location'] = users['Location'].str.replace("\";NULL","")
users['Age'] = users['Age'].fillna("0")
users['Age'] = users['Age'].str.replace("\"","")
books['ISBN'] = books['ISBN'].str.replace("\"","")
books['Book-Title'] = books['Book-Title'].str.replace("\"","")
ratings['User-ID'] = ratings['User-ID'].str.replace("\"","")
ratings['Book-Rating'] = ratings['Book-Rating'].str.replace("\"","").astype(int)
# Quality books having atleast 5 reviews

quality_ratings = ratings[ratings['Book-Rating']!=0]
quality_book = quality_ratings['ISBN'].value_counts().rename_axis('ISBN').reset_index(name = 'Count')
quality_book = quality_book[quality_book['Count']>5]['ISBN'].to_list()
quality_ratings = quality_ratings[quality_ratings['ISBN'].isin(quality_book)]
quality_ratings
# Quality Users making atleast 5 reviews

quality_user = quality_ratings['User-ID'].value_counts().rename_axis('User-ID').reset_index(name = 'Count')
quality_user = quality_user[quality_user['Count']>5]['User-ID'].to_list()
quality_ratings = quality_ratings[quality_ratings['User-ID'].isin(quality_user)]
quality_ratings
# Normalizing the Ratings

mean_rating_user = quality_ratings.groupby('User-ID')['Book-Rating'].mean().reset_index(name='Mean-Rating-User')
mean_data = pd.merge(quality_ratings, mean_rating_user, on='User-ID')
mean_data['Diff'] = mean_data['Book-Rating'] - mean_data['Mean-Rating-User']
mean_data['Square'] = (mean_data['Diff'])**2
norm_data = mean_data.groupby('User-ID')['Square'].sum().reset_index(name='Mean-Square')
norm_data['Root-Mean-Square'] = np.sqrt(norm_data['Mean-Square'])
mean_data = pd.merge(norm_data, mean_data, on='User-ID')
mean_data['Norm-Rating'] = mean_data['Diff']/(mean_data['Root-Mean-Square'])  
mean_data['Norm-Rating'] = mean_data['Norm-Rating'].fillna(0)
max_rating = mean_data.sort_values('Norm-Rating')['Norm-Rating'].to_list()[-1]
min_rating = mean_data.sort_values('Norm-Rating')['Norm-Rating'].to_list()[0]
mean_data['Norm-Rating'] = 5*(mean_data['Norm-Rating'] - min_rating)/(max_rating-min_rating)
mean_data['Norm-Rating'] = np.ceil(mean_data['Norm-Rating']).astype(int)
norm_ratings = mean_data[['User-ID','ISBN','Norm-Rating']]
mean_data.sort_values('Norm-Rating')

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(norm_ratings[['User-ID', 'ISBN', 'Norm-Rating']], reader)
benchmark = []
for algorithm in [SVD(), 
                  SVDpp(), 
                  SlopeOne(), 
                  NMF(), 
                  NormalPredictor(), 
                  KNNBaseline(), 
                  KNNBasic(), 
                  KNNWithMeans(),
                  KNNWithZScore(), 
                  BaselineOnly(),
                  CoClustering()]:
    
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
surprise_results
# Baseline

train_set, test_set = train_test_split(data, test_size=0.25)
algo = BaselineOnly(bsl_options={'method': 'sgd','learning_rate': .00005, 'n_epochs':30, 'reg':0.5})
fit = algo.fit(train_set)
pred = fit.test(test_set)
accuracy.rmse(pred)
# SVD 

algo = SVD(reg_bi = 0.5, lr_bi=0.005)
fit = algo.fit(train_set)
pred = fit.test(test_set)
accuracy.rmse(pred)
recommend = algo.trainset
users_norm = list(set(norm_ratings['User-ID'].to_list()))
books_norm = list(set(norm_ratings['ISBN'].to_list()))
norm_ratings['User-ID'].unique()
pred_users = [user for user in users_norm if recommend.knows_user(recommend.to_inner_uid(user))]
pred_books = []
for book in books_norm:
    try:
        if recommend.knows_item(recommend.to_inner_iid(book)):
            pred_books.append(book)
    except:
        pass
    
pred_users[:5]
def recommend_books(user_id, count):
    result=[]
    for b in pred_books:
        result.append([b,algo.predict(user_id,b,r_ui=4).est])
    recom = pd.DataFrame(result, columns=['ISBN','Rating'])
    merge = pd.merge(recom,books, on='ISBN' )
    return merge.sort_values('Rating', ascending=False).head(count)
recommendation = recommend_books('36938', 5)
scoring = recommendation.sort_values('Year-Of-Publication')
view = "".join(["<span><img src='"+a+"'></span>" for a in scoring['Image-Url-M'].to_list()])
scoring[['Book-Title']]
view