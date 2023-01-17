!pip install surprise

!wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip

    

import zipfile



zip_ref = zipfile.ZipFile('BX-CSV-Dump.zip', 'r')

zip_ref.extractall("BX_CSV_Dump")

zip_ref.close()
import pandas as pd

import numpy as np



from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



from surprise.reader import Reader

from surprise.dataset import Dataset

from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor

from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore

from surprise import BaselineOnly, CoClustering

from surprise.model_selection import cross_validate, train_test_split

from surprise import accuracy
# reading BX-Users and removing bad lines from the dataframe

# we will just use the following 3 columns

user = pd.read_csv('BX_CSV_Dump/BX-Users.csv', sep=';', 

                   error_bad_lines=False, encoding="latin-1")

user.columns = ['userID', 'Location', 'Age']



# reading BX-Book-Ratings and removing bad lines from the dataframe

# we will just use the following 3 columns

rating = pd.read_csv('BX_CSV_Dump/BX-Book-Ratings.csv', sep=';', 

                     error_bad_lines=False, encoding="latin-1")

rating.columns = ['userID', 'ISBN', 'bookRating']



# performing inner join on userID

# dropping unnecessary columns

df = pd.merge(user, rating, on='userID', how='inner')

df.drop(['Location', 'Age'], axis=1, inplace=True)



df.head()
data = df['bookRating'].value_counts().sort_index(ascending=False)

trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               )

# Create layout

layout = dict(title = 'Distribution Of {} book-ratings'.format(df.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
# Number of ratings per book

data = df.groupby('ISBN')['bookRating'].count().clip(upper=50)



# Create trace

trace = go.Histogram(x = data.values,

                     name = 'Ratings',

                     xbins = dict(start = 0,

                                  end = 50,

                                  size = 2))

# Create layout

layout = go.Layout(title = 'Distribution Of Number of Ratings Per Book (Clipped at 100)',

                   xaxis = dict(title = 'Number of Ratings Per Book'),

                   yaxis = dict(title = 'Count'),

                   bargap = 0.2)



# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
df.groupby('ISBN')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
# Number of ratings per user

data = df.groupby('userID')['bookRating'].count().clip(upper=50)



# Create trace

trace = go.Histogram(x = data.values,

                     name = 'Ratings',

                     xbins = dict(start = 0,

                                  end = 50,

                                  size = 2))

# Create layout

layout = go.Layout(title = 'Distribution Of Number of Ratings Per User (Clipped at 50)',

                   xaxis = dict(title = 'Ratings Per User'),

                   yaxis = dict(title = 'Count'),

                   bargap = 0.2)



# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
df.groupby('userID')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
min_book_ratings = 50

filter_books = df['ISBN'].value_counts() > min_book_ratings

filter_books = filter_books[filter_books].index.tolist()



min_user_ratings = 50

filter_users = df['userID'].value_counts() > min_user_ratings

filter_users = filter_users[filter_users].index.tolist()



df_new = df[(df['ISBN'].isin(filter_books)) & (df['userID'].isin(filter_users))]

print('The original data frame shape:\t{}'.format(df.shape))

print('The new data frame shape:\t{}'.format(df_new.shape))
reader = Reader(rating_scale=(0, 9))

data = Dataset.load_from_df(df_new[['userID', 'ISBN', 'bookRating']], reader)
benchmark = []

# Iterate over all algorithms

for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), 

                  KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:

    # Perform cross validation

    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

    

    # Get results & append algorithm name

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    benchmark.append(tmp)

    

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    
print('Using ALS')

bsl_options = {'method': 'als',

               'n_epochs': 5,

               'reg_u': 12,

               'reg_i': 5

               }

algo = KNNBaseline(bsl_options=bsl_options)

cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)
trainset, testset = train_test_split(data, test_size=0.25)

algo = KNNBaseline(bsl_options=bsl_options)

predictions = algo.fit(trainset).test(testset)

accuracy.rmse(predictions)
def get_Iu(uid):

    """ return the number of items rated by given user

    args: 

      uid: the id of the user

    returns: 

      the number of items rated by the user

    """

    try:

        return len(trainset.ur[trainset.to_inner_uid(uid)])

    except ValueError: # user was not part of the trainset

        return 0

    

def get_Ui(iid):

    """ return number of users that have rated given item

    args:

      iid: the raw id of the item

    returns:

      the number of users that have rated the item.

    """

    try: 

        return len(trainset.ir[trainset.to_inner_iid(iid)])

    except ValueError:

        return 0

    

df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])

df['Iu'] = df.uid.apply(get_Iu)

df['Ui'] = df.iid.apply(get_Ui)

df['err'] = abs(df.est - df.rui)

best_predictions = df.sort_values(by='err')[:10]

worst_predictions = df.sort_values(by='err')[-10:]
best_predictions
worst_predictions
import matplotlib.pyplot as plt

%matplotlib notebook

df_new.loc[df_new['ISBN'] == '055358264X']['bookRating'].hist()

plt.xlabel('rating')

plt.ylabel('Number of ratings')

plt.title('Number of ratings book ISBN 055358264X has received')

plt.show();