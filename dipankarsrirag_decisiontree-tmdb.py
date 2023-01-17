# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ast 

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')

df.head()
def binaryToDecimal(n): 

    num = n; 

    dec_value = 0; 

      

    # Initializing base  

    # value to 1, i.e 2 ^ 0 

    base = 1; 

      

    temp = num; 

    while(temp): 

        last_digit = temp % 10; 

        temp = int(temp / 10); 

          

        dec_value += last_digit * base; 

        base = base * 2; 

    return dec_value; 







def oneHot(ser, train = True):

    li = []

    leng = len(ser)

    for i in ser:

        if type(i) == str:

            li += [i]

            continue

        li += i

    col = list(pd.Series(li).unique())

    encoded = dict([(x, leng*[0]) for x in col])

    

    for i in range(len(ser)):

        if type(ser[i]) == str:

            ser[i] = [ser[i]]

        try:

            for j in ser[i]:

                if j in col:

                    if train:

                        encoded[j][i] = 1

                    else:

                        encoded[j][i + 3000] = 1

        except:

            print('i : ', i)

            print('j : ', j)

    

    return pd.DataFrame(encoded)



def listToNum(data, col, train = True):

    x = oneHot(list(data[col]), train)

    li = []

    for i in range(len(x)):

        li.append(int(''.join(list(map(str, list(x.iloc[i, :]))))))



    li = pd.Series(li).apply(lambda x : binaryToDecimal(x))

    data[col] = li

    return data[col]



def getData(li, string, priority = False):

    

    coll = []

    for text in li:

        if type(text) == float:

            continue

        samp = text[1:-1].split('}')

        for i in range(len(samp)):

            samp[i] += '}'

            if i > 0:

                samp[i] = samp[i][2:]

        samp = samp[:-1]

        

        for i in samp:

            dic = ast.literal_eval(i) 

            coll.append(dic[string])

        

        if len(coll) > 0:

            if priority:

                return coll[0]

            return coll

        return ['None']

        

def getYear(string):

    if type(string) == float:

       return None 

    num = int(string[-2:])

    if num > 18:

        return 1900 + num

    return 2000+num



def preProcess(df, train = True):

    data = df.copy()

    columns = ['belongs_to_collection', 'budget', 'genres', 

           'popularity', 'production_companies', 'release_date', 

           'runtime', 'original_language']

    if train:

        columns.append('revenue')

    data = data[columns]



    data['collection'] = data.belongs_to_collection.apply(lambda x : getData([x], 'name', priority = True))

    data['genres'] = data.genres.apply(lambda x : getData([x], 'name', priority = True))

    data['production_companies'] = data.production_companies.apply(lambda x : getData([x], 'name', priority = True))



    data.drop('belongs_to_collection', axis = 1, inplace = True)

    

    data.genres.fillna('Not Available', inplace = True)

    #data.genres = listToNum(data, 'genres', train)

    

    data.production_companies.fillna('Not Available', inplace = True)

    

    data.collection.fillna('Not Available', inplace = True)

    

    data.budget = data.budget.apply(lambda x : float(x))

    if train: 

        data.revenue = data.revenue.apply(lambda x : float(x))

    

    lE = LabelEncoder()

    data.production_companies = lE.fit_transform(data.production_companies)

    data.collection = lE.fit_transform(data.collection)

    data.original_language = lE.fit_transform(data.original_language)

    data.genres = lE.fit_transform(data.genres)

    

    

    data['year'] = data.release_date.apply(lambda x : getYear(x))

    data.drop('release_date', axis = 1, inplace = True)

    

    data.runtime.fillna(list(data.runtime.describe())[5], inplace = True)

    

    mx = MinMaxScaler()

    ser = pd.Series(list(mx.fit_transform(np.array(data.budget).reshape((len(data), -1))))).apply(lambda x : x[0])

    data.budget = ser

    

    return data
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import r2_score



train = preProcess(df)



y = np.array(train['revenue']).reshape((len(train), -1))

X = np.array(train.drop('revenue', axis = 1)).reshape((len(train), -1))





train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.tree import DecisionTreeRegressor



estimator = DecisionTreeRegressor()

estimator.fit(train_x, train_y)

estimator.score(train_x, train_y)
criterion = ['mse', 'mae', 'friedman_mse']

splitter = ['best', 'random']

max_depth = range(1, 500)

max_features = ['auto', 'sqrt', 'log2']





random_grid = {

    'criterion' : criterion,

    'splitter' : splitter,

    'max_depth' : max_depth,

    'max_features' : max_features

}
random_estimator = RandomizedSearchCV(

    estimator = estimator,

    param_distributions = random_grid,

    random_state = 1,

    n_jobs = -1,

    n_iter = 200

)



random_estimator.fit(train_x, train_y)

estimator = random_estimator.best_estimator_
r2_score(estimator.predict(test_x), test_y)
test = preProcess(pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv'), train = False)

test.head()
test.year.fillna(test.year.describe()[5], inplace = True)
test = np.array(test).reshape((len(test), -1))
sub = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/sample_submission.csv')

sub.head()
sub.revenue = estimator.predict(test)

sub.to_csv('submissions.csv', index = False)