import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = "/kaggle/input/emergent-kaggle-competion/train.csv" # Path was printed using the previous chunk of code

train = pd.read_csv(path, encoding='latin-1') # converts the CSV file to a pandas (pd) dataframe

# the latin-1 is the encoding that was used by the provider of the data set. If you do not include this you will get an abstract error

# that might confuse you.

del path # The path variable will not be used anymore. It makes sense to delete the reference in the environment (not necessary)

train # putting the name of a dataset at the end of the file without the print statement creates a nicer table compared to print(train)
train = train[['id', 'original_language', 'budget', 'genres', 'popularity', 'runtime', 'vote_average']] #requires an array of strings as input or a single string

train.head(5) # only prints the first 5 rows, again the lack of print statement improves the layout of the output
train.groupby('original_language').agg('count')['id'].sort_values(ascending = False).head(10)
languages = ['en', 'fr', 'it', 'ja', 'de', 'es'] # The hard coded languages we will keep

train.loc[:, 'original_language'] = np.where(train.loc[:, 'original_language'].isin(languages), train.loc[:, 'original_language'], 'other_language') # replacing languages that we don't want to keep with other

train.groupby('original_language').agg('count')['id'].sort_values(ascending = False).head(10) # check if it worked
train = pd.get_dummies(train, columns = ['original_language'], prefix = "", prefix_sep = "")

train.head(5)
import matplotlib.pyplot as plt

plt.hist(train.loc[:, 'budget'], bins=50) 

plt.ylabel('Count')

plt.xlabel('Budget'); #; prevents the output of warnings
budget = train.loc[:, 'budget']

budget[budget != 0].median() # we will impute the zeros with the median ignoring the zeros. This is not an optimal solution!

train.loc[:, 'budget_imputed'] = np.where(train.loc[:, 'budget'] == 0 | train.loc[:, 'budget'].isna(), 1, 0) # Best practise to include an indicator column that shows if the value was imputed

train.loc[:, 'budget'] = np.where(train.loc[:, 'budget'] == 0 | train.loc[:, 'budget'].isna() , 8000000, train.loc[:, 'budget'])

# The histogram also shows some extremely large values. We will cap them off at 15 000 000 (arbitrary choice)

train.loc[:, 'budget_capped'] = np.where(train.loc[:, 'budget'] > 15000000, 1,0)

train.loc[:, 'budget'] = np.where(train.loc[:, 'budget'] > 15000000, 15000000, train.loc[:, 'budget'])

# Check if it worked

plt.hist(train.loc[:, 'budget'], bins=50)

plt.ylabel('Count')

plt.xlabel('Budget');
# Not the most efficient code but it does the job quite nicely

def selectFirstGenre(String):

    return String.split("'")[5]



for i in range(len(train.loc[:, "genres"])):

    try: # in case there is no genre given

      newGenre = selectFirstGenre(train.loc[:, "genres"][i])

    except:

      newGenre = "Unknown"

    train.loc[i, 'genres'] = newGenre

train.groupby('genres').agg('count').head(7).sort_values(by = "id", ascending = False)['id']
genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Crime', 'Adventure', 'Animation'] 

train.loc[:, 'genres'] = np.where(train.loc[:, 'genres'].isin(genres), train.loc[:, 'genres'], 'Other_genre') 

train = pd.get_dummies(train, columns = ['genres'], prefix = "", prefix_sep = "")

train.head(5)
import missingno as msno 

msno.matrix(train); # ; surpresses warnings or comments
plt.hist(train.loc[:, 'runtime'], bins=50) ;

plt.ylabel('Count');

plt.xlabel('Runtime'); #; prevents the output of warnings

# We again see some extreme outliers hence these will also be capped. We will impute the missing values with the median

# train.loc[:, 'runtime'].median() median = 95

train.loc[:, 'runtime_imputed'] = np.where(train.loc[:, 'runtime'].isnull(), 1, 0)

train.loc[:, 'runtime'] = np.where(train.loc[:, 'runtime'].isnull(), 95, train.loc[:, 'runtime'])

#msno.matrix(train) # check if it worked

train.loc[:, 'runtime_capped'] = np.where(train.loc[:, 'runtime'] > 400, 1, 0)

train.loc[:, 'runtime'] = np.where(train.loc[:, 'runtime'] > 400, 400, train.loc[:, 'runtime'])
# Check

plt.hist(train.loc[:, 'runtime'], bins=50) ;

plt.ylabel('Count');

plt.xlabel('Runtime'); #; prevents the output of warnings

# We ignore the zeros
plt.hist(train.loc[:, 'popularity'], bins=50) ;

plt.ylabel('Count');

plt.xlabel('Popularity'); #; prevents the output of warnings
train.loc[:, 'popularity_imputed'] = np.where(train.loc[:, 'popularity'].isnull(), 1, 0)

train.loc[:, 'popularity'] = np.where(train.loc[:, 'popularity'].isnull(), 1.1213915, train.loc[:, 'popularity'])

train.loc[:, 'popularity_capped'] = np.where(train.loc[:, 'popularity'] > 35, 1, 0)

train.loc[:, 'popularity'] = np.where(train.loc[:, 'popularity'] > 35, 35, train.loc[:, 'popularity'])

plt.hist(train.loc[:, 'popularity'], bins=50) ;

plt.ylabel('Count');

plt.xlabel('Popularity'); #; prevents the output of warnings
def selectColumns(df):

    df = df[['id', 'original_language', 'budget', 'genres', 'popularity', 'runtime', 'vote_average']]

    return df





def cleanOriginalLanguage(df):

    languages = ['en', 'fr', 'it', 'ja', 'de', 'es'] # The hard coded languages we will keep

    df.loc[:, 'original_language'] = np.where(df.loc[:, 'original_language'].isin(languages), df.loc[:, 'original_language'], 'other_language')

    df = pd.get_dummies(df, columns = ['original_language'], prefix = "", prefix_sep = "")

    return df





def cleanBudget(df):

    df.loc[:, 'budget_imputed'] = np.where(df.loc[:, 'budget'] == 0 | df.loc[:, 'budget'].isnull(), 1, 0) # Best practise to include an indicator column that shows if the value was imputed

    df.loc[:, 'budget'] = np.where(df.loc[:, 'budget'] == 0 | df.loc[:, 'budget'].isnull(), 8000000, df.loc[:, 'budget'])

    df.loc[:, 'budget_capped'] = np.where(df.loc[:, 'budget'] > 15000000, 1,0)

    df.loc[:, 'budget'] = np.where(df.loc[:, 'budget'] > 15000000, 15000000, df.loc[:, 'budget'])

    return df





def selectFirstGenre(String):

    return String.split("'")[5]





def cleanGenres(df):

    for i in range(len(df.loc[:, "genres"])):

        try: # in case there is no genre given

          newGenre = selectFirstGenre(df.loc[:, "genres"][i])

        except:

          newGenre = "Unknown"

        df.loc[i, 'genres'] = newGenre

    genres = ['Drama', 'Comedy', 'Action', 'Documentary', 'Crime', 'Adventure', 'Animation'] 

    df.loc[:, 'genres'] = np.where(df.loc[:, 'genres'].isin(genres), df.loc[:, 'genres'], 'Other_genre') 

    df = pd.get_dummies(df, columns = ['genres'], prefix = "", prefix_sep = "")

    return df





def cleanPopularity(df):

    df.loc[:, 'popularity_imputed'] = np.where(df.loc[:, 'popularity'].isnull(), 1, df.loc[:, 'popularity'])

    df.loc[:, 'popularity'] = np.where(df.loc[:, 'popularity'].isnull(), 1.1213915, 0)

    df.loc[:, 'popularity_capped'] = np.where(df.loc[:, 'popularity'] > 35, 1, 0)

    df.loc[:, 'popularity'] = np.where(df.loc[:, 'popularity'] > 35, 35, df.loc[:, 'popularity'])

    return df



def cleanRuntime(df):

    df.loc[:, 'runtime_imputed'] = np.where(df.loc[:, 'runtime'].isnull(), 1, 0)

    df.loc[:, 'runtime'] = np.where(df.loc[:, 'runtime'].isnull(), 95, df.loc[:, 'runtime'])

    df.loc[:, 'runtime_capped'] = np.where(df.loc[:, 'runtime'] > 400, 1, 0)

    df.loc[:, 'runtime'] = np.where(df.loc[:, 'runtime'] > 400, 400, df.loc[:, 'runtime'])

    return df



   

def clean(df):

    df = selectColumns(df)

    df = cleanOriginalLanguage(df)

    df = cleanBudget(df)

    df = cleanGenres(df)

    df = cleanPopularity(df)

    df = cleanRuntime(df)

    return df







#test cleaning

path = "/kaggle/input/emergent-kaggle-competion/train.csv"

test = pd.read_csv(path, encoding='latin-1')

del path

test = clean(test)
import statsmodels.api as sm

X = train.drop(['vote_average', 'id'], axis=1)

y = train.loc[:, 'vote_average']



# Note the difference in argument order

model = sm.OLS(y, X, missing='drop').fit()

predictions = model.predict(X) # make the predictions by the model

# Print out the statistics

model.summary()
plt.hist(train.loc[:, 'vote_average'] , bins=50);

plt.ylabel('Count');

plt.xlabel('Vote Average'); #; prevents the output of warnings



plt.hist(predictions, bins=50) ;

plt.ylabel('Count');

plt.xlabel('Prediction'); #; prevents the output of warnings



# The orange displays the prediction and the blue shows the vote average
path = "/kaggle/input/emergent-kaggle-competion/train.csv"

train = pd.read_csv(path, encoding='latin-1')

train = clean(train)

X = train.drop(['vote_average', 'id'], axis=1)

y = train.loc[:, 'vote_average']



# Note the difference in argument order

model = sm.OLS(y, X, missing='drop').fit()



def estimate(df, model):

    df = clean(df)

    newX = df.drop(['vote_average', 'id'], axis=1)

    predictions = model.predict(newX)

    df.loc[:, 'Predicted'] = predictions

    df.loc[:, "Id"] = df.loc[:, 'id']

    return df[['Id', 'Predicted']]



path = "/kaggle/input/emergent-kaggle-competion/test.csv"

test = pd.read_csv(path, encoding='latin-1')

print(estimate(test, model))
path = "/kaggle/input/emergent-kaggle-competion/validate.csv"

validate = pd.read_csv(path, encoding='latin-1')

validate.loc[:, 'vote_average'] = 1



result = estimate(validate, model)

result.head(10)
result.to_csv("submission1_multiple_linear_regression.csv", index = False)