import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, log_loss

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import seaborn as sns

from scipy import stats
songs_df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
songs_df.describe()
songs_df.isnull().sum()
#Dropping Unnamed column

songs_df = songs_df.drop('Unnamed: 0', axis = 1)
#checking the data types 

songs_df.dtypes
songs_df.nunique()
songs_df =  songs_df.drop(['Track.Name', 'Artist.Name'], axis = 1)
categorical = ['Genre']

numerical= ['Beats.Per.Minute', 'Energy', 'Danceability', 'Loudness..dB..', 'Liveness', 'Valence.', 'Length.', 'Acousticness..', 

            'Speechiness.']

target = 'Popularity'
#Obtaining the counts for each category from every variable

for i in songs_df.columns:

    print(songs_df[i].value_counts())
fig = plt.figure(figsize = (18, 12)) 

count  = 1

for i in numerical:

    ax = fig.add_subplot(5, 2, count)

    ax.hist(songs_df[i])

    ax.set_title(i)

    count += 1



fig.tight_layout()

plt.show()
#correlation for numerical values

songs_corr = songs_df.corr()
# heatmap of the correlation 

plt.figure(figsize=(10,10))

plt.title('Correlation heatmap')

sns.heatmap(songs_corr,annot=True)

g = sns.PairGrid(songs_df)

g = g.map(plt.scatter)
#from out value counts we know dance_pop is our most popular, we will drop and use as referance

dummies_dropped_one = pd.get_dummies(songs_df['Genre'])

dummies_dropped_one = dummies_dropped_one.drop(columns = ['dance pop'])

#dropping genre column as we used it for our dummies

songs_df = songs_df.drop('Genre', axis = 1)

#combining our dummies with our other variables

songs_df = pd.concat([songs_df, dummies_dropped_one], axis = 1)
X = songs_df.loc[:, songs_df.columns != target]

y = songs_df[target].loc[:,]

X_1 = sm.add_constant(X, prepend = True, has_constant = 'add')

#%%

#Using SkLearn to create out training and testing data sets

X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=0)
#Statsmodels Linear Regression

method = sm.regression.linear_model.OLS(y_train.values.ravel(), X_train, has_constant = True)

result = method.fit()

print(result.summary())
#Feature Selection

cols = list(X.columns)

pmax = 1 #placeholder for new p-value max

while (len(cols)>0):

    p= []

    X_1 = X[cols]

    X_1 = sm.add_constant(X_1,prepend = True, has_constant = 'add')

    model = sm.OLS(y,X_1, hasconst = True).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols) #not idexing the constant column     

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print(selected_features_BE)
#selecting our significant variables

X = songs_df.loc[:, selected_features_BE]

y = songs_df.loc[:,target]



#scaling our variables

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)



#adding a constant

X_1 = sm.add_constant(X, prepend = True, has_constant = 'add')

#Using SkLearn to create out training and testing data sets

X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=0)
#Statsmodels Logistic Regression

method = sm.regression.linear_model.OLS(y_train.values.ravel(), X_train, hasconst = True)

result = method.fit()

print(result.summary())