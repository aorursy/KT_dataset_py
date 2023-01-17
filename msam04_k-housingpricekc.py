import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score



%matplotlib inline

data = pd.read_csv('../input/kc_house_data.csv')

print(data.info())
#Dropping id and date from the get go.

X = data[['bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
         'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
         'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price']]

X['renovated_living'] = X['sqft_living15'] - X['sqft_living']
X['renovated_lot'] = X['sqft_lot15'] - X['sqft_lot']
data[['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']].corr()
X = X.drop(['sqft_living15', 'sqft_lot15'], axis=1) #Removing these 2 columns as they are highly correlated with sqt_living, sqt_lot
print(len(list(data['zipcode'].unique())))

sns.stripplot(x="zipcode", y="price", data=data)
X = X.drop(['lat', 'long'], axis=1) #Remove these columns as we have zipcode
print(X['yr_built'].isnull().values.any())
print(X['yr_renovated'].isnull().values.any())

def get_years_ago(yr_built, yr_renovated):
    yr_later = max(yr_built, yr_renovated)
    return (2018 - yr_later)


#X['year_worked_ago'] = X[['yr_built','yr_renovated']].apply(lambda x: get_years_ago(x['yr_built'], x['yr_renovated']), axis=1)

X['age'] = 2018 - X['yr_built']


X.drop(['yr_built', 'yr_renovated'], axis=1, inplace=True)

print(X['price'].quantile(0.25))
print(X['price'].quantile(0.5))
print(X['price'].quantile(0.75))

def get_zip_category(price_value):
    if(int(price_value) <= 321950):
        return "Low"
    elif(int(price_value) > 321950 and int(price_value) <= 450000):
        return "Medium"
    else:
        return "High"
    
#X['zip_value'] = ""

#X['zip_value'] = get_zip_category(X['price'])

#X['zip_value'] = pd.Series([get_zip_category(X.iloc[x]['price'] for x in range(X.shape[0]))])
X['zip_value'] = X['price'].apply(get_zip_category)
X.drop('zipcode', axis=1, inplace=True)
print(X.info())
print(data['view'].unique())
print(data['condition'].unique())
print(data['grade'].unique())
sns.heatmap(X.corr(), mask=np.zeros_like(X.corr(), dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True)

print(X.corr())





def get_unique_pairs(corr_matrix):
    unique_pairs = set()
    for c1 in corr_matrix.columns:
        for c2 in corr_matrix.columns:
            unique_pairs.add((c1, c2))
            
    return unique_pairs

print(len(get_unique_pairs(X.corr())))


def get_redundant_pairs(corr_matrix):
    #Get diagonal and lower triangular pairs of correlation matrix
    pairs_to_drop = set()
    cols = corr_matrix.columns
    for i in range(0, corr_matrix.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

corr_vals = X.corr().abs().unstack()
corr_vals = corr_vals.drop(labels=get_redundant_pairs(X.corr())).sort_values(ascending=False)
print(corr_vals[corr_vals >= 0.75])
#Since sqft_living, sqft_above, grade and bathrooms are all highly correlated, dropping sqft_above, grade and bathrooms

#X.drop(['sqft_above', 'grade', 'bathrooms'], axis=1, inplace=True)

X.drop('sqft_above', axis=1, inplace=True)
print(X.info())
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


for col in ['bedrooms', 'sqft_living', 'sqft_lot', 'price', 'sqft_basement', 'age', 'renovated_living', 'renovated_lot', 'bathrooms'] :
    X = remove_outlier(X, col)

print(X.info())
X = pd.get_dummies(data=X, columns=['waterfront', 'view', 'condition', 'bedrooms', 'zip_value', 'grade'])
print(X.info())

import statsmodels.api as sm
import numpy as np

Y = X['price']
X.drop('price', axis=1, inplace=True)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
influence = results.get_influence()
#c is the distance and p is p-value
(c, p) = influence.cooks_distance
print(c,p)
#X['zip_code_gp'] = pd.qcut(X['zipcode'], q=10, labels=np.arange(10) + 1)
#X['zip_code_gp'] = pd.cut(X['zipcode'], bins=40, labels=np.arange(40) + 1)
#X.drop('zipcode', axis=1, inplace=True)
#X = pd.get_dummies(data=X, columns=['waterfront', 'view', 'condition', 'bedrooms', 'zipcode', 'grade'])

x_scaler = StandardScaler()

X = x_scaler.fit_transform(X)

y_scaler = StandardScaler()

y = Y.values.reshape(-1,1)

y = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
model = LinearRegression()

model.fit(X_train, y_train)

print("Model train score: {}".format(model.score(X_train, y_train)))

print("Adjusted R2 {}".format(1 - (1-model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

