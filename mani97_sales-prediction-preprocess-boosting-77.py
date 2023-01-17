import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Basic 

import numpy as np

import pandas as pd



# Plotting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
sales = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')

sales.head(1)
sales.info()
sales.isnull().sum()[sales.isnull().sum() !=0]
# rating features

sales['rating_five_count'].replace(np.nan, 0, inplace=True)

sales['rating_four_count'].replace(np.nan, 0, inplace=True)

sales['rating_three_count'].replace(np.nan, 0, inplace=True)

sales['rating_two_count'].replace(np.nan, 0, inplace=True)

sales['rating_one_count'].replace(np.nan, 0, inplace=True)



# urgency banner

sales['has_urgency_banner'].replace(np.nan, 0, inplace=True)
sales.dtypes[sales.dtypes == 'object']
count = sales['product_color'].value_counts()

count[count>3]
sales['product_color'].replace('armygreen', 'green', inplace=True)

sales['product_color'].replace('winered', 'red', inplace=True)

sales['product_color'].replace('navyblue', 'blue', inplace=True)

sales['product_color'].replace('lightblue', 'blue', inplace=True)

sales['product_color'].replace('khaki', 'green', inplace=True)

sales['product_color'].replace('gray', 'grey', inplace=True)

sales['product_color'].replace('rosered', 'red', inplace=True)

sales['product_color'].replace('skyblue', 'blue', inplace=True)

sales['product_color'].replace('coffee', 'brown', inplace=True)

sales['product_color'].replace('darkblue', 'blue', inplace=True)

sales['product_color'].replace('rose', 'red', inplace=True)

sales['product_color'].replace('fluorescentgreen', 'green', inplace=True)

sales['product_color'].replace('navy', 'blue', inplace=True)

sales['product_color'].replace('lightpink', 'pink', inplace=True)
count = sales['product_color'].value_counts()

count[count==3]
sales['product_color'].replace('orange-red', 'red', inplace=True)

sales['product_color'].replace('Black', 'black', inplace=True)

sales['product_color'].replace('lightgreen', 'green', inplace=True)

sales['product_color'].replace('White', 'white', inplace=True)
count = sales['product_color'].value_counts()

count[count==2]
sales['product_color'].replace('wine', 'red', inplace=True)

sales['product_color'].replace('Pink', 'pink', inplace=True)

sales['product_color'].replace('Army green', 'green', inplace=True)

sales['product_color'].replace('coralred', 'red', inplace=True)

sales['product_color'].replace('lightred', 'red', inplace=True)

sales['product_color'].replace('apricot', 'orange', inplace=True)

sales['product_color'].replace('navy blue', 'blue', inplace=True)

sales['product_color'].replace('burgundy', 'red', inplace=True)

sales['product_color'].replace('silver', 'grey', inplace=True)

sales['product_color'].replace('camel', 'brown', inplace=True)

sales['product_color'].replace('lakeblue', 'blue', inplace=True)

sales['product_color'].replace('lightyellow', 'yellow', inplace=True)

sales['product_color'].replace('watermelonred', 'red', inplace=True)

sales['product_color'].replace('coolblack', 'black', inplace=True)

sales['product_color'].replace('applegreen', 'green', inplace=True)

sales['product_color'].replace('mintgreen', 'green', inplace=True)

sales['product_color'].replace('dustypink', 'pink', inplace=True)
count = sales['product_color'].value_counts()

count[count==1]
sales['product_color'].replace('ivory', 'white', inplace=True)

sales['product_color'].replace('lightkhaki', 'green', inplace=True)

sales['product_color'].replace('lightgray', 'grey', inplace=True)

sales['product_color'].replace('darkgreen', 'green', inplace=True)

sales['product_color'].replace('RED', 'red', inplace=True)

sales['product_color'].replace('tan', 'brown', inplace=True)

sales['product_color'].replace('jasper', 'red', inplace=True)

sales['product_color'].replace('nude', 'white', inplace=True)

sales['product_color'].replace('army', 'brown', inplace=True)

sales['product_color'].replace('light green', 'green', inplace=True)

sales['product_color'].replace('offwhite', 'white', inplace=True)

sales['product_color'].replace('Blue', 'blue', inplace=True)

sales['product_color'].replace('denimblue', 'blue', inplace=True)

sales['product_color'].replace('Rose red', 'red', inplace=True)

sales['product_color'].replace('lightpurple', 'purple', inplace=True)

sales['product_color'].replace('prussianblue', 'blue', inplace=True)

sales['product_color'].replace('offblack', 'black', inplace=True)

sales['product_color'].replace('violet', 'purple', inplace=True)

sales['product_color'].replace('gold', 'yellow', inplace=True)

sales['product_color'].replace('wine red', 'red', inplace=True)

sales['product_color'].replace('rosegold', 'red', inplace=True)

sales['product_color'].replace('claret', 'red', inplace=True)

sales['product_color'].replace('army green', 'green', inplace=True)

sales['product_color'].replace('lightgrey', 'grey', inplace=True)
count = sales['product_color'].value_counts()

count
sales['product_color'].replace(np.nan, 'others', inplace=True)
def color(col):

    ls = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'grey', 'purple', 'orange', 'brown', 'beige']

    if col not in ls:

        if '&' in col:

            return 'dual'

        else:

            return 'others'

    return col
sales['product_color'] = sales['product_color'].apply(color)
plt.figure(figsize=(12,10))

sns.countplot(x = 'product_color', data = sales, order = sales['product_color'].value_counts().iloc[:].index)

plt.xlabel('Product Colour')

plt.ylabel('Count')

plt.show()
count = sales['product_variation_size_id'].value_counts()

count[count>3]
sales['product_variation_size_id'].replace('S.', 'S', inplace=True)

sales['product_variation_size_id'].replace('Size S', 'S', inplace=True)

sales['product_variation_size_id'].replace('XS.', 'XS', inplace=True)

sales['product_variation_size_id'].replace('s', 'S', inplace=True)

sales['product_variation_size_id'].replace('M.', 'M', inplace=True)

sales['product_variation_size_id'].replace('2XL', 'XXL', inplace=True)

sales['product_variation_size_id'].replace('Size XS', 'XS', inplace=True)

sales['product_variation_size_id'].replace('Size-XS', 'XS', inplace=True)

sales['product_variation_size_id'].replace('4XL', 'XXXXL', inplace=True)

sales['product_variation_size_id'].replace('SIZE XS', 'XS', inplace=True)
count = sales['product_variation_size_id'].value_counts()

count[count==3]
sales['product_variation_size_id'].replace('SizeL', 'L', inplace=True)

sales['product_variation_size_id'].replace('Size-S', 'S', inplace=True)
count = sales['product_variation_size_id'].value_counts()

count[count==2]
sales['product_variation_size_id'].replace('5XL', 'XXXXXL', inplace=True)

sales['product_variation_size_id'].replace('3XL', 'XXXL', inplace=True)

sales['product_variation_size_id'].replace('S(bust 88cm)', 'S', inplace=True)

sales['product_variation_size_id'].replace('Size4XL', 'XXXXL', inplace=True)

sales['product_variation_size_id'].replace('Size -XXS', 'XXS', inplace=True)

sales['product_variation_size_id'].replace('SIZE-XXS', 'XXS', inplace=True)

sales['product_variation_size_id'].replace('Size M', 'M', inplace=True)

sales['product_variation_size_id'].replace('size S', 'S', inplace=True)

sales['product_variation_size_id'].replace('S Pink', 'S', inplace=True)

sales['product_variation_size_id'].replace('Size S.', 'S', inplace=True)

sales['product_variation_size_id'].replace('Suit-S', 'S', inplace=True)
count = sales['product_variation_size_id'].value_counts()

count.count()
def size_name(size):

    ls = ["XXXS", "XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "XXXXL", "XXXXXL"]

    if size in ls:

        return size

    return "Others"
sales['product_variation_size_id'].replace(np.nan, 'Others', inplace=True)

sales['product_variation_size_id'] = sales['product_variation_size_id'].apply(size_name)
plt.figure(figsize=(12,10))

sns.countplot(x = 'product_variation_size_id', data = sales, order = sales['product_variation_size_id'].value_counts().iloc[:].index)

plt.xlabel('Product Variation Size ID')

plt.ylabel('Count')

plt.show()
sales['origin_country'].value_counts()
def origin_name(country):

    ls = ["VE", "SG", "GB", "AT"]

    if country in ls:

        return "Others"

    return country
sales['origin_country'].replace(np.nan, "Others", inplace=True)

sales['origin_country'] = sales['origin_country'].apply(origin_name)
plt.figure(figsize=(12,10))

sns.countplot(x = 'origin_country', data = sales, order = sales['origin_country'].value_counts().iloc[:].index)

plt.xlabel('Origin Country')

plt.ylabel('Count')

plt.show()
ls = sales.nunique()

ls[ls==1]
sales.drop(labels = ['currency_buyer', 'theme', 'crawl_month'], axis=1, inplace=True)
collect_tags = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')

print('Total number of tags: ', collect_tags.shape[0])
# Return percentage of tags present for a product



def tag_number(tags):

    ls = tags.split(',')

    return len(ls)/collect_tags.shape[0]
sales['tags_percentage'] = sales['tags'].apply(tag_number)
sales.drop(labels = ['tags'], axis=1, inplace=True)
sales.dtypes[sales.dtypes == 'object']
sales.drop(labels = ['title', 'title_orig', 'merchant_profile_picture', 'product_url', 'product_picture', 'product_id', 'merchant_id', 

                     'merchant_info_subtitle', 'merchant_name', 'merchant_title', 'shipping_option_name', 'urgency_text'], 

           axis=1, 

           inplace=True)
sales.drop(labels = ['rating_count'], axis=1, inplace=True)
# product color

dummies_color = pd.get_dummies(sales['product_color'], drop_first=True) # give us the one hot ecoded features

dummies_color.drop(labels = 'others', axis=1, inplace=True) # remove the 'others' feature as n-1 encoded features represents n features
# product variation size id

dummies_variation = pd.get_dummies(sales['product_variation_size_id'])

dummies_variation.drop(labels = ['Others'], axis = 1, inplace=True)
dummies_origin = pd.get_dummies(sales['origin_country'])

dummies_origin.drop(labels=['Others'], axis = 1, inplace=True)

# concatenating all the one hot encoded features for the three categorical variables above



feat_onehot = pd.concat([dummies_color, dummies_variation, dummies_origin, sales['units_sold']], axis=1)

feat_onehot.head(1)
feat_onehot_corr = feat_onehot.corr()



feat_onehot_corr['units_sold'].sort_values(ascending=False)
sales.drop(labels = ['product_color', 'product_variation_size_id', 'origin_country'], 

           axis=1, 

           inplace=True)
sales_corr = sales.corr()



plt.figure(figsize = (18, 16))

sns.heatmap(sales_corr, annot=True, cmap='Blues_r')

plt.title('Correlation between features')

plt.show()
sales_corr['units_sold'].sort_values(ascending=False)
# separating the independent and dependent variables



y = sales['units_sold']

X = sales.drop(labels = ['units_sold'], axis = 1)

print("Shape of X is {} and that of y is {}".format(X.shape, y.shape))
# Splitting the dataset 



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)



print('Shape of training set ', X_train.shape)

print('Shape of test set ', X_test.shape)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_regression



# feature selection

def select_features(X_train, y_train, X_test):

    # configure to select all features

    fs = SelectKBest(score_func=mutual_info_regression, k='all')

    # learn relationship from training data

    fs.fit(X_train, y_train)

    # transform train input data

    X_train_fs = fs.transform(X_train)

    # transform test input data

    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs

 
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)



plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

plt.tick_params(color='white', labelcolor='white')

plt.xlabel('Features', color='white')

plt.ylabel('Score of Features', color='white')

plt.show()
def select_features(X_train, y_train, X_test):

    # configure to select all features

    fs = SelectKBest(score_func=mutual_info_regression, k=8)

    # learn relationship from training data

    fs.fit(X_train, y_train)

    # transform train input data

    X_train_fs = fs.transform(X_train)

    # transform test input data

    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs
# Selecting features



X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)



print('Shape of Training set with the best features: ', X_train_fs.shape)
cols = fs.get_support(indices=True)



print('Best columns that we are using for our model\n')

for i in cols:

    print (sales.columns[i])
# Importing models

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



# Regression Metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



# Cross validation

from sklearn.model_selection import cross_val_score
regressors = [LinearRegression(),

             DecisionTreeRegressor(random_state=1),

             RandomForestRegressor(n_estimators = 10, random_state=1)]



df = pd.DataFrame(columns = ['Name', 'Train Score', 'Test Score', 'Mean Absolute Error', 'Mean Squared Error', 

                             'Cross Validation Score (Mean Accuracy)', 'R2 Score'])
for regressor in regressors:

    regressor.fit(X_train_fs, y_train)

    y_pred = regressor.predict(X_test_fs)

    

    # print classifier name

    s = str(type(regressor)).split('.')[-1][:-2]

    

    # Train Score

    train = regressor.score(X_train_fs, y_train)

    

    # Test Score

    test = regressor.score(X_test_fs, y_test)

    

    # MAE score

    mae = mean_absolute_error(y_test, y_pred)

    

    # MSE Score

    mse = mean_squared_error(y_test, y_pred)

    

    accuracy = cross_val_score(estimator = regressor, X = X_train_fs, y = y_train, cv=10)

    cv = accuracy.mean()*100

    

    r2 = r2_score(y_test, y_pred)

    

    df = df.append({'Name': s, 'Train Score': train, 'Test Score': test, 'Mean Absolute Error': mae, 

                    'Mean Squared Error': mse, 'Cross Validation Score (Mean Accuracy)': cv,

                   'R2 Score': r2},

                  ignore_index=True)
df
# Making Polynomial Features

from sklearn.preprocessing import PolynomialFeatures



poly_reg = PolynomialFeatures(degree = 3)

X_train_poly = poly_reg.fit_transform(X_train_fs)

X_test_poly = poly_reg.fit_transform(X_test_fs)



# Fitt PolyReg to training set

regressor = LinearRegression()

regressor.fit(X_train_poly, y_train)



# Predicting test values

y_pred = regressor.predict(X_test_poly)



df = df.append({'Name': str(type(regressor)).split('.')[-1][:-2] + ' (Poly)', 

                'Train Score': regressor.score(X_train_poly, y_train), 

                'Test Score': regressor.score(X_test_poly, y_test), 

                'Mean Absolute Error': mean_absolute_error(y_test, y_pred), 

                'Mean Squared Error': mean_squared_error(y_test, y_pred), 

                'Cross Validation Score (Mean Accuracy)': cross_val_score(estimator = regressor, X = X_train_fs, y = y_train, cv=10).mean()*100,

                'R2 Score': r2_score(y_test, y_pred)},

                  ignore_index=True)
# Scaling

from sklearn.preprocessing import StandardScaler



# Applying feature scaling for this

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train_fs)

X_test_sc = sc.fit_transform(X_test_fs)



regressor = SVR(kernel='rbf')

regressor.fit(X_train_sc, y_train)



# Predicting test values

y_pred = regressor.predict(X_test_sc)



df = df.append({'Name': str(type(regressor)).split('.')[-1][:-2], 

                'Train Score': regressor.score(X_train_sc, y_train), 

                'Test Score': regressor.score(X_test_sc, y_test), 

                'Mean Absolute Error': mean_absolute_error(y_test, y_pred), 

                'Mean Squared Error': mean_squared_error(y_test, y_pred), 

                'Cross Validation Score (Mean Accuracy)': cross_val_score(estimator = regressor, X = X_train_sc, y = y_train, cv=10).mean()*100,

                'R2 Score': r2_score(y_test, y_pred)},

                  ignore_index=True)
df
from sklearn.model_selection import GridSearchCV



reg = RandomForestRegressor(random_state=1)



param_grid = { 

    'n_estimators': np.arange(4, 30, 2),

    'max_depth' : [4,5,6,7,8],

}
CV_reg = GridSearchCV(estimator=reg, param_grid=param_grid, cv= 5)

CV_reg.fit(X_train_fs, y_train)
CV_reg.best_params_
regressor = RandomForestRegressor(n_estimators=18, random_state=1, max_depth=4)



regressor.fit(X_train_fs, y_train)



# Predicting test values

y_pred = regressor.predict(X_test_fs)



df = df.append({'Name': str(type(regressor)).split('.')[-1][:-2] + ' (after GridSearchCV)', 

                'Train Score': regressor.score(X_train_fs, y_train), 

                'Test Score': regressor.score(X_test_fs, y_test), 

                'Mean Absolute Error': mean_absolute_error(y_test, y_pred), 

                'Mean Squared Error': mean_squared_error(y_test, y_pred), 

                'Cross Validation Score (Mean Accuracy)': cross_val_score(estimator = regressor, X = X_train_fs, y = y_train, cv=10).mean()*100,

                'R2 Score': r2_score(y_test, y_pred)},

                  ignore_index=True)
from sklearn.ensemble import VotingRegressor



regressor = VotingRegressor([('lr',LinearRegression()), ('rf', RandomForestRegressor(n_estimators=18, random_state=1, max_depth=4))])



regressor.fit(X_train_fs, y_train)



# Predicting test values

y_pred = regressor.predict(X_test_fs)



df = df.append({'Name': str(type(regressor)).split('.')[-1][:-2], 

                'Train Score': regressor.score(X_train_fs, y_train), 

                'Test Score': regressor.score(X_test_fs, y_test), 

                'Mean Absolute Error': mean_absolute_error(y_test, y_pred), 

                'Mean Squared Error': mean_squared_error(y_test, y_pred), 

                'Cross Validation Score (Mean Accuracy)': cross_val_score(estimator = regressor, X = X_train_fs, y = y_train, cv=10).mean()*100,

                'R2 Score': r2_score(y_test, y_pred)},

                  ignore_index=True)
df
