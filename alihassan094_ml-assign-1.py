# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        url = os.path.join(dirname, filename)
        print(url)

# Any results you write to the current directory are saved as output.
import os
import sys
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report

# I Just need the Close
# url='/kaggle/input/dataset-tree/Tree Training Dataset.csv'

def read_data(url):

    data = pd.read_csv(url)
    
    # sort the values by symbol and then date
#     data.sort_values(by = ['symbol','datetime'], inplace = True)
    cols = ['Date','Price','Open','High','Low','Vol.','Change']
    data = data[cols]
    
#     top_row = pd.DataFrame({'Date':['June 22, 2020'], 'Price':[0], 'Open':[0], 'High':[0],'Low':[0],'Vol.':[0],'Change':[0]})
#     top_row = pd.DataFrame({'Date':['June 22, 2020']})
    
#     # Concat with old DataFrame and reset the Index.
#     data = pd.concat([top_row, data]).reset_index(drop = True)

    send_data = data.loc[:,:].values
    
    return cols, data, send_data
    
def add_change_in_price(data):
    # calculate the change in price
    data['change'] = data['Price'].diff()
    return data
cols, price_data, array_data = read_data(url)
# price_data = add_change_in_price(price_data)
print('Features are:', cols)
price_data = price_data.iloc[::-1]
print(price_data)
print(price_data.shape)
for i in range (len(cols)):
    print(type(price_data[cols[i]][5]))
def calculate_rsi(data):
    # Calculate the 14 day RSI
    n = 14

    # First make a copy of the data frame twice
    up_df, down_df = data[['Change']].copy(), data[['Change']].copy()

    # # For up days, if the change is less than 0 set to 0.
    up_df.loc['Change'] = up_df.loc[(up_df['Change'] < 0), 'Change'] = 0

    # # For down days, if the change is greater than 0 set to 0.
    down_df.loc['Change'] = down_df.loc[(down_df['Change'] > 0), 'Change'] = 0

    # # We need change in price to be absolute.
    down_df['Change'] = down_df['Change'].abs()

    # # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
    ewma_up = up_df['Change'].transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df['Change'].transform(lambda x: x.ewm(span = n).mean())

    # # Calculate the Relative Strength
    relative_strength = ewma_up / ewma_down

    # # Calculate the Relative Strength Index
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    # # Add the info to the data frame.
    data['down_days'] = down_df['Change']
    data['up_days'] = up_df['Change']
    data['RSI'] = relative_strength_index

    # print(price_data.shape)

    # price_data
    return data
price_data = calculate_rsi(price_data)

# Display the head.
price_data.head(30)
def calculate_stos(data):
    # Calculate the Stochastic Oscillator
    n = 14

    # Make a copy of the high and low column.
    low_14, high_14 = data['Low'].copy(), data['High'].copy()

    # low_14 = low_14.apply(pd.to_numeric, errors='coerce')
    # high_14 = high_14.apply(pd.to_numeric, errors='coerce')

    # # # Group by symbol, then apply the rolling function and grab the Min and Max.
    # low_14 = low_14.transform(lambda x: x.rolling(window = n).min())
    # high_14 = high_14.transform(lambda x: x.rolling(window = n).max())
    high_14 = high_14.rolling(n).max()
    low_14 = low_14.rolling(n).min()

    # # # # Calculate the Stochastic Oscillator.
    k_percent = 100 * ((data['Price'] - low_14) / (high_14 - low_14))

    # # # Add the info to the data frame.
    data['low_14'] = low_14
    data['high_14'] = high_14
    data['k_percent'] = k_percent
    return data

price_data = calculate_stos(price_data)
# Display the head.
price_data.tail(1)
def calculate_william_r(data):
    # Calculate the Williams %R
    n = 14

    # Make a copy of the high and low column.
    # low_14, high_14 = price_data['Low'].copy(), price_data['High'].copy()

    # # Group by symbol, then apply the rolling function and grab the Min and Max.
    # low_14 = low_14.transform(lambda x: x.rolling(window = n).min())
    # high_14 = high_14.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())
    low_14, high_14 = data['Low'].copy(), data['High'].copy()

    high_14 = high_14.rolling(n).max()
    low_14 = low_14.rolling(n).min()

    # # Calculate William %R indicator.
    r_percent = ((high_14 - data['Price']) / (high_14 - low_14)) * - 100

    # # Add the info to the data frame.
    data['r_percent'] = r_percent
    return data
price_data = calculate_william_r(price_data)
# Display the head.
price_data.head(14)
def calculate_macd(data):
    # Calculate the MACD
    ema_26 = data['Price'].transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = data['Price'].transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26
    print(price_data.shape)
    # Calculate the EMA
    ema_9_macd = macd.ewm(span = 9).mean()

    # # Store the data in the data frame.
    data['MACD'] = macd
    data['MACD_EMA'] = ema_9_macd
    
    return data
price_data = calculate_macd(price_data)
# Print the head.
price_data.head(30)
def calculate_price_roc(data):
    # Calculate the Price Rate of Change
    n = 9

    # Calculate the Rate of Change in the Price, and store it in the Data Frame.
    data['Price_Rate_Of_Change'] = data['Price'].transform(lambda x: x.pct_change(periods = n))
    
    return data
    
price_data = calculate_price_roc(price_data)
# Print the first 30 rows
price_data.head(33)
# Create a column we wish to predict
def create_prediction(data):
    # Group by the `Symbol` column, then grab the `Close` column.
    price_groups = data['Price']

    # Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
    price_groups = price_groups.transform(lambda x : np.sign(x.diff()))

    # add the data to the main dataframe.
    data['Prediction'] = price_groups

    # for simplicity in later sections I'm going to make a change to our prediction column. To keep this as a binary classifier I'll change flat days and consider them up days.
    data.loc[data['Prediction'] == 0.0] = 1.0
    
    return data

# OPTIONAL CODE: Dump the data frame to a CSV file to examine the data yourself.
# price_data.to_csv('final_metrics.csv')
price_data = create_prediction(price_data)
# print the head
price_data.tail(10)

# We need to remove all rows that have an NaN value.
print('Before NaN Drop we have {} rows and {} columns'.format(price_data.shape[0], price_data.shape[1]))

# Any row that has a `NaN` value will be dropped.
price_data = price_data.dropna()

# Display how much we have left now.
print('After NaN Drop we have {} rows and {} columns'.format(price_data.shape[0], price_data.shape[1]))

# Print the head.
price_data.head()
features = ['RSI','low_14','high_14','k_percent','r_percent','MACD','MACD_EMA','Price_Rate_Of_Change']
collection = ['Date','RSI','low_14','high_14','k_percent','r_percent','MACD','MACD_EMA','Price_Rate_Of_Change']
target = ['Prediction']
x_collect = price_data[collection]
x_data = price_data[features]
y_data = price_data[target]
x_collect

price_data.shape
top_row = pd.DataFrame({'Date':['June 22, 2020'],'RSI':[0],'low_14':[0], 'high_14':[0],'k_percent':[0],
                        'r_percent':[0],'MACD':[0],'MACD_EMA':[0],'Price_Rate_Of_Change':[0]})
    
price_data1 = price_data
price_data1 = price_data1.iloc[::-1]

# Concat with old DataFrame and reset the Index.
df = price_data1[collection]

# df.drop(df.tail(1).index,inplace=True)
df = pd.concat([top_row, df])
# df = df.iloc[::-1]
print(df.shape)
df
df = df.iloc[::-1]
df
df = df[features]
my = df.loc[:,:].values
my
dg = pd.DataFrame(my)
# dg = dg.iloc[::-1]
dg
dg
# dg = dg.reindex(index=dg.index[::-1])
# dg = dg.iloc[::-1]

# dg.loc[278:278,:]
# price_data.loc[1:1,:]
price_data
dg = dg.iloc[::-1]
dg.head(5)
price_data1 = price_data
price_data1 = price_data1.iloc[::-1]

# price_data1[features] = dg

price_data1.head(5)
# 52.182112	1671.7	1761.0	66.517357	-33.482643	3.448006	3.101512	
x_data
y_data
plot_data = pd.DataFrame(x_data)
plot_data["Pre"] = y_data
sns.pairplot(plot_data, hue='Pre', palette='OrRd')
def split_data(x1, y1):
    # Split X and y into X_
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, random_state = 0)
    return X_train, X_test, y_train, y_test

def apply_randomforest(x, y):
    # Create a Random Forest Classifier
    rand_frst_clf = RandomForestClassifier(n_estimators = 200, oob_score = True, criterion = "gini", random_state = 0)
    
    X_train, X_test, y_train, y_test = split_data(x, y)
    # Fit the data to the model
    rand_frst_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = rand_frst_clf.predict(X_test)
    
    correct = accuracy_score(y_test, y_pred, normalize = True) * 100.0
    print('Correct Prediction (%): ', correct)
    
    return correct, y_pred
pred_score, y_pred = apply_randomforest(x_data, y_data)
y_pred
count = np.zeros(price_data.shape[0])
# add=0
for i in range (price_data.shape[0]):
    count[i] = i
# count
x_array = x_data.loc[:,:].values

for i in range (x_array.shape[1]):
    plt.scatter(x_array[:,i], count)
# plt.scatter(y_data, count)
from sklearn.decomposition import PCA
pca_ml1 = PCA(n_components=4)
pca_fit = pca_ml1.fit_transform(x_data)

pca_df = pd.DataFrame(data = pca_fit
             , columns = ['pc1', 'pc2', 'pc3', 'pc4'])
pca_df.head(5)
sns.pairplot(pca_df)
plt.show()
pca_df["Pre"] = y_data
sns.pairplot(pca_df, hue='Pre', palette='OrRd')

count.shape
pca_fit[:,1].shape
rf_pca_pred_score, rf_pca_pred = apply_randomforest(pca_df, y_data)
def apply_knn(x, y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    
    X_train, X_test, y_train, y_test = split_data(x, y)
    # Train the model using the training sets
    
    model.fit(X_train,y_train)

    #Predict Output
    pred = model.predict(X_test)
    correct = accuracy_score(y_test, pred, normalize = True) * 100.0
    print('Correct Prediction (%): ', correct)
    return correct, pred
knn_pred_score, pred = apply_knn(x_data, y_data)
def apply_NB(x, y):
    from sklearn.naive_bayes import GaussianNB
    
    model = GaussianNB()
    
    X_train, X_test, y_train, y_test = split_data(x, y)
    
    # Train the model using the training sets
    model.fit(X_train,y_train)

    #Predict Output
    pred = model.predict(X_test)
    correct = accuracy_score(y_test, pred, normalize = True) * 100.0
    print('Correct Prediction (%): ', correct)
    return correct, pred
NB_pred_score, pred = apply_NB(x_data, y_data)
def apply_lda(x, y):
#     from sklearn.lda import LDA
    
#     model = LDA()
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis()

    X_train, X_test, y_train, y_test = split_data(x, y)
    
    print(X_train.shape)
    print(y_train.shape)
    
    # Train the model using the training sets
    model.fit(X_train,y_train.values.ravel())
#     model.fit(X_train,y_train)
#     model.fit(X_train)
    lda_fit = model.transform(X_train)
    lda_fit=0
    
    #Predict Output
    pred = model.predict(X_test)
    correct = accuracy_score(y_test, pred, normalize = True) * 100.0
    print('Correct Prediction (%): ', correct)

    
    return correct, lda_fit, pred
lda_pred_score, lda_fit, lda_pred = apply_lda(x_data, y_data)
def apply_LG(x, y):
    
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

    X_train, X_test, y_train, y_test = split_data(x, y)
    
    # Train the model using the training sets
    model.fit(X_train,y_train)
#     LG_fit = model.fit(X_train,y_train).transform(X_train)
    
    #Predict Output
    pred = model.predict(X_test)
#     correct = accuracy_score(y_test, pred, normalize = True) * 100.0
#     print('Correct Prediction (%): ', correct)
#     return correct, pred
    return y_test, pred
# LG_pred_score, LG_pred = apply_LG(x_data, y_data)
y_te, LG_pred = apply_LG(x_data, y_data)
LG_pred
# correct = accuracy_score(y_test, pred, normalize = True) * 100.0
#     print('Correct Prediction (%): ', correct)

def apply_svm(x, y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X_train, X_test, y_train, y_test = split_data(x, y)
    
    # Train the model using the training sets
#     model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    model = SVC()
    model.fit(X_train, y_train)
    
    #Predict Output
    pred = model.predict(X_test)
    correct = accuracy_score(y_test, pred, normalize = True) * 100.0
    print('Correct Prediction (%): ', correct)
    return correct, pred
svm_pred_score, svm_pred = apply_svm(x_data, y_data)
svm_pred_score
def apply_decisiontree(x, y):
    from sklearn import tree
    
    X_train, X_test, y_train, y_test = split_data(x, y)
    
    # Train the model using the training sets
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    tree.plot_tree(model)
    
    #Predict Output
    pred = model.predict(X_test)
    correct = accuracy_score(y_test, pred, normalize = True) * 100.0
    print('Correct Prediction (%): ', correct)
    return correct, pred
dt_pred_score, dt_pred = apply_decisiontree(x_data, y_data)
dt_pred_score
plot_data = price_data.tail(50)
plot_data
def apply_randomforest_test(x, y):
    # Create a Random Forest Classifier
    rand_frst_clf = RandomForestClassifier(n_estimators = 200, oob_score = False, criterion = "gini", random_state = 0)
    
    X_train, X_test, y_train, y_test = split_data(x, y)
    # Fit the data to the model
    rand_frst_clf.fit(X_train, y_train)

    # Make predictions
    pred = rand_frst_clf.predict(X_test)
    
    correct = accuracy_score(y_test, pred, normalize = True) * 100.0
    print('Correct Prediction (%): ', correct)
    
    return correct, pred

y11 = price_data.tail(1)[features]
y11
price_data.tail(2)
y11 = x_data.loc[2:2,:]
# rsi=52.182112 low_14=1671.7 high_14=1761.0 k_percent=66.517357 r_percent=-33.482643 MACD=3.448006 MACD_EMA=3.101512 Price_roc=0.018595
y11
price_data.head(15)
z_score, z_pred = apply_randomforest_test(x_data, y_data)
z_score
count_lda_pred = np.zeros(lda_pred.shape)
for i in range (lda_pred.shape[0]):
    count_lda_pred[i] = i 
plt.scatter(lda_pred, count_lda_pred)
plot_data = pd.DataFrame(x_data)
plot_data['Prediction'] = y_data

# sns.relplot(x='Change', y='Prediction',  data=price_data)
# sns.catplot(x='Date', y='Prediction',  data=plot_data)
sns.regplot(x='RSI', y='Prediction',  data=plot_data)
sns.regplot(x='r_percent', y='Prediction',  data=plot_data)

g = sns.FacetGrid(price_data, hue="Prediction", hue_kws={"marker": ["^", "v"]})
# g = sns.FacetGrid(plot_data, col="Prediction", row="Date")
g.map(plt.scatter, "RSI", "r_percent", alpha=.7)
g.add_legend();
sns.pairplot(plot_data, hue='Prediction', palette='OrRd')
g = sns.FacetGrid(plot_data, col="Prediction", height=4, aspect=.5)
g.map(sns.barplot, "RSI", "r_percent");
x_data.loc[2:2,:]
# rsi=52.182112 low_14=1671.7 high_14=1761.0 k_percent=66.517357 r_percent=-33.482643 MACD=3.448006 MACD_EMA=3.101512 Price_roc=0.018595
price_data.tail()