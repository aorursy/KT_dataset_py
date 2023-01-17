!pip install cufflinks plotly 

!pip install plotly

!pip install chart_studio 
import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np     # linear algebra

import seaborn as sb

import copy



import plotly

import plotly.express as px

import chart_studio.plotly as py

import matplotlib.pyplot as plt
import chart_studio

chart_studio.tools.set_credentials_file(username='cavanferns', api_key='wvTuwmntu6XCYiSG2pKx')
edgr_data = pd.read_csv("../input/Edible_grains.csv") 

print(edgr_data.shape)
edgr_data.head(5)
edgr_data.info()
edgr_data.columns
# To find if there are negative values in the dataset:



num = edgr_data._get_numeric_data()

np.sum((num < 0).values.ravel())
# Convert negative values to 0:



num[num < 0] = 0

# num.head(10)
# Update the existing database with the new one(removed the negative values and replaced with 0)



edgr_data.update(num)



num = edgr_data._get_numeric_data()

np.sum((num < 0).values.ravel())



# edgr_data.head(10)
edgr_data.head(10)
edgr_data.Variety.value_counts()
cool = edgr_data.Variety.str.contains('C')
cool1 = len(edgr_data[cool])

hot = (len(edgr_data['Variety'])) - len(edgr_data[cool])
import plotly.graph_objects as go



labels = ['Hot Cereals', 'Cool Cereals']

values = [3,74]



# pull is given as a fraction of the pie radius

fig = go.Figure(data = [go.Pie(labels = labels, values = values, pull = [0.2, 0])])

fig.update(layout_title_text='Hot v/s Cool Cereals')

fig.show()
product_count = pd.DataFrame(edgr_data['Producer'].value_counts(dropna = False).reset_index())

product_count.columns = ['Producer', 'Number of Products']

# product_count

# edgr_data['Producer'].value_counts()



product_count["Producer"].replace({"K": "Kellogs", "G": "Green Light Foods",

                                  "P":"Periyar Foods", "R":"Ran Impex Inc",

                                  "Q":"Quaker Oats", "N":"Nestle Products",

                                  "A":"Agra Foods"}, inplace=True)



product_count
# Visualization of number of products sold per Producer:



import plotly.graph_objects as go



x = product_count['Producer']

y = product_count['Number of Products']



# Use the hovertext kw argument for hover text

fig = go.Figure(data = [go.Bar(x = x, y = y, hovertext = ['Kellogs sells 23 products', 

                                                          'Green Light Foods sells 22 products',

                                                         'Periyar Food Products sells 9 products',

                                                         'Ran Impex Inc sells 8 products',

                                                         'Quaker Oats sells 8 products',

                                                         'Nestle Products sells 6 products',

                                                         'Agra Food Products sells 1 product'], )])



# Customize aspect

fig.update_traces(marker_color='RGB(163,102,210)', marker_line_color='RGB(170,73,195)', marker_line_width=0.5, opacity=0.7)

fig.update_layout(title_text = 'Number of products sold per Producer')

fig.show()
producer_rating = edgr_data.groupby('Producer')['ratings'].mean().reset_index()

producer_rating.columns = ['Producer', 'Average Rating']

# producer_rating

producer_rating["Producer"].replace({"K": "Kellogs", "G": "Green Light Foods",

                                  "P":"Periyar Foods", "R":"Ran Impex Inc",

                                  "Q":"Quaker Oats", "N":"Nestle Products",

                                  "A":"Agra Foods"}, inplace=True)

producer_rating
# Visualization of product ranking:



fig = px.bar(producer_rating, x = 'Producer', y = 'Average Rating')

fig.show()
edgr_data['Rank'] = edgr_data['ratings'].rank(ascending = False) 

edgr_data = edgr_data.set_index('Rank')

# edgr_data.head(10)

edgr_data = edgr_data.sort_index()

edgr_data.head(10)
# Visualization of product ranking



edgr_data = edgr_data.sort_values(['ratings'],ascending = False).reset_index(drop = True)

plt.figure(figsize = (20,26))

sb.barplot(x = edgr_data["ratings"], y = edgr_data["grain_name"])

plt.xlabel("Ratings", fontsize = 15)

plt.ylabel("Product names", fontsize = 15)

plt.title("Product Ratings", fontsize = 20)

plt.show()
# Compute the correlation matrix

corr = edgr_data.iloc[:,~edgr_data.columns.isin(['Rank','name','producer','Variety','weight', 'cups'])].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize = (16, 12))



# Generate a custom diverging colormap

cmap = sb.diverging_palette(220, 10, as_cmap = True)



# Draw the heatmap with the mask and correct aspect ratio

sb.heatmap(corr, mask = mask, cmap = cmap, vmax =.3, center = 0,

            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
health_edgr_data = edgr_data
# creating a new Dataframe with a new column: 'Healthy : (Y/N)' :



health_edgr_data['Healthy'] = np.where((edgr_data['sugars_content'] <= 5) 

                               & (edgr_data['sodium_content'] <= 0.3) 

                               & (edgr_data['fat_content'] <= 3)  

                               &(edgr_data['calories_content'] >= 50), 'Y','N')



health_edgr_data['Healthy'].value_counts()
health_edgr_data['Rank'] = health_edgr_data['ratings'].rank(ascending = False) 

health_edgr_data = health_edgr_data.set_index('Rank')

# edgr_data.head(10)

health_edgr_data = health_edgr_data.sort_index()

# health_edgr_data.head()

health_edgr_data[['grain_name', 'ratings', 'Healthy']]
cool_healthy = health_edgr_data



# choosing rows based on 'cool' Variety 

cool = cool_healthy["Variety"] == "C"

  

# choosing rows based on healthy option

healthy = cool_healthy["Healthy"] == "Y"

  

# filtering data on basis of both filters 

cool_healthy.where(cool & healthy, inplace = True) 

cool_healthy = cool_healthy.dropna()

# display



cool_healthy[['grain_name', 'Variety', 'ratings', 'Healthy']]
edgr_data.drop(columns=['Rank'])
# %matplotlib inline



# edgr_data2 = edgr_data

# notnorm = edgr_data2.drop(columns=['grain_name', 'Producer', 'Variety', 'Healthy'])



# plt.figure(figsize=(16,6))

# plt.ylim(0,0.70)



# sb.kdeplot(notnorm['calories_content'])

# sb.kdeplot(notnorm['protein_content']) 

# sb.kdeplot(notnorm['fat_content'])

# sb.kdeplot(notnorm['sodium_content'])

# sb.kdeplot(notnorm['fiber_content'])

# sb.kdeplot(notnorm['hydrated_carbon'])

# sb.kdeplot(notnorm['sugars_content'])

# sb.kdeplot(notnorm['potassium_content'])

# sb.kdeplot(notnorm['vit_&_min'])
#For all coulmns find outliers



def outlierCount(data):    #function to count the outliers in a data frame

    dataNum = data._get_numeric_data()

    outlierCountDF = pd.DataFrame()

    for (columnName, columnData) in dataNum.iteritems():

        q1 = columnData.quantile(0.25)

        q3 = columnData.quantile(0.75)

        iqr = q3 - q1

        UB = q3 + 1.5 * iqr

        LB = q1 - 1.5 * iqr

        outlierCount = columnData[~columnData.between(LB,UB)].count()

        outlierCountDF = outlierCountDF.append([[columnName,outlierCount]],ignore_index = True)

    outlierCountDF.columns = ['Variable','Outlier Count']

    return outlierCountDF



outlierCount(edgr_data)    #function call passing the data
from plotly.offline import init_notebook_mode, iplot



trace0 = go.Box(y = edgr_data['calories_content'], name = 'calories', marker = dict(color = 'rgb(214, 12, 140)'))

trace1 = go.Box(y = edgr_data['protein_content'], name = 'protein', marker = dict(color = 'RGB(255,101,80)'))

trace2 = go.Box(y = edgr_data['sodium_content'], name = 'sodium', marker = dict(color = 'RGB(255,169,80)'))

trace3 = go.Box(y = edgr_data['fiber_content'], name = 'fiber', marker = dict(color = 'RGB(111,169,80)'))

trace4 = go.Box(y = edgr_data['hydrated_carbon'], name = 'hydrated carbon', marker = dict(color = 'RGB(111,169,177)'))

trace5 = go.Box(y = edgr_data['potassium_content'], name = 'potassiun', marker = dict(color = 'RGB(111,59,177)'))

trace6 = go.Box(y = edgr_data['vit_&_min'], name = 'vit and min', marker = dict(color = 'RGB(215,118,92)'))

trace7 = go.Box(y = edgr_data['weight'], name = 'weight', marker = dict(color = 'RGB(37,134,221)'))

                      

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]

layout = go.Layout(title = "Visualization of Outliers:", plot_bgcolor= 'rgba(0, 0, 0, 0.20)')



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
edgr_data_num = edgr_data._get_numeric_data()
# function to cap the outliers:



def outliercap(data):

    data_num = data._get_numeric_data()

    for column in data_num:            

        q1 = data_num[column].quantile(0.25)    #q1

        q3 = data_num[column].quantile(0.75)    #q3

        iqr = q3 - q1

        ub = q3 + 1.5 * iqr

        lb = q1 - 1.5 * iqr

        

        data_num[column] = data_num[column].replace(data_num[data_num[column] > ub][column], q3)

        data_num[column] = data_num[column].replace(data_num[data_num[column] < lb][column], q1)

    return data_num



# market_data2 = copy.deepcopy(market_data)

capped_outliers = outliercap(edgr_data)

# capped_outliers.describe()
# Visualization of Outliers after Capping the values:



trace0 = go.Box(y = capped_outliers['calories_content'], name = 'calories', marker = dict(color = 'rgb(214, 12, 140)'))

trace1 = go.Box(y = capped_outliers['protein_content'], name = 'protein', marker = dict(color = 'RGB(255,101,80)'))

trace2 = go.Box(y = capped_outliers['sodium_content'], name = 'sodium', marker = dict(color = 'RGB(255,169,80)'))

trace3 = go.Box(y = capped_outliers['fiber_content'], name = 'fiber', marker = dict(color = 'RGB(111,169,80)'))

trace4 = go.Box(y = capped_outliers['hydrated_carbon'], name = 'hydrated carbon', marker = dict(color = 'RGB(111,169,177)'))

trace5 = go.Box(y = capped_outliers['potassium_content'], name = 'potassiun', marker = dict(color = 'RGB(111,59,177)'))

trace6 = go.Box(y = capped_outliers['vit_&_min'], name = 'vit and min', marker = dict(color = 'RGB(215,118,92)'))

trace7 = go.Box(y = capped_outliers['weight'], name = 'weight', marker = dict(color = 'RGB(37,134,221)'))

                      

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]

layout = go.Layout(title = "Visualization of Outliers after capping:", plot_bgcolor= 'rgba(0, 0, 0, 0.20)')



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
capped_outliers.shape
# Update and create a new dataframe with the capped outliers:



edgr_data_capped = edgr_data.drop(columns = ['Healthy','Rank'])

edgr_data_capped.update(capped_outliers)
mlm_data = edgr_data_capped._get_numeric_data()
from sklearn.preprocessing import Normalizer



norm = Normalizer()



norm_data = norm.fit_transform(mlm_data)

norm_data = pd.DataFrame(norm_data)

norm_data.columns = mlm_data.columns

norm_data.head()
norm_data.describe()
# Visualization of Normalised data:



plt.figure(figsize=(20,6))





sb.kdeplot(norm_data['calories_content'])

sb.kdeplot(norm_data['protein_content']) 

sb.kdeplot(norm_data['fat_content'])

sb.kdeplot(norm_data['sodium_content'])

sb.kdeplot(norm_data['fiber_content'])

sb.kdeplot(norm_data['hydrated_carbon'])

sb.kdeplot(norm_data['sugars_content']) 

sb.kdeplot(norm_data['potassium_content'])

sb.kdeplot(norm_data['vit_&_min'])

sb.kdeplot(norm_data['fiber_content'])

sb.kdeplot(norm_data['ratings'])
# Using Linear Regression:



from sklearn.model_selection import train_test_split   #for spliting the dataset

from sklearn.linear_model import LinearRegression   #for linear regression
norm_data.head(3)
X = norm_data.drop('ratings', axis = 1)

Y = norm_data.ratings
print(X.shape)

print(Y.shape)
# r2 for test data:



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



#Model Intialization

reg = LinearRegression()



#Data Fitting

reg = reg.fit(X_train, Y_train)

print('Coefficients: ', reg.coef_)

print('Intercept: ', reg.intercept_)
Y_pred = reg.predict(X_test)     #for test set



# Y_pred = reg.predict(X_train)       #for training set
# Model Evaluation:



from sklearn.metrics import r2_score, mean_squared_error



rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))    #for test set



# rmse = np.sqrt(mean_squared_error(Y_train, Y_pred))       #for training set



r2 = r2_score(Y_test, Y_pred)       #for test set



# r2 = r2_score(Y_train, Y_pred)      #for training set



print('RMSE = ', rmse)

print('R2 Score = ', r2*100)
#residual plot



x = [i for i in range(1, len(Y_pred) + 1)]

x_plot = plt.scatter(x, (Y_pred - Y_test), c = 'b')

plt.plot(x, [0]*len(Y_pred), c = 'r')

plt.title('Residual Plot')