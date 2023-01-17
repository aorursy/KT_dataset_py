# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()
df.info(verbose=True)
df.describe(include='all').transpose()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',size='price', data=df)
# import Geopandas

import geopandas as gpd



# import street map

street_map = gpd.read_file('/kaggle/input/nyc-shapefile/geo_export_cd000852-2825-4046-9168-5b498fd4c7ca.shp')
# create figure and axes, assign to subplot

fig, ax = plt.subplots(1,1,figsize=(10,10))



#plotting the price as a function of longitude and latitude

g=df.plot(x='longitude',y='latitude', c='price',kind='scatter',cmap='jet',ax=ax,zorder=5)



# add .shp mapfile to axes

street_map.plot(ax=ax,alpha=0.6,color='black')

plt.show()
sns.distplot(df['price'],kde = False)

plt.show()
#adding a small constant to the price column to prevent log(0) issue and assign a new column

df['log_price'] = np.log(df['price']+1)
sns.distplot(df['log_price'],kde = False)

plt.show()
# create figure and axes, assign to subplot

fig, ax = plt.subplots(1,1,figsize=(10,10))



#plotting the price as a function of longitude and latitude

df.plot(x='longitude',y='latitude', c='log_price',kind='scatter',cmap='jet',ax=ax,zorder=5)



# add .shp mapfile to axes

street_map.plot(ax=ax,alpha=0.6,color='black',legend=True)

plt.show()
plt.figure(figsize=(15,10))



#utilizing Sebaorn boxplot. Categorizing log_price into neighbourhood_group for differnt room types

sns.boxplot(x="room_type", y="log_price",hue="neighbourhood_group",data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#creating bins

bins = [-np.inf, 100, 200, 300, 400, np.inf]



#assigning names to each bin

names = ['0 to 100', '100 to 200', '200 to 300', '300 to 500', '500 and above']



#creating a new column containing bin information

df['num_review_cat'] = pd.cut(df['number_of_reviews'], bins, labels=names)
g=sns.catplot(x="num_review_cat", y="log_price", hue="neighbourhood_group", data=df, kind="violin")

g.fig.set_size_inches(10,8)
corr_matrix=df.corr()

#creating a mask to block the upper triangle of coorelation matrix

mask = np.zeros_like(corr_matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)]= True



#plotting correlation

f, ax = plt.subplots(figsize=(10, 10)) 

heatmap = sns.heatmap(corr_matrix, 

                      mask = mask,

                      square = True,

                      linewidths = .5,

                      cmap = 'coolwarm',

                      cbar_kws = {'shrink': .4, 

                                'ticks' : [-1, -.5, 0, 0.5, 1]},

                      vmin = -1, 

                      vmax = 1,

                      annot = True,

                      annot_kws = {'size': 12})



#add the column names as labels

ax.set_yticklabels(corr_matrix.columns, rotation = 0)

ax.set_xticklabels(corr_matrix.columns)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
sns.regplot(x='longitude',y='log_price',data=df, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
fig,ax=plt.subplots(2,figsize=(9,9))

#Creating a histogram for availablity

sns.distplot(df["availability_365"], kde=False,ax=ax[0])



#Creating a box plot for availablity

sns.boxplot("availability_365",data=df,ax=ax[1])
#first making the df['name'] series lower-case. Then, apply 'str.split' to split each sentence in each row by whitespace.

#After that drop any NaN values in the series and then covert the series to a list.



split_name_list=df['name'].str.lower().str.split().dropna().tolist()
#see the first 10 elements in the list

split_name_list[:10]
flatten_name_list = [val for sublist in split_name_list for val in sublist]
#show the first 10 elements in the flattened list

flatten_name_list[:10]
#importing nltk library to use stopwords

import nltk

#need to first download the stopwords using the follwoing link: nltk.download('stopwords')



from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stop_words
filtered_sentence = [w for w in flatten_name_list if not w in stop_words]
#showing the first 10 elements of the filtered element.

filtered_sentence[0:10]
#import counter from collections package

from collections import Counter

c = Counter(filtered_sentence)

most_common_word=c.most_common(10)

print(dict(most_common_word))

plt.figure(figsize=(10,10))

plt.bar(dict(most_common_word).keys(), dict(most_common_word).values())
df.head()
#getting rid of unncessary columns from dataframe

#we are going to use log_price instead of price

df=df.drop(["price","num_review_cat"],axis=1)
#creating a new dataframe for the hypothesis check

new_df=df[['name','host_name','last_review','log_price']]



#melting the dataframe to easily visualize the above mentioned features

#variables: features, values: corresponding value for each feature. id_vars: log_price

new_df_melt=pd.melt(new_df,id_vars=['log_price'])



#sampling 20% records from dataframe

new_dfSample = new_df_melt.sample(int(len(new_df)*0.2))
plt.figure(figsize=(20,10))

#creating a scatterplot. Hue is for each variable (feature).

g=sns.scatterplot(x="value", y="log_price", hue="variable",data=new_dfSample)

#not showing any xticklables because they are all different for each feature

g.set(xticklabels=[])
df=df.drop(["id","host_id","last_review","host_name","name"],axis=1)
df.head()
#separating labels and predictors

X=df.drop('log_price',axis=1)

y=df['log_price'].values



#splitting train (75%) and test set (25%)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.head()
#Selecting numerical dataframe in train set

X_train_num=X_train.select_dtypes(include=np.number)



#Selecting categorical dataframe in train set

X_train_cat=df.select_dtypes(exclude=['number'])
X_train_num.head()
X_train_cat.head()
#importing necessary modules from sklearn

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler



#creating a pipeline for numerical attribute. Pipeline: median imputer + MinMaxScaler

num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('mm_scaler',MinMaxScaler()),])
#importing label encoder

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



#applying label encoding for categorical features in train set

X_train_cat=X_train_cat.apply(LabelEncoder().fit_transform)
X_train_cat.head()
#importing ColumnTransformer and OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



#separating numerical and categorical attributes in the train set

num_attribs = list (X_train_num)

cat_attribs = list (X_train_cat)



#creating a full pipeline: numerical + categorical

full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),("cat",OneHotEncoder(handle_unknown='ignore'),cat_attribs)])



#fit and transform the train set using the full pipeline

X_train_prepd = full_pipeline.fit_transform(X_train)
X_train_prepd
#importing linear regression model and mean_squared_error metrics

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



#definiing linear regressor

lin_reg=LinearRegression()



#feeding in X_train and y_train for model fitting

lin_reg.fit(X_train_prepd,y_train)



#making predictions on train set

lin_predictions = lin_reg.predict(X_train_prepd)



#getting MSE and RMSE values

lin_mse=mean_squared_error(y_train,lin_predictions)

lin_rmse=np.sqrt(lin_mse)



print("Mean squared error: %.3f" % lin_mse)

print("Root mean squared error: %.3f" % lin_rmse)
#importing cross_val_score to perform k-fold validation

from sklearn.model_selection import cross_val_score



#performing 10-fold validation

scores=cross_val_score(lin_reg, X_train_prepd, y_train, scoring = "neg_mean_squared_error",cv=10)



# skleanr's cross validation expect a utility function, so the greater the better. 

# That is why putting a negative sign

lin_rmse_scores=np.sqrt(-scores)

print("Scores:", lin_rmse_scores)

print("Mean: %.3f" % lin_rmse_scores.mean())

print("STD: %.3f" % lin_rmse_scores.std())
plt.scatter(y_train,lin_predictions, label='Predictions')

plt.plot(y_train,y_train,'r',label='Perfect prediction line')

plt.xlabel("Log_price in the train set")

plt.ylabel("Predicted log_price")

plt.legend()
#importing decision tree model

from sklearn.tree import DecisionTreeRegressor



#defining decision tree regressor

tree_reg=DecisionTreeRegressor()

#feeding X_train and y_train into the regressor

tree_reg.fit(X_train_prepd,y_train)



#making predictions on train set

tree_predictions = tree_reg.predict(X_train_prepd)



#getting MSE and RMSE values

tree_mse=mean_squared_error(y_train,tree_predictions)

tree_rmse=np.sqrt(tree_mse)



print("Mean squared error: %.3f" % tree_mse)

print("Root mean squared error: %.3f" % tree_rmse)
#performing 10-fold validation

scores=cross_val_score(tree_reg, X_train_prepd, y_train, scoring = "neg_mean_squared_error",cv=10)



# skleanr's cross validation expect a utility function, so the greater the better. 

# That is why putting a negative sign

tree_rmse_scores=np.sqrt(-scores)

print("Scores:", tree_rmse_scores)

print("Mean: %.3f" % tree_rmse_scores.mean())

print("STD: %.3f" %  tree_rmse_scores.std())
#adding max_depth of 15 limitation

tree_reg=DecisionTreeRegressor(max_depth=15)

#fitting the train set

tree_reg.fit(X_train_prepd,y_train)



#making predictions

tree_predictions = tree_reg.predict(X_train_prepd)



#getting MSE and RMSE values

tree_mse=mean_squared_error(y_train,tree_predictions)

tree_rmse=np.sqrt(tree_mse)



print("Mean squared error: %.3f" % tree_mse)

print("Root mean squared error: %.3f" % tree_rmse)
plt.scatter(y_train,tree_predictions, label='Predictions')

plt.plot(y_train,y_train,'r',label='Perfect prediction line')

plt.xlabel("Log_price in the train set")

plt.ylabel("Predicted log_price")

plt.legend()
#setting environment path to correctly run graphviz in Jupyter notebook

os.environ["PATH"] += os.pathsep + 'C:/Users/jeong/Anaconda3/Library/bin/graphviz/bin/'
dum_X_train=pd.get_dummies(X_train,columns=cat_attribs)

#skipping the sclaing part for easier understanding.

dum_X_train = dum_X_train.fillna(dum_X_train.median())
#limiting the maximum tree depth to 3

tree_reg_vis=DecisionTreeRegressor(max_depth=3)

#refitting the train set

tree_reg_vis.fit(dum_X_train,y_train)
from sklearn.tree import export_graphviz

from sklearn import tree

import graphviz



#export_graphviz(tree_reg, out_file = "d1.dot",impurity =False, filled=True)

outfile = tree.export_graphviz(tree_reg_vis, out_file = 'tree.dot',feature_names=dum_X_train.columns.tolist())



from subprocess import call
#converting the dot file to png image file

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
#importing random forest regressor

from sklearn.ensemble import RandomForestRegressor



#choose 30 number of trees

forest_reg=RandomForestRegressor(n_estimators = 30, random_state = 42)



#fitting the training set

forest_reg.fit(X_train_prepd,y_train)



#making predictions

forest_predictions=forest_reg.predict(X_train_prepd)



#getting MSE and RMSE values

forest_mse=mean_squared_error(y_train,forest_predictions)

forest_rmse=np.sqrt(forest_mse)



print("Mean squared error: %.3f" % forest_mse)

print("Root mean squared error: %.3f" % forest_rmse)
plt.scatter(y_train,forest_predictions, label='Predictions')

plt.plot(y_train,y_train,'r',label='Perfect prediction line')

plt.xlabel("Log_price in the train set")

plt.ylabel("Predicted log_price")

plt.legend()
#performing 5fold validation

scores=cross_val_score(forest_reg, X_train_prepd, y_train, scoring = "neg_mean_squared_error",cv=5)



# skleanr's cross validation expect a utility function, so the greater the better. 

# That is why putting a negative sign

forest_rmse_scores=np.sqrt(-scores)

print("Scores:", forest_rmse_scores)

print("Mean: %.3f" % forest_rmse_scores.mean())

print("STD: %.3f" % forest_rmse_scores.std())
#to show the pair of tuples in feature_importances

for feature in zip(X_train.columns, forest_reg.feature_importances_):

    print(feature)



feats = {} # a dict to hold feature_name: feature_importance

for feature, importance in zip(X_train.columns, forest_reg.feature_importances_):

    feats[feature] = importance #add the name/value pair 
#creating a dataframe from 'feats' dict. Setting dict keys as an index and renaming name of the importance value column

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Feature importance'})

#sort the importance value by an descending order and create a bar plot

importances.sort_values(by='Feature importance',ascending=False).plot(kind='bar')
#Do NOT fit the test data. Only transform using the pipeline

X_test_prepd=full_pipeline.transform(X_test)
#Making predictions using random forest model trained above

final_predictions = forest_reg.predict(X_test_prepd)



#getting mse and rmse values

final_mse = mean_squared_error(y_test,final_predictions)

final_rmse = np.sqrt(final_mse)



print("Mean squared error: %.3f" % final_mse)

print("Root mean squared error: %.3f" % final_rmse)
plt.scatter(y_test,final_predictions, label='Predictions')

plt.plot(y_test,y_test,'r',label='Perfect prediction line')

plt.xlabel("Log_price in the test set")

plt.ylabel("Predicted log_price")

plt.legend()
#making predictions on the test set using the Decision tree model with pruning

final_predictions_tree = tree_reg.predict(X_test_prepd)



#getting mse and rmse values

final_mse_tree = mean_squared_error(y_test,final_predictions_tree)

final_rmse_tree = np.sqrt(final_mse_tree)



print("Mean squared error: %.3f" % final_mse_tree)

print("Root mean squared error: %.3f" % final_rmse_tree)
plt.scatter(y_test,final_predictions_tree, label='Predictions')

plt.plot(y_test,y_test,'r',label='Perfect prediction line')

plt.xlabel("Log_price in the test set")

plt.ylabel("Predicted log_price")

plt.legend()