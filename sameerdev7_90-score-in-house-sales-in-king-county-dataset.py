#importng libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization
from matplotlib import pyplot as plt #data visualization
%matplotlib inline
sns.set_style("darkgrid")

import warnings
warnings.filterwarnings("ignore")


#importing data and segregating columns using the 'names' parameter
house = pd.read_csv("../input/kc_house_data.csv")

#displaying the dataset
house.tail(10)
set(house['view'])
#describing the dataset
house.describe()
# Looking for nulls
print(house.isnull().any())
# Inspecting type
print(house.dtypes)
# Dropping the id and date columns
house = house.drop(['id', 'date'],axis=1)
#creating a pairplot for data visualization of various features
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(house[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=5)
g.set(xticklabels=[]);
#plotting a correlation matrix
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in house.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = house.columns.difference(str_list) 
# Create Dataframe containing only numerical features
house_num = house[num_list]
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)
#seperating features and target
x=house.iloc[:,1:].values
y=house['price']

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split,cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
#using the LightGBM algorithm as the model
import lightgbm as lgb
params={'objective':'regression',
        'metric':'mae'}
reg_lgm = lgb.LGBMRegressor(learning_rate=0.05,**params, n_estimators=1000)
reg_lgm.fit(x_train, y_train)
#checking score of the model
reg_lgm.score(x_test,y_test)
#viewing feature importances
reg_lgm.feature_importances_
#predicting the test set result
y_pred_lgm=reg_lgm.predict(x_test)
#checking the cross val score of the model
results = cross_val_score(reg_lgm, x_train, y_train, cv=5, n_jobs=-1)
results.mean()
results.std()