import pandas as pd
df=pd.read_csv('../input/cardekho-dataset/car data.csv')
df.head()
df.shape
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
#missing values or null values
df.isnull().sum()
df.describe()
df.columns
# Removed Car name as it will not be much helpful due to the variant and object data 
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()
final_dataset['current_year']=2020
final_dataset.head()
final_dataset['no_year']=final_dataset['current_year']-final_dataset['Year']
final_dataset.head()
#dropping non required col
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.drop(['current_year'],axis=1,inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.corr()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(final_dataset)
corrmat = final_dataset.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(20,20))
#plotting heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')
## independent and dependent feature
x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
y.head()
#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization 
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh') 
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.head()
x_train.shape
from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()
import numpy as np
#hyperparameter
n_estimators=[int(x)for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', 
                               n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(x_train,y_train)
prediction = rf_random.predict(x_test)
prediction
sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)
