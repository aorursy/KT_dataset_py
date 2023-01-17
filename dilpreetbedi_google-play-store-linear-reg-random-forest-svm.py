import numpy as np

import pandas as pd
data= pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
data.head()
data.shape
data.isnull().sum()
data.drop(labels = ['App','Last Updated','Current Ver','Android Ver','Genres'], axis = 1, inplace = True)
data['Type'].fillna(data['Type'].mode()[0], inplace=True)



data['Type']=data['Type'].map(lambda i:0 if i== 'Free' else 1)

# filling missing values of categorical variables with mode



data['Content Rating'].fillna(data['Content Rating'].mode()[0], inplace=True)

# cleaning size of installation

def change_size(size):

    if 'M' in size:

        x = size[:-1]

        x = float(x)*1000000

        return(x)

    elif 'k' == size[-1:]:

        x = size[:-1]

        x = float(x)*1000

        return(x)

    else:

        return None



data["Size"] = data["Size"].map(change_size)



#filling Size which had NA

data.Size.fillna(method = 'ffill', inplace = True)



# converting to Int

data['Size']=data['Size'].astype(int)
# Cleaning Installs



# Removing '+'

data['Installs']=data['Installs'].map(lambda i:i.replace('+','0'))

# Removing "," ( comma )

data['Installs']=data['Installs'].map(lambda i:i.replace(',',''))

# Replacing char with na

data['Installs']=data['Installs'].replace('Free','Nan')

# converting to float

data['Installs']=data['Installs'].apply(lambda x: float(x))

# imputing all the na/missing values using mean

data['Installs'].fillna(data['Installs'].mean(), inplace=True)

# converting to Interger

data['Installs']=data['Installs'].apply(lambda x: int(x))
data['Content Rating'] = data['Content Rating'].map({'Everyone': 4, 'Everyone 10+': 4, 'Teen': 3, 'Mature 17+': 3,

       'Adults only 18+' : 2, 'Unrated' : 2})

#Price



# Removing '$'

data['Price']=data['Price'].map(lambda i:i.replace('$',''))

# replacing char with na

data['Price']=data['Price'].replace('Everyone','Nan')

# converting to float

data['Price']=data['Price'].apply(lambda x: float(x))

# Imputing the missing value

data['Price'].fillna(data['Price'].mean(), inplace=True)

# converting to integer



data['Price']=data['Price'].astype(int)

# Reviews



#replacing M with zeros

data['Reviews']=data['Reviews'].map(lambda i:i.replace('.0M','000000'))

# Converting to intergers

data['Reviews']=data['Reviews'].apply(lambda x: int(x))

# Cleaning Rating



def rating_clean(Rating):

    if Rating > 5:

        return 5

    else:

        return Rating



data['Rating'] = data['Rating'].map(rating_clean).astype(float)





data['Rating'].fillna(data['Rating'].mean(), inplace=True)

                                                                   
 #converting Categorical column 

data2=pd.get_dummies(data, columns=['Category'], drop_first=True)
# **Normalization**



## *** formula is norm = ((value - minimum Value)/ (Maximum value - Minimum Value))



#data['Reviews']=(data['Reviews']-data['Reviews'].min())/(data['Reviews'].max()-data['Reviews'].min())

#data['Size']=(data['Size']-data['Size'].min())/(data['Size'].max()-data['Size'].min())

#data['Installs']=(data['Installs']-data['Installs'].min())/(data['Installs'].max()-data['Installs'].min())

#data['Type']=(data['Type']-data['Type'].min())/(data['Type'].max()-data['Type'].min())

#data['Price']=(data['Price']-data['Price'].min())/(data['Price'].max()-data['Price'].min())
x1=data2[['Reviews','Size','Installs','Price','Content Rating']]

x2=data2.drop(labels=['Reviews','Size','Installs','Price','Content Rating'],axis=1)



from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(x1[['Reviews','Size','Installs','Price','Content Rating']])

new = std_scale.transform(x1[['Reviews','Size','Installs','Price','Content Rating']])



new = pd.DataFrame(data=new, columns=['Reviews','Size','Installs','Price','Content Rating'])



nwdata=pd.concat([new,x2],axis=1)

nwdata.dtypes
nwdata.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.pairplot(data)


plt.hist(nwdata['Rating'], bins=10)

plt.show()



plt.hist(nwdata['Reviews'], bins=10)

plt.show()



plt.hist(nwdata['Size'], bins=10)

plt.show()



plt.hist(nwdata['Installs'], bins=10)

plt.show()



plt.hist(nwdata['Type'], bins=10)

plt.show()



plt.hist(nwdata['Price'], bins=10)

plt.show()



plt.hist(nwdata['Content Rating'], bins=10)

plt.show()



f, ax = plt.subplots(figsize=(15, 20))

sns.countplot(y="Category", data=data);
import sklearn

from sklearn import metrics

from sklearn.model_selection import train_test_split
X = nwdata.drop(labels = ['Rating'],axis = 1)

y = nwdata.Rating

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30)
### Feature Importance



from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

model = ExtraTreesRegressor()

model.fit(X,y)





#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
# Extracting Important features



x=X[['Reviews','Size','Installs', 'Price', 'Content Rating','Category_FAMILY', 'Category_MEDICAL', 'Category_LIFESTYLE', 'Category_HEALTH_AND_FITNESS', 'Category_FINANCE']]

# Splitting the data with new x 



# In above x we have not consider Ratings, so we can split directly 



x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.30)

from sklearn.linear_model import LinearRegression 



model = LinearRegression()

model.fit(x_train,y_train)

Results_x = model.predict(x_test)
# MAE

print(metrics.mean_absolute_error(y_test,Results_x))



#MSE

print(metrics.mean_squared_error(y_test,Results_x))



#RSME

print(np.sqrt(metrics.mean_squared_error(y_test,Results_x)))

plt.scatter(y_test, Results_x)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()



n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

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
# Create the random grid

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

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(x_train,y_train)
rf_random.best_params_
predictions=rf_random.predict(x_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

from sklearn import svm
model2 = svm.SVR()

model2.fit(x_train,y_train)



svm_pred = model2.predict(x_test)
sns.distplot(y_test-svm_pred)
plt.scatter(y_test,svm_pred)
# MAE

print('MAE:',metrics.mean_absolute_error(y_test,svm_pred))



#MSE

print('MSE:',metrics.mean_squared_error(y_test,svm_pred))





#RSME

print('RSME:',np.sqrt(metrics.mean_squared_error(y_test,svm_pred)))





# MAE

print('MAE:',''  'LR:',metrics.mean_absolute_error(y_test,Results_x),'RF:', metrics.mean_absolute_error(y_test, predictions),'SVM:',metrics.mean_absolute_error(y_test,svm_pred) )



#MSE

print('MSE:',''  'LR:',metrics.mean_squared_error(y_test,Results_x),'RF:', metrics.mean_squared_error(y_test, predictions),'SVM:',metrics.mean_squared_error(y_test,svm_pred) )



#RSME

print('MSE:',''  'LR:',np.sqrt(metrics.mean_squared_error(y_test,Results_x)),'RF:',np.sqrt(metrics.mean_squared_error(y_test, predictions)),'SVM:',np.sqrt(metrics.mean_squared_error(y_test,svm_pred)) )


