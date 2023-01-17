import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import mean_squared_error,accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df.head()
df.info()
df.isnull().any()

df.fillna(value=df.mean(),inplace=True)

df.isnull().any()

# df['type'].replace("new", 1, inplace=True)

# df['type'].replace("old", 0, inplace=True)
# plt.plot(df.feature6, df.rating)

sns.regplot(x="feature8", y="rating", data=df)
#Compute the correlation matrix



%matplotlib inline

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.heatmap(df.corr())

s=df.corr()

s['rating']
# X=check.drop(['id','type','feature3','feature5','feature7'], axis=1)

X=df[['feature1','feature2','feature3','feature5','feature6','feature9']]

# X=df[['feature6','feature2']]

y=df['rating']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(random_state = 42)



# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

# max_depth.append(None)

# # Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree

# bootstrap = [True, False]

# # Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}
# # Use the random grid to search for best hyperparameters

# # First create the base model to tune

# rf = RandomForestRegressor()

# # Random search of parameters, using 3 fold cross validation, 

# # search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model

# rf_random.fit(X_train, y_train)
# rf_random.best_params_
from sklearn.ensemble import RandomForestClassifier



clf=RandomForestClassifier(n_estimators= 1000,min_samples_split= 2,

 min_samples_leaf= 1,

 max_features= 'log2',

 max_depth= 110,

 bootstrap= True,

 n_jobs=-1)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
ans=mean_squared_error(y_pred, y_test)

ans**0.5
# z=pd.DataFrame()

# z['id']=
# sc=StandardScaler()

# check=sc.fit_transform(X_train)

# X_train=pd.DataFrame(check,columns=['feature1','feature2','feature3','feature5','feature6','feature9','feature10'])

# sns.regplot(x="feature1", y="rating", data=check)

# X_train=(X_train-X_train.mean())/X_train.std()
#Answer Through Linear Regression:0.79



# lr=LinearRegression()

# lr.fit(X_train,y_train)
# Answer Through kNN:0.87



# knn=KNeighborsClassifier(n_neighbors=5)

# knn.fit(X_train,y_train)
#Answer Through Random Forest:0.85



# from sklearn.ensemble import RandomForestClassifier



# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# clf.fit(X_train, y_train)
# from sklearn.grid_search import GridSearchCV

# param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}

# grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2,n_jobs=-1)

# grid.fit(X_test,y_test)
# grid.best_params_
#Answer Through SVM:0.85



# clf=SVC(gamma='auto')

# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)

# ## ans=y_pred.round().astype(int)

# ans=mean_squared_error(y_pred, y_test)

# ans**0.5

# ## np.round_(y_pred,decimals=0)
# accuracy_score(y_pred,y_test)
df2=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

df2.fillna(value=df2.mean(), inplace=True)

X_new=df2[['feature1','feature2','feature3','feature5','feature6','feature9']]

pred=clf.predict(X_new)

# ans=pred.round().astype(int)

pred
final=pd.DataFrame()

final['id']=df2['id']

final['rating']=pred

final.head()

final.to_csv('predictions4.csv',index=False)