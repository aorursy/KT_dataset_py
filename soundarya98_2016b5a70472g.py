import pandas as pd

import numpy as np

import sklearn as skl

import seaborn as sns

import matplotlib.pyplot as plt

df=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

# df3=df.replace(to_replace='old', value=0, inplace=False)

# df3=df3.replace(to_replace='new', value=1, inplace=False)

# df3

t=[]

t = pd.get_dummies(df['type'])

t

df['type2']=t['new']

df['type2-2']=t['old']
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 20))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
features=['id', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'type2', 'type2-2']

test=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

t=[]

t = pd.get_dummies(test['type'])

test['type2']=t['new']

test['type2-2']=t['old']



test.fillna(value=test.mean(),inplace=True)

df.fillna(value=df.mean(),inplace=True)



# test3=test.replace(to_replace='old', value=0, inplace=False)

# test3=test3.replace(to_replace='new', value=1, inplace=False)
# from scipy import stats

# import numpy as np

# z = np.abs(stats.zscore(df[features]))

# df_without_outliers = df[(z <3.8).all(axis=1)]
# df[z>5]
# df_without_outliers.shape
# df.describe()
# df_without_outliers.info()
# df_without_outliers_mean=df_without_outliers[df_without_outliers.columns.difference(['rating', 'type'])]

# test2=(test.loc[:, test.columns != 'type'])

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# X_train = scaler.fit_transform(df_without_outliers_mean)

# X_test = scaler.transform(test2)



# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# X_train = scaler.fit_transform(df_without_outliers[features])

# X_test = scaler.transform(test[features])



# from sklearn.preprocessing import RobustScaler

# scaler = RobustScaler()

# X_train = scaler.fit_transform(df_without_outliers[features])

# X_test = scaler.transform(test[features])
X_train=df[features]

Y_train=df['rating']

X_test=test[features]
# def score(x):

#     for i in range(len(x)):

#         if(x[i]<0.5):

#             x[i]=0

#         elif(x[i]<1.5):

#             x[i]=1

#         elif(x[i]<2.5):

#             x[i]=2

#         elif(x[i]<3.5):

#             x[i]=3

#         elif(x[i]<4.5):

#             x[i]=4

#         elif(x[i]<5.5):

#             x[i]=5

#         else:

#             x[i]=6

#     return x
from sklearn.ensemble import ExtraTreesRegressor

# Create the model with 150 trees

model = ExtraTreesRegressor(n_estimators=2000, n_jobs=-1)

# Fit on training data

model.fit(X_train, Y_train)
rf_predictions = model.predict(X_test)

rf_predictions=np.round(rf_predictions).astype(int)

rf_predictions=pd.DataFrame(data=rf_predictions)

# # np.round(answer['rating'])

answer=pd.concat([test['id'], rf_predictions], axis=1)

answer.columns=['id', 'rating']



answer.to_csv('answer_eval_lab.csv', index=False)

answer
rf_predictions
# num_est=[100, 200, 300]

# from sklearn.model_selection import validation_curve

# train_scoreNum, test_scoreNum = validation_curve(

#                                 RandomForestClassifier(),

#                                 X = X_train, y = df['rating'], 

#                                 param_name = 'n_estimators', 

#                                 param_range = num_est, cv = 3)
# train_mean = np.mean(train_scoreNum, axis=1)

# train_std = np.std(train_scoreNum, axis=1)



# test_mean = np.mean(test_scoreNum, axis=1)

# test_std = np.std(test_scoreNum, axis=1)
# plt.plot(num_est, train_mean, label="Training score", color="black")

# plt.plot(num_est, test_mean, label="Cross-validation score", color="dimgrey")
# max_depth=[10, 12, 16]

# train_scoreNum, test_scoreNum = validation_curve(

#                                 RandomForestClassifier(),

#                                 X = X_train, y = df['rating'], 

#                                 param_name = 'max_depth', 

#                                 param_range = num_est, cv = 3)

# train_mean = np.mean(train_scoreNum, axis=1)

# train_std = np.std(train_scoreNum, axis=1)



# test_mean = np.mean(test_scoreNum, axis=1)

# test_std = np.std(test_scoreNum, axis=1)

# plt.plot(max_depth, train_mean, label="Training score", color="black")

# plt.plot(max_depth, test_mean, label="Cross-validation score", color="dimgrey")
# df2=pd.DataFrame(data=X_train[0:,0:],columns=features)

# # selected_feat= df2.columns[(sel.get_support())]

# fi = pd.DataFrame({'feature': list(df2.columns),

#                    'importance': model.feature_importances_}).\

#                     sort_values('importance', ascending = False)



# # Display

# fi
f1=pd.read_csv("answer_eval_lab.csv")
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [100, 150, 200, 250, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150]

# Number of features to consider at every split

max_features = ['log2', 'sqrt']

# Maximum number of levels in tree

max_depth = [10, 12, 15, 20]

max_depth.append(None)



# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'bootstrap': bootstrap}

# pprint(random_grid)

# {'bootstrap': [True, False],

#  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

#  'max_features': ['auto', 'sqrt'],

#  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
# # Use the random grid to search for best hyperparameters

# # First create the base model to tune

# rf = ExtraTreesRegressor()

# # Random search of parameters, using 3 fold cross validation, 

# # search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)

# # Fit the random search model

# rf_random.fit(X_train, Y_train)
rf_random.best_params_