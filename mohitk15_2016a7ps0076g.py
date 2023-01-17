import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

%matplotlib inline
train_data = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

test_data = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
train_data.head()
train_data.info()
train_data.isnull().sum()
train_data.shape
train_data = train_data.fillna(train_data.mean())

test_data = test_data.fillna(train_data.mean())
train_data.isnull().sum()
train_data.isnull().any().any()
train_data=train_data.drop('id',axis=1)
train_data.drop_duplicates(inplace=True)

train_data.shape
# Compute the correlation matrix

corr = train_data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
sns.boxplot(data=train_data,x='rating',y='feature1')
sns.boxplot(data=train_data,x='rating',y='feature2')
sns.boxplot(data=train_data,x='rating',y='feature3')
sns.boxplot(data=train_data,x='rating',y='feature4')
sns.boxplot(data=train_data,x='rating',y='feature5')
sns.boxplot(data=train_data,x='rating',y='feature6')
sns.boxplot(data=train_data,x='rating',y='feature7')
sns.boxplot(data=train_data,x='rating',y='feature8')
sns.boxplot(data=train_data,x='rating',y='feature9')
sns.boxplot(data=train_data,x='rating',y='feature10')
sns.boxplot(data=train_data,x='rating',y='feature11')
sns.boxplot(data=train_data,y='rating',x='type')
test_data = test_data.fillna(train_data.mean())

test_data.isnull().sum()
df = train_data
features = ['feature3','feature5','feature6','type','feature9','feature1','feature2','feature10','feature3','feature6','feature8']

features1 = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','type','feature10','feature11']

features = features1.copy()

num_features = features.copy()

num_features.remove('type')

num_features_all = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']
#for knowing which feature is best for predicting

import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X_be = df[num_features_all]  #independent columns

Y_be = df['rating']    #target column i.e price range

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=11)

fit = bestfeatures.fit(X_be,Y_be)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_be.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(11,'Score'))  #print 10 best features
class_weights = {0:2.5,  1:1.5,  2:3.5,  4:6, 5:2}
# removing outliers, but error increased after doing this so commented

# for i in num_features:

#     Q1 = df[i].quantile(0.25)

#     Q3 = df[i].quantile(0.75)

#     IQR = Q3 - Q1

#     filter = (df[i] >= Q1 - 1.5 * IQR) & (df[i] <= Q3 + 1.5 *IQR)

#     df=df.loc[filter]
X = df[features].copy()

Y = df['rating'].copy()

X_test = test_data[features].copy()

X_id = test_data['id'].copy()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X[num_features])

X_test_scaled = scaler.fit_transform(X_test[num_features])
# Hot encoding

X_encoded = pd.get_dummies(X['type'])

X_test_encoded = pd.get_dummies(X_test['type'])
# X_encoded
X_new = np.concatenate([X_scaled,X_encoded.values],axis=1)

X_test_new =  np.concatenate([X_test_scaled,X_test_encoded.values],axis=1)

X_new
X_test = X_test_new

X_test.shape
from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(X_new,Y,test_size=0.2,random_state=0,stratify=Y)
Y_train.value_counts(),Y_val.value_counts()
# from sklearn.model_selection import validation_curve

# n_est = [10,20,30,40,50,60,70,80,90,100]

# degree = np.arange(0, 21)

# train_score, val_score = validation_curve(RandomForestClassifier(), X=X_train, y=Y_train,param_name = 'min_samples_split', 

#                                 param_range = n_est, cv = 3)



# plt.plot(train_score, color='blue', label='training score')

# plt.plot(val_score, color='red', label='validation score')

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=40,weights='distance',algorithm='brute').fit(X_train, Y_train)

mean_squared_error(Y_val, neigh.predict(X_val))**(0.5)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)

clf.fit(X_train, Y_train)  

mean_squared_error(Y_val, clf.predict(X_val))**(0.5)
from sklearn.linear_model import LinearRegression

reg_lr = LinearRegression().fit(X_train,Y_train)

Y_pred_lr = reg_lr.predict(X_val)

mean_squared_error(Y_val, Y_pred_lr)**(0.5)
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(random_state=100,max_depth=20,n_estimators=1400,class_weight=class_weights).fit(X_train,Y_train)

Y_pred_2 = clf2.predict(X_val)

mean_squared_error(Y_val, Y_pred_2)**(0.5)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)

clf.score(X_val, Y_val) 
# using result of one model as a feature for another

from sklearn.ensemble import RandomForestRegressor

clf2 = RandomForestRegressor(random_state=100,max_depth=40,n_estimators=2000).fit(X_train,Y_train)

Y_pred_2 = clf2.predict(X_val).round()

mean_squared_error(Y_val, Y_pred_2)**(0.5)



# clf2 = KNeighborsClassifier(n_neighbors=40,weights='distance').fit(X_train, Y_train)

# Y_pred_2 = clf2.predict(X_val)

# mean_squared_error(Y_val, neigh.predict(X_val))**(0.5)



# clf2 = LinearRegression().fit(X_train,Y_train)

# Y_pred_2 = reg_lr.predict(X_val)

# mean_squared_error(Y_val, Y_pred_lr)**(0.5)
Pred = pd.DataFrame(Y_pred_2,columns=['r'])

Y_ok = pd.concat([pd.DataFrame(X_val),Pred],axis=1)
X_train_ok = pd.concat([pd.DataFrame(X_train),pd.DataFrame(clf2.predict(X_train))],axis=1)

clf2 = RandomForestRegressor(random_state=100,max_depth=20,n_estimators=1400).fit(X_train_ok,Y_train)
Y_pred_3 = clf2.predict(Y_ok).round()

mean_squared_error(Y_val, Y_pred_3)**(0.5)
Y_out = pd.DataFrame(Y_pred_3,columns=['rating'])

Y_out['rating'].value_counts()
####test

# clf2 = ExtraTreesRegressor(random_state=100,n_estimators=2000).fit(X_new,Y)

# Y_pred_2 = clf2.predict(X_test)

# Pred = pd.DataFrame(Y_pred_2,columns=['r'])

# Y_ok = pd.concat([pd.DataFrame(X_test),Pred],axis=1)

# X_train_ok = pd.concat([pd.DataFrame(X_new),pd.DataFrame(clf2.predict(X_new))],axis=1)

# clf2 = ExtraTreesRegressor(random_state=100,n_estimators=2000).fit(X_train_ok,Y)

# Y_pred_3 = clf2.predict(Y_ok).round()
# Y_out = pd.DataFrame(Y_pred_3,columns=['rating'])

# Y_out = pd.concat([X_id,Y_out],axis=1)

# Y_out['rating'].value_counts()
# clf2 = RandomForestRegressor(random_state=0,n_estimators=2000).fit(X_train,Y_train)

# Y_pred_2 = clf2.predict(X_val).round()

# mean_squared_error(Y_val, Y_pred_2)**(0.5)
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestClassifier

clf2 = ExtraTreesRegressor(random_state=100,n_estimators=2000).fit(X_train,Y_train)

Y_pred_2 = clf2.predict(X_val).round()

mean_squared_error(Y_val, Y_pred_2)**(0.5)
clf2 = ExtraTreesRegressor(random_state=100,n_estimators=2000).fit(X_new,Y)

Y_pred_2 = clf2.predict(X_test).round()
Y_out = pd.DataFrame(Y_pred_2,columns=['rating'])
Y_out = pd.concat([X_id,Y_out],axis=1)
Y_out['rating'].value_counts()
Y_out.to_csv('output15.csv',index=False)
clf2 = ExtraTreesRegressor(random_state=100,n_estimators=2500).fit(X_train,Y_train)

Y_pred_2 = clf2.predict(X_val).round()

mean_squared_error(Y_val, Y_pred_2)**(0.5)
clf2 = ExtraTreesRegressor(random_state=100,n_estimators=2500).fit(X_new,Y)

Y_pred_2 = clf2.predict(X_test).round()
Y_out = pd.DataFrame(Y_pred_2,columns=['rating'])

Y_out = pd.concat([X_id,Y_out],axis=1)

Y_out['rating'].value_counts()
Y_out.to_csv('output13.csv',index=False)
# from tqdm import tqdm_notebook as tqdm

# best_score = 1

# best_clf = RandomForestClassifier()

# for estimators in tqdm([100,500,1000,1200,1300,1400,1500,1600,1700]):

#     for depth in tqdm([10,20,40,50,60,70]):

#         for random_state in tqdm([0,4,16,42,50,65,100,90,5,2,16,32,64]):

#             clf = RandomForestClassifier(n_estimators = estimators, max_depth = depth,random_state = random_state)

#             score = np.sqrt(mean_squared_error(Y_val,clf.fit(X_train, Y_train).predict(X_val).round(0)))

#             if(score<best_score):

#                 best_clf = clf

#                 best_score = score            