# IDEAS FROM

# https://www.kaggle.com/valentinw/simple-data-exploration-and-visualization

# https://www.kaggle.com/iamprateek/submission-to-mercari-price-suggestion-challenge

# https://www.kaggle.com/huguera/mercari-data-analysis

# https://www.kaggle.com/rimunoz/titanic/





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS

import squarify 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



def transform_category_name(category_name):

    try:

        main, sub1, sub2= category_name.split('/')

        return main, sub1, sub2

    except:

        return np.nan, np.nan, np.nan

print('OK')



# load dataset

train = pd.read_csv('../input/train.tsv', sep = "\t")

test = pd.read_csv('../input/test.tsv', sep = "\t")



train = train.sample(frac=0.10, replace=False)

#test  = test.sample(frac=0.25, replace=False)



# Store our ID for easy access

testId = test['test_id']



#train.head(1)

print(train.shape)

print(test.shape)
from sklearn.feature_extraction.text import CountVectorizer



nrow_train = train.shape[0]

nrow_test  = test.shape[0]

print(str(nrow_train) + "-" + str(nrow_test))

################################# ITEM DESCRIPTION #############

text_a = train['item_description'].fillna('NA')

text_b =  test['item_description'].fillna('NA')

text = text_a.append(text_b)



vect = CountVectorizer(max_features = 20,stop_words='english')

dtm = vect.fit_transform(text)

item_dtm = pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()) 



print('item')

################################# NAME #############

text_a = train['name'].fillna('NA')

text_b =  test['name'].fillna('NA')

text = text_a.append(text_b)



vect = CountVectorizer(max_features = 10,stop_words='english')

dtm = vect.fit_transform(text)

name_dtm = pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()) 



print('name')

################################# CATEGORY NAME #############

text_a = train['category_name'].fillna('NA')

text_b =  test['category_name'].fillna('NA')

text = text_a.append(text_b)



vect = CountVectorizer(max_features = 5,stop_words='english')

dtm = vect.fit_transform(text)

cat_dtm = pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()) 



print('cat')

################################# BRAND NAME #############

text_a = train['brand_name'].fillna('NA')

text_b =  test['brand_name'].fillna('NA')

text = text_a.append(text_b)



vect = CountVectorizer(max_features = 5,stop_words='english')

dtm = vect.fit_transform(text)

brand_dtm = pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()) 



print('brand')



text_dtm = item_dtm

print(text_dtm.shape)

text_dtm = pd.concat([text_dtm.reset_index(drop=True), name_dtm.reset_index(drop=True)], axis=1)

print(text_dtm.shape)

text_dtm = pd.concat([text_dtm.reset_index(drop=True), cat_dtm.reset_index(drop=True)], axis=1)

print(text_dtm.shape)

text_dtm = pd.concat([text_dtm.reset_index(drop=True), brand_dtm.reset_index(drop=True)], axis=1)

print(text_dtm.shape)



train_text_dtm = text_dtm.iloc[:nrow_train,].reset_index(drop=True)

test_text_dtm  = text_dtm.iloc[nrow_train:,].reset_index(drop=True)



nrow_train1 = train_text_dtm.shape[0]

nrow_test1  = test_text_dtm.shape[0]

print(str(nrow_train1) + "-" + str(nrow_test1))

print(train_text_dtm.info())

print(test_text_dtm.info())
np.sum(train_text_dtm)
def if_null(row):

    if row == row:

        return 1

    else:

        return 0

print('OK')
full_data = [train, test]

for dataset in full_data:

    dataset['category_main'], dataset['category_sub1'], dataset['category_sub2'] = zip(*dataset['category_name'].apply(transform_category_name))

    dataset['item_description_len'] = dataset['item_description'].str.len()

    dataset['name_len'] = dataset['name'].str.len()

    

    dataset['has_descrip'] = 1

    dataset.loc[dataset.item_description=='No description yet', 'has_descrip'] = 0

    dataset['has_brand'] = dataset.brand_name.apply(lambda row : if_null(row))

    

    dataset['contains_brand_new'] = dataset['item_description'].str.contains("Brand New")

    dataset['contains_brand_new'] = dataset['contains_brand_new'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['contains_free_shipping'] = dataset['item_description'].str.contains("free shipping")

    dataset['contains_free_shipping'] = dataset['contains_free_shipping'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['contains_price_firm'] = dataset['item_description'].str.contains("Price firm")

    dataset['contains_price_firm'] = dataset['contains_price_firm'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['contains_rm'] = dataset['item_description'].str.contains("[rm]")

    dataset['contains_rm'] = dataset['contains_rm'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['is_Adidas'] = dataset['brand_name'].str.contains("Adidas")

    dataset['is_Adidas'] = dataset['is_Adidas'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_American_Eagle'] = dataset['brand_name'].str.contains("American Eagle")

    dataset['is_American_Eagle'] = dataset['is_American_Eagle'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Apple'] = dataset['brand_name'].str.contains("Apple")

    dataset['is_Apple'] = dataset['is_Apple'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Bath__Body_Works'] = dataset['brand_name'].str.contains("Bath & Body Works")

    dataset['is_Bath__Body_Works'] = dataset['is_Bath__Body_Works'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Coach'] = dataset['brand_name'].str.contains("Coach")

    dataset['is_Coach'] = dataset['is_Coach'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Disney'] = dataset['brand_name'].str.contains("Disney")

    dataset['is_Disney'] = dataset['is_Disney'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_FOREVER_21'] = dataset['brand_name'].str.contains("FOREVER 21")

    dataset['is_FOREVER_21'] = dataset['is_FOREVER_21'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Funko'] = dataset['brand_name'].str.contains("Funko")

    dataset['is_Funko'] = dataset['is_Funko'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_LuLaRoe'] = dataset['brand_name'].str.contains("LuLaRoe")

    dataset['is_LuLaRoe'] = dataset['is_LuLaRoe'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Lululemon'] = dataset['brand_name'].str.contains("Lululemon")

    dataset['is_Lululemon'] = dataset['is_Lululemon'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Michael_Kors'] = dataset['brand_name'].str.contains("Michael Kors")

    dataset['is_Michael_Kors'] = dataset['is_Michael_Kors'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Nike'] = dataset['brand_name'].str.contains("Nike")

    dataset['is_Nike'] = dataset['is_Nike'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Nintendo'] = dataset['brand_name'].str.contains("Nintendo")

    dataset['is_Nintendo'] = dataset['is_Nintendo'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Old_Navy'] = dataset['brand_name'].str.contains("Old Navy")

    dataset['is_Old_Navy'] = dataset['is_Old_Navy'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_PINK'] = dataset['brand_name'].str.contains("PINK")

    dataset['is_PINK'] = dataset['is_PINK'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Rae_Dunn'] = dataset['brand_name'].str.contains("Rae Dunn")

    dataset['is_Rae_Dunn'] = dataset['is_Rae_Dunn'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Sephora'] = dataset['brand_name'].str.contains("Sephora")

    dataset['is_Sephora'] = dataset['is_Sephora'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Sony'] = dataset['brand_name'].str.contains("Sony")

    dataset['is_Sony'] = dataset['is_Sony'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Under_Armour'] = dataset['brand_name'].str.contains("Under Armour")

    dataset['is_Under_Armour'] = dataset['is_Under_Armour'].map( {True: 1, False: 0} ).astype(float)

    dataset['is_Victoria_Secret'] = dataset['brand_name'].str.contains("Victoria's Secret")

    dataset['is_Victoria_Secret'] = dataset['is_Victoria_Secret'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['item_description'] = dataset['item_description'].str.lower()

    

    

    ### VAR IDEAS FROM https://www.kaggle.com/lopuhin/eli5-for-mercari

    dataset['contains_gb'] = dataset['item_description'].str.contains("gb ")

    dataset['contains_gb'] = dataset['contains_gb'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['contains_14k'] = dataset['item_description'].str.contains("14k ")

    dataset['contains_14k'] = dataset['contains_14k'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['contains_unlocked'] = dataset['item_description'].str.contains("unlocked")

    dataset['contains_unlocked'] = dataset['contains_unlocked'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['contains_carat'] = dataset['item_description'].str.contains("carat")

    dataset['contains_carat'] = dataset['contains_carat'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['is_vitamix'] = dataset['brand_name'].str.contains("vitamix")

    dataset['is_vitamix'] = dataset['is_vitamix'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['is_david_yurman'] = dataset['brand_name'].str.contains("david yurman")

    dataset['is_david_yurman'] = dataset['is_david_yurman'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['is_hatchimal'] = dataset['name'].str.contains("hatchimal")

    dataset['is_hatchimal'] = dataset['is_hatchimal'].map( {True: 1, False: 0} ).astype(float)

    

    dataset['is_dockatot'] = dataset['name'].str.contains("dockatot")

    dataset['is_dockatot'] = dataset['is_dockatot'].map( {True: 1, False: 0} ).astype(float)   

    



    #####################################################################

    dataset['category_main'] = dataset['category_main'].str.replace(' ','_')

    dataset['category_sub1'] = dataset['category_sub1'].str.replace(' ','_')

    dataset['category_sub2'] = dataset['category_sub2'].str.replace(' ','_')

    

    dataset['category_main'] = dataset['category_main'].str.replace('&','')

    dataset['category_sub1'] = dataset['category_sub1'].str.replace('&','')

    dataset['category_sub2'] = dataset['category_sub2'].str.replace('&','')

    

   

 



train = pd.concat([train, pd.get_dummies(train.category_main, prefix_sep='', prefix='M_')], axis=1)

test  = pd.concat([test , pd.get_dummies(test.category_main , prefix_sep='', prefix='M_')], axis=1)

train = pd.concat([train, pd.get_dummies(train.item_condition_id, prefix_sep='', prefix='M_')], axis=1)

test  = pd.concat([test , pd.get_dummies(test.item_condition_id , prefix_sep='', prefix='M_')], axis=1)



#dataset = pd.concat([dataset, pd.get_dummies(dataset.category_sub1, prefix_sep='', prefix='S1_')], axis=1)

#dataset = pd.concat([dataset, pd.get_dummies(dataset.category_sub2, prefix_sep='', prefix='S2_')], axis=1)

    

train.head(3)
# Feature Selection

drop_elements = ['train_id', 'name', 'category_name', 'brand_name', 'item_description', 'category_main', 'category_sub1', 'category_sub2', 'item_condition_id']

train = train.drop(drop_elements, axis = 1)

drop_elements = ['test_id', 'name', 'category_name', 'brand_name', 'item_description', 'category_main', 'category_sub1', 'category_sub2', 'item_condition_id']

test  = test.drop(drop_elements, axis = 1)



train.head(3)
print (train.info())
full_data = [train, test]

for dataset in full_data:   

    dataset['item_description_len']   = dataset['item_description_len'].fillna(0)

    dataset['contains_brand_new']     = dataset['contains_brand_new'].fillna(0).astype(int)

    dataset['contains_free_shipping'] = dataset['contains_free_shipping'].fillna(0).astype(int)

    dataset['contains_price_firm']    = dataset['contains_price_firm'].fillna(0).astype(int)

    dataset['contains_rm']    = dataset['contains_rm'].fillna(0).astype(int)

    dataset['has_brand']    = dataset['has_brand'].fillna(0).astype(int)

    dataset['is_Adidas'] = dataset['is_Adidas'].fillna(0).astype(int)

    dataset['is_American_Eagle'] = dataset['is_American_Eagle'].fillna(0).astype(int)

    dataset['is_Apple'] = dataset['is_Apple'].fillna(0).astype(int)

    dataset['is_Bath__Body_Works'] = dataset['is_Bath__Body_Works'].fillna(0).astype(int)

    dataset['is_Coach'] = dataset['is_Coach'].fillna(0).astype(int)

    dataset['is_Disney'] = dataset['is_Disney'].fillna(0).astype(int)

    dataset['is_FOREVER_21'] = dataset['is_FOREVER_21'].fillna(0).astype(int)

    dataset['is_Funko'] = dataset['is_Funko'].fillna(0).astype(int)

    dataset['is_LuLaRoe'] = dataset['is_LuLaRoe'].fillna(0).astype(int)

    dataset['is_Lululemon'] = dataset['is_Lululemon'].fillna(0).astype(int)

    dataset['is_Michael_Kors'] = dataset['is_Michael_Kors'].fillna(0).astype(int)

    dataset['is_Nike'] = dataset['is_Nike'].fillna(0).astype(int)

    dataset['is_Nintendo'] = dataset['is_Nintendo'].fillna(0).astype(int)

    dataset['is_Old_Navy'] = dataset['is_Old_Navy'].fillna(0).astype(int)

    dataset['is_PINK'] = dataset['is_PINK'].fillna(0).astype(int)

    dataset['is_Rae_Dunn'] = dataset['is_Rae_Dunn'].fillna(0).astype(int)

    dataset['is_Sephora'] = dataset['is_Sephora'].fillna(0).astype(int)

    dataset['is_Sony'] = dataset['is_Sony'].fillna(0).astype(int)

    dataset['is_Under_Armour'] = dataset['is_Under_Armour'].fillna(0).astype(int)

    dataset['is_Victoria_Secret'] = dataset['is_Victoria_Secret'].fillna(0).astype(int)

    

    dataset['contains_gb'] = dataset['contains_gb'].fillna(0).astype(int)

    dataset['contains_14k'] = dataset['contains_14k'].fillna(0).astype(int)    

    dataset['contains_unlocked'] = dataset['contains_unlocked'].fillna(0).astype(int) 

    dataset['contains_carat'] = dataset['contains_carat'].fillna(0).astype(int)

    dataset['is_vitamix'] = dataset['is_vitamix'].fillna(0).astype(int)

    dataset['is_david_yurman'] = dataset['is_david_yurman'].fillna(0).astype(int)

    dataset['is_hatchimal'] = dataset['is_hatchimal'].fillna(0).astype(int)

    dataset['is_dockatot'] = dataset['is_dockatot'].fillna(0).astype(int)





print (train.info())

#print(train.describe())

print (test.info())

#print(test.describe())
np.sum(train)
# Feature Selection 2

drop_elements = ['is_vitamix' ,'is_david_yurman','is_hatchimal','is_dockatot']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)



train.head(3)
full_data = [train_text_dtm, test_text_dtm]

for dataset in full_data:  

    for column in dataset:

        #print(column)

        dataset[column]   = dataset[column].fillna(0).astype(int)

print(train_text_dtm.info())

print(test_text_dtm.info())
from sklearn import preprocessing



#yy_train = (np.log(train['price']+1))

yy_train = train['price']

x_train = (train.drop('price',axis=1))

x_train = x_train.iloc[:, 0:].values



x_train = pd.concat([pd.DataFrame(x_train).reset_index(drop=True), train_text_dtm.reset_index(drop=True)], axis=1)

test = pd.concat([pd.DataFrame(test).reset_index(drop=True), test_text_dtm.reset_index(drop=True)], axis=1)



print(x_train.shape)

print(test.shape)

print(x_train.info())

print(test.info())





# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

data_test = sc.transform(test)



import numpy as np

#x_train = np.delete(x_train, 4, axis=1)

#data_test = np.delete(data_test, 4, axis=1)

#x_train   = x_train.astype(int)

#data_test = data_test.astype(int)



data_train = x_train



print(data_train.__class__.__name__)

print(data_test.__class__.__name__)

#print(pd.DataFrame(yy_train).describe())

#print(pd.DataFrame(data_train).describe())

#print(pd.DataFrame(data_test).describe())



data_test
import numpy as np



def rmsle(h, y): 

    """

    Compute the Root Mean Squared Log Error for hypthesis h and targets y



    Args:

        h - numpy array containing predictions with shape (n_samples, n_targets)

        y - numpy array containing targets with shape (n_samples, n_targets)

    """

    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

print('ok')
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_train, yy_train, test_size = 0.2, random_state = 0)

print('ok')
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, Lasso, LassoCV, ElasticNetCV

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct,ConstantKernel                                            

                                              



import tensorflow as tf

import tensorflow.contrib.learn as learn

#Some useful packages

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn   import metrics

from sklearn.model_selection import train_test_split



def trainModels(model, X_train, y_train, X_test):

    if(model == 'linear'):

        cl = LinearRegression()   

    if(model == 'SGD'):

        cl= SGDRegressor()

    if(model == 'Ridge'):

        cl = Ridge(solver='auto',

        fit_intercept=True,

        alpha=0.5,

        max_iter=100,

        normalize=False,

        tol=0.05)

    if(model == 'RidgeCV'):

        cl = RidgeCV()        

    if(model == 'Lasso'):

        cl = LassoCV()

    if(model == 'ElasticNet'):

        cl = ElasticNetCV()

    if(model == 'SVR'):

        cl = SVR(kernel='linear', C=1e4) 

    if(model == 'NeuralNet'):

        cl = MLPRegressor(solver='lbfgs', 

                                       alpha=1e-5, 

                                       hidden_layer_sizes=(80,50,20), 

                                       random_state=1)

    if(model == 'RandomForest'):

        cl = RandomForestRegressor(n_estimators=500,oob_score=True, max_features = 5, max_depth = 3)

    if(model == 'ExtraTrees'):

        cl = ExtraTreesRegressor(n_estimators=500)

        

    if(model == 'GradientBoosting'):

        cl = GradientBoostingRegressor(n_estimators=1000, 

                                                  learning_rate=0.05, 

                                                  min_samples_leaf=50, 

                                                  min_samples_split=20, 

                                                  loss='huber')



    cl.fit(X_train, y_train)

    pred = cl.predict(X_test)

    #score = np.sqrt(metrics.mean_squared_error(y_test,pred))

    score = rmsle(pred,y_test)

    return score, cl



print('OK')
a = train.drop('price',axis=1).columns.tolist()

b = train_text_dtm.columns.tolist()

lista = a + b

print(len(a))

print(len(b))

print(len(lista))

clf = ExtraTreesRegressor()

clf = clf.fit(X_train, y_train)

clf.feature_importances_  

#model = SelectFromModel(clf, prefit=True)

importances = clf.feature_importances_

std = np.std([f.feature_importances_ for f in clf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]

features = lista
print(len(indices))

print(len(features))

print(len(importances))
# Print the feature ranking

print("Feature ranking:")



cum_imp = 0

sel_features = list()

sel_features_idx = list()

for f in range(X_train.shape[1]):

    cum_imp = cum_imp + importances[indices[f]]

    print('rank '+(str(f)) + " var: "+ (str(indices[f])) + " "+ features[indices[f]] + "--\t\t" + str(importances[indices[f]]) + "--\t" + str(cum_imp))

    if(cum_imp <= 0.8):

        sel_features.append(features[indices[f]])

        sel_features_idx.append(indices[f])



plt.figure()

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), indices)

plt.xlim([-1, X_train.shape[1]])

plt.show()



print(sel_features)

print(sel_features_idx)
# Feature Selection 3

X_train    = X_train[:,sel_features_idx]

X_test     = X_test[:,sel_features_idx]

data_test = data_test[:,sel_features_idx]



X_train
s, cl_linear = trainModels('linear', X_train, y_train, X_test)

print('linear:\t' + str(s))

s, cl_SGD = trainModels('SGD', X_train, y_train, X_test)

print('SGD:\t' + str(s))

s, cl_Ridge = trainModels('Ridge', X_train, y_train, X_test)

print('Ridge:\t' + str(s))

s, cl_RidgeCV = trainModels('RidgeCV', X_train, y_train, X_test)

print('RidgeCV:\t' + str(s))

s, cl_Lasso = trainModels('Lasso', X_train, y_train, X_test)

print('Lasso:\t' + str(s))

s, cl_ElasticNet = trainModels('ElasticNet', X_train, y_train, X_test)

print('ElasticNet:\t' + str(s))

#s, cl_SVR = trainModels('SVR', X_train, y_train, X_test)

#print('SVR:\t' + str(s))

s, cl_NeuralNet = trainModels('NeuralNet', X_train, y_train, X_test)

print('NeuralNet:\t' + str(s))

s, cl_RandomForest = trainModels('RandomForest', X_train, y_train, X_test)

print('RandomForest:\t' + str(s))

s, cl_ExtraTrees = trainModels('ExtraTrees', X_train, y_train, X_test)

print('ExtraTrees:\t' + str(s))

s, cl_GradientBoosting = trainModels('GradientBoosting', X_train, y_train, X_test)

print('GradientBoosting:\t' + str(s))





candidate_regressor = cl_GradientBoosting

candidate_regressor.fit(X_train, y_train)

pred = candidate_regressor.predict(X_test)

#y_test = np.exp(y_test)-1

#pred = np.exp(pred)-1

score = rmsle(pred,y_test)

print(score)

result = candidate_regressor.predict(data_test)





# Generate Submission File 

Submission = pd.DataFrame({ 'test_id': testId, 'price': result })



print(Submission['price'].mean())

print(Submission.head(3))

Submission.to_csv("Submission.csv", index=False)

print('OK')
print(X_train.shape[0])

print(X_train.shape[1])
from numpy import array

from matplotlib import pyplot



from keras.models import Sequential

from keras.layers import Dense

from keras import backend

from keras.layers import Dropout

 

def rmse_k(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))



def rmsle_k(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(backend.log(y_pred + 1) - backend.log(y_true + 1)), axis=-1))



# create model

drop = 0.3

model = Sequential()

model.add(Dense(units = 12, input_dim=12, kernel_initializer = 'uniform', activation='relu'))

model.add(Dropout(drop))

model.add(Dense(units = 10, kernel_initializer = 'uniform', activation='relu'))

model.add(Dropout(drop))

model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dropout(drop))

model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dropout(drop))

model.add(Dense(units = 1,  kernel_initializer = 'uniform'))

model.compile(loss='mse', optimizer='adam', metrics=[rmsle_k])

# train model

history = model.fit(X_train, y_train, epochs=30, batch_size=1000, verbose=0)

# plot metrics

pyplot.plot(history.history['rmsle_k'])

pyplot.show()
pred = model.predict(X_test)

pred = backend.cast_to_floatx(pred)

price_test = backend.cast_to_floatx(y_test)

#pred = np.exp(pred) -1

#price_test = np.exp(price_test) -1



score = rmsle(pred,price_test)

print(score)



#result = model.predict(data_test)

#result = result[:,0]



# Generate Submission File 

#Submission = pd.DataFrame({ 'test_id': testId, 'price': result })



#print(Submission['price'].mean())

#print(Submission.head(3))
#GradientBoostingRegressor OPTIMIZATION

from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.model_selection import cross_validation, metrics   #Additional scklearn functions

from sklearn.model_selection  import GridSearchCV   #Perforing grid search



def rmsle_gd(y_true, y_pred): return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean()) 



parametros = {'n_estimators':      [1000],

              'min_samples_leaf':  [50],

              'min_samples_split': [20,50]

              #'learning_rate':     [0.05, 0.1]

             }

print(parametros)



custom_score = 'neg_mean_squared_error'



gsearch1 = GridSearchCV(estimator = 

                        GradientBoostingRegressor(loss='huber'), 

                            param_grid = parametros,

                            n_jobs= 1,

                            iid=False,

                            scoring=custom_score,

                            cv=3)

#gsearch1.fit(X_train, y_train)





#best_parameters = gsearch1.best_params_

#best_accuracy = gsearch1.best_score_



#print(best_parameters)

#print(best_accuracy)

#gsearch1.grid_scores_
from numpy import array

from matplotlib import pyplot



from keras.models import Sequential

from keras.layers import Dense

from keras import backend

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection  import GridSearchCV 

# Importing the Keras libraries and packages

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Conv1D, MaxPooling1D

    

def rmse_k(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))



def rmsle_k(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(backend.log(y_pred + 1) - backend.log(y_true + 1)), axis=-1))



# create model

def build_classifier(drop, optimizer):

    model = Sequential()

    model.add(Dense(units = 40, input_dim=44, kernel_initializer = 'uniform', activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(units = 20, kernel_initializer = 'uniform', activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

    model.add(Dropout(drop))

    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    model.add(Dropout(drop))

    model.add(Dense(units = 1,  kernel_initializer = 'uniform'))

    model.compile(loss='mse', optimizer=optimizer, metrics=[rmsle_k])

    return model



# 118603

# 44

X_train1 = np.expand_dims(X_train, axis=2) # reshape (569, 30) to (569, 30, 1) 

X_test1  = np.expand_dims(X_test, axis=2) # reshape (569, 30) to (569, 30, 1) 

def build_classifier_CNN(drop, optimizer):

    classifier = Sequential()

    classifier.add(Conv1D(44, (5), input_shape = (44,1), activation = 'relu'))

    classifier.add(Dropout(drop))

    classifier.add(MaxPooling1D(pool_size = (3)))

    classifier.add(Conv1D(20, (5), activation = 'relu'))

    classifier.add(MaxPooling1D(pool_size = (3)))

    classifier.add(Flatten())

    classifier.add(Dense(units = 15, activation = 'relu'))

    classifier.add(Dropout(drop))

    classifier.add(Dense(units = 10, activation = 'relu'))

    classifier.add(Dense(units = 1,  kernel_initializer = 'uniform'))

    classifier.compile(optimizer = optimizer, loss = 'msle', metrics = [rmsle_k])

    return classifier





classifier = KerasClassifier(build_fn = build_classifier)

classifier_CNN = KerasClassifier(build_fn = build_classifier_CNN)



parameters = {'batch_size': [5000], #, 10000],

              'epochs': [30],#, 40],

              'optimizer': ['adam'],#, 'rmsprop'],

              'drop': [0.2]#,0.3]

             }

#'rmsprop'

custom_score = 'neg_mean_squared_log_error'



grid_search = GridSearchCV(estimator = classifier_CNN,

                           param_grid = parameters,

                           scoring = custom_score,

                           cv = 3)

#grid_search = grid_search.fit(X_train1, y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_

best_parameters

best_accuracy
best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_



print(best_parameters)

print(best_accuracy)

grid_search.grid_scores_
build_fn = build_classifier_CNN(drop = 0.2,optimizer = 'adam')

cl_CNN = KerasClassifier(build_fn)

history = cl_CNN.fit(X_train1, y_train, epochs=30, batch_size=2000, verbose=2,  validation_data = (X_test1, y_test))

pyplot.plot(history.history['rmsle_k'])

pyplot.show()
pred = cl_CNN.predict(X_train1)

print(pd.DataFrame(pred).describe())

pred = backend.cast_to_floatx(pred)

price_test = backend.cast_to_floatx(y_train)

print(pred)

score = rmsle(pred,price_test)

print(score)



#data_test1  = np.expand_dims(data_test, axis=2) # reshape (569, 30) to (569, 30, 1) 

#result = classifier_CNN.predict(data_test1)

#result = result[:,0]



# Generate Submission File 

#Submission = pd.DataFrame({ 'test_id': testId, 'price': result })



#print(Submission['price'].mean())

#print(Submission.head(3))