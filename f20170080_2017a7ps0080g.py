import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize

from sklearn.metrics import mean_squared_error

import math
os.chdir('../input/eval-lab-1-f464-v2')
!ls
train_raw = pd.read_csv("./train.csv")

test_raw = pd.read_csv("./test.csv")



train_cols = train_raw.columns

test_cols = test_raw.columns

test_index = test_raw['id']
print(train_raw.shape)

train_raw['type'] = (train_raw['type'] == 'old') * 1

train_raw.head(10)
print(test_raw.shape)

test_raw['type'] = (test_raw['type'] == 'old') * 1

test_raw.head(10)
train_raw.describe()
test_raw.describe()
print(np.isnan(train_raw).any())



imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(train_raw)  

SimpleImputer(missing_values=np.nan, strategy='mean')

train_raw = imp.transform(train_raw)



print()

print(np.isnan(train_raw).any())
print(np.isnan(test_raw).any())



imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(test_raw)  

SimpleImputer(missing_values=np.nan, strategy='mean')

test_raw = imp.transform(test_raw)



print()

print(np.isnan(test_raw).any())
train_test_ratio = 0.2
train_raw = pd.DataFrame(train_raw, columns=train_cols)

test_raw = pd.DataFrame(test_raw, columns=test_cols)
# banned = ['id','rating']

banned = ['rating', 'feature8', 'feature9', 'feature10', 'feature11']

allowed = ['feature6', 'feature3', 'type']

allowed = ['feature1', 'feature2','feature3', 'feature6', 'feature7', 'id' ]



# banned = ['rating', 'feature8', 'feature9', 'feature10', 'feature11']

banned = ['rating', 'type']

# banned = ['rating', 'feature8', 'feature9', 'feature10', 'feature11', 'type']



# X_train = train_raw.loc[:,[i in allowed for i in train_raw.columns]]

# y_train = train_raw.loc[:,[i in ['rating'] for i in train_raw.columns]]

# X_test = test_raw.loc[:,[i in allowed for i in test_raw.columns]]



X_train = train_raw.loc[:,[i not in banned for i in train_raw.columns]]

y_train = train_raw.loc[:,[i in ['rating'] for i in train_raw.columns]]

X_test = test_raw.loc[:,[i not in banned for i in test_raw.columns]]

                        

train_cols = X_train.columns

test_cols = X_test.columns    





X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=train_test_ratio, shuffle=True)
# X_train = normalize_df(X_train, X_train.columns)

# X_val = normalize_df(X_val, X_test.columns)

# test_raw = normalize_df(test_raw, test_raw.columns)



# from sklearn.preprocessing import StandardScaler  

# scaler = StandardScaler()  

# scaler.fit(X_train)



# X_train = scaler.transform(X_train)  

# X_val = scaler.transform(X_val)  

# X_test = scaler.transform(X_test) 



# X_train = pd.DataFrame(X_train, columns=train_cols)

# X_val = pd.DataFrame(X_val, columns=train_cols)

# X_test = pd.DataFrame(X_test, columns=test_cols)
from sklearn.preprocessing import MinMaxScaler



# scaler = MinMaxScaler()

# X_scaled = scaler.fit_transform(X_train)



# X_train = scaler.transform(X_train)  

# X_val = scaler.transform(X_val)  

# X_test = scaler.transform(X_test) 



# X_train = pd.DataFrame(X_train, columns=train_cols)

# X_val = pd.DataFrame(X_val, columns=train_cols)

# X_test = pd.DataFrame(X_test, columns=test_cols)
print(X_train.shape)

print(X_val.shape)

print(y_train.shape)

print(y_val.shape)
X_train
y_train
X_val
y_val
labels = set()



for i in range(y_train.shape[0]):

    label = y_train.iloc[i,0]

    if(label not in labels):

        labels.add(label)

        

labels = list(sorted(labels))



labels_counts = []



for label in labels:

    count_i = np.count_nonzero(y_train == label)

    labels_counts.append(count_i)

    print(count_i,"\t :", label )

    

# print(labels)

# print(labels_counts)
def plot_corr(df):

    f = plt.figure(figsize=(12, 7))

    plt.matshow(df.corr(), fignum=f.number)

    plt.xticks(range(df.shape[1]), df.columns, fontsize=12)

    plt.yticks(range(df.shape[1]), df.columns, fontsize=12)

    cb = plt.colorbar()

    cb.ax.tick_params(labelsize=14)

    plt.title('Correlation Matrix', fontsize=20);

    

plot_corr(X_train)
rs = np.random.RandomState(0)

corr = train_raw.corr()

corr.style.background_gradient(cmap='coolwarm')
from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier()

model.fit(X_train,y_train)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers



#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

feat_importances.nlargest(100).plot(kind='barh')

plt.show()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 10 best features



bestfeatures = SelectKBest(score_func=chi2, k='all')

fit = bestfeatures.fit(X_train,y_train)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_train.columns)
#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(20,'Score'))  #print 10 best features
plt.figure(figsize=[10,5])



plt.plot(featureScores['Score'], 'o-')

plt.grid()



plt.plot()

plt.show()
import seaborn as sns



#get correlations of each features in dataset

corrmat = train_raw.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(12,12))



#plot heat map

g=sns.heatmap(train_raw[top_corr_features].corr(),annot=True,cmap="RdYlGn")
y_train = y_train.squeeze()

y_val = y_val.squeeze()
def generate_submission(model, fname):

    pred_test = model.predict(X_test)

    pred_test = np.round(pred_test)

    pred_test = np.maximum(np.zeros_like(pred_test), pred_test)



    pred_test = pd.DataFrame(pred_test)

    pred_test.index = test_index

    pred_test.to_csv(fname)
# generate_submission(gbr1, 'sub.csv')
from sklearn.ensemble import GradientBoostingRegressor
for i in range(1):

    

    n_estimators = 1500

    max_depth = 13

    learning_rate = 0.005

    warm_start = True

    subsample = 0.5



    params = {

                'n_estimators': n_estimators, 

                'max_depth': max_depth, 

                'learning_rate': learning_rate, 

                'warm_start' : warm_start,

                'subsample' : subsample,

                'loss': 'ls',

             }

    

    gbr1 = GradientBoostingRegressor(**params)

    gbr1.fit(X_train, y_train)



    pred_val = np.round(gbr1.predict(X_val))



    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    min_test_score = 100

    

    for ii, y_pred in enumerate(gbr1.staged_predict(X_val)):

        test_score[ii] = gbr1.loss_(y_val, y_pred)

        

        if(test_score[ii] < min_test_score):

            min_test_score = test_score[ii]

            actual_error = np.sqrt(min_test_score)

#             print("Error %.4f at %d iteration." % (actual_error, ii))

            

#             if(ii > n_estimators * .8):

#                 generate_submission(gbr1, 'temp'+str(i)+'.csv')

        

    error_min = math.sqrt(np.min(test_score))

    error_end = math.sqrt(test_score[-1])

    

    pred_test = gbr1.predict(X_test)

    pred_test = np.round(pred_test)

    pred_test = np.maximum(np.zeros_like(pred_test), pred_test)



    pred_test = pd.DataFrame(pred_test)

    pred_test.index = test_index

   

    print("-----------------------------------------------------------------------------------------------------")

    print('i = ', i)

    print(params)

    print("End error: %.4f" % error_end)

    print("Min error: %.4f" % error_min)

    print("At iter: %d" % np.argmin(test_score))

    

    

    plt.figure(figsize=(12, 6))

        

    # compute test set deviance

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



    for i, y_pred in enumerate(gbr1.staged_predict(X_val)):

        test_score[i] = gbr1.loss_(y_val, y_pred)



    train_error = np.sqrt(gbr1.train_score_)

    test_error = np.sqrt(test_score)



    plt.title('Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, train_error, 'b-',

             label='Training Set Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, test_error, 'r-',

             label='Test Set Deviance')

    plt.legend(loc='upper right')

    plt.xlabel('Boosting Iterations')

    plt.ylabel('Deviance')

    plt.grid()

    plt.show()

    plt.close()

    

    

#     fname = 'Sample' + str(i) + '.csv'

#     pred_test.to_csv(fname)
!mkdir -p Temps



for i in range(1):

    

    n_estimators = 1500

    max_depth = 9

    learning_rate = 0.02

    warm_start = True

    subsample = 0.5



    params = {

                'n_estimators': n_estimators, 

                'max_depth': max_depth, 

                'learning_rate': learning_rate, 

                'warm_start' : warm_start,

                'subsample' : subsample,

                'loss': 'ls',

             }

    

    gbr2 = GradientBoostingRegressor(**params)

    gbr2.fit(X_train, y_train)



    pred_val = np.round(gbr2.predict(X_val))



    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    min_test_score = 100

    

    for ii, y_pred in enumerate(gbr2.staged_predict(X_val)):

        test_score[ii] = gbr2.loss_(y_val, y_pred)

        

#         if(test_score[ii] < min_test_score):

#             min_test_score = test_score[ii]

#             actual_error = np.sqrt(min_test_score)

#             print("Error %.4f at %d iteration." % (actual_error, ii))

            

#             if(ii > n_estimators * .8):

#                 generate_submission(gbr2, 'temp'+str(i)+'.csv')

        

    error_min = math.sqrt(np.min(test_score))

    error_end = math.sqrt(test_score[-1])

    

    pred_test = gbr2.predict(X_test)

    pred_test = np.round(pred_test)

    pred_test = np.maximum(np.zeros_like(pred_test), pred_test)



    pred_test = pd.DataFrame(pred_test)

    pred_test.index = test_index

   

    print("-----------------------------------------------------------------------------------------------------")

    print('i = ', i)

    print(params)

    print("End error: %.4f" % error_end)

    print("Min error: %.4f" % error_min)

    print("At iter: %d" % np.argmin(test_score))

    

    

    plt.figure(figsize=(12, 6))

        

    # compute test set deviance

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



    for i, y_pred in enumerate(gbr2.staged_predict(X_val)):

        test_score[i] = gbr2.loss_(y_val, y_pred)



    train_error = np.sqrt(gbr2.train_score_)

    test_error = np.sqrt(test_score)



    plt.title('Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, train_error, 'b-',

             label='Training Set Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, test_error, 'r-',

             label='Test Set Deviance')

    plt.legend(loc='upper right')

    plt.xlabel('Boosting Iterations')

    plt.ylabel('Deviance')

    plt.grid()

    plt.show()

    plt.close()

    

    

#     fname = 'Sample' + str(i) + '.csv'

#     pred_test.to_csv(fname)


for i in range(1):

    

    n_estimators = 1000

    max_depth = 7

    learning_rate = 0.05

    warm_start = True

    subsample = 0.5



    params = {

                'n_estimators': n_estimators, 

                'max_depth': max_depth, 

                'learning_rate': learning_rate, 

                'warm_start' : warm_start,

                'subsample' : subsample,

                'loss': 'ls',

             }

    

    gbr3 = GradientBoostingRegressor(**params)

    gbr3.fit(X_train, y_train)



    pred_val = np.round(gbr3.predict(X_val))



    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    min_test_score = 100

    

    for ii, y_pred in enumerate(gbr3.staged_predict(X_val)):

        test_score[ii] = gbr3.loss_(y_val, y_pred)

        

#         if(test_score[ii] < min_test_score):

#             min_test_score = test_score[ii]

#             actual_error = np.sqrt(min_test_score)

#             print("Error %.4f at %d iteration." % (actual_error, ii))

            

#             if(ii > n_estimators * .8):

#                 generate_submission(gbr3, 'temp'+str(i)+'.csv')

        

    error_min = math.sqrt(np.min(test_score))

    error_end = math.sqrt(test_score[-1])

    

    pred_test = gbr3.predict(X_test)

    pred_test = np.round(pred_test)

    pred_test = np.maximum(np.zeros_like(pred_test), pred_test)



    pred_test = pd.DataFrame(pred_test)

    pred_test.index = test_index

   

    print("-----------------------------------------------------------------------------------------------------")

    print('i = ', i)

    print(params)

    print("End error: %.4f" % error_end)

    print("Min error: %.4f" % error_min)

    print("At iter: %d" % np.argmin(test_score))

    

    

    plt.figure(figsize=(12, 6))

        

    # compute test set deviance

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



    for i, y_pred in enumerate(gbr3.staged_predict(X_val)):

        test_score[i] = gbr3.loss_(y_val, y_pred)



    train_error = np.sqrt(gbr3.train_score_)

    test_error = np.sqrt(test_score)



    plt.title('Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, train_error, 'b-',

             label='Training Set Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, test_error, 'r-',

             label='Test Set Deviance')

    plt.legend(loc='upper right')

    plt.xlabel('Boosting Iterations')

    plt.ylabel('Deviance')

    plt.grid()

    plt.show()

    plt.close()

    

    

#     fname = 'Sample' + str(i) + '.csv'

#     pred_test.to_csv(fname)
# 110, 10, 0





from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor



dtr = DecisionTreeRegressor(

            min_samples_split = 110,

            min_samples_leaf = 10,

            min_weight_fraction_leaf = 0.0

                )



dtr.fit(X_train, y_train)



pred_val = dtr.predict(X_val)

val_error = np.sqrt(mean_squared_error(pred_val, y_val))

val_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis(

                        solver='lsqr',

                        shrinkage = 0.1,

 )





lda.fit(X_train, y_train)  



pred_val = lda.predict(X_val)

val_error = np.sqrt(mean_squared_error(pred_val, y_val))

val_error
from sklearn.ensemble import AdaBoostRegressor
!mkdir -p Temps



   

n_estimators = 250

learning_rate = 0.0005

loss='linear'



params = {

        'n_estimators': n_estimators, 

        'learning_rate' : learning_rate,

        'loss' : loss

    }



abr1 = AdaBoostRegressor(**params)

abr1.fit(X_train, y_train)



pred_val = np.round(abr1.predict(X_val))



test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

min_test_score = 100



for ii, y_pred in enumerate(abr1.staged_predict(X_val)):

    test_score[ii] = mean_squared_error(y_val, y_pred)



    if(test_score[ii] < min_test_score):

        min_test_score = test_score[ii]

        actual_error = np.sqrt(min_test_score)

        print("Error %.4f at %d iteration." % (actual_error, ii))



        if(ii > n_estimators * .8):

            generate_submission(abr1, 'temp'+str(i)+'.csv')



error_min = math.sqrt(np.min(test_score))

error_end = math.sqrt(test_score[-1])



pred_test = abr1.predict(X_test)

pred_test = np.round(pred_test)

pred_test = np.maximum(np.zeros_like(pred_test), pred_test)



pred_test = pd.DataFrame(pred_test)

pred_test.index = test_index



print("-----------------------------------------------------------------------------------------------------")

print(params)

print("End error: %.4f" % error_end)

print("Min error: %.4f" % error_min)

print("At iter: %d" % np.argmin(test_score))



##### PLOTTING



plt.figure(figsize=(12, 6))

train_pred = np.round(abr1.predict(X_train))



train_error = np.sqrt(mean_squared_error(y_train, train_pred))

test_error = np.sqrt(test_score)



#     print(test_errors)

print("Train error : ", train_error)



plt.title('?')

plt.plot(np.arange(params['n_estimators']) + 1, test_error, 'r-', label='Validation Loss')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Loss')

plt.grid()

plt.show()

plt.close()



pred_val = abr1.predict(X_val)

val_error = np.sqrt(mean_squared_error(pred_val, y_val))

print(val_error)



#     fname = 'Sample' + str(i) + '.csv'

#     pred_test.to_csv(fname)
from sklearn.ensemble import VotingRegressor



# weights = [100,40,40,40,20,10]

weights = [100,100,100]



vtr = VotingRegressor(

                [

                    ('gbr1', gbr1), 

                    ('gbr2', gbr2), 

                    ('gbr3', gbr3), 

#                     ('abr1', abr1),

#                     ('dtr', dtr),

#                     ('lda', lda)

                ],

                weights

            )





vtr.fit(X_train, y_train)

pred_val = vtr.predict(X_val)

val_error = np.sqrt(mean_squared_error(pred_val, y_val))

val_error
def generate_submission(model, fname):

    pred_test = model.predict(X_test)

    pred_test = np.round(pred_test)

    pred_test = np.maximum(np.zeros_like(pred_test), pred_test)



    pred_test = pd.DataFrame(pred_test)

    pred_test.index = test_index

    pred_test.to_csv(fname)
generate_submission(vtr, "submission.csv")