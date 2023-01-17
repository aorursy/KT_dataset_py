import pandas as pd



maths = pd.read_csv('../input/student-mat.csv') 

portug = pd.read_csv('../input/student-por.csv')



print(maths.head())
sample = maths.loc[0,:]

print(sample)
totalDataSet = pd.concat([maths,portug])
import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

lst = ['school','sex','address','Dalc']

fig = plt.figure(figsize=(10, 20))

plt.rcParams.update({'font.size': 15})



for x,y in enumerate(lst):

    plt.subplot(len(lst),1,1+x) 

    plt.xlabel(y)

    plt.ylabel("G3")

    totalDataSet.groupby(y)['G3'].mean().plot(kind='bar')

import seaborn as sns

sns.set()

sns.heatmap(totalDataSet.corr(),linewidths=.5)
ave = sum(totalDataSet.G3)/float(len(totalDataSet))

totalDataSet['average'] = ['above average' if i > ave else 'under average' for i in totalDataSet.G3]

sns.swarmplot(x=totalDataSet.Dalc, y =totalDataSet.G3, hue = totalDataSet.average)

totalDataSet.drop('average',axis=1);
ax2 = pd.value_counts(totalDataSet['Dalc']).sort_values(ascending=False).plot.bar()

ax2.set_xlabel('Number of Weekdays spent Drinking')

ax2.set_ylabel('Number of Students')
outcomes = totalDataSet['G3']

features_raw = totalDataSet.drop(['G3','G2','G1'],axis=1)
features_raw.hist(alpha=0.5, figsize=(16, 10))
skewed=['Dalc','Walc','absences','failures','traveltime']

features_log_transformed = pd.DataFrame(data=features_raw)

features_log_transformed[skewed] = features_raw[skewed].apply(lambda x:np.log(x+1))



features_log_transformed[skewed].hist(alpha=0.5, figsize=(16, 10))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']

features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)

features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])



display(features_log_minmax_transform.head(n=5))
features_final = pd.get_dummies(features_log_minmax_transform)

encoded = list(features_final.columns)

print("{} total features after one-hot encoding".format(len(encoded)))

print(encoded)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features_final,outcomes, test_size = 0.2, random_state=42)



print("Training set has {} samples".format(X_train.shape[0]))

print("Testing set has {} samples".format(X_test.shape[0]))
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error



train_rms = sqrt(mean_squared_error(y_train, y_train_pred))

test_rms = sqrt(mean_squared_error(y_test, y_test_pred))





print("The Root mean Squared Error for the training set is", train_rms)

print("The Root mean Squared Error for the testing set is ", test_rms)





mae_train = mean_absolute_error(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)

print('Mean Absolute Error for Training Set: %f' % mae_train)

print('Mean Absolute Error for Testing Set: %f' % mae_test)



print("Cross val score for training set",cross_val_score(model, X_train, y_train, cv=5).mean())

print("Cross val score for testing set",cross_val_score(model, X_test, y_test, cv=5).mean())
def model_Creator_Tester(name,model,X_train,X_test,y_train,y_test):

    print(name)

    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)



    from sklearn.metrics import mean_squared_error

    from math import sqrt

    from sklearn.model_selection import cross_val_score

    from sklearn.metrics import mean_absolute_error



    train_rms = sqrt(mean_squared_error(y_train, y_train_pred))

    test_rms = sqrt(mean_squared_error(y_test, y_test_pred))





    print("The Root mean Squared Error for the training set is", train_rms)

    print("The Root mean Squared Error for the testing set is ", test_rms)





    mae_train = mean_absolute_error(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)

    print('Mean Absolute Error for Training Set: %f' % mae_train)

    print('Mean Absolute Error for Testing Set: %f' % mae_test)

    return train_rms,test_rms,mae_train,mae_test;

    #print("Cross val score for training set",cross_val_score(model, X_train, y_train, cv=5).mean())

    #print("Cross val score for testing set",cross_val_score(model, X_test, y_test, cv=5).mean())
from sklearn.svm import SVC

from xgboost import XGBClassifier

import lightgbm as lgb

names = ["Linear_Regression","XGB","SVM","LGB"]

models = [LinearRegression(),XGBClassifier(),SVC(gamma='auto'),lgb.LGBMRegressor()]

results = {}

for x,y in zip(names,models):

    print("\n",y,"\n")

    results[x]=model_Creator_Tester(x,y,X_train,X_test,y_train,y_test)
def color_gradient ( val, beg_rgb, end_rgb, val_min = 0, val_max = 1):

    val_scale = (1.0 * val - val_min) / (val_max - val_min)

    return ( beg_rgb[0] + val_scale * (end_rgb[0] - beg_rgb[0]),

             beg_rgb[1] + val_scale * (end_rgb[1] - beg_rgb[1]),

             beg_rgb[2] + val_scale * (end_rgb[2] - beg_rgb[2]))
#print(results)

def print_results(results):

    titles = ["Root Mean Square for Training Set","Root Mean Square for Testing Set","Mean Absolute Error for Training Set","Mean Absolute Error for Testing Set"]

    fig = plt.figure(figsize=(10, 10))

    plt.rcParams.update({'font.size': 10})

    grad_beg, grad_end = ( 0.1, 0.1, 0.1), (1, 1, 0)

    for i,k in enumerate(results):

        tempVals = []

        for j in results.keys():

            #print(i,j)

            #print(results[j][i])

            tempVals.append(results[j][i])

        print(tempVals)

        print(results.keys())

        plt.subplot(len(titles)/2.,len(titles)/2.,1+i)

        col_list = [ color_gradient( val,

                                 grad_beg,

                                 grad_end,

                                 min( tempVals),

                                 max(tempVals)) for val in tempVals]



        plt.bar(results.keys(),tempVals,color = col_list)

        plt.title(titles[i])

print_results(results)
from sklearn.model_selection import GridSearchCV

params={'Linear_Regression':{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True,False]},

        'XGB':{

                'boster':['gbtree'],

                'eta':[0.05,0.1,0.25,0.5,0.8],

                'gamma':[0.05,0.1,0.25,0.5,0.8],

                #'reg_alpha': [0.05,0.1,0.25,0.5,0.8],

                #'reg_lambda': [0.05,0.1,0.25,0.5,0.8],

                'max_depth':[3,6,10],

                'subsample':[0.1,0.25,0.5,0.8]

        },

        'SVM':{'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1],'kernel':['rbf','linear']},

       'LGB':{'boosting_type': ['gbdt'],

                'num_leaves': [20,50,80],

                'learning_rate': [0.05,0.1,0.25,0.5,0.8],

                'subsample_for_bin': [10,100,500],

                'min_child_samples': [20,50,100],

                'reg_alpha': [0.05,0.1,0.25,0.5,0.8],

                'reg_lambda': [0.05,0.1,0.25,0.5,0.8]

             }

        }



names = ["Linear_Regression","SVM","LGB","XGB"]



models = [LinearRegression(),SVC(),lgb.LGBMRegressor(),XGBClassifier()]

def grid_model_Creator_Tester(name,model,X_train,X_test,y_train,y_test):

    print(name)

    model.fit(X_train,y_train)

    best_model = model.best_params_

    print(best_model)

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)



    from sklearn.metrics import mean_squared_error

    from math import sqrt

    from sklearn.model_selection import cross_val_score

    from sklearn.metrics import mean_absolute_error



    train_rms = sqrt(mean_squared_error(y_train, y_train_pred))

    test_rms = sqrt(mean_squared_error(y_test, y_test_pred))





    print("The Root mean Squared Error for the training set is", train_rms)

    print("The Root mean Squared Error for the testing set is ", test_rms)





    mae_train = mean_absolute_error(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)

    print('Mean Absolute Error for Training Set: %f' % mae_train)

    print('Mean Absolute Error for Testing Set: %f' % mae_test)

    return train_rms,test_rms,mae_train,mae_test;
results = {}

for x,y in zip(names,models):

    results[x]=grid_model_Creator_Tester(x,GridSearchCV(y,params[x]),X_train,X_test,y_train,y_test)

print_results(results)