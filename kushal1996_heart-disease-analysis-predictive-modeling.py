import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings



warnings.filterwarnings("ignore")

print(os.listdir("../input"))
df = pd.read_csv(r'../input/heart.csv')

df.head()
df.shape
df.isnull().sum()
df.dtypes
df.describe()
plt.style.use('fivethirtyeight')
sns.pairplot(df , hue = 'target' ,

             vars = ['age' , 'trestbps' , 'chol' , 'thalach' , 'oldpeak'] )

plt.show()
plot_data = ['cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'ca' ,'thal' , 'target']

plt.figure(1 , figsize = (15 , 10))

n = 0

for i in plot_data:

    n += 1

    plt.subplot(2 , 4 , n)

    plt.subplots_adjust(hspace= 0.5 , wspace =0.5)

    sns.countplot(x = i , hue = 'sex' , data = df )

    plt.legend()

plt.show()
plot_data = ['cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'ca' ,'thal' , 'sex']

plt.figure(1 , figsize = (15 , 10))

n = 0

for i in plot_data:

    n += 1

    plt.subplot(2 , 4 , n)

    plt.subplots_adjust(hspace= 0.5 , wspace =0.5)

    sns.countplot(x = i , hue = 'target' , data = df )

    plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 5))

sns.distplot(a = df['age'] , bins = 40 , color = 'red' )

plt.title('Histogram of Age')

plt.show()
plot_data = ['trestbps' , 'chol' , 'thalach' , 'oldpeak' ]

plt.figure(1 , figsize = (15 , 8))

n = 0

for x in plot_data:

    n += 1

    plt.subplot(2 , 2 , n)

    sns.distplot(a = df[x] , bins = 50  , color = 'red')  

    

plt.show()
plot_data = ['trestbps' , 'chol' , 'thalach' , 'oldpeak' ]

plt.figure(1 , figsize = (15 , 8))

n = 0

for x in plot_data:

    n += 1

    plt.subplot(2 , 2 , n)

    sns.distplot(a = df[x][df['sex'] == 1] , bins = 50  , color = 'red' , label = 'male')

    sns.distplot(a = df[x][df['sex'] == 0] , bins = 50 , label ='female' )

    plt.legend()

    

plt.show()
plot_data = ['trestbps' , 'chol' , 'thalach' , 'oldpeak' ]

plt.figure(1 , figsize = (15 , 8))

n = 0

for x in plot_data:

    n += 1

    plt.subplot(2 , 2 , n)

    sns.distplot(a = df[x][df['target'] == 1] , bins = 50  , color = 'red' ,

                 label = 'heart disease = True')

    sns.distplot(a = df[x][df['target'] == 0] , bins = 50 ,

                 label ='heart disease = False' )

    plt.legend()

    

plt.show()
plt.figure(1 , figsize = (15 , 10 ))

plt.subplot(2 , 1 , 1)

plt.scatter(x = 'age' , y = 'trestbps' , data = df.where(df['trestbps'] <= 135) ,

            s = 200)

plt.scatter(x = 'age' , y = 'trestbps' , data = df.where(df['trestbps'] > 135) ,

            s = 200)

plt.scatter(x = 'age' , y = 'trestbps' , data = df.where(df['trestbps'] < 83) ,

            s = 200)

for critical in [84 , 136]:

    plt.plot(df['age'] , np.ones((df.shape[0] , 1))*critical , 'r-' , alpha = 0.5)



plt.annotate('dangerously high blood pressure line.', xy=(30, 135), xytext=(28, 160),

            arrowprops=dict(facecolor='black', shrink=0.05),

            )

plt.annotate('dangerously low blood pressure line.', xy=(76, 84), xytext=(68, 100),

            arrowprops=dict(facecolor='black', shrink=0.05),

            )

plt.xlabel('age')

plt.ylabel('resting blood pressure')



plt.subplot(2 , 1 , 2)

plt.scatter(x = 'age' , y = 'trestbps' , s = 200 , data = df[df['target'] == 0] ,

            label = 'Heart disease == False' , alpha = 0.5)

plt.scatter(x = 'age' , y = 'trestbps' , s = 200 , data = df[df['target'] == 1] ,

           label = 'Heart disease == True' , alpha = 0.5)

plt.xlabel('age')

plt.ylabel('resting blood pressure')

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 10))

plt.subplot(2, 1 , 1)

plt.scatter(x = 'age' , y = 'chol' , data = df , s = 200 ,

           alpha = 0.5 , label = 'normal serum cholesterol')

plt.scatter(x = 'age' , y = 'chol' , data = df.where(df['chol'] > 200) , s = 200 ,

           alpha = 0.5 , label = 'high serum cholesterol')

plt.plot(df['age'] , np.ones((df.shape[0] , 1))*221 , 'r-' , alpha =0.5 )

plt.annotate('danger line', xy=(32, 221), xytext=(30,400 ),

            arrowprops=dict(facecolor='black', shrink=0.05)

            )

plt.xlabel('age')

plt.ylabel('serum cholesterol in mg')

plt.legend()



plt.subplot(2 , 1 , 2)

plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

for i in [0 , 1]:

    plt.scatter(x = 'age' , y = 'chol' , data = df.where(df['target'] == i) , s = 200 , 

               alpha = 0.5 , label = i )

plt.xlabel('age')

plt.ylabel('serum cholesterol in mg')

plt.title('0 = No Heart disease , 1 = Heart disease')

plt.legend()

plt.show()
def bpm_issue(age , thalach):

    bpm = []

    for a , t in zip(age , thalach):

        if t > (220 - a):

            bpm.append(1)

        else :

            bpm.append(0)

    return bpm

bpm = bpm_issue(df['age'] , df['thalach'])

df['bpm'] = bpm
plt.figure(1 , figsize = (15 , 10))

plt.subplot(2 , 1 , 1)

plt.scatter(x = 'age' , y = 'thalach' , data = df[df['bpm'] == 0],

            label = 'normal' , alpha = 0.5 , s = 200)

plt.scatter(x = 'age' , y = 'thalach' , data = df[df['bpm'] == 1],

            label = 'dangerously high heart beat rate' , s = 200)

plt.xlabel('age')

plt.ylabel('maximum heart rate achieved')

plt.legend()

plt.subplot(2 , 1 , 2 )

plt.scatter(x = 'age' , y = 'thalach' , data = df[df['target'] == 0],

           label = 'heart disease = False' , alpha = 0.5 , s = 200)



plt.scatter(x = 'age' , y = 'thalach' , data = df[df['target'] == 1],

           label = 'heart disease = True' , alpha = 0.5 , s = 200)

plt.xlabel('age')

plt.ylabel('maximum heart rate achieved')

plt.legend()

plt.show()
plt.figure(1 , figsize  = (15 , 10))

plt.subplot(2 , 1 , 1)

plt.scatter(x = 'age' , y = 'oldpeak' , data = df.where(df['oldpeak'] >= 2) , s = 200,

           alpha = 0.5 , label = 'normal')

plt.plot(df['age'] , np.ones((df.shape[0] , 1))*2 , '-')

plt.scatter(x = 'age' , y = 'oldpeak' , data = df.where(df['oldpeak'] < 2) , s = 200,

           alpha = 0.5 , label = 'does not indicates a reversible ischaemia')



plt.annotate('ST depression of at least 2 mm to significantly indicate reversible ischaemia.', 

             xy=(30, 2), xytext=(30, 4),arrowprops=dict(facecolor='black', shrink=0.05))

plt.xlabel('age')

plt.ylabel('oldpeak')

plt.legend()



plt.subplot(2 , 1 , 2)

plt.scatter(x = 'age' ,y = 'oldpeak' , data = df.where(df['target'] == 0 ) , s = 200 ,

           marker = 'o' , alpha = 0.5 , label = 'heart disease == False')

plt.scatter(x = 'age' ,y = 'oldpeak' , data = df.where(df['target'] == 1 ) , s = 200 ,

           marker = '+' , label = 'heart disease == True' )

plt.xlabel('age')

plt.ylabel('oldpeak')

plt.legend()

plt.show()
from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix , accuracy_score , classification_report

x = df.iloc[: , :-1].values

y = df.iloc[: , -1].values

x_train , x_test , y_train , y_test = train_test_split(x , y ,

                                                      test_size = 0.3,

                                                      random_state = 134)

# import random

# def tunner(n_estimators = [] , max_features = [] , 

#           max_dept = [] , min_samples_split = [] , 

#           min_samples_leaf = []  , iteration = 20 ):

#     acc = []

#     iterr = []

#     params = dict()

#     for i in range(iteration):

#         algo = rfc(min_samples_split = random.choice(min_samples_split) ,

#                   min_samples_leaf = random.choice(min_samples_leaf) , 

#                   max_depth = random.choice(max_dept) , 

#                   max_features = random.choice(max_features), 

#                   n_estimators = random.choice(n_estimators))

#         algo.fit(x_train , y_train)

#         pred =  algo.predict(x_test)

#         acc.append(accuracy_score(y_test , pred))

#         iterr.append(i)

#         params[i] = algo.get_params()

    

#     print('Best Params :\n{}\nAccuracy\n{}'.format(params.get(acc.index(max(acc))),

#                                                    max(acc)))

   



# tunner(n_estimators = [  1000 , 500 , 100 , 10 , 50 ],

#        max_features = [9 , 7 , 2 , 3 , 12 , 13 , 5],

#        max_dept = [8 , 9 , 5 , 20 , 16 , 14 , 30],

#        min_samples_split = [2 , 10 , 3, 5 , 6],

#        min_samples_leaf = [2 , 10 , 3 , 5 , 6],

#       iteration = 20)

'''

output of tunner()

Best Params :

{'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': 5,

'max_features': 12, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0,

'min_impurity_split': None, 'min_samples_leaf': 5, 'min_samples_split': 5,

'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': None, 

'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}



Accuracy

0.945054945054945

'''
algo = rfc(bootstrap = True , class_weight = None , criterion = 'gini' , max_depth = 5,

          max_features = 12 , max_leaf_nodes = None , min_impurity_decrease = 0.0 , 

          min_impurity_split = None , min_samples_leaf = 5 , min_samples_split = 5 , 

          min_weight_fraction_leaf = 0.0 , n_estimators = 50 , n_jobs = None , 

          oob_score = False , random_state = None , verbose = 0 , warm_start = False )



algo.fit(x_train , y_train)

preds = algo.predict(x_test)



print('confusion matrix :  \n{}\n\nreport : \n{}\n\naccuracy : {}'.format(confusion_matrix(y_test , preds),

                                                                     classification_report(y_test,preds),

                                                                     accuracy_score(y_test , preds)))