# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/my-new-flights/my_new_flights.csv')
df = df.sample(frac=0.01, replace=True, random_state=1)  # 50k
df.head()
df.shape
# เนื่องจากไม่ได้ใช้ข้อมูล cancelled และข้อมูลมีมากไม่เหมาะการทำ onehot 

df = df.drop(columns=['CANCELLATION_REASON', 'CANCELLED','ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 

                      'TAIL_NUMBER', 'ARRIVAL_TIME', 'FLIGHT_NUMBER','DIVERTED',

                      'ELAPSED_TIME','AIR_TIME','WHEELS_ON','TAXI_IN','AIR_SYSTEM_DELAY', 

                      'SECURITY_DELAY','AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])
df = df.dropna()

df= df.reset_index(drop=True)
df.shape
#เปลี่ยนคอลัมน์ให้เป็น classification

df['FLIGHT_DELAY'] = np.where(df['ARRIVAL_DELAY'] > 0, 1,0)

del df['ARRIVAL_DELAY']
df.groupby('FLIGHT_DELAY').size()
#Day of month

df2 = df[df['FLIGHT_DELAY'] ==1]



day_tmp = []

for n in  df2['DAY'].tolist() :

    if n < 11 :

        day_tmp.append("begin")

    elif n < 21:

        day_tmp.append("middle")

    else:

         day_tmp.append("end")



df2['DAY_CLASS'] = day_tmp

df2['DAY_CLASS'].value_counts()   



plt.bar(df2['DAY_CLASS'].value_counts().index.tolist(),

df2['DAY_CLASS'].value_counts().values.tolist(),

color=['blue'])

plt.title("Class label distribution")

plt.ylabel('Frequency')

plt.xlabel('Class label')

plt.tight_layout()

#เนื่องจาก การ plotค่าของ day_tmpที่แบ่งclass เป็น 3 ช่วง พบว่าไม่มีความแตกต่างกันอย่างมีนัยสำคัญ ซึ่งจะถูกดรอปออกในภายหลัง
#Day of week

plt.bar(df2['DAY_OF_WEEK'].value_counts().index.tolist(),

df2['DAY_OF_WEEK'].value_counts().values.tolist(),

color=['blue'])

plt.title("Class label distribution")

plt.ylabel('Frequency')

plt.xlabel('Class label')

plt.tight_layout()



# 3 class 4,5= d_high, 1,2,3,7=d_medium  6= d_low
day_tmp = []

for n in  df['DAY_OF_WEEK'].tolist() :

    if n in [4,5] :

        day_tmp.append('d_high')

    elif n in [1, 2, 3, 7]  :

        day_tmp.append('d_medium')

    else: 

        day_tmp.append('d_low')

    



df['day_delay'] = day_tmp

df['day_delay'].value_counts()   
plt.bar(df2['MONTH'].value_counts().index.tolist(),

df2['MONTH'].value_counts().values.tolist(),

color=['blue'])

plt.title("Class label distribution")

plt.ylabel('Frequency')

plt.xlabel('Class label')

plt.tight_layout()
month_tmp = []

for n in  df['MONTH'].tolist() :

    if n in [9,10,11] :

        month_tmp.append('M_low')

    elif n in [2,4,5] :

        month_tmp.append('M_medium')

    else: 

        month_tmp.append('M_high')

        

df['month_class'] = month_tmp

df['month_class'].value_counts()   
# SCHEDULED_DEPARTURE

def time_to_string(n):

    if n  < 100 :

        return('0')

    elif n < 200 :

        return('1')

    elif n < 300 :

        return('2')

    elif n < 400 :

        return('3')    

    elif n < 500 :

        return('4')        

    elif n < 600 :

        return('5')

    elif n < 700 :

        return('6')

    elif n < 800 :

        return('7')

    elif n < 900 :

        return('8')    

    elif n < 1000 :

        return('9')

    elif n < 1100 :

        return('10')

    elif n < 1200 :

        return('11')

    elif n < 1300 :

        return('12')

    elif n < 1400 :

        return('13')

    elif n < 1500 :

        return('14')    

    elif n < 1600 :

        return('15')        

    elif n < 1700 :

        return('16')

    elif n < 1800 :

        return('17')

    elif n < 1900 :

        return('18')

    elif n < 2000 :

        return('19')    

    elif n < 2100 :

        return('20')

    elif n < 2200 :

        return('21')

    elif n < 2300 :

        return('22')

    else: 

        return('23')

    

time_tmp = []

for n in  df2['SCHEDULED_DEPARTURE'].tolist() :

        time_tmp.append(time_to_string(n))



        

df2['time_tmp'] = time_tmp

df2['time_tmp'].value_counts()   



plt.bar(df2['time_tmp'].value_counts().index.tolist(),

df2['time_tmp'].value_counts().values.tolist(),

color=['blue'])

plt.title("Class label distribution")

plt.ylabel('Frequency')

plt.xlabel('Class label')

plt.tight_layout()
hour_tmp = []

for n in  df['SCHEDULED_DEPARTURE'].tolist() :

    n = time_to_string(n)

    if n in ['17','15', '19', '18', '16', '13'] :

        hour_tmp.append('H_high')

    elif n in ['14', '12', '11', '10'] :

        hour_tmp.append('H_medium')

    elif n in ['8', '20', '9', '7','6', '21'] :

        hour_tmp.append('H_low')

    else: 

        hour_tmp.append('H_lowest')

        

df['hour_class'] = hour_tmp

df['hour_class'].value_counts()  
df.head()
df.columns
#drop year, month, day, day of week, airline  เพราะว่าเราทำเป็นclass แล้ว

df = df.drop(columns=['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE'])
df.dtypes

#ทำ object type ให้เป็นอยู่ในรูปแบบ onehot



from  sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories='auto')

feature_arr = ohe.fit_transform(df[['month_class', 'hour_class','day_delay']]).toarray() # list of one-hot-encoder

#print(feature_arr)

feature_labels = ohe.categories_  # list of new column name  

#print(feature_labels)

#feature_labels = np.array(feature_labels).ravel() # no effect

feature_labels =  np.concatenate((feature_labels), axis=None)

#print(feature_labels)

features = pd.DataFrame(feature_arr, columns=feature_labels)

df = pd.concat([features,df], axis=1, sort=False)
import seaborn as sns

%matplotlib inline



# calculate the correlation matrix

corr = df.corr()



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
numeric_cols = ['DEPARTURE_DELAY',

                'TAXI_OUT','WHEELS_OFF','SCHEDULED_TIME',

                'DISTANCE','last_dep_delay','last_arr_delay', 'SCHEDULED_DEPARTURE','DEPARTURE_TIME', 'SCHEDULED_ARRIVAL']



for col in numeric_cols:

    df[col] = pd.to_numeric(df[col])
for col in numeric_cols:

  #  print("Column name:  " + col)

    q1= df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3-q1

    lower_bound = q1 -(1.5 * iqr)

    upper_bound = q3 +(1.5 * iqr)

 #   print('q1 = {}'.format(q1))

 #   print('q3 = {}'.format(q3))

 #   print('iqr = {}'.format(iqr))

#     print('lower bound = {}, upper bound = {}'.format(lower_bound, upper_bound))

    outlier_row_indice = df[(df[col] < lower_bound) | (df[col]>upper_bound)].index

#     print('number of outliers = {}'.format(len(outlier_row_indice)))

#     print('indices of outliers = ', outlier_row_indice.to_list())

#    print("######################################\n")
print(numeric_cols)

fig, axes = plt.subplots(figsize=(18, 10), nrows=3, ncols=3, squeeze=0)

i=0

for ax, col in zip(axes.reshape(-1), numeric_cols):

      ax.boxplot(df[col], labels=[col], sym='k.')
y = df['FLIGHT_DELAY'].tolist()



del df['FLIGHT_DELAY']

del df['month_class']

del df['hour_class']

del df['day_delay']
df.columns

df.dtypes
#df = pd.concat([df,features], axis=1)

X = df.iloc[:, :].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier





lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75]



for learning_rate in lr_list:

    gb_clf = GradientBoostingClassifier(n_estimators=10, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)

    gb_clf.fit(X_train, y_train)



    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))

    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)

gb_clf2.fit(X_train, y_train)

predictions = gb_clf2.predict(X_test)



print("Confusion Matrix:")

print(confusion_matrix(y_test, predictions))



print("Classification Report")

print(classification_report(y_test, predictions))

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

#create new a knn model

knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors

params_knn = {'n_neighbors': np.arange(1, 5)}

#use gridsearch to test all values for n_neighbors

knn_gs = GridSearchCV(knn, params_knn, cv=5)

#fit model to training data

knn_gs.fit(X_train, y_train)



#save best model

knn_best = knn_gs.best_estimator_

#check best n_neigbors value

print(knn_gs.best_params_)
from sklearn.ensemble import RandomForestClassifier

#create a new random forest classifier

rf = RandomForestClassifier()

#create a dictionary of all values we want to test for n_estimators

params_rf = {'n_estimators': [100, 200]}

#use gridsearch to test all values for n_estimators

rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data

rf_gs.fit(X_train, y_train)



#save best model

rf_best = rf_gs.best_estimator_

#check best n_estimators value

print(rf_gs.best_params_)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import metrics
params = {

    'penalty':['l2'], # l1 is Lasso, l2 is Ridge

    'class_weight' : ['dict', 'balanced'],

    'C': [0.01,0.10,0.25,0.50,0.75,1.0],#np.linspace(0.00002,1,100),

    'solver' : ['newton-cg', 'lbfgs',  'sag', 'saga']

}



lr = LogisticRegression()

lr_gs = GridSearchCV(lr, params, cv=3, verbose=1).fit(X_train, y_train)



print ("Best Params", lr_gs.best_params_)

print ("Best Score", lr_gs.best_score_)



lr_best = LogisticRegression(C= 0.75, class_weight = 'dict', penalty = 'l2', solver = 'lbfgs')

lr_best.fit(X_train, y_train)
print('knn: {}'.format(knn_best.score(X_test, y_test)))

print('rf: {}'.format(rf_best.score(X_test, y_test)))

print('log_reg: {}'.format(lr_best.score(X_test, y_test)))

from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(X_train, y_train) 

clf.score(X_test, y_test)
from sklearn.ensemble import VotingClassifier

#create a dictionary of our models

estimators=[('gb', gb_clf2), ('clf', clf), ('log_reg', lr_best)]

#create our voting classifier, inputting our models

ensemble = VotingClassifier(estimators, voting='hard')
#fit model to training data

ensemble.fit(X_train, y_train)

#test our model on the test data

ensemble.score(X_test, y_test)
from catboost import CatBoostClassifier



params = {'loss_function':'Logloss', # objective function

          'eval_metric':'AUC', # metric

          'verbose': 200, # output to stdout info about training process every 200 iterations

          'random_seed': 1

         }

classifier = CatBoostClassifier(**params)

classifier.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)

          eval_set=(X_valid, y_valid), # data to validate on

          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score

          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)

         );
# Predicting the Test Set results

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred.round())

print(cm)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1))
X.shape