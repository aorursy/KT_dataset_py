# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data1= pd.read_csv("../input/ipl-predictions2020/matches.csv")
data1.head()
data1.info()
data1.columns
data1.apply(lambda x:sum(x.isnull()))
data1.shape
num1_col=['id','season',"dl_applied","win_by_runs","win_by_wickets"]

cat1_col=data1.columns.difference(num1_col)
num1_col =['id','season',"dl_applied","win_by_runs","win_by_wickets"]

cat1_col = data1.columns.difference(num1_col)



data1[cat1_col] = data1[cat1_col].apply(lambda x: x.astype('category'))

data1[num1_col] = data1[num1_col].apply(lambda x: x.astype('float'))

data1.dtypes
data1.head()
num_data1 = data1.loc[:,num1_col]

cat_data1 = data1.loc[:,cat1_col]
from sklearn.impute import SimpleImputer

import numpy as np

imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_num.fit(num_data1)

num_data1 = pd.DataFrame(imp_num.transform(num_data1),columns=num1_col)

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

cat_data1 = pd.DataFrame(imp.fit_transform(cat_data1),columns=cat1_col)

print(num_data1.isnull().sum())

print(cat_data1.isnull().sum())
matches_data = pd.concat([num_data1,cat_data1],axis=1)

matches_data.head()
matches_data[pd.isnull(matches_data['winner'])]


matches_data.loc[241,'winner']
matches_data.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Delhi Capitals']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','DC1'],inplace=True)





matches_data.head()
sum(matches_data.player_of_match=='Yuvraj Singh')


encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'DC1':14},

          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'DC1':14},

          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'DC1':14},

          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'DC1':14,'Draw':15}}

matches_data.replace(encode, inplace=True)

matches_data.head(2)
#Find cities which are null

matches_data[pd.isnull(matches_data['city'])]
#team_match_total=matches.groupby(["team1"]).size()+matches.groupby(["team2"]).size()

#print(team_match_total.get_value(1))

xx=matches_data.groupby(["toss_winner"]).size()

print(xx.get_value(1))

yy=matches_data.groupby(["winner"]).size()

print(yy.get_value(1))
#team_match_total=matches.groupby(["team1"]).size()+matches.groupby(["team2"]).size()

#print(team_match_total.get_value(1))

x1=matches_data.groupby(["player_of_match"]).size()

print(x1.get_value(10))

y1=matches_data.groupby(["winner"]).size()

print(y1.get_value(10))
#we maintain a dictionary for future reference mapping teams

dicVal = encode['winner']

print(dicVal['MI']) #key value

print(list(dicVal.keys())[list(dicVal.values()).index(1)]) #find key by value search

matches_data.toss_decision.value_counts().plot(kind='bar')

matches = matches_data[['team1','team2','city','toss_decision','toss_winner','win_by_runs','venue','winner']]

matches.head(2)

df = pd.DataFrame(matches)

df.info()
#31 cities

df["city"].unique()
matches_data['player_of_match'].unique()
#35 venues

df["venue"].unique()
cat_list=df["city"]

encoded_data, mapping_index = pd.Series(cat_list).factorize()

print(encoded_data)

print(mapping_index)

print(mapping_index.get_loc("Bengaluru"))
cat_list1=df["venue"]

encoded_data1, mapping_index1 = pd.Series(cat_list1).factorize()

print(encoded_data1)

print(mapping_index1)

print(mapping_index1.get_loc("Rajiv Gandhi Intl. Cricket Stadium"))
cat_list2=df["toss_decision"]

encoded_data2, mapping_index2 = pd.Series(cat_list2).factorize()

#print(encoded_data2)

print(mapping_index2)

print(mapping_index2.get_loc("bat"))
#Find some stats on the match winners and toss winners

temp1=df['toss_winner'].value_counts(sort=True)

temp2=df['winner'].value_counts(sort=True)

temp3=df['win_by_runs'].value_counts(sort=True)

#Mumbai won most toss and also most matches

print('No of toss winners by each team')

for idx, val in temp1.iteritems():

    print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))

print('No of match winners by each team')

for idx, val in temp2.iteritems():

    print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))

#DC1 is delhi capitals
import matplotlib.pyplot as plt

#df['toss_winner'].hist(bins=50)
#shows that Mumbai won most matches followed by Chennai

#df['winner'].hist(bins=50)
fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel('toss_winner')

ax1.set_ylabel('Count of toss winners')

ax1.set_title("toss winners")

temp1.plot(kind='bar')



ax2 = fig.add_subplot(122)

temp2.plot(kind = 'bar')

ax2.set_xlabel('winner')

ax2.set_ylabel('Count of match winners')

ax2.set_title("Match winners")



ax3 = fig.add_subplot(121)

temp3.plot(kind = 'bar')

ax3.set_xlabel('win by runs')

ax3.set_ylabel('Count of runs')

ax3.set_title("Match winners")
df.apply(lambda x: sum(x.isnull()),axis=0) 

#find the null values in every column
#Find cities which are null

df[pd.isnull(df['city'])]
#building predictive model , convert categorical to numerical data

from sklearn.preprocessing import LabelEncoder

var_mod = ['city','toss_decision','venue']

le = LabelEncoder()

for i in var_mod:

    df[i] = le.fit_transform(df[i])

df.head(5)
df.shape
x = df.loc[:, df.columns != 'winner']

y = df.loc[:, df.columns == 'winner']

x.head()
#spliting validation set



from sklearn.model_selection import train_test_split



X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=20, random_state=0)
X_train.columns
Y_train = Y_train.astype('int')

from sklearn.linear_model import LogisticRegression

reg= LogisticRegression()

reg.fit(X_train, Y_train)
y_pred=reg.predict(X_validation)
y_pred
from sklearn.metrics import classification_report

print(classification_report(Y_validation.winner.astype('int'),y_pred))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



Xtrain = scaler.transform(X_train)

Xtest = scaler.transform(X_validation) 
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(Xtrain, Y_train)
from sklearn.model_selection import GridSearchCV



#svc_grid = SVC(class_weight='balanced',random_state=1234)

 



param_grid = {

    'n_neighbors': [1,2,3,4,5,6,7],

    'weights':['uniform','distance'],

    'metric':['euclidean','manhattan']



}



 

knn = GridSearchCV(estimator = classifier, param_grid = param_grid, cv = 10,verbose=3,n_jobs=-1)

knn.fit(Xtrain, Y_train)
#knn.best_estimator_

knn.best_params_

#knn.best_score_
y_pred1=classifier.predict(Xtest)

print(classification_report(Y_validation.winner.astype('int'),y_pred1))
from sklearn.ensemble import RandomForestClassifier



regressor = RandomForestClassifier(n_estimators=15, random_state=0)

regressor.fit(Xtrain, Y_train)

y_pred2 = regressor.predict(Xtest)

y_pred2
#y_pred1=classifier.predict(X_validation)

#print(classification_report(Y_validation.winner.astype('int'),y_pred2))

#print(classification_report (Y_validation.winner.astype('int'), np.argmax(y_pred2)))

print(classification_report(Y_validation.winner.astype('int'),y_pred2))
from sklearn.svm import SVC # "Support Vector Classifier" 

clf = SVC(kernel='linear') 

clf.fit(Xtrain,Y_train) 
y_pred3 = clf.predict(Xtest)

y_pred3
print(classification_report(Y_validation.winner.astype('int'),y_pred3))
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, y_pred2))

print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, y_pred2))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, y_pred2)))