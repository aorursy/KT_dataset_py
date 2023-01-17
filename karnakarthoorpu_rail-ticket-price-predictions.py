import numpy as np

import pandas as pd

import seaborn as sbn

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
full_data = pd.read_csv('../input/renfe.csv')
full_data.head()
full_data.isnull().sum()
full_data['train_class'] = full_data['train_class'].fillna('N_A')

full_data['fare'] = full_data['fare'].fillna('N_A')
full_data.drop(['Unnamed: 0','insert_date'],axis = 1,inplace=True)
fig1,ax1 = plt.subplots()



plt.pie(full_data.origin.value_counts().values,labels = full_data.origin.value_counts().index,

        autopct='%1.1f%%', startangle=90, pctdistance=0.85)



#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
fig1,ax1 = plt.subplots()





plt.pie(full_data.destination.value_counts().values,labels = full_data.destination.value_counts().index,

        autopct='%1.1f%%', startangle=90, pctdistance=0.85)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
full_data['time_dif'] = (pd.to_datetime(full_data.end_date) - pd.to_datetime(full_data.start_date)).astype('timedelta64[h]')
full_data.head()
full_data = pd.get_dummies(full_data,columns = ['origin','destination','train_type','train_class','fare'],

                                                  prefix = ["origin_",'dest','tt','tc','fare'])

full_data.drop(['start_date','end_date'],axis = 1, inplace=True)
train_data = full_data[full_data['price'].notnull()]

test_data = full_data[full_data['price'].isnull()]


print(f"shape of train_data = {train_data.shape}")

print(f"shape of test_data = {test_data.shape}")

print(f"shape of full_data: {full_data.shape} ")
y_labels = train_data['price']

train_data.drop(['price'],axis = 1, inplace = True)

test_data.drop(['price'],axis = 1,inplace = True)
x_train,x_test,y_train,y_test = train_test_split(train_data,y_labels,random_state =45,test_size = 0.33 )
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score



model_rf = RandomForestRegressor(max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2)

model_gb = GradientBoostingRegressor(loss='quantile', learning_rate=0.0001, n_estimators=50, max_features='log2', min_samples_split=2, max_depth=1)

model_ada_tree_backing = DecisionTreeRegressor(max_features='sqrt', splitter='random', min_samples_split=4, max_depth=3)

model_ab = AdaBoostRegressor(model_ada_tree_backing, learning_rate=0.1, loss='square', n_estimators=1000)

def fit_predict(model,X_train,X_test,Y_train,Y_test):

    model.fit(X_train,Y_train)

    return r2_score(model.predict(X_test),Y_test)

    
print("accuracy of Randome forest = ",fit_predict(model_rf,x_train,x_test,y_train,y_test))

print("accuracy of gradient_boosting = ",fit_predict(model_gb,x_train,x_test,y_train,y_test))

print("accuracy of addaptive Boosting = ",fit_predict(model_ab,x_train,x_test,y_train,y_test))