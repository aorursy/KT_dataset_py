import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')

test = pd.read_csv('../input/test_V2.csv')
# shape of train data

train.shape
#shape of test data.



test.shape
train.columns
train.info()
test.info()
train.isnull().sum()
train[train.winPlacePerc.isnull()]

train.drop(2744604,inplace=True)
train.isnull().sum()
# Number of matches played in  train  data.

#matchId

len(train.matchId.unique())
train.groupby('matchId')['matchId'].count()
#visualization

group = train[train['matchId']=='a329ac99449ad7']['groupId'].value_counts().sort_values(ascending=False)

plt.figure()

plt.bar(group.index,group.values)

plt.xticks(rotation=90)

plt.xlabel('GroupId')

plt.ylabel('Count')

plt.title('Number of Group Members in One Match')

plt.show()



print('Min number of group members is: ',min(group.values))

print('Max number of group members is: ',max(group.values))
import seaborn as sns

print("The average person kills {:.4f} players, 99% of people have {} kills or less, while the most kills ever recorded is {}.".format(train['kills'].mean(),train['kills'].quantile(0.99), train['kills'].max()))

data = train.copy()

data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'

plt.figure(figsize=(10,5))

sns.countplot(data['kills'].astype('str').sort_values())

plt.title("Kill Count",fontsize=15)

plt.show()
data = train.copy()

data = data[data['kills']==0]

plt.figure(figsize=(10,5))

plt.title("Damage Dealt by 0 killers",fontsize=15)

sns.distplot(data['damageDealt'])

plt.show()
print("{} players ({:.4f}%) have won without a single kill!".format(len(data[data['winPlacePerc']==1]), 100*len(data[data['winPlacePerc']==1])/len(train)))



data1 = train[train['damageDealt'] == 0].copy()

print("{} players ({:.4f}%) have won without dealing damage!".format(len(data1[data1['winPlacePerc']==1]), 100*len(data1[data1['winPlacePerc']==1])/len(train)))
def visualize(col_name, num_bin=10):

    '''

    Function for visualization

    '''

    title_name = col_name[0].upper() + col_name[1:]

    f, ax = plt.subplots()

    plt.xlabel(title_name)

    plt.ylabel('log Count')

    ax.set_yscale('log')

    train.hist(column=col_name,ax=ax,bins=num_bin)

    plt.title('Histogram of ' + title_name)

    tmp = train[col_name].value_counts().sort_values(ascending=False)



    print('Min value of ' + title_name + ' is: ',min(tmp.index))

    print('Max value of ' + title_name + ' is: ',max(tmp.index))
visualize('roadKills')
visualize('assists')
visualize('teamKills')
visualize('killStreaks')
visualize('longestKill',num_bin=100)
sns.jointplot(x="winPlacePerc", y="kills", data=train,ratio=3, color="r")

plt.show()
print("The average person walks for {:.1f}m, 99% of people have walked {}m or less, while the marathoner champion walked for {}m.".format(train['walkDistance'].mean(), train['walkDistance'].quantile(0.99), train['walkDistance'].max()))
data = train.copy()

data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]

plt.figure(figsize=(15,10))

plt.title("Walking Distance Distribution",fontsize=15)

sns.distplot(data['walkDistance'])

plt.show()
print("{} players ({:.4f}%) walked 0 meters. This means that they die before even taking a step or they are ready (more possible).".format(len(data[data['walkDistance'] == 0]), 100*len(data1[data1['walkDistance']==0])/len(train)))
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=train, ratio=3, color="lime")

plt.show()
visualize('DBNOs',num_bin=50)
plt.subplots(figsize=(12, 4))

sns.distplot(train.swimDistance,bins=10)
plt.figure(figsize=(12,4))

sns.distplot(train['weaponsAcquired'], bins=100)

plt.show()
visualize('revives',num_bin=40)
plt.subplots(figsize=(12,4))

sns.distplot(train.heals, bins=20)
# Create the dummy variable for categorical variable present in our data set.





#train



train=pd.get_dummies(train,columns=['matchType'])



#test



test=pd.get_dummies(test,columns=['matchType'])
train.info()
#feature selection

from sklearn.ensemble import RandomForestRegressor
#before that first drop some unnecessary columns.





train.drop(['Id','groupId','matchId'],axis=1,inplace=True)



test_id=test['Id']

test.drop(['Id','groupId','matchId'],axis=1,inplace=True)
train.info()

test.info()
sample = 400000

df_sample = train.sample(sample)
y=df_sample['winPlacePerc']

df = df_sample.drop(columns = ['winPlacePerc'])

df.info()
from sklearn.model_selection import train_test_split



X_train,X_valid,y_train,y_valid=train_test_split(df,y,test_size=0.3,random_state=40)
from sklearn.metrics import mean_absolute_error
def score(m : RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 

           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)


m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)



m1.fit(X_train, y_train)

score(m1)
importance=m1.feature_importances_
data=pd.DataFrame(sorted(zip(m1.feature_importances_, df.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 8))

sns.barplot(x="Value", y="Feature", data=data.sort_values(by="Value", ascending=False))
#create new model based on these features(test).

new_data=data.sort_values(by='Value',ascending=False)[:25]
plt.subplots(figsize=(15,8))

sns.barplot(x='Value',y='Feature',data=new_data)
cols=new_data.Feature.values
#recreate for validation set

X_train,X_valid,y_train,y_valid=train_test_split(df[cols],y,test_size=0.3,random_state=40)
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)

m1.fit(X_train, y_train)

score(m1)


from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor



model = RandomForestRegressor(n_estimators=10)

model.fit(X_train, y_train)

accuracy = model.score(X_valid,y_valid)

print('accuracy Random forest',accuracy*100,'%')



y_pred = model.predict(X_valid)



# getting the r2_score

r2 = r2_score(y_valid, y_pred)

print("The r2 score :", r2)

model1 = LinearRegression()

model1.fit(X_train, y_train)

accuracy = model1.score(X_valid,y_valid)

print('accuracy Linear Regressor',accuracy*100,'%')



y_pred1 = model.predict(X_valid)



# getting the r2_score

r2 = r2_score(y_valid, y_pred1)

print("The r2 score :", r2)



model2 = GradientBoostingRegressor(learning_rate=0.8)

model2.fit(X_train, y_train)

accuracy = model2.score(X_valid,y_valid)

print('accuracy Gradient',accuracy*100,'%')



y_pred2 = model2.predict(X_valid)





# getting the r2_score

r2 = r2_score(y_valid, y_pred2)

print("The r2 score :", r2)

y_final=train['winPlacePerc']

df_final = train.drop(columns = ['winPlacePerc'])

df_final.shape
X_train,X_valid,y_train,y_valid=train_test_split(df_final,y_final,test_size=0.3,random_state=40)
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)

m1.fit(X_train, y_train)

score(m1)
# Replace all the infnite value from our test data. In Case ?



test.replace([np.inf, -np.inf], np.nan)

test.isnull().sum()
predictions = np.clip(a = m1.predict(test), a_min = 0.0, a_max = 1.0)

pred_df = pd.DataFrame({'Id' : test_id, 'winPlacePerc' : predictions})



# Create submission file

pred_df.to_csv("submission.csv", index=False)
final_output=pd.read_csv('submission.csv')
final_output.head()