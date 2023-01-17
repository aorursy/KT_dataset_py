import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../working"))


train = pd.read_csv('../input/train_V2.csv')
train.head()
#ID Metrics: matchId, groupId, Id(Player unique ID)
print("Number of matches played:" , train['matchId'].nunique())
print("Number of unique players:" , train['Id'].nunique())
#Number of matches played in diffrent GameModes/matchType
train.groupby(['matchType']).matchId.nunique().sort_values(axis=0)
#Number of max,min,avg,etc... group size in matches in diffrent GameModes/matchType
train.groupby(['matchType','matchId']).groupId.nunique().groupby(['matchType']).describe()
#['squad-fpp', 'duo', 'solo-fpp', 'squad', 'duo-fpp', 'solo','normal-squad-fpp', 'crashfpp', 'flaretpp', 'normal-solo-fpp','flarefpp', 'normal-duo-fpp', 'normal-duo', 'normal-squad','crashtpp', 'normal-solo']
#There is no player who played in more than one matchtype
matchTypes=train.matchType.unique()
matchTypes
for i in matchTypes:
    for j in matchTypes:
        if i!=j:
            temp1 = train[train.matchType == i][['Id']]
            temp2 = train[train.matchType == j][['Id']]
            print(i,"AND",j,"   :",temp1.merge(temp2,on='Id',how='inner').size)


#Distance Metrics: walkDistance,swimDistance,rideDistance
#walkDistance Metric distribution
def roundofvalue(x):
    return pd.Series([np.round(i,2) for i in x],index = x.index)
x=train[['matchType','walkDistance']].groupby('matchType').describe()
x['walkDistance']['count']  =   roundofvalue(x['walkDistance']['count'])
x['walkDistance']['mean']  =   roundofvalue(x['walkDistance']['mean'])
x['walkDistance']['std']  =   roundofvalue(x['walkDistance']['std'])
x['walkDistance']['25%']  =   roundofvalue(x['walkDistance']['25%'])
x['walkDistance']['50%']  =   roundofvalue(x['walkDistance']['50%'])
x['walkDistance']['75%']  =   roundofvalue(x['walkDistance']['75%'])
x['walkDistance']['max']  =   roundofvalue(x['walkDistance']['max'])

#swimDistance Metric distribution
def roundofvalue(x):
    return pd.Series([np.round(i,2) for i in x],index = x.index)
y=train[['matchType','swimDistance']].groupby('matchType').describe()
y['swimDistance']['count']  =   roundofvalue(x['walkDistance']['count'])
y['swimDistance']['mean']  =   roundofvalue(x['walkDistance']['mean'])
y['swimDistance']['std']  =   roundofvalue(x['walkDistance']['std'])
y['swimDistance']['25%']  =   roundofvalue(x['walkDistance']['25%'])
y['swimDistance']['50%']  =   roundofvalue(x['walkDistance']['50%'])
y['swimDistance']['75%']  =   roundofvalue(x['walkDistance']['75%'])
y['swimDistance']['max']  =   roundofvalue(x['walkDistance']['max'])

#rideDistance Metric distribution
def roundofvalue(x):
    return pd.Series([np.round(i,2) for i in x],index = x.index)
z=train[['matchType','rideDistance']].groupby('matchType').describe()
z['rideDistance']['count']  =   roundofvalue(x['walkDistance']['count'])
z['rideDistance']['mean']  =   roundofvalue(x['walkDistance']['mean'])
z['rideDistance']['std']  =   roundofvalue(x['walkDistance']['std'])
z['rideDistance']['25%']  =   roundofvalue(x['walkDistance']['25%'])
z['rideDistance']['50%']  =   roundofvalue(x['walkDistance']['50%'])
z['rideDistance']['75%']  =   roundofvalue(x['walkDistance']['75%'])
z['rideDistance']['max']  =   roundofvalue(x['walkDistance']['max'])
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

display_side_by_side(x,y,z)


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))

# Load the example planets dataset
df_distance = train[['matchType','walkDistance','swimDistance','rideDistance']]

# Plot the orbital period with horizontal boxes
sns.boxplot(x="walkDistance", y="matchType", data=df_distance,whis="range", palette="vlag")

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
# Load the example planets dataset
df_distance = train[['matchType','walkDistance','swimDistance','rideDistance']]
fig, ax =plt.subplots(1,3,figsize=(25, 10)) 
# Plot the orbital period with horizontal boxes
sns.boxplot(x="walkDistance", y="matchType", data=df_distance,whis="range", palette="vlag", ax=ax[0])
sns.boxplot(x="swimDistance", y="matchType", data=df_distance,whis="range", palette="vlag", ax=ax[1])
sns.boxplot(x="rideDistance", y="matchType", data=df_distance,whis="range", palette="vlag", ax=ax[2])
fig.show()

vif.sort_values('VIF Factor',ascending = False)
train = train[train['winPlacePerc'].isna() == False]
features = train.drop(['winPlacePerc'] , axis = 1)._get_numeric_data()  
#train[['maxPlace','winPoints','rankPoints','killPoints','killPlace','DBNOs','headshotKills','rideDistance']]
target   = train[['winPlacePerc']]

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size = 0.3)
#import linear regression classifier, initialize and fit the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

y_pred = regressor.predict(X_test)
X_test.head()
# The next is to evaluate the classifier using metrics such as the mean square error 
# and the coefficient of determination R square

from sklearn.metrics import mean_squared_error,r2_score

#The coefficients 
print(features.columns.values)
print('Coefficients: \n', regressor.coef_)

#The mean squared error
print('The mean squared error: {:2f}'.format(mean_squared_error(y_test , y_pred)))

#Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(r2_score(y_test ,y_pred)))


test = pd.read_csv('../input/test_V2.csv')
test_pred = regressor.predict(test._get_numeric_data())

submission_df = pd.concat([test['Id'], pd.DataFrame(test_pred)], axis=1, sort=False)
submission_df.rename(columns={"0":"winPlacePerc"})
submission_df.head()
submission_df.to_csv('sample_submission.csv',index =False)
