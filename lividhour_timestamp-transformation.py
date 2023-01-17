from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("../input/advertising.csv")
df = df[['Timestamp', 'Clicked on Ad']]
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
df['Timestamp'] = df['Timestamp'].astype(np.int64)//10**9
df.head()
target = df['Clicked on Ad']
train = df['Timestamp'].values.reshape(-1, 1)
logreg = LogisticRegression(solver='lbfgs')
cross_val_score(logreg, train, target, cv=10, scoring='accuracy')               
df = pd.read_csv("../input/advertising.csv")
df = df[['Timestamp', 'Clicked on Ad']]
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day     
df['Hour'] = df['Timestamp'].dt.hour   
df['Minute'] = df['Timestamp'].dt.minute  
df['Second'] = df['Timestamp'].dt.second 
df["Weekday"] = df['Timestamp'].dt.dayofweek 

df.head()

train = pd.get_dummies(df, columns = ['Year', 'Month' ,'Day', 'Hour', 'Minute', 'Second', 'Weekday'], drop_first=True)
train.drop(['Clicked on Ad', 'Timestamp'], axis = 1, inplace=True)

rfecv = RFECV(estimator=logreg, step=1, cv=10, scoring='accuracy')
rfecv.fit(train, target)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Best features:', ", ".join(list(train.columns[rfecv.support_])))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

train = train[list(train.columns[rfecv.support_])]
cross_val_score(logreg, train, target, cv=10, scoring='accuracy').mean()  
df = pd.read_csv("../input/advertising.csv")
df = df[['Timestamp', 'Clicked on Ad']]
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day     
df['Hour'] = df['Timestamp'].dt.hour   
df['Minute'] = df['Timestamp'].dt.minute  
df['Second'] = df['Timestamp'].dt.second 
df["Weekday"] = df['Timestamp'].dt.dayofweek 

df['Month_sin'] = np.sin(2*np.pi*(df['Month']-1)/12)
df['Month_cos'] = np.cos(2*np.pi*(df['Month']-1)/12)

df['Day_sin'] = np.sin(2*np.pi*(df['Day']-1)/30)
df['Day_cos'] = np.cos(2*np.pi*(df['Day']-1)/30)

df['Hour_sin'] = np.sin(2*np.pi*(df['Hour'])/24)
df['Hour_cos'] = np.cos(2*np.pi*(df['Hour'])/24)

df['Minute_sin'] = np.sin(2*np.pi*(df['Minute'])/60)
df['Minute_cos'] = np.cos(2*np.pi*(df['Minute'])/60)

df['Second_sin'] = np.sin(2*np.pi*(df['Second'])/60)
df['Second_cos'] = np.cos(2*np.pi*(df['Second'])/60)

df['Weekday_sin'] = np.sin(2*np.pi*(df['Weekday'])/7)
df['Weekday_cos'] = np.cos(2*np.pi*(df['Weekday'])/7)

df.head()
train = df.drop(['Clicked on Ad', 'Timestamp', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Weekday'], axis = 1)

std = StandardScaler()
scaled = std.fit_transform(train[['Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Hour_sin', 'Hour_cos', 'Minute_sin','Minute_cos', 'Second_sin', 'Second_cos', 'Weekday_sin', 'Weekday_cos']])
scaled = pd.DataFrame(scaled, columns=['Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Hour_sin', 'Hour_cos', 'Minute_sin','Minute_cos', 'Second_sin', 'Second_cos', 'Weekday_sin', 'Weekday_cos'])
train[scaled.columns] = scaled[scaled.columns]
train.describe()

train.describe()
rfecv = RFECV(estimator=logreg, step=1, cv=10, scoring='accuracy')
rfecv.fit(train, target)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Best features:', ", ".join(list(train.columns[rfecv.support_])))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

train = train[list(train.columns[rfecv.support_])]
cross_val_score(logreg, train, target, cv=10, scoring='accuracy').mean() 