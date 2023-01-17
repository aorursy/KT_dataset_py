import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#read in the data
df = pd.read_csv("../input/ks-projects-201801.csv")
df.head()
#split the launched column into separate dates an times, and add into the dataframe
df_split = pd.DataFrame(df.launched.str.split(' ').tolist(), columns = ['start','time'])
frames = [df, df_split]
df = pd.concat([df.reset_index(drop=True), df_split.reset_index(drop=True)], axis=1)

#convert to readily used datetime values
df['deadline'] = pd.to_datetime(df['deadline'])
df['start'] = pd.to_datetime(df['start'])
set(map(type, df.start.values.tolist()))
set(map(type, df.deadline.values.tolist()))

#now create a column and a simple equation to calculate the total days each campaign went
df['days'] = df['deadline'] - df['start']
df.head()
#clean up the days column a bit
df['days'] = df['days'] / np.timedelta64(1, 'D')
#now to drop categorical columns that are not useful for one reason or another
feat = df.drop(['ID', 'name', 'currency', 'launched', 'time', 'usd pledged','deadline', 'start'], axis = 1)
feat.head()
#checking to make sure there are no NaN
feat.isnull().sum()
#how many 'days' outliers are there?
feat.nlargest(n=10, columns='days')
#now we drop the 'days' outliers using my inelegent solution
feat= feat.drop(index=[319002, 2842, 48147, 94579, 75397, 247913, 273779,])
sns.pairplot(data=feat)
plt.figure(figsize=(10,5))
sns.violinplot(x='state', y='days', data=feat)
#seems clean enough to get some dummies for categorical variables
dummies = pd.get_dummies(data=feat, columns=['category', 'main_category', 'country'])
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(dummies.drop('state',axis=1), dummies['state'], test_size=0.25, random_state=101)
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=50, n_jobs=-1)
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)
print(classification_report(y_test, forest_predictions))
print('\n')
print(confusion_matrix(y_test, forest_predictions))