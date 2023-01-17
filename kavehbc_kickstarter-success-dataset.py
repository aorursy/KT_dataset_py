import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("ks-projects-201612.csv", encoding='iso-8859-1', low_memory=False)

# to check if the dataset is loaded correctly.
df.head()
#The last four columns are unnamed, so I remove it.
df = df.iloc[:,:-4]

#Stripping white space in the column names
df.columns = [col.strip() for col in df.columns.tolist()]

#Renaming a column name for easier usage in Pandas
df.rename(columns = {'usd pledged':'usd_pledged'}, inplace = True)

#Converting the data types of the dataset from string to the proper formats
df['usd_pledged'] = pd.to_numeric(df.usd_pledged, downcast='float', errors='coerce')
df['goal'] = pd.to_numeric(df.goal, downcast='float', errors='coerce')
df['backers'] = pd.to_numeric(df.backers, downcast='integer', errors='coerce')
df['launched'] = pd.to_datetime(df.launched, errors='coerce')
df['deadline'] = pd.to_datetime(df.deadline, errors='coerce')

#Checking the columns with null values
df.isnull().sum()
df = df.dropna(axis=0, subset=['name', 'category', 'deadline', 'goal', 'launched', 'backers', 'usd_pledged'])
df.isnull().sum()
#Now failed & successful
print("Initial states: ", df.state.unique())
df.loc[df.state == 'canceled','state'] = 'failed'
df.loc[df.state == 'suspended','state'] = 'failed'
df = df[(df.state != 'live')]
print("Merged states: ", df.state.unique())
features = df.copy()
features['success_label'] = np.where(features.state == 'successful', 1, 0)
features['name_length'] = features.name.str.len()
features['contains_title'] = pd.get_dummies(features.name.str.istitle(), drop_first=True)
features['running_time'] = (features.deadline.dt.date - features.launched.dt.date).dt.days

features = features.drop(columns=['ID','name','category', 'backers', 'pledged', 'usd_pledged', 'main_category','currency','deadline','launched','state','country'])
features.head()
total_features = features.shape[0]
suc_label = len(features.success_label[features['success_label']==1])
fai_label = len(features.success_label[features['success_label']==0])

print("Total: ", total_features)
print("Success Label: ", suc_label, " | " , suc_label/total_features*100 , "%")
print("Failed Label", fai_label, " | " , fai_label/total_features*100 , "%")
#Training and Test set are defined
X = features.drop(columns=['success_label'])
y = features.success_label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Logistic Regression Classification
lr = LogisticRegression()

fit = lr.fit(X_train, y_train)

y_predicted = lr.predict(X_test)

print('\n Confusion Matrix of Classification')
print(pd.crosstab(y_predicted, y_test))

# Cross Validation
scores = cross_val_score(lr, X, y, cv=10)

print(scores)