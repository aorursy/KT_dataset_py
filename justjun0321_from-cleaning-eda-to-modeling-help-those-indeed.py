import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
train.head(3)
#Check the ratio of hacdor & hacapo
len(train.loc[train.hacdor == 1])/len(train.loc[train.hacapo == 1])
# Slicing the dataset
train = train[['v2a1','hacdor','rooms','hacapo','v14a','refrig','v18q','r4h1','r4h3','r4m1','r4m3','tamhog',
               'tamviv','pisonotiene','cielorazo','abastaguano','noelec','epared1',
               'epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3','dis','idhogar','instlevel1',
               'instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9',
               'bedrooms','overcrowding','Target']]
df = train[['epared1','epared2','epared3']]
x = df.stack()
train['epared'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
train['epared'] = train['epared'].apply(lambda x : 1 if x == 'epared1' else (2 if x == 'epared2' else 3))
df = train[['etecho1','etecho2','etecho3']]
x = df.stack()
train['etecho'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
train['etecho'] = train['epared'].apply(lambda x : 1 if x == 'etecho1' else (2 if x == 'etecho2' else 3))
df = train[['eviv1','eviv2','eviv3']]
x = df.stack()
train['eviv'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
train['eviv'] = train['epared'].apply(lambda x : 1 if x == 'eviv1' else (2 if x == 'eviv2' else 3))
train.drop(['epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3'],axis=1,inplace=True)
train.head()
train.info()
train.idhogar.value_counts()
train.drop('idhogar',axis=1,inplace=True)
pd.isnull(train).sum()
train.v2a1.describe()
sns.boxplot(train.v2a1)
train['unavailable_v2a1'] = train.v2a1.apply(lambda x: 1 if pd.isnull(x) else 0)
train['v2a1'] = train['v2a1'].fillna(130000)
sns.countplot(train.Target)
plt.title('Distribution of Target')
corr = train.corr()
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
X = train.drop('Target',axis=1)
y = train[['Target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create the model and assign it to the variable model.
model = DecisionTreeClassifier()

# Fit the model.
model.fit(X_train,y_train)
# Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X_test)

# Calculate the accuracy and assign it to the variable acc.
print('The accuracy for the model is:', accuracy_score(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='2.0f')
plt.xlabel('Predicted label')
plt.ylabel('True label')
features = X_train.columns[:X_train.shape[1]]
importances = model.feature_importances_
indices = np.argsort(importances)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
model = DecisionTreeClassifier()

param_dist = {"max_depth": [3,5,7,9,10,15,20,None],
              "min_samples_split": [2,5,10,15],
              "min_samples_leaf": [1,3,5]}

Search = RandomizedSearchCV(model, param_distributions=param_dist)

# Fit the model on the training data
Search.fit(X_train, y_train)

# Make predictions on the test data
preds = Search.best_estimator_.predict(X_test)

print('The accuracy for the model is:', accuracy_score(y_test,preds))
sns.heatmap(confusion_matrix(y_test,preds),annot=True,fmt='2.0f')
plt.xlabel('Predicted label')
plt.ylabel('True label')
features = X_train.columns[:X_train.shape[1]]
importances = Search.best_estimator_.feature_importances_
indices = np.argsort(importances)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
test_df = test[['v2a1','hacdor','rooms','hacapo','v14a','refrig','v18q','r4h3','r4m1','r4m3','tamhog',
               'tamviv','pisonotiene','cielorazo','abastaguano','noelec','epared1','bedrooms','overcrowding','r4h1',
               'epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3','dis','instlevel1',
               'instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9']]

df = test_df[['epared1','epared2','epared3']]
x = df.stack()
test_df['epared'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
test_df['epared'] = test_df['epared'].apply(lambda x : 1 if x == 'epared1' else (2 if x == 'epared2' else 3))

df = test_df[['etecho1','etecho2','etecho3']]
x = df.stack()
test_df['etecho'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
test_df['etecho'] = test_df['epared'].apply(lambda x : 1 if x == 'etecho1' else (2 if x == 'etecho2' else 3))

df = test_df[['eviv1','eviv2','eviv3']]
x = df.stack()
test_df['eviv'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
test_df['eviv'] = test_df['epared'].apply(lambda x : 1 if x == 'eviv1' else (2 if x == 'eviv2' else 3))


test_df.drop(['epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3'],axis=1,inplace=True)

test_df['unavailable_v2a1'] = test_df.v2a1.apply(lambda x: 1 if pd.isnull(x) else 0)
test_df['v2a1'] = test_df['v2a1'].fillna(0)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction = model.predict(test_df)
submission1 = pd.read_csv('../input/sample_submission.csv')
submission1['Target'] = prediction
submission1.to_csv('submission1.csv',index=False)
model = DecisionTreeClassifier()

param_dist = {"max_depth": [3,5,7,9,10,15,20,None],
              "min_samples_split": [2,5,10,15],
              "min_samples_leaf": [1,3,5]}

Search = RandomizedSearchCV(model, param_distributions=param_dist)

# Fit the model on the training data
Search.fit(X_train, y_train)

# Make predictions on the test data
preds = Search.best_estimator_.predict(test_df)

submission2 = pd.read_csv('../input/sample_submission.csv')
submission2['Target'] = prediction
submission2.to_csv('submission2.csv',index=False)