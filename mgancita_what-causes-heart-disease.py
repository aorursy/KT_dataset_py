import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../input/heart.csv')
df.head()
df.shape
df.isna().sum()
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))

fig.suptitle("Categorical Variable Distributions", fontsize=16)



x = ['Female','Male']

y = df.sex.value_counts(sort = False).values

axes[0][0].bar(x,y)

axes[0][0].set_title('Sex')



x = ['Typical','Atypical','Non-Anginal','Aysyptomatic']

y = df.cp.value_counts(sort = False).values

axes[0][1].bar(x,y)

axes[0][1].set_title('Chest Pain')



x = ['Healthy','Unhealthy']

y = df.fbs.value_counts(sort = False).values

axes[0][2].bar(x,y)

axes[0][2].set_title('Fasting Blood Sugar')



x = ['Regular','Abnormality','Severe']

y = df.restecg.value_counts(sort = False).values

axes[0][3].bar(x,y)

axes[0][3].set_title('Electrocardiographic')





x = ['No','Yes']

y = df.exang.value_counts(sort = False).values

axes[1][0].bar(x,y)

axes[1][0].set_title('Exercise induced Angina')



x = ['Downward','Flat','Upward']

y = df.slope.value_counts(sort = False).values

axes[1][1].bar(x,y)

axes[1][1].set_title('ST excercise peak')



x = ['None','Normal','Fixed Defect','Reversable Defect']

y = df.thal.value_counts(sort = False).values

axes[1][2].bar(x,y)

axes[1][2].set_title('Thalium Stress Test')



x = ['No','Yes']

y = df.target.value_counts(sort = False).values

axes[1][3].bar(x,y)

axes[1][3].set_title('Heart Disease')



plt.show()
df['max_heart_rate'] = 220 - df['age']

df['peak_to_max_ratio'] = df['thalach']/df['max_heart_rate']
continuous_df = df[['age','trestbps','chol','thalach','oldpeak','ca','max_heart_rate','peak_to_max_ratio','target']]
sns.pairplot(continuous_df)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

sns.violinplot(x="target", y="age", data=df,color = 'white',edgecolor = 'black',ax=axes[0][0]).set_title('Age')

sns.swarmplot(x="target", y="age", data=df,ax = axes[0][0])



sns.violinplot(x="target", y="trestbps", data=df,color = 'white',edgecolor = 'black',ax = axes[0][1]).set_title('Resting Blood Pressure')

sns.swarmplot(x="target", y="trestbps", data=df,ax = axes[0][1])



sns.violinplot(x="target", y="chol", data=df,color = 'white',edgecolor = 'black',ax = axes[0][2]).set_title('Cholesterol')

sns.swarmplot(x="target", y="chol", data=df,ax = axes[0][2])



sns.violinplot(x="target", y="thalach", data=df,color = 'white',edgecolor = 'black',ax = axes[1][0]).set_title('Max Heart Rate Achieved')

sns.swarmplot(x="target", y="thalach", data=df,ax = axes[1][0])



sns.violinplot(x="target", y="oldpeak", data=df,color = 'white',edgecolor = 'black',ax = axes[1][1]).set_title('ST Depression Peak')

sns.swarmplot(x="target", y="oldpeak", data=df,ax = axes[1][1])



sns.violinplot(x="target", y="peak_to_max_ratio", data=df,color = 'white',edgecolor = 'black',ax = axes[1][2]).set_title('Peak Heart Rate to Max Heart Rate Ratio')

sns.swarmplot(x="target", y="peak_to_max_ratio", data=df,ax = axes[1][2])
fig,ax = plt.subplots(figsize=(16, 10))

sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import metrics
X = df.drop('target',axis = 1)

y = df.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 123)
log = LogisticRegression()

log.fit(X_train,y_train)

print('Accuracy for Logistic Regression: %0.4f' %log.score(X_test,y_test))
rf = RandomForestClassifier(n_estimators = 1000,random_state = 123)

rf.fit(X_train,y_train)

rf.score(X_test,y_test)

print('Accuracy for Random Forest: %0.4f' %rf.score(X_test,y_test))
feature_importance = pd.DataFrame(sorted(zip(rf.feature_importances_, X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(10, 6))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))

plt.title('Random Forest Feature Importance')

plt.tight_layout()
X = df.drop(['target','age','thalach'],axis = 1)

y = df.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 123)
rf = RandomForestClassifier(n_estimators = 1000,random_state = 123)

rf.fit(X_train,y_train)

rf.score(X_test,y_test)

print('Accuracy for Random Forest: %0.4f' %rf.score(X_test,y_test))
feature_importance = pd.DataFrame(sorted(zip(rf.feature_importances_, X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(10, 6))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))

plt.title('Random Forest Feature Importance')

plt.tight_layout()
test_accuracy = []

rf = RandomForestClassifier(random_state = 123,n_jobs = -1)

forest_sizes = range(100,1001,100)

for forest_size in forest_sizes:

    rf.set_params(n_estimators = forest_size)

    rf.fit(X_train,y_train)

    test_accuracy.append(rf.score(X_test,y_test))
plt.plot(forest_sizes, test_accuracy)

plt.title('N_Estimators Accuracy')

plt.xlabel('Number of Trees (By Hundred)')

plt.ylabel('Accuracy')

plt.show()
test_accuracy = []

rf = RandomForestClassifier(n_estimators = 600,random_state = 123,n_jobs = -1)

depths = range(5,15)

for depth in depths:

    rf.set_params(max_depth = depth)

    rf.fit(X_train,y_train)

    test_accuracy.append(rf.score(X_test,y_test))
plt.plot(depths,test_accuracy)

plt.title('Max_Depth Accuracy')

plt.xlabel('Maximum Depth')

plt.ylabel('Accuracy')

plt.show()
test_accuracy = []

rf = RandomForestClassifier(n_estimators = 600,max_depth = 10,random_state = 123,n_jobs = -1)

splits = range(2,15)

for split in splits:

    rf.set_params(min_samples_split = split)

    rf.fit(X_train,y_train)

    test_accuracy.append(rf.score(X_test,y_test))
plt.plot(splits,test_accuracy)

plt.title('Min_Samples_Split Accuracy')

plt.xlabel('Minimum Sample Split')

plt.ylabel('Accuracy')

plt.show()
test_accuracy = []

rf = RandomForestClassifier(n_estimators = 600,max_depth = 10,min_samples_split = 10,random_state = 123,n_jobs = -1)

features = range(2,14)

for feature in features:

    rf.set_params(max_features = feature)

    rf.fit(X_train,y_train)

    test_accuracy.append(rf.score(X_test,y_test))
plt.plot(features, test_accuracy)

plt.title('Max_Features Accuracy')

plt.xlabel('Maximum Features')

plt.ylabel('Accuracy')

plt.show()
rf = RandomForestClassifier(n_estimators = 600,max_depth = 10,min_samples_split = 10,max_features = 3,random_state = 123,n_jobs = -1)

print(rf)
rf.fit(X_train,y_train)

rf.score(X_test,y_test)
feature_importance = pd.DataFrame(sorted(zip(rf.feature_importances_, X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(10, 6))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))

plt.title('Random Forest Feature Importance')

plt.tight_layout()