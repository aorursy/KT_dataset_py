import pandas as pd

import numpy as np
df = pd.read_csv('../input/creditcard.csv')
df.info()
df.head(20)
df.groupby('Class').count()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(20,5))

sns.countplot(x='Class', data=df)
from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics

from sklearn.model_selection import train_test_split
X = df.drop(['Class'], axis=1)
Y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
df[df.Class==1].Amount.describe()
df.Time.max()/86400
plt.figure(figsize=(20,10))

ax = plt.subplot()

ax.set_xlim(0, 2000)

sns.distplot(df[(df.Class==1) & (df.Amount < 2000)].Amount, bins=100, color='r')

sns.distplot(df[(df.Class==0) & (df.Amount < 2000)].Amount, bins=100, color='b')
plt.figure(figsize=(20,10))

ax = plt.subplot()

sns.distplot(df[(df.Class==1) & (df.Amount < 2)].Amount, bins=100, color='r')

sns.distplot(df[(df.Class==0) & (df.Amount < 2)].Amount, bins=100, color='b')

sns.distplot(df[(df.Amount < 2)].Amount, bins=100, color='g')
# Make a new feature denoting a micro transaction

df['Micro TXN'] = df.Amount <= 1
df['Micro TXN in 1K TXN'] = df['Micro TXN'].rolling(1000).sum()
df.dropna(inplace=True)

df.drop(['Micro TXN'], axis=1, inplace=True)
df.head(10)
df['Micro TXN in 1K TXN'].describe()
X = df.drop(['Class'], axis=1)
Y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
df['Large TXN'] = df.Amount > 250
X = df.drop(['Class'], axis=1)

Y = df.Class

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
df.drop(['Large TXN'], axis=1, inplace=True)
from sklearn.manifold import TSNE
X_TSNE = TSNE().fit_transform(X)
X_TSNE
vis_x = X_TSNE[:, 0]

vis_y = X_TSNE[:, 1]
plt.figure(figsize=(15,15))

plt.scatter(vis_x, vis_y, c=Y.as_matrix())
from sklearn.cluster import KMeans
kmeans = KMeans(random_state=0).fit(X)
clusters = kmeans.predict(X)
plt.figure(figsize=(15,15))

plt.scatter(vis_x, vis_y, c=clusters)
distance_from_centroids = kmeans.transform(X)
distance_from_centroids.shape
df = pd.concat([df, pd.DataFrame(X_TSNE, index=X.index, columns=['TSNE_0', 'TSNE_1'])], axis=1)
df.head()
df = pd.concat([df, pd.DataFrame(distance_from_centroids, index=X.index, columns=['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6', 'Cluster7', 'Cluster8'])], axis=1)
df.head()
X = df.drop(['Class'], axis=1)

Y = df.Class

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
df['TimeOfDay'] = df.Time % 86400
df.head()
X = df.drop(['Time', 'Class'], axis=1)

Y = df.Class

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
names=X.columns

f_imp = sorted(zip(names, map(lambda x: round(x, 4), rf.feature_importances_)), 

               key=lambda x: x[1],

             reverse=True)
pd.DataFrame(f_imp)
X = df.drop(['Time', 'Class', 'V15', 'Cluster6', 'Micro TXN in 1K TXN', 'TSNE_1', 'V23'], axis=1)

Y = df.Class

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
resampled_df_normal = df[df.Class == 0].sample(n=490, random_state=0)
resampled_df_fraud = df[df.Class==1]
resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
X = df.drop(['Time', 'Class'], axis=1)

Y = df.Class
resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

resampled_Y = resampled_df.Class

X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
resampled_df_normal = df[df.Class == 0].sample(n=2500, random_state=0)
resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

resampled_Y = resampled_df.Class

X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
sample_size = []

precision = []

recall = []

fone = []

for size in range(500, 283500, 500):

    print('Running : size = %d' % (size))

    sample_size.append(size)

    resampled_df_normal = df[df.Class == 0].sample(n=size, random_state=0)

    resampled_df_fraud = df[df.Class==1]

    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

    resampled_Y = resampled_df.Class

    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

    rf = RandomForestClassifier(random_state=0,n_jobs=-1)

    rf.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    y_pred = rf.predict(X_test)

    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])

    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])

    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])

    
ss = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)
ss.columns = ['Precision', 'Recall', 'F1']
ss.index = range(500, 283500, 500)
ss.head(15)
ss.plot(figsize=(20,10))
ss.F1.idxmax()
ss.loc[270000:280000]
ss['Recall MA'] = ss.Recall.rolling(20).mean()
ss['F1 MA'] = ss.F1.rolling(20).mean()
ss.plot(figsize=(20,10))
ss['F1 MA'].idxmax()
ss.loc[200500:213500]
resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)

resampled_df_fraud = df[df.Class==1]

resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

resampled_Y = resampled_df.Class

X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
reloaded_df = pd.read_csv('../input/creditcard.csv')
reloaded_df_normal = reloaded_df[reloaded_df.Class == 0].sample(n=204000, random_state=0)

reloaded_df_fraud = reloaded_df[reloaded_df.Class==1]

reloaded_df = pd.concat([reloaded_df_normal, reloaded_df_fraud])

reloaded_X = reloaded_df.drop(['Class'], axis=1)

reloaded_Y = reloaded_df.Class

X_train, X_test, y_train, y_test = train_test_split(reloaded_X, reloaded_Y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
key_index = []

precision = []

recall = []

fone = []

for key in range(2,40):

    print('Running : key = %d' % (key))

    key_index.append(key)

    resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)

    resampled_df_fraud = df[df.Class==1]

    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

    resampled_Y = resampled_df.Class

    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

    rf = RandomForestClassifier(n_estimators=key, random_state=0,n_jobs=-1)

    rf.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    y_pred = rf.predict(X_test)

    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])

    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])

    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])

    
opt_param = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)

opt_param.columns = ['Precision', 'Recall', 'F1']

opt_param.index = key_index

opt_param.plot(figsize=(20,10))
opt_param.F1.idxmax()
resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)

resampled_df_fraud = df[df.Class==1]

resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

resampled_Y = resampled_df.Class

X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

rf = RandomForestClassifier(n_estimators=16, random_state=0)

rf.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
key_index = []

precision = []

recall = []

fone = []

for key in range(2,50):

    print('Running : key = %d' % (key))

    key_index.append(key)

    resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)

    resampled_df_fraud = df[df.Class==1]

    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

    resampled_Y = resampled_df.Class

    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

    rf = RandomForestClassifier(n_estimators=16, max_depth=key, random_state=0,n_jobs=-1)

    rf.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    y_pred = rf.predict(X_test)

    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])

    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])

    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])
opt_param = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)

opt_param.columns = ['Precision', 'Recall', 'F1']

opt_param.index = key_index

opt_param.plot(figsize=(20,10))
opt_param.F1.max()
key_index = []

precision = []

recall = []

fone = []

for key in range(1,42):

    print('Running : key = %d' % (key))

    key_index.append(key)

    resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)

    resampled_df_fraud = df[df.Class==1]

    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

    resampled_Y = resampled_df.Class

    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

    rf = RandomForestClassifier(n_estimators=16, max_features=key, random_state=0,n_jobs=-1)

    rf.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    y_pred = rf.predict(X_test)

    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])

    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])

    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])
opt_param = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)

opt_param.columns = ['Precision', 'Recall', 'F1']

opt_param.index = key_index

opt_param.plot(figsize=(20,10))
opt_param.F1.max()
opt_param.F1.idxmax()
resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)

resampled_df_fraud = df[df.Class==1]

resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

resampled_Y = resampled_df.Class

X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)

rf = RandomForestClassifier(n_estimators=16, max_features=14, random_state=0)

rf.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=1)

resampled_df_fraud = df[df.Class==1]

resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])

resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)

resampled_Y = resampled_df.Class

X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=2)

rf = RandomForestClassifier(n_estimators=16, max_features=14, random_state=3)

rf.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=4)

y_pred = rf.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_pred))
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')