import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/capstone-car-accident-serveity/Data_Collisions.csv')
df.head()
df.shape
col_data = df[['SEVERITYCODE', 'X', 'Y', 'ADDRTYPE', 'COLLISIONTYPE',
               'PERSONCOUNT', 'VEHCOUNT', 'JUNCTIONTYPE',  'WEATHER', 'ROADCOND', 'LIGHTCOND', 
               'SPEEDING', 'UNDERINFL', 'INATTENTIONIND']]
col_data.head()
for col in col_data.columns:
    if ((col_data[col].value_counts()/len(col_data[col])) > 0.8).any() == True:
        print(col)
def list_count(columns, df):
    for col in columns:
        print(col)
        print(df[col].value_counts())
        print()

data_columns = ['SEVERITYCODE','ADDRTYPE', 'COLLISIONTYPE', 'JUNCTIONTYPE', 'WEATHER', 
 'ROADCOND','LIGHTCOND', 'SPEEDING', 'UNDERINFL', 'INATTENTIONIND']

#Use value_counts() method in each column
list_count(data_columns, col_data)
filterCond = (col_data.LIGHTCOND == 'Other') | (col_data.LIGHTCOND == 'Unknown') | \
                      (col_data.LIGHTCOND == 'Dark - Unknown Lighting') |\
                      (col_data.ROADCOND == 'Other') | (col_data.ROADCOND == 'Unknown') | \
                      (col_data.WEATHER == 'Other') | (col_data.WEATHER == 'Unknown') | \
                      (col_data.JUNCTIONTYPE == 'Other') | (col_data.JUNCTIONTYPE == 'Unknown') | \
                      (col_data.COLLISIONTYPE == 'Other')
col_data = col_data.drop(col_data[filterCond].index)
col_data["LIGHTCOND"] = col_data["LIGHTCOND"].replace("Dark - Street Lights Off", "Dark - No Street Lights")
col_data["UNDERINFL"] = col_data["UNDERINFL"].replace("N", 0)
col_data["UNDERINFL"] = col_data["UNDERINFL"].replace("0", 0)
col_data["UNDERINFL"] = col_data["UNDERINFL"].replace("1", 1)
col_data["UNDERINFL"] = col_data["UNDERINFL"].replace("Y", 1)
col_data["INATTENTIONIND"] = col_data["INATTENTIONIND"].replace("Y", 1)
col_data["SPEEDING"] = col_data["SPEEDING"].replace("Y", 1)
# Check the columns which has NaN values
col_data.isna().sum()
col_data['UNDERINFL'] = col_data['UNDERINFL'].fillna(0)
col_data['INATTENTIONIND'] = col_data['INATTENTIONIND'].fillna(0)
col_data['SPEEDING'] = col_data['SPEEDING'].fillna(0)
col_data.dropna(inplace=True)
col_data.info()
col_data['SEVERITYCODE'].unique()
# Rename severitycode to 0,1
col_data["SEVERITYCODE"] = col_data["SEVERITYCODE"].replace(1, 0)
col_data["SEVERITYCODE"] = col_data["SEVERITYCODE"].replace(2, 1)
# One hot encoding for the relevant dataset
feature = pd.concat([pd.get_dummies(col_data['WEATHER']), 
                     pd.get_dummies(col_data['ROADCOND']),
                     pd.get_dummies(col_data['LIGHTCOND'])], axis=1)
feature.head()
col_data.columns
import seaborn as sns
sns.countplot(x="ADDRTYPE", hue="SEVERITYCODE", data=col_data)
plt.figure(figsize=(10,5))
ax= sns.countplot(x="COLLISIONTYPE", hue="SEVERITYCODE", data=col_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.figure(figsize=(8,5))
sns.countplot(y="JUNCTIONTYPE", hue="SEVERITYCODE", data=col_data)
plt.figure(figsize=(10,5))
ax= sns.countplot(x="WEATHER", hue="SEVERITYCODE", data=col_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.figure(figsize=(8,5))
sns.countplot(y="ROADCOND", hue="SEVERITYCODE", data=col_data)
plt.figure(figsize=(8,5))
sns.countplot(y="LIGHTCOND", hue="SEVERITYCODE", data=col_data)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.countplot(x="SPEEDING", hue="SEVERITYCODE", data=col_data, ax=axes[0])
sns.countplot(x="UNDERINFL", hue="SEVERITYCODE", data=col_data, ax=axes[1])
sns.countplot(x="INATTENTIONIND", hue="SEVERITYCODE", data=col_data, ax=axes[2])
# !conda install -c conda-forge folium=0.5.0 --yes
import folium

print('Folium installed and imported!')
from folium import plugins
seattle_long= -122.335167
seattle_lat= 47.608013
seattle_map = folium.Map(location=[seattle_lat, seattle_long], zoom_start=4)
# let's start again with a clean copy of the map of Seattle

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, in zip(col_data.Y, col_data.X):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        #popup=label,
    ).add_to(incidents)

# display map
seattle_map
# Defining X matrix and y vector
X = feature
y = col_data['SEVERITYCODE'].values
# Normalizing and splitting data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
import matplotlib.pyplot as plt
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
k = 4
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))
# predicted y
yhat_knn = neigh.predict(X_test)

# jaccard
jaccard_knn = jaccard_similarity_score(y_test, yhat_knn)
print("KNN Jaccard index: ", jaccard_knn)

# f1_score
f1_score_knn = f1_score(y_test, yhat_knn, average='weighted')
print("KNN F1-score: ", f1_score_knn)
from sklearn.tree import DecisionTreeClassifier
severityTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
severityTree.fit(X_train, y_train)
# predicted y
yhat_dt = severityTree.predict(X_test)

# jaccard
jaccard_dt = jaccard_similarity_score(y_test, yhat_dt)
print("DT Jaccard index: ", jaccard_dt)

# f1_score
f1_score_dt = f1_score(y_test, yhat_dt, average='weighted')
print("DT F1-score: ", f1_score_dt)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01).fit(X_train,y_train)
LR
yhat_lg = LR.predict(X_test)
yhat_lg_prob = LR.predict_proba(X_test)

# jaccard
jaccard_lg = jaccard_similarity_score(y_test, yhat_lg)
print("LR Jaccard index: ", jaccard_lg)

# f1_score
f1_score_lg = f1_score(y_test, yhat_lg, average='weighted')
print("LR F1-score: ", f1_score_lg)

# logloss
logloss_lg = log_loss(y_test, yhat_lg_prob)
print("LR log loss: ", logloss_lg)
from sklearn import svm
# training
clf = svm.SVC()
clf.fit(X_train, y_train)
# predicted y
yhat_svm = clf.predict(X_test)

# jaccard
jaccard_svm = jaccard_similarity_score(y_test, yhat_svm)
print("SVM Jaccard index: ", jaccard_svm)

# f1_score
f1_score_svm = f1_score(y_test, yhat_svm, average='weighted')
print("SVM F1-score: ", f1_score_svm)