import numpy as np
import pandas as pd
import datetime as dt

from datetime import date

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import pickle
df = pd.read_csv("../input/fraudulent_claims_data.csv")
df.head()
df.shape
df.info()
df['fraud_reported'].value_counts()
df['collision_type']=df['collision_type'].replace("?","Not Applicable")
df['police_report_available']=df['police_report_available'].replace("?","Unknown")
df['property_damage']=df['property_damage'].replace("?","Not Applicable")
df[['incident_date','policy_bind_date']] = df[['incident_date','policy_bind_date']].apply(pd.to_datetime)
df['duration_bw_inception_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
df.head()
df=df.drop(['policy_number','policy_bind_date','insured_zip','authorities_contacted','umbrella_limit','insured_hobbies', 
            'capital-gains', 'capital-loss','injury_claim', 'property_claim', 'vehicle_claim', 'incident_date', 'auto_model',
            'policy_csl','insured_education_level','insured_occupation', '_c39','incident_location', 
            'policy_state','incident_type','collision_type','property_damage', 'policy_deductable', 'policy_annual_premium', 
            'number_of_vehicles_involved', 'bodily_injuries', 'incident_state', 'incident_city', 'incident_hour_of_the_day', 'total_claim_amount',
            'auto_year', 'auto_make', 'duration_bw_inception_incident'],axis=1)
# year_ = date.today().year
# df['auto_year'] = year_ - df['auto_year']
df.shape
df.columns
# x = df.drop(['fraud_reported'],axis=1)
# y = df['fraud_reported']
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(), [3,4,5,6,7,8,9,12,15,17])], remainder='passthrough')
# df = np.array(ct.fit_transform(df))
df = pd.get_dummies(df,columns=['insured_sex','insured_relationship',
                                'incident_severity',
                                'police_report_available'],drop_first=True)
x = df.drop(['fraud_reported'],axis=1)
y = df['fraud_reported']
x_upsample, y_upsample  = SMOTE().fit_resample(x, y)

print(x_upsample.shape)
print(y_upsample.shape)
y_upsample.value_counts()
sc=StandardScaler()
x_scale=sc.fit_transform(x_upsample)
#0.95
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
x_scaled=pca.fit_transform(x_scale)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_upsample,test_size=0.3)
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
svm = SVC()
xgb = XGBClassifier()
for i in [rf, knn, svm]:
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    print(accuracy_score(y_test,y_pred))
# import pickle

# Pkl_Filename = "grid_xgboost.pkl"  

# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(grid, file)
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(x)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
# kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
# y_kmeans = kmeans.fit_predict(x)
# plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()
#Hyperparamenter tuning
from sklearn.model_selection import GridSearchCV
parameters={'criterion':['gini','entropy'], 'max_depth': np.arange(1,30)}
grid=GridSearchCV(rf,parameters)
grid.fit(x_train,y_train)
model=grid.best_estimator_
grid.best_score_
param_grid = {"kernel": ['rbf','sigmoid'],
             "C":[0.1,0.5,1.0],
             "random_state":[0,100,200,300]}
grid = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5,  verbose=3)
grid.fit(x_train,y_train)
model2 = grid.best_estimator_
grid.best_score_
param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 10, 1)}

grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5,  verbose=3,n_jobs=-1)
grid.fit(x_train,y_train)
model3 = grid.best_estimator_
grid.best_score_
# import pickle

# Pkl_Filename = "grid_xgboost.pkl"  

# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(grid, file)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
labelEncoder1 = LabelEncoder()
labelEncoder2 = LabelEncoder()
labelEncoder3 = LabelEncoder()
labelEncoder4 = LabelEncoder()
labelEncoder5 = LabelEncoder()
labelEncoder6 = LabelEncoder()
labelEncoder7 = LabelEncoder()
labelEncoder8 = LabelEncoder()
labelEncoder9 = LabelEncoder()
labelEncoder10 = LabelEncoder()
labelEncoder11 = LabelEncoder()
# 'insured_sex','incident_type','collision_type','insured_relationship',
#                                 'incident_severity','incident_state','incident_city','property_damage',
#                                 'police_report_available', 'auto_make'
labelEncoder1.fit(df['insured_sex'])
# labelEncoder2.fit(df['incident_type'])
# labelEncoder3.fit(df['collision_type'])
labelEncoder4.fit(df['insured_relationship'])
labelEncoder5.fit(df['incident_severity'])
# labelEncoder6.fit(df['incident_state'])
# labelEncoder7.fit(df['incident_city'])
# labelEncoder8.fit(df['property_damage'])
labelEncoder9.fit(df['police_report_available'])
# labelEncoder10.fit(df['auto_make'])
labelEncoder11.fit(df['fraud_reported'])
df['insured_sex'] = labelEncoder1.transform(df['insured_sex'])
# df['incident_type'] = labelEncoder2.transform(df['incident_type'])
# df['collision_type'] = labelEncoder3.transform(df['collision_type'])
df['insured_relationship'] = labelEncoder4.transform(df['insured_relationship'])
df['incident_severity'] = labelEncoder5.transform(df['incident_severity'])
# df['incident_state'] = labelEncoder6.transform(df['incident_state'])
# df['incident_city'] = labelEncoder7.transform(df['incident_city'])
# df['property_damage'] = labelEncoder8.transform(df['property_damage'])
df['police_report_available'] = labelEncoder9.transform(df['police_report_available'])
# df['auto_make'] = labelEncoder10.transform(df['auto_make'])
df['fraud_reported'] = labelEncoder11.transform(df['fraud_reported'])
df.info()
x = np.array(df.drop(['fraud_reported'],1).astype(float))
y = np.array(df['fraud_reported'])
x_upsample, y_upsample  = SMOTE().fit_resample(x, y)

print(x_upsample.shape)
print(y_upsample.shape)
sc=StandardScaler()
x_scale=sc.fit_transform(x_upsample)
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
x_scaled=pca.fit_transform(x_scale)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_upsample,test_size=0.3)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(x_scale)
correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(x))
