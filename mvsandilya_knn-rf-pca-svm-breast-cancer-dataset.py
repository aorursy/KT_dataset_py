import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
cancer = pd.read_csv('../input/breast_cancer.csv')
cancer.head()
cancer.info() #missing values can be checked here & also the datatype of the variables
# how is the cancer spread in the data
t= cancer.groupby('diagnosis')
t.count()
from sklearn.preprocessing import StandardScaler # need to scale feature set to fit KNN
scaler = StandardScaler() # initialise a scaler object to run on a dataframe
# create a df randomnly of 9 features
random_df = cancer[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean']]
scaler.fit(random_df.drop('diagnosis',axis=1)) # run the above scaler method on the selected dataframe
scaled_features = scaler.transform(random_df.drop('diagnosis',axis=1)) #scaled features is the new transformed df with nomralized values
#on which the KNN algo can be run
from sklearn.model_selection import train_test_split #Basic practice of train/test splitting the data
X_train, X_test, y_train, y_test = train_test_split(scaled_features,random_df['diagnosis'],
                                                    test_size=0.30, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20) # initialise the KNN classifier with neighbours=20 in this case
knn.fit(X_train,y_train)
pred = knn.predict(X_test) #run the KNN model on the test data
unique, counts = np.unique(pred, return_counts=True) # checking the variable spread in the prediction dataset
print (unique, counts)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train) # This has been run on scaled features
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight 
clf.fit(X_train,y_train)
clf_pred = clf.predict(X_test)
print(confusion_matrix(y_test,clf_pred))
print(classification_report(y_test,clf_pred)) #M (Malignant) is 1 here, assumed by the model
plt.figure(figsize = (18,18))
sns.heatmap(cancer.corr(), cmap='coolwarm', annot = True)
# creating a DF with the above selected features only
chosen = cancer[['diagnosis' , 'radius_mean', 'texture_mean' ,'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se']]
scaler.fit(chosen.drop('diagnosis',axis=1))
scaled_features = scaler.transform(chosen.drop('diagnosis',axis=1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,chosen['diagnosis'],
                                                    test_size=0.30, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
# creating a DF with the above selected features only
chosen = cancer[['diagnosis' , 'radius_mean', 'texture_mean' ,'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se']]
from sklearn.model_selection import train_test_split
X = chosen.drop('diagnosis',axis=1)
y = chosen['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
df = cancer.drop('diagnosis',axis=1)
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
type(scaled_data)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['diagnosis'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
reformed = pd.DataFrame(x_pca)
reformed.info()
df1 = pd.DataFrame(cancer['diagnosis'])
df1.info()
final_df = reformed.join(df1)
# We are creating a df which has only the principal components and the "Target"variable so that we can a ML algo on this dataframe
final_df.info()
final_df.columns = ['PCA-1', 'PCA-2', 'Target']
from sklearn.model_selection import train_test_split
X = final_df[['PCA-1', 'PCA-2']]
y = final_df['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.30, random_state=101)
X_train.info()
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
