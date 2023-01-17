import numpy as np   
from sklearn.linear_model import LinearRegression
import pandas as pd    
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
vehicle_data = pd.read_csv('/kaggle/input/vehicle-silhouettes/vehicle.csv')
vehicle_data.shape
vehicle_data.info()
vehicle_data.apply(lambda x: sum(x.isnull()))
vehicle_data['class'].value_counts()
#Label encode the target class
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
vehicle_data['class'] = labelencoder.fit_transform(vehicle_data['class'])
vehicle_data['class'].value_counts()
vehicle_data.describe().transpose()
vehicle_data.head()
vehicle_data.isnull().sum()
sns.heatmap(vehicle_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
vehicle_data.fillna(vehicle_data.mean(), inplace=True)
sns.heatmap(vehicle_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
vehicle_data.isnull().sum()
num_features=[col for col in vehicle_data.select_dtypes(np.number).columns]

plt.figure(figsize=(20,20))
for i,col in enumerate(num_features,start=1):
    plt.subplot(5,4,i);
    sns.boxplot(vehicle_data['class'],vehicle_data[col]);
plt.show()
vehicle_data.drop(vehicle_data[vehicle_data['radius_ratio']>276].index,axis=0,inplace=True)
vehicle_data.drop(vehicle_data[vehicle_data['pr.axis_aspect_ratio']>77].index,axis=0,inplace=True)
vehicle_data.drop(vehicle_data[vehicle_data['max.length_aspect_ratio']>14.5].index,axis=0,inplace=True)
vehicle_data.drop(vehicle_data[vehicle_data['max.length_aspect_ratio']<2.5].index,axis=0,inplace=True)
vehicle_data[vehicle_data['scaled_variance']>292]
vehicle_data.drop(vehicle_data[vehicle_data['scaled_variance.1']>989.5].index,axis=0,inplace=True)
vehicle_data.drop(vehicle_data[vehicle_data['scaled_radius_of_gyration.1']>87].index,axis=0,inplace=True)
vehicle_data.drop(vehicle_data[vehicle_data['skewness_about']>19.5].index,axis=0,inplace=True)
vehicle_data.drop(vehicle_data[vehicle_data['skewness_about.1']>40].index,axis=0,inplace=True)
print("Shape of the dataset after fixing the outliers:",vehicle_data.shape)
sns.pairplot(vehicle_data,diag_kind='kde', hue='class')
plt.show()
num_features=[col for col in vehicle_data.select_dtypes(np.number).columns ]

plt.figure(figsize=(20,20))
for i,col in enumerate(num_features,start=1):
    plt.subplot(5,4,i);
    sns.distplot(vehicle_data[col])
plt.show()
plt.figure(figsize=(20,4))
sns.heatmap(vehicle_data.corr(),annot=True)
plt.show()
scaler = StandardScaler()
scaled_df = scaler.fit_transform(vehicle_data.drop(columns = 'class'))
X = scaled_df
y = vehicle_data['class']

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.3,random_state = 10)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
# Training an SVC using the actual attributes(scaled)

model = SVC(gamma = 'auto')

model.fit(X_train,Y_train)

score_using_actual_attributes = model.score(X_test, Y_test)

print(score_using_actual_attributes)
model = SVC(C=1, kernel="rbf", gamma='auto')

scores = cross_val_score(model, X, y, cv=10)

CV_score = scores.mean()
print(CV_score)
pca = PCA().fit(scaled_df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
print(np.cumsum(pca.explained_variance_ratio_))
plt.bar(list(range(1,19)),pca.explained_variance_ratio_,alpha=0.5,align='center')
plt.ylabel('cum of variation explained')
plt.xlabel('eigen value')
plt.show()
plt.step(list(range(1,19)),np.cumsum(pca.explained_variance_ratio_),where= 'mid')
plt.ylabel('cum of variation explained')
plt.xlabel('eigen values')
plt.show()
pca = PCA(n_components=8)

X = pca.fit_transform(scaled_df)
Y = vehicle_data['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=10)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
# Training an SVC using the PCs instead of the actual attributes 
model = SVC(gamma= 'auto')

model.fit(X_train,Y_train)

score_PCs = model.score(X_test, Y_test)

print(score_PCs)
model = SVC(C=1, kernel="rbf", gamma='auto')

scores = cross_val_score(model, X, y, cv=10)

CV_score_pca = scores.mean()
print(CV_score_pca)
matrix = pd.DataFrame({'SVC' : ['All scaled attributes', '8 Principle components'],
                      'Accuracy' : [score_using_actual_attributes,score_PCs],
                      'Cross-validation score' : [CV_score,CV_score_pca]})
matrix