import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
%matplotlib inline
dataset = pd.read_csv('vehicle.csv')
dataset.head()
dataset.shape
dataset.describe().transpose()
dataset.dtypes
dataset['class'].value_counts()
dataset.groupby('class').size()
dataset.plot(kind='box', figsize=(20,10))
plt.show()
dataset.hist(figsize=(15,15))
plt.show()
dataset.isnull().sum()
dataset.info()
for i in dataset.columns[:-1]:
    median_value = dataset[i].median()
    dataset[i] = dataset[i].fillna(median_value)
dataset.info()
for col_name in dataset.columns[:-1]:
    q1 = dataset[col_name].quantile(0.25)
    q3 = dataset[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr
    
    dataset.loc[ (dataset[col_name] < low) | (dataset[col_name] > high), col_name] = dataset[col_name].median()
    
dataset.plot(kind='box', figsize=(20,10))
sns.pairplot(dataset,diag_kind='kde')
dataset.corr()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_df = scaler.fit_transform(dataset.drop(columns = 'class'))
X = scaled_df
y = dataset['class']

X_train, X_test, Y_train, Y_test = train_test_split(X,y, random_state = 10)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
# Training an SVC using the actual attributes(scaled)

model = SVC(gamma = 'auto')

model.fit(X_train,Y_train)

score_using_actual_attributes = model.score(X_test, Y_test)

print(score_using_actual_attributes)
model = SVC()

params = {'C': [0.01, 0.1, 0.5, 1], 'kernel': ['linear', 'rbf'], 'gamma' : ['auto', 'scale' ]}

model1 = GridSearchCV(model, param_grid=params, verbose=5)

model1.fit(X_train, Y_train)

print("Best Hyper Parameters:\n", model1.best_params_)
model = SVC(C=1, kernel="rbf", gamma='auto')

scores = cross_val_score(model, X, y, cv=10)

CV_score = scores.mean()
print(CV_score)
from sklearn.decomposition import PCA

pca = PCA().fit(scaled_df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
print(np.cumsum(pca.explained_variance_ratio_))
pca = PCA(n_components=8)

X = pca.fit_transform(scaled_df)
Y = dataset['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)
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
result = pd.DataFrame({'SVC' : ['All scaled attributes', '8 Principle components'],
                      'Accuracy' : [score_using_actual_attributes,score_PCs],
                      'Cross-validation score' : [CV_score,CV_score_pca]})
result