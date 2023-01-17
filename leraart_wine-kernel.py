import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
path = '../input/winequality-red.csv'
df = pd.read_csv(path, delimiter=',')
df.head()
df.describe()
df.head()
df.sample(10)
df.shape
df.info()
df.hist();
num_bins = 10
df.hist(bins=num_bins, figsize=(20,15));
import seaborn as sns
corr=df.corr()
df.corr()
import seaborn as sns
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df.plot(kind='scatter', x='pH', y='alcohol',alpha = 0.5,color = 'red', figsize=(9,9))
plt.xlabel('pH')             
plt.ylabel('alcohol')
plt.title('pH & alcohol')        
plt.show()
Y = df.iloc[:,:8].values
X = df.iloc[:,10:11].values
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train) 
X.size
Y.size
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target = ['quality']
df.isnull()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)
print(y_prediction[:5])
print('*'*40)
print(y_test[:5])
y_test.describe()

from math import sqrt
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)
regressor = DecisionTreeRegressor(max_depth=50)
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)
y_prediction[:5]
y_test
RMSE1 = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE1)
from sklearn.tree import DecisionTreeClassifier
data_classifier = df.copy()
data_classifier.head()
data_classifier['quality'].dtype

data_classifier['quality_label'] = (data_classifier['quality'] > 6.5)*1
data_classifier['quality_label']

features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']
X = data_classifier[features]
y = data_classifier[target_classifier]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
wine_quality_classifier = DecisionTreeClassifier(max_leaf_nodes=20, random_state=0)
wine_quality_classifier.fit(X_train, y_train)
prediction = wine_quality_classifier.predict(X_test)
print(prediction)
print()
print(y_test['quality_label'])
accuracy_score(y_true=y_test, y_pred=prediction)
from sklearn.tree import DecisionTreeClassifier, export_graphviz
export_graphviz(wine_quality_classifier, out_file='wine.dot', feature_names=features, filled=True)
!dot -Tpng 'wine.dot' -o 'wine.png'
![images] ("wine.png")
from sklearn.linear_model import LogisticRegression
data_classifier.head()
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']
X = data_classifier[features]
y = data_classifier[target_classifier]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
prediction = logistic_regression.predict(X_test)
print(prediction)
print(y_test)
accuracy_score(y_true=y_test, y_pred=prediction)
#1 - плохое
#2 - нормальное
#3 - превосходное
#1,2,3 --> плохое
#4,5,6,7 --> нормальное
#8,9,10 --> превосходное
reviews = []
for i in df['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
df['Reviews'] = reviews
x = df.iloc[:,:11]
y = df['Reviews']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
from sklearn.decomposition import PCA
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.4)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)
print(rf.feature_importances_)
from sklearn.metrics import confusion_matrix, accuracy_score
acc_score = accuracy_score(y_test, rf_predict)
print(acc_score)
export_graphviz(rf, out_file='tree.dot', feature_names=features, filled=True)
