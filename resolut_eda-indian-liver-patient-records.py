# import libs 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# machine learning 
import sklearn
from sklearn import preprocessing
from scipy.stats import pearsonr

# machine learning  - supervised
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

# machine learning  - unsupervised
from sklearn import decomposition
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# visualization and plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/indian_liver_patient.csv')
df.shape
# general information about dataset
df.info()
# let's look on first entries in the data
df.head(3)
# let's look on target variable - classes imbalanced?
df['Dataset'].value_counts()
# what are the missing values? 
df[df["Albumin_and_Globulin_Ratio"].isnull()]
# fill with median/mean/max/min or ?
df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)
# encode gender
le = preprocessing.LabelEncoder()
le.fit(df.Gender.unique())
df['Gender_Encoded'] = le.transform(df.Gender)
df.drop(['Gender'], axis=1, inplace=True)
# correlation plots
g = sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")
# calculate correlation coefficients for two variables
print(pearsonr(df['Total_Bilirubin'], df['Direct_Bilirubin']))
# calculate correlation coefficients for all dataset
correlations = df.corr()

# and visualize
plt.figure(figsize=(10, 10))
g = sns.heatmap(correlations, cbar = True, square = True, annot=True, fmt= '.2f', annot_kws={'size': 10})

# based on correlation, you can exclude some highly correlated features
# pair grid allows to visualize multi-dimensional datasets
g = sns.PairGrid(df, hue="Dataset", vars=['Age','Gender_Encoded','Total_Bilirubin','Total_Protiens'])
g.map(plt.scatter)
plt.show()
# prepare train and test set
X = df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
        'Albumin', 'Albumin_and_Globulin_Ratio','Gender_Encoded']]
y = df[['Dataset']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# let's try random forest on the data
rf = RandomForestClassifier(n_estimators=25, random_state=2018)
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)

random_forest_score      = round(rf.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(rf.score(X_test, y_test) * 100, 2)

print('Random Forest Score: ', random_forest_score)
print('Random Forest Test Score: ', random_forest_score_test)
print('Accuracy: ', accuracy_score(y_test,rf_predicted))
print('\nClassification report: \n', classification_report(y_test,rf_predicted))

g = sns.heatmap(confusion_matrix(y_test,rf_predicted), annot=True, fmt="d")
# perform PCA for dataset to simplify visualizations
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_decomposed = pca.transform(X)
plt.figure( figsize=(10,5))
plt.scatter(X_decomposed[:,0], X_decomposed[:,1], c=y.values.ravel(), edgecolor='black', s=100)
plt.show()
# predict kmeans
kmeans = KMeans(n_clusters=2)
pred_kmeans = kmeans.fit_predict(X_decomposed)

# predict gmm
gmm = GaussianMixture(n_components=2).fit(X_decomposed)
gmm = gmm.fit(X)
pred_gmm = gmm.predict(X)
print('Adjusted Rand Score:', adjusted_rand_score(y.values.ravel(), pred_kmeans))
plt.figure( figsize=(10,5))
plt.scatter(X_decomposed[:,0], X_decomposed[:,1], c=pred_kmeans, edgecolor='black', s=100)
plt.show()

print('Adjusted Rand Score:', adjusted_rand_score(y.values.ravel(), pred_gmm))
plt.figure( figsize=(10,5))
plt.scatter(X_decomposed[:,0], X_decomposed[:,1], c=pred_gmm, edgecolor='black', s=100)
plt.show()
# use all features now

# predict kmeans
kmeans = KMeans(n_clusters=2)
pred_kmeans = kmeans.fit_predict(X)

# predict gmm
gmm = GaussianMixture(n_components=2).fit(X)
gmm = gmm.fit(X)
pred_gmm = gmm.predict(X)
print('Adjusted Rand Score', adjusted_rand_score(y.values.ravel(), pred_kmeans))
print('Adjusted Rand Score', adjusted_rand_score(y.values.ravel(), pred_gmm))