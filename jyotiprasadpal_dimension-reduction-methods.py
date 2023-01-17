import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn import decomposition

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from sklearn.metrics import accuracy_score,confusion_matrix
cancer = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

cancer.head()
cancer.drop('Unnamed: 32', axis=1)

cancer.diagnosis.unique()
cancer.columns
#drop 'id' column - godd practice to drop columns such as id, name, etc as they bear no fruit in model building.

X = cancer.loc[:, ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']]



y = cancer.loc[:, 'diagnosis']
#scaling of variables

sc = StandardScaler()

scaled_X = sc.fit_transform(X.values)

pd.DataFrame(scaled_X, columns=X.columns).head()
#encode target variable y

le = LabelEncoder()

y = le.fit_transform(y)

pd.DataFrame(y, columns=['diagnosis']).head()
pca = decomposition.PCA()

pca.fit_transform(scaled_X)
print(X.shape)
#Information content by all new indep variables

pca.explained_variance_ratio_
#do a PCA plot and find the correct number of componenets from Elbo

df1 = pd.DataFrame({'Information':pca.explained_variance_ratio_,

                    'PCs':['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',

                          'PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20',

                          'PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29','PC30']})



plt.figure(figsize = (25,6))

sns.barplot(x = 'PCs',y = 'Information',data = df1)
#we will do PCA with only 5 components now as they seem to provide 80% of the information.

pca1 = decomposition.PCA(n_components=5)

pca_5var = pca1.fit_transform(scaled_X)
pca1.explained_variance_ratio_
np.sum(pca1.explained_variance_ratio_)*100
new_X = pd.DataFrame(pca_5var,columns=['PC1','PC2','PC3','PC4','PC5'])

new_X.head()
new_X.corr()
fig, ax = plt.subplots(figsize=(10, 5))

sns.heatmap(new_X.corr(), annot= True, fmt='.10f', ax=ax)
#let's compare Logistic Regession with PCA

new_X_train, new_X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.2, random_state = 42)



logreg = LogisticRegression(solver='lbfgs')

logreg.fit(new_X_train, y_train)



y_pred_test = logreg.predict(new_X_test)

print(confusion_matrix(y_test,y_pred_test))

print(accuracy_score(y_test,y_pred_test))
#let's compare Logistic Regession without PCA when we have all of the original features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



logreg2 = LogisticRegression(solver='lbfgs')

logreg2.fit(X_train, y_train)



y_pred_test2 = logreg2.predict(X_test)

print(confusion_matrix(y_test, y_pred_test2))

print(accuracy_score(y_test, y_pred_test2))
# factor loading = PC loadings

pca1.components_
lda = LinearDiscriminantAnalysis()

new_X_train_lda = lda.fit_transform(X_train, y_train)
lda.explained_variance_ratio_
new_X_train_lda_df = pd.DataFrame(new_X_train_lda,columns=['LDA1'])

new_X_train_lda_df.head()
lda.coef_
lg2 = LogisticRegression()

lg2.fit(new_X_train_lda, y_train)



new_x_test_lda = lda.transform(X_test)

y_test_pred_lda = lg2.predict(new_x_test_lda)



print(confusion_matrix(y_test, y_test_pred_lda))

print(accuracy_score(y_test, y_test_pred_lda))
cancer['diagnosis'].value_counts()