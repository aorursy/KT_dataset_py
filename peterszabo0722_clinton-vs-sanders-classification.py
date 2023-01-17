# importing python packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.metrics import f1_score

from sklearn.metrics import jaccard_score
# loading primary results



df = pd.read_csv("../input/2016-us-election/primary_results.csv")

df.head()
df.info()
# detecting missing values in each column



missing_values = df.isnull()



for column in missing_values.columns.values.tolist():

    print(column)

    print(missing_values[column].value_counts())

    print("")
# removing 'fips' column 



df = df.drop(columns = ['fips'])

df.head()
# searching for duplicated rows



duplicated_rows = df[df.duplicated()]

duplicated_rows
# creating dataframe for Democrats



df_dem = df.loc[df['party'] == 'Democrat']

df_dem.head()
# relevant candidates



df_dem = df_dem.loc[df_dem['candidate'].isin(['Hillary Clinton', 'Bernie Sanders'])]
# sorting candidates by total votes in "battleground states" 



df_dem = df_dem.loc[df_dem['state_abbreviation'].isin(['CO', 'FL', 'IA', 'MI', 'NV', 'NH', 'NC', 'OH', 'PA', 'VA', 'WI'])]

df_dem_sorted = df_dem.groupby(by = ['candidate']).sum().sort_values(by = ['votes'], ascending = False)

df_dem_sorted.head()
# plotting results



df_dem_sorted.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes (in battleground states)', fontsize = 16)

plt.xlabel('Votes (million)', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_CO = df_dem.loc[df_dem['state_abbreviation'] == 'CO']

df_dem_CO = df_dem_CO.groupby(by = ['candidate']).sum()

df_dem_CO.head()
# plotting results



df_dem_CO.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Colorado', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_FL = df_dem.loc[df_dem['state_abbreviation'] == 'FL']

df_dem_FL = df_dem_FL.groupby(by = ['candidate']).sum()

df_dem_FL.head()
# plotting results



df_dem_FL.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Florida', fontsize = 16)

plt.xlabel('Votes (million)', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_IA = df_dem.loc[df_dem['state_abbreviation'] == 'IA']

df_dem_IA = df_dem_IA.groupby(by = ['candidate']).sum()

df_dem_IA.head()
# plotting results



df_dem_IA.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Iowa', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_MI = df_dem.loc[df_dem['state_abbreviation'] == 'MI']

df_dem_MI = df_dem_MI.groupby(by = ['candidate']).sum()

df_dem_MI.head()
# plotting results



df_dem_MI.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Michigan', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_NV = df_dem.loc[df_dem['state_abbreviation'] == 'NV']

df_dem_NV = df_dem_NV.groupby(by = ['candidate']).sum()

df_dem_NV.head()
# plotting results



df_dem_NV.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Nevada', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_NH = df_dem.loc[df_dem['state_abbreviation'] == 'NH']

df_dem_NH = df_dem_NH.groupby(by = ['candidate']).sum()

df_dem_NH.head()
# plotting results



df_dem_NH.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in New Hampshire', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_NC = df_dem.loc[df_dem['state_abbreviation'] == 'NC']

df_dem_NC = df_dem_NC.groupby(by = ['candidate']).sum()

df_dem_NC.head()
# plotting results



df_dem_NC.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in North Carolina', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_OH = df_dem.loc[df_dem['state_abbreviation'] == 'OH']

df_dem_OH = df_dem_OH.groupby(by = ['candidate']).sum()

df_dem_OH.head()
# plotting results



df_dem_OH.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Ohio', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_PA = df_dem.loc[df_dem['state_abbreviation'] == 'PA']

df_dem_PA = df_dem_PA.groupby(by = ['candidate']).sum()

df_dem_PA.head()
# plotting results



df_dem_PA.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Pennsylvania', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_VA = df_dem.loc[df_dem['state_abbreviation'] == 'VA']

df_dem_VA = df_dem_VA.groupby(by = ['candidate']).sum()

df_dem_VA.head()
# plotting results



df_dem_VA.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Virginia', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
df_dem_WI = df_dem.loc[df_dem['state_abbreviation'] == 'WI']

df_dem_WI = df_dem_WI.groupby(by = ['candidate']).sum()

df_dem_WI.head()
# plotting results



df_dem_WI.plot(kind = 'barh', figsize = (12, 8))



plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12)

plt.title('2016 Democratic Party Primaries Votes in Wisconsin', fontsize = 16)

plt.xlabel('Votes', fontsize = 14)

plt.ylabel('Candidates', fontsize = 14)



plt.show()
# getting statistical summary of numeric-typed columns



df_dem.describe()
# checking correlations



df_dem.corr()
df_dem['state'] = pd.get_dummies(df_dem['state'])

df_dem['county'] = pd.get_dummies(df_dem['county'])
X = df_dem[['state', 'county', 'votes']].values

X[0:5]
y = df_dem['candidate'].values

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# fitting and predicting(k = 13)



knn = KNeighborsClassifier(n_neighbors = 13).fit(X_train, y_train)



yhat_knn = knn.predict(X_test)

yhat_knn[0:5]
#fitting and predicting



tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6).fit(X_train, y_train)



yhat_tree = tree.predict(X_test)

yhat_tree[0:5]
# fitting and predicting



lr = LogisticRegression(C = 0.01, solver = 'liblinear').fit(X_train, y_train)



yhat_lr = lr.predict(X_test)

yhat_lr[0:5]
# fitting and predicting



svm_lin = svm.SVC(kernel = 'linear').fit(X_train, y_train)



yhat_svm_lin = svm_lin.predict(X_test)

yhat_svm_lin[0:5]
# fitting and predicting



svm_poly = svm.SVC(kernel = 'poly').fit(X_train, y_train)



yhat_svm_poly = svm_poly.predict(X_test)

yhat_svm_poly[0:5]
# fitting and predicting



svm_rbf = svm.SVC(kernel = 'rbf').fit(X_train, y_train)



yhat_svm_rbf = svm_rbf.predict(X_test)

yhat_svm_rbf[0:5]
# fitting and predicting



svm_sig = svm.SVC(kernel = 'sigmoid').fit(X_train, y_train)



yhat_svm_sig = svm_sig.predict(X_test)

yhat_svm_sig[0:5]
print("KNN evaluation\n")

print("F1 score: ", f1_score(y_test, yhat_knn, average = 'weighted'))

print("Jaccard index: ", jaccard_score(y_test, yhat_knn, average = 'weighted'))
print("Decision Tree evaluation\n")

print("F1 score: ", f1_score(y_test, yhat_tree, average = 'weighted'))

print("Jaccard index: ", jaccard_score(y_test, yhat_tree, average = 'weighted'))
print("Logistic Regression evaluation\n")

print("F1 score: ", f1_score(y_test, yhat_lr, average = 'weighted'))

print("Jaccard index: ", jaccard_score(y_test, yhat_lr, average = 'weighted'))
print("SVM evaluation using linear kernel\n")

print("F1 score: ", f1_score(y_test, yhat_svm_lin, average = 'weighted'))

print("Jaccard index: ", jaccard_score(y_test, yhat_svm_lin, average = 'weighted'))
print("SVM evaluation using polynomial kernel\n")

print("F1 score: ", f1_score(y_test, yhat_svm_poly, average = 'weighted'))

print("Jaccard index: ", jaccard_score(y_test, yhat_svm_poly, average = 'weighted'))
print("SVM evaluation using RBF kernel\n")

print("F1 score: ", f1_score(y_test, yhat_svm_rbf, average = 'weighted'))

print("Jaccard index: ", jaccard_score(y_test, yhat_svm_rbf, average = 'weighted'))
print("SVM evaluation using sigmoid kernel\n")

print("F1 score: ", f1_score(y_test, yhat_svm_sig, average = 'weighted'))

print("Jaccard index: ", jaccard_score(y_test, yhat_svm_sig, average = 'weighted'))