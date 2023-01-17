# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
input_path = "../input/Test_Data.xlsx"

df = pd.read_excel(input_path, sheet_name ='Sheet2')

df.head(5)
df.rename(index=str, columns = {'DESCRIPTION\n----------------\nDATE & TIME\n(DD/MM/YYYY)':'time'}, inplace = True)

df.rename(str.strip, axis = 'columns', inplace = True)

df.head()
df.dtypes
df.iloc[:,1:] = df.iloc[:,1:].apply(lambda x: pd.to_numeric(x, errors='coerce'))
missing_data = df.isnull()

missing_data.sum()
missing_col = []

for column in missing_data.columns.values.tolist():

    if(missing_data[column].sum() > 0):

        print("Column: ",column)

        print("Missing Data: {} ({:.2f}%)".format(missing_data[column].sum(), (missing_data[column].sum() * 100/ len(df))))

        print("Data Type: ",df[column].dtypes)

        print("")

        missing_col.append(column)
fig1 = plt.figure(figsize=(40,20))

i = 0

for column in missing_col:

    bad = missing_data[column].sum()

    good = len(df) - missing_data[column].sum()

    x = [bad, good]

    labels = ["Missing Data", "Good Data"]

    explode = (0.1, 0)

    i = i+1

    ax = fig1.add_subplot(4,5,i)

    ax.pie(x,explode = explode, labels = labels, shadow = True,autopct='%1.1f%%', colors = ['#ff6666', '#99ff99'],rotatelabels = True, textprops={'fontsize': 18})

    centre_circle = plt.Circle((0,0),0.3,color='black', fc='white',linewidth=0)

    fig = plt.gcf()

    fig.gca().add_artist(centre_circle)

    ax.axis('equal')

    ax.set_title(column, fontsize = 25)

plt.tight_layout()

plt.show()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(df.iloc[:,1:])
df.iloc[:,1:] = imputer.transform(df.iloc[:,1:])
df.head()
missing_data = df.isnull()

missing_data.sum()
df.describe()
df.iloc[:,1:5].describe()
for i in range(1,5):

    print('{0} : min: {1:.2f}\t max: {2:.2f}'.format(df.columns.values[i], df.iloc[:,i].min(), df.iloc[:,i].max()))

    print()    
fig2 = plt.figure(figsize = (20,20))



for i in range(1,5):

    ax = fig2.add_subplot(4,1,i)

    df.iloc[:,i].plot(kind = 'line')

    ax.set_title(df.columns.values[i], fontsize = 18)

plt.show()
mean = 57.5

std = 12.73
test_array = np.random.normal(loc = mean, scale = std, size = (10,4))

test_df_1 = pd.DataFrame(test_array, columns = df.iloc[:,1:5].columns.values)



test_array = np.random.normal(loc = mean, scale = std, size = (12,4))

test_df_2 = pd.DataFrame(test_array, columns = df.iloc[:,1:5].columns.values)



test_array = np.random.normal(loc = mean, scale = std, size = (8,4))

test_df_3 = pd.DataFrame(test_array, columns = df.iloc[:,1:5].columns.values)



test_array = np.random.normal(loc = mean, scale = std, size = (5,4))

test_df_4 = pd.DataFrame(test_array, columns = df.iloc[:,1:5].columns.values)



test_1 = df.iloc[0:1000,1:5].copy()

test_2 = df.iloc[1010:2000,1:5].copy()

test_3 = df.iloc[2012:3500,1:5].copy()

test_4 = df.iloc[3508:4000,1:5].copy()

test_5 = df.iloc[4005:5000,1:5].copy()

test_df = pd.concat([test_1, test_df_1, test_2, test_df_2, test_3, test_df_3, test_4, test_df_4, test_5], ignore_index=True)

test_df.reset_index(drop = True, inplace=True)
test_df.describe()
fig2 = plt.figure(figsize = (20,20))



for i in range(1,5):

    ax = fig2.add_subplot(4,1,i)

    test_df.iloc[:,i-1].plot(kind = 'line', color = 'red')

    df.iloc[:,i].plot(kind = 'line', color = 'green')

    ax.set_title(df.columns.values[i] + ' (With Noise)', fontsize = 18)

    plt.legend(['Noise', 'Original'], fontsize = 16)

plt.show()
# from pandas.plotting import scatter_matrix

# scatter_matrix(df.iloc[:,1:5], alpha=0.2, figsize=(40,20), diagonal='kde')

# plt.show()
from sklearn.svm import OneClassSVM

svm = OneClassSVM(kernel='rbf', degree=3, gamma=0.3, nu = 0.001, max_iter=-1)
X = df.iloc[:,1:5].values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

X = imputer.fit_transform(X)



X_test = test_df.iloc[:,:].values

X_test = sc.transform(X_test)

X_test = imputer.transform(X_test)
svm.fit(X)

y_pred = svm.predict(X_test)
y_pred = ["Outlier" if item == -1 else "Good Data" for item in y_pred]
sns.countplot(y_pred)

plt.show()
from collections import Counter

results_svm = Counter(y_pred)

for key, value in results_svm.items():

    print(key, '=', value)
test_df['outlier_status'] = y_pred

test_df.head()
def display_results(classifier_name = 'After Detection'):



    fig3 = plt.figure(figsize = (20,20))



    for i in range(1,5):

        ax = fig3.add_subplot(4,1,i)

        test_df.iloc[:,i-1].plot(kind = 'line', color = 'red')

        df.iloc[:,i].plot(kind = 'line', color = 'green')

        ax.scatter(x = test_df[test_df['outlier_status'] == 'Outlier'].index, y = test_df[test_df['outlier_status'] == 'Outlier'].iloc[:,i-1].values, s = 80)

        ax.set_title(df.columns.values[i] + ' (' + classifier_name + ')', fontsize = 18)

        plt.legend(['Noise', 'Original', 'Detected Outliers'], fontsize = 12)

    plt.show()
display_results('OneClass SVM')
# Perform Dimensionality Reduction (Kernel PCA)

from sklearn.decomposition import KernelPCA

pca = KernelPCA(n_components = 2, kernel = 'rbf')

X_pca = pca.fit_transform(X)



# For Normal Algorithms

def predict_pca(classifier):

    try:

        y_pca = classifier.fit_predict(X_pca)

    except AttributeError: # For LOF only

        classifier.fit(X_pca)

        y_pca = classifier.predict(X_pca)

    y_pca = [-1 if item == -1 else 1 for item in y_pca]

    

    return np.array(y_pca)



# For DBSCAN(Clustering)

def predict_clusters(classifier):

    classifier = classifier.fit(X)

    y_pca = list(classifier.labels_)

    y_pca = [-1 if item == -1 else 1 for item in y_pca]

    return np.array(y_pca)

    

# Plot Scatterplot Graph    

def plot_data(classifier, clustering = False, title = ''):    

    

    if clustering == True:

        y_pca = predict_clusters(classifier)

    else:

        y_pca = predict_pca(classifier)

    #print(y_pca)

    fig = plt.figure(figsize = (15,8))

    plt.scatter(X_pca[y_pca == -1,0],X_pca[y_pca == -1,1], color = 'red')

    plt.scatter(X_pca[y_pca == 1,0],X_pca[y_pca == 1,1], color = 'green')

    plt.legend(labels = ['Outliers', 'Good Data'], fontsize = 16)

    plt.title('Outliers Detection Scatterplot - ' + title, fontsize = 20)

    plt.tight_layout()

    plt.show()
plot_data(svm, title = 'OneClassSVM')
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples = 10)
db = db.fit(X_test)
print("Number of Clusters = ",max(db.labels_)+1)
sns.countplot(db.labels_)

plt.show()
y_pred = list(db.labels_)

y_pred = ["Outlier" if item == -1 else "Good Data" for item in y_pred]

sns.countplot(y_pred)

plt.show()
results_db = Counter(y_pred)

for key, value in results_db.items():

    print(key, '=', value)
test_df['outlier_status'] = y_pred

display_results('DBSCAN')
plot_data(db, clustering = True, title = 'DBSCAN')
from sklearn.ensemble import IsolationForest

iforest = IsolationForest(n_estimators = 2000, 

                          contamination = 0.01,

                          random_state = 42,

                          behaviour = 'new').fit(X)
y_pred = iforest.predict(X_test)
y_pred = ["Outlier" if item == -1 else "Good Data" for item in y_pred]

sns.countplot(y_pred)

plt.show()
results_iforest = Counter(y_pred)

for key, value in results_iforest.items():

    print(key, '=', value)
test_df['outlier_status'] = y_pred

display_results('Isolation Forest')
plot_data(iforest, title = 'Isolation Forest')
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors = 25,                        

                         metric='minkowski',

                         contamination = 0.006)
y_pred = lof.fit_predict(X_test)
y_pred = ["Outlier" if item == -1 else "Good Data" for item in y_pred]

sns.countplot(y_pred)

plt.show()
results_lof = Counter(y_pred)

for key, value in results_lof.items():

    print(key, '=', value)
test_df['outlier_status'] = y_pred

display_results('Local Outlier Factor')
plot_data(lof, title = 'Local Outlier Factor (LOF)')
report = pd.DataFrame(columns = ['Algorithm', 'Good Data', 'Outliers', 'Contamination Ratio'])

report['Algorithm'] = ['OneClassSVM', 'DBSCAN', 'Isolation Forest', 'Local Outlier Factor']



#ADD Results to Report

report['Good Data'] = [results_svm['Good Data'], results_db['Good Data'], results_iforest['Good Data'], results_lof['Good Data']]

report['Outliers'] = [results_svm['Outlier'], results_db['Outlier'], results_iforest['Outlier'], results_lof['Outlier']]

report['Contamination Ratio'] = [results_svm['Outlier']/results_svm['Good Data'], results_db['Outlier']/results_db['Good Data'], results_iforest['Outlier']/results_iforest['Good Data'], results_lof['Outlier']/results_lof['Good Data']]



report
report[['Algorithm', 'Good Data', 'Outliers']].plot(kind = 'bar', x = 'Algorithm', cmap = 'Dark2')

plt.title('Outliers Detection Algorithms Performance')

plt.ylabel('Counts')

plt.show()
report[['Algorithm', 'Outliers']].plot(kind = 'bar', x = 'Algorithm', cmap = 'coolwarm')

plt.title('Outliers Detected')

plt.ylabel('Counts')

plt.legend().remove()

plt.show()
report[['Algorithm', 'Contamination Ratio']].plot(kind = 'bar', x = 'Algorithm', cmap = 'Accent')

plt.title('Contamination Detected')

plt.ylabel('Percentage(%)')

plt.legend().remove()

plt.show()