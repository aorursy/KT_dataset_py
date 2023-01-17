import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.colors as colors



from sklearn.utils import resample

from sklearn.preprocessing import scale

from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, StratifiedKFold



from sklearn.svm import SVC



from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



file_path = "/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv"

#file_path1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

df = pd.read_csv(file_path)



df.head()
df.info()
df.rename({"default.payment.next.month" : "DEFAULT"}, axis="columns", inplace=True)

df.head()
df.drop(columns='ID', inplace=True)

df.head(3)
df.describe().T
df.info()
print("Unique values of each column\n")

for cols in ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'DEFAULT']:

    print(cols, " : ", df[cols].unique())
len(df.loc[(df['EDUCATION']==0) | (df['MARRIAGE']==0)])
len(df)
len(df.loc[(df['EDUCATION']==0) | (df['MARRIAGE']==0)]) / len(df) * 100
df_msno = df.loc[(df['EDUCATION']==0) | (df['MARRIAGE']==0)]

df_msno.shape
df =  df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]

df.shape
df['DEFAULT'].value_counts()
df_default = df[df['DEFAULT']==1]

df_no_default = df[df['DEFAULT']==0]



display(len(df_default), len(df_no_default))
sns.countplot(df['SEX']);
sns.countplot(df['MARRIAGE']);
sns.countplot(df['EDUCATION']);
sns.distplot(df['LIMIT_BAL']);
default_by_gender = pd.crosstab(df['SEX'], df['DEFAULT'])

sns.heatmap(default_by_gender, annot=True, fmt='2d');
default_by_gender.plot(kind='barh', stacked=True);
plt.figure(figsize=(18,18))

sns.pairplot(df)

plt.show()
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(), annot=True, fmt='.2f', square=True)

plt.show()
df_no_default_downsampled = resample(df_no_default,

                                    replace=False,

                                    n_samples=1000,

                                    random_state=24)

len(df_no_default_downsampled)
df_default_downsampled = resample(df_default,

                                    replace=False,

                                    n_samples=1000,

                                    random_state=24)

len(df_default_downsampled)
df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled])

len(df_downsample)
X = df_downsample.drop(columns='DEFAULT', axis=1).copy()

X.shape
y = df_downsample['DEFAULT'].copy()

y.shape
X_encoded = pd.get_dummies(X, columns=['SEX','MARRIAGE','EDUCATION','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'])

X_encoded.head()
X_encoded.shape
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=24)
X_train.head(3)
X_test.head(3)
X_train_scaled = scale(X_train)

X_test_scaled = scale(X_test)
clf_svc = SVC(C=1.0,

              kernel='rbf',

             gamma='auto',

             probability=True)
clf_svc.fit(X_train_scaled, y_train)
y_pred = clf_svc.predict(X_test_scaled)
print("Classification Report : \n")

print(classification_report(y_pred, y_test))
plot_confusion_matrix(clf_svc,

                     X_test_scaled,

                     y_test,

                     values_format='d',

                     display_labels=['No Default','Default'])
plot_roc_curve(clf_svc,

               X_test_scaled,

               y_test)
plot_precision_recall_curve(clf_svc,

                            X_test_scaled,

                            y_test)
param_grid = [{

    'C' : [0.5, 1.0, 10, 100],

    'gamma' : ['scale', 1, 0.1, 0.01, 0.001, 0.0001],

    'kernel' : ['rbf']

}]
clf_svc_tuned = GridSearchCV(SVC(),

                             param_grid,

                             cv=5,

                             scoring='accuracy',

                             verbose=2

)
clf_svc_tuned.fit(X_train_scaled, y_train)
clf_svc_tuned.best_estimator_
clf_svc_tuned.best_params_
y_pred_tuned = clf_svc_tuned.predict(X_test_scaled)
print("Classification Report : \n")

print(classification_report(y_pred_tuned, y_test))
plot_confusion_matrix(clf_svc_tuned,

                     X_test_scaled,

                     y_test,

                     values_format='d',

                     display_labels=['No Default','Default'])
plot_roc_curve(clf_svc_tuned,

               X_test_scaled,

               y_test)
plot_precision_recall_curve(clf_svc_tuned,

                            X_test_scaled,

                            y_test)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
pvar = np.round(pca.explained_variance_ratio_*100, decimals=1)

labels = [str(x) for x in range(1, len(pvar)+1)]



plt.bar(x=range(1, len(pvar)+1), height=pvar)

plt.tick_params(axis='x',

               which='both',

               bottom=False,

               top=False,

               labelbottom=False)

plt.ylabel('Percentage of Explained Variance')

plt.xlabel('Principal Components')

plt.title('Scree Plot')

plt.show()
train_pc1_coords = X_train_pca[:,0]

train_pc2_coords = X_train_pca[:,1]



pca_train_scaled = np.column_stack((train_pc1_coords, train_pc2_coords))
clf_svc_tuned.fit(pca_train_scaled, y_train)
X_test_pca = pca.transform(X_train_scaled)
test_pc1_coords = X_test_pca[:,0]

test_pc2_coords = X_test_pca[:,1]



pca_test_scaled = np.column_stack((train_pc1_coords, train_pc2_coords))



x_min = test_pc1_coords.min() - 1

x_max = test_pc1_coords.max() - 1



y_min = test_pc2_coords.min() - 1

y_max = test_pc2_coords.max() - 1



xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),

                     np.arange(start=y_min, stop=y_max, step=0.1))



Z = clf_svc_tuned.predict(np.column_stack((xx.ravel(), yy.ravel())))

Z = Z.reshape(xx.shape)



fig, ax = plt.subplots(figsize=(10,10))

ax.contourf(xx, yy, Z, alpha=0.1)



cmap = colors.ListedColormap(['#e41a1c','#4daf4a'])



scatter = ax.scatter(test_pc1_coords,

                     test_pc2_coords,

                     c=y_train,

                     cmap=cmap,

                     s=100,

                     edgecolors='k',

                     alpha=0.7)



legend = ax.legend(scatter.legend_elements()[0],

                   scatter.legend_elements()[1],

                   loc='upper right')



legend.get_texts()[0].set_text('No Default')

legend.get_texts()[1].set_text('Default')



ax.set_ylabel('PC2')

ax.set_xlabel('PC1')

ax.set_title('Decision Boundary')



plt.show()