import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



from sklearn.metrics import classification_report
data_path = '/kaggle/input/heart-disease-uci-dataset/heart.csv'



df = pd.read_csv(data_path)

df.head()
df.info()
df.describe()
df.columns
new_columns = [col.replace(' ', '_') for col in df.columns]

df.columns = new_columns

new_columns
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), cmap="YlGnBu");
df_presence = df[df.Presence == 2]

df_absence = df[df.Presence == 1]



df_sex_0 = df[df.Sex == 0]

df_sex_1 = df[df.Sex == 1]
sex_0 = df_presence[df_presence.Sex == 0]

sex_1 = df_presence[df_presence.Sex == 1]

plt.figure(figsize=(15,5))

plt.title("Age distribution of the presence of heart disease by gender")

sns.kdeplot(data=sex_0['Age'], shade=True, label='Sex_0');

sns.kdeplot(data=sex_1['Age'], shade=True, label='Sex_1');
plt.subplots(figsize=(15, 5))

plt.title('Female and man with heart disease')

sns.countplot(y="Sex", data=df_presence, color="c");
plt.subplots(figsize=(15, 5))

sns.countplot(y="Chest_Pain_type", data=df_presence, color="c");
pain_1 = df_presence[df_presence.Chest_Pain_type == 1]

pain_2 = df_presence[df_presence.Chest_Pain_type == 2]

pain_3 = df_presence[df_presence.Chest_Pain_type == 3]

pain_4 = df_presence[df_presence.Chest_Pain_type == 4]



plt.figure(figsize=(15,5))

plt.title("Chest pain distribution of the presence of heart disease by gender")

sns.kdeplot(data=pain_1['Age'], shade=True, label='Pain 1');

sns.kdeplot(data=pain_2['Age'], shade=True, label='Pain 2');

sns.kdeplot(data=pain_3['Age'], shade=True, label='Pain 3');

sns.kdeplot(data=pain_4['Age'], shade=True, label='Pain 3');
plt.figure(figsize=(15,5))

plt.title("Mean resting blood pressure for sex 0")

plt.xlabel('Age')

plt.ylabel('Resting blood pressure')

data = df[df.Sex == 0]



sns.lineplot(

    data=data[data.Presence == 2].groupby(['Age'])['Resting_Blood_pressure'].mean(), 

    label='Presence'

);



sns.lineplot(

    data=data[data.Presence == 1].groupby(['Age'])['Resting_Blood_pressure'].mean(), 

    label='Absence'

);
plt.figure(figsize=(15,5))

plt.title("Mean resting blood pressure for sex 1")

plt.xlabel('Age')

plt.ylabel('Resting blood pressure')

data = df[df.Sex == 1]



sns.lineplot(

    data=data[data.Presence == 2].groupby(['Age'])['Resting_Blood_pressure'].mean(), 

    label='Presence'

);



sns.lineplot(

    data=data[data.Presence == 1].groupby(['Age'])['Resting_Blood_pressure'].mean(), 

    label='Absence'

);
plt.figure(figsize=(15, 10))

sns.scatterplot(x="Age", y="Serum_Cholestoral_in_mg/dl", hue="Presence", palette="Set1", data=df);
num_features = len(df.columns) - 1

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)
cov_mat = np.cov(X_train_std.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('Eigen vals: {}'.format(eigen_vals))
tot = sum(eigen_vals)

var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)



plt.figure(figsize=(15,5))

plt.bar(range(num_features), var_exp, alpha=.5, align='center', label='Single variance')

plt.step(range(num_features), cum_var_exp, where='mid', label='Cumsum variance')

plt.ylabel('Factor of variance')

plt.xlabel('Main components')

plt.legend(loc='best')

plt.show()
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

eigen_pairs.sort(key = lambda k: k[0], reverse=True)

W = np.hstack((eigen_pairs[0][1][:, np.newaxis], 

               eigen_pairs[1][1][:, np.newaxis]))

print('Matrix W: \n{}'.format(W))
X_train_pca = X_train_std.dot(W)

X_test_pca = X_test_std.dot(W)
def run_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('Report:\n{}'.format(classification_report(y_test, y_pred)))

    print('Score: {}'.format(model.score(X_test, y_test)))
lr = LogisticRegression()

run_model(lr, X_train=X_train_pca, y_train=y_train, X_test=X_test_pca, y_test=y_test)
svc = SVC(kernel='linear', C=1.0, random_state=0)

run_model(svc, X_train=X_train_pca, y_train=y_train, X_test=X_test_pca, y_test=y_test)
knn = KNeighborsClassifier(n_neighbors=20, p=2, metric='minkowski')

run_model(knn, X_train=X_train_pca, y_train=y_train, X_test=X_test_pca, y_test=y_test)