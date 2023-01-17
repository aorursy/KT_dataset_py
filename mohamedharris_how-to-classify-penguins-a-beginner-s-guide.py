import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #simple data visualization

%matplotlib inline

import seaborn as sns #some advanced data visualizations

import warnings

warnings.filterwarnings('ignore') # to get rid of warnings

plt.style.use('seaborn-white') #defining desired style of viz



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

original = df.copy()
print('Dataset has', df.shape[0] , 'rows and', df.shape[1], 'columns')
df.info()
df.describe()
df.isnull().sum()
df.head(10)
plt.rcParams['figure.figsize'] = (10,7)
df['species'].value_counts(normalize = True).plot(kind = 'bar', color = 'seagreen', linewidth = 1, edgecolor = 'k')

plt.title('Penguin Species')

plt.xlabel('Species')

plt.ylabel('% (100s)')

plt.xticks(rotation = 360)

plt.show()
df['island'].value_counts(normalize = True).plot(kind = 'bar', color = 'seagreen', linewidth = 1, edgecolor = 'k')

plt.title('Islands where Penguins live')

plt.xlabel('Island')

plt.ylabel('% (100s)')

plt.xticks(rotation = 360)

plt.show()
df['sex'].value_counts(normalize = True).plot(kind = 'bar', color = 'seagreen', linewidth = 1, edgecolor = 'k')

plt.title('Penguins - Sex')

plt.xlabel('Sex')

plt.ylabel('% (100s)')

plt.xticks(rotation = 360)

plt.show()
def ecdf(x):

    n = len(x)

    a = np.sort(x)

    b = np.arange(1, 1 + n) / n

    plt.subplot(211)

    plt.plot(a, b, marker = '.', linestyle = 'None', c = 'seagreen')

    mean_x = np.mean(x)

    plt.axvline(mean_x, c = 'k', label = 'Mean')

    plt.title('ECDF')

    plt.legend()

    plt.show()

    plt.subplot(212)

    sns.distplot(x, color = 'r')

    plt.title('Probability Density Function')

    plt.show()
ecdf(df['culmen_length_mm'])
ecdf(df['culmen_depth_mm'])
ecdf(df['flipper_length_mm'])
ecdf(df['body_mass_g'])
def box(f):

    sns.boxplot(y = f, x = 'species', hue = 'sex',data = df)

    plt.title(f)

    plt.show()
box('culmen_length_mm')
box('culmen_depth_mm')
box('flipper_length_mm')
box('body_mass_g')
sns.pairplot(df, hue = 'species')

plt.show()
new_df = original.copy()



new_df['culmen_length_mm'].fillna(np.mean(original['culmen_length_mm']), inplace = True)

new_df['culmen_depth_mm'].fillna(np.mean(original['culmen_depth_mm']), inplace = True)

new_df['flipper_length_mm'].fillna(np.mean(original['flipper_length_mm']), inplace = True)

new_df['body_mass_g'].fillna(np.mean(original['body_mass_g']), inplace = True)

new_df['sex'].fillna(original['sex'].mode()[0], inplace = True)
new_df.head()
new_df.isnull().sum()
print('Skewness of numeric variables')

print('-' * 35)



for i in new_df.select_dtypes(['int64', 'float64']).columns.tolist():

    print(i, ' : ',new_df[i].skew())
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
new_df['culmen_length_mm'] = mms.fit_transform(new_df['culmen_length_mm'].values.reshape(-1, 1))

new_df['culmen_depth_mm'] = mms.fit_transform(new_df['culmen_depth_mm'].values.reshape(-1, 1))

new_df['flipper_length_mm'] = mms.fit_transform(new_df['flipper_length_mm'].values.reshape(-1, 1))

new_df['body_mass_g'] = mms.fit_transform(new_df['body_mass_g'].values.reshape(-1, 1))
new_df.head()
new_df.describe()
new_df_dummy = pd.get_dummies(new_df, columns = ['sex', 'island'], drop_first = True)
new_df_dummy['species'].unique()
new_df_dummy['species'].replace({'Adelie' : 0,

                                'Chinstrap' : 1,

                                'Gentoo': 2}, inplace = True)
sns.heatmap(new_df_dummy.corr(), annot = True, cmap = 'Blues')
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
X = new_df_dummy.drop(columns = ['species', 'sex_FEMALE', 'sex_MALE'])

Y = new_df_dummy['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 123)
LR = LogisticRegression()

LR.fit(X_train, Y_train)



pred = LR.predict(X_test)
print('Accuracy : ', accuracy_score(Y_test, pred))

print('F1 Score : ', f1_score(Y_test, pred, average = 'weighted'))
models = []

models.append(('LR', LogisticRegression()))

models.append(('DT', DecisionTreeClassifier()))

models.append(('RF', RandomForestClassifier()))

models.append(('kNN', KNeighborsClassifier()))

models.append(('SVC', SVC()))
for name, model in models:

    kfold = KFold(n_splits = 5, random_state = 42)

    cv_res = cross_val_score(model, X_train, Y_train, scoring = 'accuracy', cv = kfold)

    print(name, ' : ', cv_res.mean())
svc = SVC()

svc.fit(X_train, Y_train)



pred = LR.predict(X_test)
print('Accuracy : ', accuracy_score(Y_test, pred))

print('F1 Score : ', f1_score(Y_test, pred, average = 'weighted'))

print('Precision : ', precision_score(Y_test, pred , average = 'weighted'))

print('Recall : ', recall_score(Y_test, pred, average = 'weighted'))
confusion_matrix(Y_test, pred)