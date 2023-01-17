import pandas as pd
adult_train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult_train.shape
adult_train.head()
adult_train.isnull().sum().sort_values(ascending = False).head(5)
adult_train = adult_train.dropna()
adult_train.isnull().sum().sort_values(ascending = False).head(5)
def make_data_numeric(dataset, column_names):

    new_dataset = dataset.copy()

    for i in range(len(column_names)):

        curr_column = dataset[column_names[i]].unique().tolist()

        mapping = dict(zip(curr_column, range(len(curr_column))))

        new_dataset.replace({column_names[i]: mapping}, inplace = True)

        

    return new_dataset
train_categoric_column_names = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']



adult_train = make_data_numeric(adult_train, train_categoric_column_names)
adult_train.describe()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
correlation = adult_train.corr()



plt.figure(figsize=(16,16))

matrix = np.triu(correlation)

sns.heatmap(correlation, annot=True, mask = matrix, vmin = -0.5, vmax = 0.5, center = 0, cmap= 'coolwarm')
choosen_columns = ["education.num", "age", "hours.per.week", "capital.gain", "sex", "marital.status", "capital.loss", "relationship", "income"]



adult_train = adult_train[choosen_columns].copy()
X_adult_train = adult_train.drop(["income"], axis = 1)

Y_adult_train = adult_train["income"]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
#Preparação das configurações para o GridSearch

k_range = list(range(5, 31))

p_options = list(range(1,3))

param_grid = dict(n_neighbors=k_range, p=p_options)
knn = KNeighborsClassifier(n_neighbors=5)



grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs = -2)  
grid.fit(X_adult_train, Y_adult_train)

print(grid.best_estimator_)

print(grid.best_score_)
from sklearn.model_selection import cross_val_score
acc = 0

best_i = 0

for i in range(4, 8, 1):

    columns = choosen_columns[0:i]

    new_X_adult_train = adult_train[columns].copy()

    knn_model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=26, p=1,

                     weights='uniform')

    score = np.mean(cross_val_score(knn_model, new_X_adult_train, Y_adult_train, cv=10))

    if score > acc:

        best_i = i

        acc = score
print(best_i, score)
final_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=26, p=1,

                     weights='uniform')



final_columns = ["education.num", "age", "hours.per.week", "capital.gain", "sex", "marital.status", "capital.loss"]



X_adult_train = adult_train[final_columns].copy()



final_knn.fit(X_adult_train, Y_adult_train)
#Leitura do arquivo csv

adult_test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



#Transformação das colunas categóricas em valores numéricos

test_categoric_column_names = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']



adult_test = make_data_numeric(adult_test, test_categoric_column_names)



#Seleção das features desejadas

X_adult_test = adult_test[final_columns].copy()
Y_adult_test = final_knn.predict(X_adult_test)
#Restora a label para o formato original

def label_transform(y):

    pred_array = []

    for i in range(len(y)):

        if (y[i] == 0):

            pred_array.append('<=50K')

        else:

            pred_array.append('>50K')

    return pred_array
label_array = label_transform(Y_adult_test)

label_df = pd.DataFrame({'income': label_array})
label_df.to_csv("submission.csv", index = True, index_label = 'Id')