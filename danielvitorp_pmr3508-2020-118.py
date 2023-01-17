import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
test_data = pd.read_csv('../input/adult-pmr3508/test_data.csv')
train_data = pd.read_csv('../input/adult-pmr3508/train_data.csv')
test_data.head()
train_data.describe()
train_data.head()
train_data.info()
from sklearn import preprocessing

corr = train_data.apply(preprocessing.LabelEncoder().fit_transform).corr()
corr.head()
sns.heatmap(corr)
# exemplo de gráfico pairplot

sns.pairplot(train_data)
def contar_valores(column):
    
    return print(train_data[f'{column}'].value_counts())
columns = ['age', 'fnlwgt', 'education.num', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']

for column in columns:
    contar_valores(column)
def contar_valores(column):
    
    return train_data[f'{column}'].value_counts().plot(title= f'{column}', kind="bar")
columns = ['age', 'fnlwgt', 'education.num', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
contar_valores(columns[2])
contar_valores(columns[3])
contar_valores(columns[4])
contar_valores(columns[5])
contar_valores(columns[6])
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

SEED = 301
np.random.seed(SEED)
cv = KFold(n_splits = 10, shuffle = True)
columns = ["age", 'fnlwgt', "education.num", "occupation", "relationship",
               "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country"]

#exemplo:
num_test_data = test_data.apply(preprocessing.LabelEncoder().fit_transform)
x_test = num_test_data[columns]
def data_preprocessing(dataset):
    
    # pré-processamento
    num_train_data = dataset.apply(preprocessing.LabelEncoder().fit_transform)

    x = num_train_data[columns]
    y = dataset.income
    
    return x, y
def modeloKNN(dataset):
    
    x, y = data_preprocessing(dataset)    

    knn = KNeighborsClassifier(n_neighbors=25)
    
    cross_score = cross_val_score(knn, x, y, cv=cv).mean()

    
    
    return print(f'Média da Acurácia com Validação Cruzada : {cross_score}')
modeloKNN(train_data)
columns = ["age", "education.num", "occupation", "relationship",
               "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country"]
modeloKNN(train_data)
def sum_groups(cluster):
    
    k = 0
    j = 0
    
    for value in cluster:
        if value == '<=50K':
            k += 1
        else:
            j += 1
            
    return k, j
primeiro_cluster = train_data.loc[(train_data['education.num'] > 9)].income
primeiro_cluster.values

fig,ax = plt.subplots()
size = 0.3

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([1, 2])

ax.pie(sum_groups(primeiro_cluster.values), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))


ax.set(aspect="equal", title='Distribuição da renda para Tempo de educação > que 9')
plt.legend(['<=50K', 'Outros'])

primeiro_cluster = train_data.loc[(train_data['education.num'] <= 9)].income
primeiro_cluster.values

fig, ax = plt.subplots()


cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([1, 2])

ax.pie(sum_groups(primeiro_cluster.values), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))


ax.set(aspect="equal", title='Distribuição da renda para Tempo de educação < que 9')
plt.legend(['<=50K', 'Outros'])
plt.show()
primeiro_cluster = train_data.loc[(train_data['race'] != 'White')].income
primeiro_cluster.values

fig, ax = plt.subplots()


cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([1, 2])

ax.pie(sum_groups(primeiro_cluster.values), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))


ax.set(aspect="equal", title='Distribuição da renda para race dif de White')
plt.legend(['<=50K', 'Outros'])

primeiro_cluster = train_data.loc[(train_data['race'] == 'White')].income
primeiro_cluster.values

fig, ax = plt.subplots()


cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([1, 2])

ax.pie(sum_groups(primeiro_cluster.values), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))


ax.set(aspect="equal", title='Distribuição da renda para race White')
plt.legend(['<=50K', 'Outros'])
plt.show()
numeros = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorias = ['occupation', 'relationship', 'sex']
train_data.drop_duplicates().head()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
train_data_imputed_num = imputer.fit_transform(train_data[numeros])

train_data_imputed_num = pd.DataFrame(train_data_imputed_num, columns=numeros)
train_data_imputed_num
from sklearn.preprocessing import OrdinalEncoder

train_data_encoded = train_data[categorias].apply(preprocessing.LabelEncoder().fit_transform)
train_data_encoded
from sklearn.impute import SimpleImputer

train_data_imputed_cat = imputer.fit_transform(train_data_encoded[categorias])

train_data_imputed_cat = pd.DataFrame(train_data_imputed_cat, columns=categorias)
train_data_imputed_cat
new_train_data = pd.concat([train_data_imputed_num, train_data_imputed_cat], axis= 1, join='inner')
new_train_data
def modeloKNN(data):
    
    x = data
    y = train_data.income
    
    knn = KNeighborsClassifier(n_neighbors=25)
    
    cross_score = cross_val_score(knn, x, y, cv=cv).mean()

    return print(f'Média da Acurácia com Validação Cruzada : {cross_score}')
modeloKNN(new_train_data)
from sklearn.impute import MissingIndicator, KNNImputer

def trainPreprocessing(imputer):
    
    train_data_imputed_num = imputer.fit_transform(train_data[numeros])
    train_data_imputed_num = pd.DataFrame(train_data_imputed_num, columns=numeros)

    train_data_imputed_cat = imputer.fit_transform(train_data_encoded[categorias])
    train_data_imputed_cat = pd.DataFrame(train_data_imputed_cat, columns=categorias)

    trainDataPreprocessing = pd.concat([train_data_imputed_num, train_data_imputed_cat], axis= 1, join='inner')
    
    return trainDataPreprocessing
knnImputer = KNNImputer(n_neighbors = 2, weights = 'uniform')

trainDataPreprocessing = trainPreprocessing(knnImputer)

modeloKNN(trainDataPreprocessing)
from sklearn.pipeline import FeatureUnion

transformer = FeatureUnion(transformer_list=[
    ('features', KNNImputer(n_neighbors = 2, weights = 'uniform')),
    ('indicators', MissingIndicator())
])

trainDataPreprocessing = trainPreprocessing(transformer)
modeloKNN(trainDataPreprocessing)
x = trainDataPreprocessing
y = train_data.income

parameters = {
    'weights': ('uniform', 'distance'),
    'n_neighbors': [5,10,15,20]
}


for weight in parameters['weights']:
    for n_neighbor in parameters['n_neighbors']:
        
        transformer = FeatureUnion(transformer_list=[
            ('features', KNNImputer(n_neighbors = n_neighbor, weights = weight)),
            ('indicators', MissingIndicator())
        ])
        
        testDataPreprocessing = trainPreprocessing(transformer)
        print(f'Com weights = {weight} e n_neighbor = {n_neighbor}:')
        modeloKNN(testDataPreprocessing)
from sklearn.model_selection import GridSearchCV

x = testDataPreprocessing
y = train_data.income

parameters = {
    'weights': ('uniform', 'distance'),
    'n_neighbors': [5,10,15,20]
}


for weight in parameters['weights']:
    for n_neighbor in parameters['n_neighbors']:
        
        knnImputer = KNNImputer(n_neighbors = n_neighbor, weights = weight)

        
        testDataPreprocessing = trainPreprocessing(knnImputer)
        print(f'Com weights = {weight} e n_neighbor = {n_neighbor}:')
        modeloKNN(testDataPreprocessing)
transformer = FeatureUnion(transformer_list=[
            ('features', KNNImputer(n_neighbors = 20, weights = 'uniform')),
            ('indicators', MissingIndicator())
        ])

trainDataPreprocessing = trainPreprocessing(transformer)
from sklearn.preprocessing import StandardScaler

modeloKNN(StandardScaler().fit_transform(trainDataPreprocessing))
from sklearn.preprocessing import Normalizer

modeloKNN(Normalizer().fit_transform(trainDataPreprocessing))
from sklearn.preprocessing import RobustScaler

modeloKNN(RobustScaler().fit_transform(trainDataPreprocessing))
def preprocessingData(test_data):
    
    imputer = FeatureUnion(transformer_list=[
            ('features', KNNImputer(n_neighbors = 10, weights = 'uniform')),
            ('indicators', MissingIndicator())
        ])
    
    test_data_imputed_num = imputer.fit_transform(test_data[numeros])
    test_data_imputed_num = pd.DataFrame(test_data_imputed_num, columns=numeros)
    
    test_data_encoded = test_data[categorias].apply(preprocessing.LabelEncoder().fit_transform)

    test_data_imputed_cat = imputer.fit_transform(test_data_encoded[categorias])
    test_data_imputed_cat = pd.DataFrame(test_data_imputed_cat, columns=categorias)
    
    testData = pd.concat([test_data_imputed_num, test_data_imputed_cat], axis= 1, join='inner')
    
    testData = RobustScaler().fit_transform(testData)
    
    return testData
trainData = preprocessingData(train_data)
from sklearn.model_selection import GridSearchCV

x = trainData
y = train_data.income

parameters = {
    'weights': ('uniform', 'distance'),
    'n_neighbors': [5,10,15,20, 25, 30],
    'leaf_size': [20, 25, 30, 35, 40],
    'p': [1, 2]
    
}

knn = KNeighborsClassifier()

gridKNN = GridSearchCV(estimator = knn, param_grid = parameters)
gridKNN.fit(x, y)
gridKNN.best_score_
gridKNN.best_estimator_
def predictKNN(data):
    
    x = data
    y = train_data.income
    
    knn = KNeighborsClassifier(n_neighbors=30, leaf_size = 20, p = 1)
    
    knn.fit(x,y)
    
    knn.predict(testData)
    
    predict = knn.predict(testData)

    return predict
testData = preprocessingData(test_data)
    
predict = predictKNN(trainData)
prediction = pd.DataFrame({'Income': predict})
prediction.index.names = ['Id']

prediction.to_csv(path_or_buf=r'./prediction.csv', sep=',')

prediction