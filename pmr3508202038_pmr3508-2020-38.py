import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
columnNames = ['Id', 'age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
                'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week',
                'native.country', 'income']
df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = '?')
df_train.head()
df_shape = df_train.shape
print("Número de Linhas:", df_shape[0])
print("Número de colunas:", df_shape[1])
df_train.describe()
df_train.describe(exclude=[np.number])
"""
Função para descobrir a quantidade de dados faltantes em cada coluna
Recebe pandas.dataFrame
Retorna lista com quantidade de dados faltantes por coluna
"""
def missing_data(dataFrame):
    quantity = pd.Series(dataFrame.isnull().sum(), name='qty')
    frequency = pd.Series(100*quantity/(dataFrame.count() + quantity), name = 'freq', dtype='float16')
    missingData = pd.concat([quantity, frequency], axis=1)
    return missingData
missing_data(df_train)
# Retirando linhas do df_train com NaN values
new_df_train = df_train.dropna()
missing_data(new_df_train)
new_df_train.shape
new_df_train.head()
columns_name = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
df_analyse = new_df_train.copy()
for i in range(len(columns_name)):
    curr_column = df_analyse[columns_name[i]].unique().tolist()
    mapping = dict(zip(curr_column, range(len(curr_column))))
    df_analyse.replace({columns_name[i]: mapping}, inplace = True)
df_analyse.shape
df_analyse.describe()
df_analyse.head()
correlation = df_analyse.corr()

plt.figure(figsize=(16,16))
matrix = np.triu(correlation)
sns.heatmap(correlation, annot=True, mask = matrix, vmin = -0.5, vmax = 0.5, center = 0, cmap= 'coolwarm')
knn_columns = ["age", "education.num", "capital.gain", "hours.per.week", "income"]
knn_df = df_analyse[knn_columns].copy()
knn_df.head()
neighbors_num = 7
X = knn_df.drop(['income'], axis = 1)
y = knn_df['income']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(neighbors_num)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
from sklearn import metrics
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

def best_k(X, y, max_neigh):
    k = 1
    max_mean = 0
    max_std = 0
    max_k = -1
    while (k < max_neigh):
        KNNclf = KNeighborsClassifier(n_neighbors = k)
        curr_scores = cross_val_score(KNNclf, X, y)
        curr_score_mean = curr_scores.mean()
        curr_score_std = curr_scores.std()
        print(" K = %d: Accuracy: %0.2f (+/- %0.2f)" % (k, curr_score_mean, curr_score_std * 2))
        if (curr_score_mean > max_mean):
            max_mean = curr_score_mean
            max_std = curr_score_std
            max_k = k
        k += 2
    print ("Max k = %d Accuracy: %0.2f (+/- %0.2f)" % (max_k, max_mean, max_std * 2))
    return max_k
max_neigh = 33
max_k = best_k(X, y, max_neigh)
#Create KNN Classifier
knn = KNeighborsClassifier(max_k)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Leitura do arquivo test_data.csv
df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values = '?')
# Verificar formato do arquivo
df_test.head()
# Guardar apenas as colunas cujas propriedades são de interesse
knn_test_columns = ["age", "education.num", "capital.gain", "hours.per.week"]
knn_test_df = df_test[knn_test_columns].copy()
# Verificar se dados foram selecionados com sucesso
knn_test_df.head()
# Verificar se há dados faltantes
missing_data(knn_test_df)
y_pred = knn.predict(knn_test_df)
# Função para refazer troca de valores 0 e 1 para '<=50K' e '>50K' respectivamente
# Recebe numpy.array y_pred type int
# Retorna numpy.array pred_array type String
def pred_transform(y_pred):
    pred_array = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            pred_array.append('<=50K')
        else:
            pred_array.append('>50K')
    return pred_array
pred_array = pred_transform(y_pred)
df_pred = pd.DataFrame({'income': pred_array})
df_pred.head()
df_pred.to_csv("./submission.csv", index = True, index_label = 'Id')
