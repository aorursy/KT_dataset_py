import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



%matplotlib inline
df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")
print('Tamanho do DataFrame: ', df_train.shape)

df_train.head()
df_train.info()
def count_missing_values(data):

    '''

    Count missing values for each feature and return a sorted DataFrame with the resuls

    '''



    missing_count = []

    for column in data.columns:

        missing_count.append(data[column].isna().sum())

    missing_count = np.asarray(missing_count)

    missing_count = pd.DataFrame({'feature': data.columns, 'count': missing_count,

                                'freq. [%]': 100*missing_count/data.shape[0]}, index=None)

    missing_count.sort_values('count', ascending=False, inplace=True, ignore_index=True)

    return missing_count
display(count_missing_values(df_train).head())

display(count_missing_values(df_test).head())
datas = ['workclass', 'native.country', 'occupation']



for data in datas:

    

    value = df_train[data].describe().top

    df_train[data] = df_train[data].fillna(value)

    

    value = df_test[data].describe().top

    df_test[data] = df_test[data].fillna(value)
display(count_missing_values(df_train).head())

display(count_missing_values(df_test).head())
df_train_analysis = df_train.copy()
# Importando o LabelEncoder

from sklearn.preprocessing import LabelEncoder



# Instanciando o LabelEncoder

le = LabelEncoder()



# Modificando o nosso dataframe, transformando a variável de classe em 0s e 1s

df_train_analysis['income'] = le.fit_transform(df_train_analysis['income'])
df_train_analysis['income']
def plot_feature_frequencies(data, feature, kind='line'):

    '''

    Plot frequencie for income <=50k and >50k for a specific feature

    '''

    

    less_50 = data.loc[data['income'] == 0, feature].value_counts().rename('<=50K')

    more_50 = data.loc[data['income'] == 1, feature].value_counts().rename('>50K')

    plot_data = pd.concat([less_50, more_50], axis=1)

    plot_data.plot(xlabel = feature, ylabel = 'Frequency', kind=kind)

    return plot_data
mask = np.triu(np.ones_like(df_train_analysis.corr(), dtype=np.bool))



plt.figure(figsize=(10,10))



sns.heatmap(df_train_analysis.corr(), square = True, annot=True, vmin=-1, vmax=1, cmap='mako')

plt.show()
sns.catplot(x="income", y="fnlwgt", kind="boxen", data=df_train_analysis);
sns.catplot(x="income", y="education.num", kind="boxen", data=df_train_analysis);
sns.catplot(x="income", y="hours.per.week", kind="boxen", data=df_train_analysis);
obj_data = df_train_analysis.drop(['age', 'fnlwgt', 'education.num','capital.gain', 'capital.loss', 'hours.per.week'], axis = 1)

obj_data = list(obj_data)

numdata = df_train_analysis

numdata = numdata[obj_data].apply(le.fit_transform)
mask = np.triu(np.ones_like(numdata.corr(), dtype=np.bool))



plt.figure(figsize=(10,10))



sns.heatmap(numdata.corr(), square = True, annot=True, vmin=-1, vmax=1,cmap='autumn')

plt.show()
plot_feature_frequencies(df_train_analysis, 'workclass', 'bar').head()
plot_feature_frequencies(df_train_analysis, 'race', 'bar').head()
plot_feature_frequencies(df_train_analysis, 'education', 'bar').head()
plot_feature_frequencies(df_train_analysis, 'occupation', 'bar').head()
plot_feature_frequencies(df_train_analysis, 'native.country', 'bar').head()
plot_feature_frequencies(df_train_analysis, 'marital.status', 'bar').head()
plot_feature_frequencies(df_train_analysis, 'sex', 'bar').head()
#=================================================

# Encoda os dados STRING em classificação ordinal

#=================================================

stringfeatures = ["marital.status","education","occupation", "relationship", "race", "sex", "workclass", "native.country"]



numdata = df_train

numdata[stringfeatures] = numdata[stringfeatures].apply(le.fit_transform) 

numdatatest = df_test

numdatatest[stringfeatures] = numdatatest[stringfeatures].apply(le.fit_transform) 

#============================

# Normalização

#============================

normalizar = ["age", "education.num", "marital.status","occupation","relationship",

           "race", "sex", "native.country", "workclass", "hours.per.week"]



for col in normalizar:

    numdata[col] = (numdata[col] - numdata[col].min()) / (numdata[col].max() - numdata[col].min())

    numdatatest[col] = (numdatatest[col] - numdatatest[col].min()) / (numdatatest[col].max() - numdatatest[col].min())
#==================================

# Seleção das Features Relevantes

#==================================

features = ["age", "education.num", "marital.status","relationship", "occupation",

           "sex", "capital.gain", "capital.loss", "hours.per.week"]

X_train = numdata.filter(items=features)

Y_train = numdata.income

X_test = numdatatest.filter(items=features)
display(X_train.head())

display(X_test.head())
Y_train.head()
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)
for k_value in range(5, 45, 5):

    clf = KNeighborsClassifier(k_value, p=1, weights='uniform', n_jobs=-1)

    score = np.mean(cross_val_score(clf, X_train, Y_train, cv=10))

    

    print(f'Para {k_value} vizinhos, a acurácia foi de {100*score:.5f}%')
max_k = 0

max_score = 0.

knn_clf = None



for k_value in range(20, 35):

    clf = KNeighborsClassifier(k_value, p=1, weights='uniform', n_jobs=-1)

    score = np.mean(cross_val_score(clf, X_train, Y_train, cv=10))

    

    if score > max_score:

        max_score = score

        max_k = k_value

        knn_clf = clf

    

print(f'O melhor número de vizinhos encontrado foi {max_k}, com acurácia de {max_score*100:.5f}%')
knn_clf.fit(X_train, Y_train)

Y_hat_test = knn_clf.predict(X_test)
result_data = pd.DataFrame({'income': Y_hat_test})

display(result_data.head())

result_data.to_csv('submission.csv', index = True, index_label = 'Id')