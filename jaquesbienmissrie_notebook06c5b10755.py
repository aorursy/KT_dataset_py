import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn import preprocessing as prep

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import numpy as np
nomes = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

            "Hours per week", "Country", "Target"]



train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv", names = nomes,

        sep= r'\s*,\s*',

        engine= 'python',

        na_values= "?")



train_data = train_data.iloc[1:]

train_data.shape
train_data.head()
nomes = ["Id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

            "Hours per week", "Country", "Target"]

test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv", names = nomes,

        sep= r'\s*,\s*',

        engine= 'python',

        na_values= "?")



test_data.shape
test_data.head()
plt.title('Age')

plt.ylabel('Adults')

plt.xlabel('Age')

plot_data = train_data["Age"].value_counts()

plot_data.plot(kind = "line", sharex=False)

#plt.legend()

plt.show()
plot_data = train_data["Workclass"].value_counts().copy()

plot_data.plot(kind = "bar", fontsize=12)

plt.title('Workclass')

plt.ylabel('Adults')

plt.show()
plot_data = train_data["Education"].value_counts().copy()

plot_data.plot(kind = "bar", fontsize=10)

plt.title('Education')

plt.ylabel('Adults')

plt.show()
plot_data = train_data["Martial Status"].value_counts().copy()

plot_data.plot(kind = "bar")

plt.title('Martial Status')

plt.ylabel('Adults')

plt.show()

plot_data = train_data["Occupation"].value_counts().copy()

plot_data.plot(kind = "bar", fontsize=10)



plt.title('Occupation')

plt.ylabel('Adults')

plt.show()
plot_data = train_data["Relationship"].value_counts().copy()

plot_data.plot(kind = "pie", fontsize=8, ylabel="")

plt.title('Relationship')

plt.show()
plot_data = train_data["Race"].value_counts().copy()

plot_data.plot(kind = "pie", legend=True, fontsize=0, ylabel="")



plt.title('Race')

plt.show()
plot_data = train_data["Sex"].value_counts().copy()

plot_data.plot(kind = "pie", ylabel="", autopct='%1.0f%%', radius=1)



plt.title('Sex')

plt.show()
plot_data = train_data["Country"].value_counts().copy()



us_data = plot_data[0]

mex_data = plot_data[1]

others_data = 0

i = 2

while i < len(plot_data): 

    others_data += plot_data[i]

    i += 1

    

x = ['United States', 'Mexico','Others']

y = [us_data, mex_data, others_data]



plt.bar(x, y)

plt.title('Countries (US | Mex | Others)')

plt.show()

    

plot_data.iloc[2:].plot(kind = "bar", fontsize=8)



plt.title('Countries (Others)')

plt.ylabel('Adults')

plt.show()
def bar_plot(data, hue, by="Target", size=20, normalize_by_index = True):

    index = data[by].unique()

    

    columns = data[hue].unique()

  

    data_to_plot = pd.DataFrame({'index': index})

    

    for column in columns:

        temp = []

        for unique in index:

            filtered_data = data[data[by] == unique]

            filtered_data = filtered_data[filtered_data[hue] == column]

            

            temp.append(filtered_data.shape[0])

        data_to_plot = pd.concat([data_to_plot, pd.DataFrame({column: temp})], axis = 1)

        

    data_to_plot = data_to_plot.set_index('index', drop = True)

    

    if normalize_by_index:

        for row in index:

            data_to_plot.loc[row] = data_to_plot.loc[row].values/data_to_plot.loc[row].values.sum()

    

    ax = data_to_plot.plot.bar(rot=0, figsize = (14,7), alpha = 0.9, cmap = 'Wistia', xlabel="", fontsize=size)
bar_plot(train_data, 'Education')
bar_plot(train_data, 'Race')
bar_plot(train_data, 'Sex')
train_data.isnull().sum()
moda_workclass = train_data['Workclass'].describe().top

train_data['Workclass'] = train_data['Workclass'].fillna(moda_workclass)



moda_occupation = train_data['Occupation'].describe().top

train_data['Occupation'] = train_data['Occupation'].fillna(moda_occupation)



moda_country = train_data['Country'].describe().top

train_data['Country'] = train_data['Country'].fillna(moda_country)
moda_workclass = test_data['Workclass'].describe().top

test_data['Workclass'] = test_data['Workclass'].fillna(moda_workclass)



moda_occupation = test_data['Occupation'].describe().top

test_data['Occupation'] = test_data['Occupation'].fillna(moda_occupation)



moda_country = test_data['Country'].describe().top

test_data['Country'] = test_data['Country'].fillna(moda_country)
train_data = train_data.apply(prep.LabelEncoder().fit_transform)

test_data = test_data.apply(prep.LabelEncoder().fit_transform)



train_data.head()

features = ["Age", "Workclass", "Education", "Occupation", "Race", "Capital Gain", "Capital Loss", "Hours per week"]



x_train = train_data[features]

y_train = train_data.Target



x_test = test_data[features]

y_test = test_data.Target
knn = KNeighborsClassifier(n_neighbors = 5)



scores = cross_val_score(knn, x_train, y_train, cv=10)

scores.mean()
inf = 1

sup = 35



scores_media = []

aux = 0

k_max = 0



i = 0

for k in range(inf, sup):

    knn = KNeighborsClassifier(n_neighbors = k)

    scores = cross_val_score(knn, x_train, y_train, cv=10)

    scores_media.append(scores.mean())



    if scores_media[i] > aux:

        k_max = k

        aux = scores_media[i]



    i = i + 1



x = np.arange(1, sup)



knn = KNeighborsClassifier(n_neighbors = k_max)



scores = cross_val_score(knn, x_train, y_train, cv=10)

y = scores.mean()



plt.figure(figsize=(10, 5))

plt.plot(x, scores_media, '--', color = 'red', linewidth = 2)

plt.plot(k_max, y, 'o')



plt.xlabel('K')

plt.ylabel('Acurácia')

plt.title('Acurácia da predição  X  Valor de K')



print(k_max)
knn = KNeighborsClassifier(n_neighbors = 20)



scores = cross_val_score(knn, x_train, y_train, cv=10)

score = scores.mean()



print('Acurácia para k = {0} : {1:2.2f}%'.format(k_max, 100 * score))
knn.fit(x_train, y_train)

predict_knn = knn.predict(x_test)
valores_originais = {0: '<=50K', 1: '>50K'}

predicao = np.array([valores_originais[i] for i in predict_knn], dtype=object)



submission = pd.DataFrame()



print(len(test_data.index), len(predicao))



submission[0] = test_data.index

submission[1] = predicao

submission.columns = ['Id', 'Income']



submission.to_csv('final_results.csv',index = False)