import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv("../input/adult-census-income/adult.csv")

data.head()
data.info()
data.hist(bins=20, figsize=(15,12), edgecolor='black',grid=True)

plt.show()
data.loc[data['native.country']=="?"]
data.isin(['?']).sum()
data = data.replace('?', np.NAN)
for col in ['workclass', 'occupation', 'native.country']:

    data[col].fillna(data[col].mode()[0], inplace=True)
data.isnull().values.any()
data.isnull().sum()
data['income'].value_counts()
import seaborn as sns

sns.countplot(x='income',data=data, palette="cool")

plt.title('Income Values')
sns.boxplot(x='income', y='age', data=data, palette='hot')

plt.title('Age vs Income')
sns.boxplot(x='income',y='hours.per.week', data=data, palette='seismic')

plt.title('hours vs Income')
sns.countplot(data['sex'],hue=data['income'], palette='coolwarm')

plt.title('Sex vs Income')
sns.countplot(data['occupation'],hue=data['income'], palette='winter')

plt.xticks(rotation=90)

plt.title('Occupation vs Income')
data['income']=data['income'].map({'<=50K':0, '>50K': 1})
sns.FacetGrid(data, col='income').map(sns.distplot, "age")
sns.barplot(x='education.num', y='income', data=data)

plt.title('Education vs Income')
sns.barplot(x="workclass",y="income",data=data)

plt.xticks(rotation=90)

plt.title('Workclass vs Income')
sns.barplot(x="education",y="income",data=data)

plt.xticks(rotation=90)

plt.title('Education vs Income')

sns.barplot(x='marital.status',y='income', data=data)

plt.xticks(rotation=90)

plt.title('Education vs Income')
data['relationship'].unique()
sns.barplot(x='relationship',y='income', data=data)

plt.xticks(rotation=90)

plt.title('Relationship vs Income')
sns.barplot(x='race', y='income', data=data)

plt.xticks(rotation=90)

plt.title('Race vs Income')
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in data.columns:

    if data[col].dtypes == 'object':

        data[col] = label_encoder.fit_transform(data[col])
data.dtypes
data.head()
corr = data.corr()

plt.figure(figsize=(20,12))

sns.heatmap(corr, annot=True, cmap='coolwarm')
corr['income'].sort_values(ascending = False)
previsores = data.iloc[:,0:14]

classe = data.iloc[:,14]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),

 [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')



previsores = column_tranformer.fit_transform(previsores).toarray()

previsores
from sklearn.model_selection import train_test_split

previsores_train, previsores_teste, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.30, random_state=0)
#append results

lista=[]
# best parameters for models:

# Labelencoder (79.23%) 

# Labelencoder and Scaler (82.17%)

# Labelencoder, Scaler and OneHotEncoder (84.58%)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix



classificador = LogisticRegression(random_state=1, solver='lbfgs')

classificador.fit(previsores_train, classe_train)



previsoes = classificador.predict(previsores_teste)



precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test,previsoes)



print('Logistic Regression: ', precisao * 100)

lista.append(precisao)
# best parameters for models:

# Labelencoder (79.50%) 

# Labelencoder and Scaler (80.38%)

# Labelencoder, Scaler and OneHotEncoder (54.50%)



from sklearn.naive_bayes import GaussianNB



classificador = GaussianNB()

classificador.fit(previsores_train, classe_train)



previsoes = classificador.predict(previsores_teste)



precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test,previsoes)



print('Naive Bayes: ', precisao * 100)

lista.append(precisao)

# best parameters for models:

# Labelencoder (80.86%) 

# Labelencoder and Scaler (80.84%)

# Labelencoder, Scaler and OneHotEncoder (81.28%)



from sklearn.tree import DecisionTreeClassifier



classificador = DecisionTreeClassifier()

classificador.fit(previsores_train, classe_train)



previsoes = classificador.predict(previsores_teste)



precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test,previsoes)



print('Tree Classifier: ', precisao * 100)

lista.append(precisao)

# best parameters for models:

# Labelencoder (85.55%) 

# Labelencoder and Scaler (85.51%)

# Labelencoder, Scaler and OneHotEncoder (85.33%)



from sklearn.ensemble import RandomForestClassifier



classificador = RandomForestClassifier(n_estimators=350, criterion='entropy', random_state=0)

classificador.fit(previsores_train, classe_train)



previsoes = classificador.predict(previsores_teste)



precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test,previsoes)



print('Random Forest: ', precisao * 100)

lista.append(precisao)

# best parameters for models:

# Labelencoder (86.16%) 

# Labelencoder and Scaler (86.16%)

# Labelencoder, Scaler and OneHotEncoder (86.22%)



from sklearn.ensemble import GradientBoostingClassifier



classificador = GradientBoostingClassifier()

classificador.fit(previsores_train, classe_train)



previsoes = classificador.predict(previsores_teste)



precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test,previsoes)



print('GBC: ', precisao * 100)

lista.append(precisao)
# best parameters for models:

# Labelencoder (75.82%) 

# Labelencoder and Scaler (84.46%)

# Labelencoder, Scaler and OneHotEncoder (84.68%)



from sklearn.svm import SVC



classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0, gamma='auto')

classificador.fit(previsores_train, classe_train)



previsoes = classificador.predict(previsores_teste)



precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test,previsoes)



print('SVM: ', precisao * 100)

lista.append(precisao)
# best parameters for models:

# Labelencoder (80.53%) 

# Labelencoder and Scaler (84.78%)

# Labelencoder, Scaler and OneHotEncoder (83.07%)



from sklearn.neural_network import MLPClassifier



classificador = MLPClassifier(verbose = False,

                              max_iter=1000,

                              tol = 0.0000010,

                              solver = 'adam',

                              hidden_layer_sizes=(100),

                              activation='relu')

classificador.fit(previsores_train, classe_train)



previsoes = classificador.predict(previsores_teste)



precisao = accuracy_score(classe_test, previsoes)

matriz = confusion_matrix(classe_test,previsoes)



print('MLP Classifier: ', precisao * 100)

lista.append(precisao)
fig, ax = plt.subplots()

y_grafico =['Logistic Regression',

          'Naive Bayes',

          'Tree Classifier',

           'Random Forest',

            'GBC',

            'SVM',

            'MLP Classifier'

           ]



x_grafico = lista 

sns.barplot(x=x_grafico,y=y_grafico)

for y,x in enumerate(x_grafico):

    ax.annotate("{:.2f}%".format(x * 100), xy=(x,y))

    ax.set_xlim(0, 1)

plt.xlabel('Accuracy')

plt.title('List Best Models')