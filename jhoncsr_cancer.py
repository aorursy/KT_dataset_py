import pandas



data_df = pandas.read_csv("../input/data.csv")
data_df.head()
data_df = data_df.drop('Unnamed: 32', axis = 1)

data_df = data_df.drop('id', axis = 1)
data_df.head()
len(data_df)
data_df.describe()
data_df.describe()

plt.hist(data_df['diagnosis'])

plt.title('Diagnosis (M=1 , B=0)')

plt.show()
from sklearn.preprocessing import LabelEncoder 



labelencoder = LabelEncoder()

data_df['diagnosis'] = labelencoder.fit_transform(data_df['diagnosis'])

data_df.head()
import seaborn as sms

import matplotlib.pyplot as plt



g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'radius_mean', bins=10)



#nota-se quanto maior o raio medio, maior a possibilidade de ser maligino.
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'texture_mean', bins=10)

#nota-se quando menor a textura, maior chance de ser benigno, mas a diferença n tras muita luz pro problema
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'perimeter_mean', bins=10)

#quanto maior o perimetro, maior possibilidade de ser maligno, perimetros abaixo de 75 raramente são malignos
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'area_mean', bins=10)

#regra se repete na area, maior area, maior letalidade
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'smoothness_mean', bins=10)

#nada pode ser analisado.
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'compactness_mean', bins=10)

#quanto mais concentrado o cancer é, maior sua letalidade, logo maior sua probabilidade de ser maligno
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'concavity_mean', bins=10)

#na concavidade, podemos ver que entre 0.1 a 0.2 maior chance de ser maligno
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'concave points_mean', bins=10)

#nos pontos concavidade, podemos ver que entre 0.05 a 0.10 maior chance de ser maligno
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'symmetry_mean', bins=20)

#media simetrica, parece que a media simetrtica dos benignos beiram sempre abaixo de 0.20, mas malignos tbm apresentam

#grande amostra nessa media
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_mean', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'radius_se', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'texture_se', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'perimeter_se', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'area_se', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'smoothness_se', bins=10)

#nada a ser analisado.

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'compactness_se', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'concavity_se', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'concave points_se', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'symmetry_se', bins=10)

#classe se aparentemente não ajuda na analise
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_se', bins=10)

#grafico não me aprensenta nada
g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'radius_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'texture_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'area_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'smoothness_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'compactness_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'concavity_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'concave points_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'symmetry_worst', bins=10)

g = sms.FacetGrid(data_df, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_worst', bins=10)



#ambos não auxiliam em nada a minha analise
data_df.drop(

['smoothness_se',

'fractal_dimension_se',

'symmetry_se',

'texture_se',

'compactness_se',

'concave points_se',

'fractal_dimension_mean',

'symmetry_mean',

'fractal_dimension_worst',

'symmetry_worst'],axis = 1,inplace=True)
data_df.head()
classe = data_df['diagnosis']

atributos = data_df.drop('diagnosis', axis=1)
classe.head()
atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, classe_train, classe_test = train_test_split(atributos, classe, test_size = 0.25)

atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=3, random_state=0)

model = tree.fit(atributos_train, classe_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

classe_pred
acc = accuracy_score(classe_test, classe_pred)

print("Probabilidade é: ", format(acc))