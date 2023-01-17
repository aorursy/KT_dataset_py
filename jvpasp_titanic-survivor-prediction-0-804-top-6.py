import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score, KFold

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC



import warnings
PATH_DATA = '../input/titanic/'

warnings.filterwarnings('ignore')
df = (pd.read_csv(f'{PATH_DATA}train.csv'

                  , sep = ','

                  , header = 0)

      .sample(frac = 1, random_state = 0))



df.head()
def Actualizar_df(df):

    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand = False)

    

    a = df.Title.value_counts().reset_index().transpose().to_numpy()

    df.Title = df.Title.map(lambda x: x if x in a[0][a[1] > 10].tolist() else 'Others')

    

    df['Familia'] = df['SibSp'] + df['Parch'] + 1

    df.Familia = df.Familia.map(lambda x: 'Singel' if x == 1 else ('Parejas' if x == 2 else ('Famila Numerosa' if x >= 5 else 'Hasta 2 hijos')))



    medias = df.groupby(['Title'], as_index = True).mean()['Age'].sort_values(ascending = True).to_dict()

    

    df.fillna({'Age': df.Title.map(medias)

               , 'Embarked': df.Embarked.value_counts().index.tolist()[0]}, inplace = True)

    

    return df



df = Actualizar_df(df)



df.head()
def plot(df, col):

    df_temp = df.groupby([col], as_index = True).mean()['Survived'].sort_values(ascending = True)

    

    Grafico = df_temp.plot(kind = 'barh'

                       , width = 0.5

                       , color = plt.get_cmap('Blues')(np.linspace(start = 0.15

                                           , stop = 0.85

                                           , num = len(df_temp)))

                       , stacked = True

                       , legend = False

                       , fontsize = 10)

    

    Grafico.set_xlim([0, 0.85])

    Grafico.set_ylabel('')

    Grafico.grid(axis = 'x',alpha = 0.25)

    Grafico.set_xticklabels(['{:3.0f} %'.format(x * 100) for x in Grafico.get_xticks()])

    [spine.set_visible(False) for spine in Grafico.spines.values()]

    Grafico.spines['left'].set_visible(True)



    plt.tick_params(left = False, bottom = False)

    plt.title(col)

    df_temp = None 

    

    

df['Edad'] = df.Age.map(lambda x: 'Niños' if x <= 15 else ('Adultos' if x <= 65 else 'Ancianos'))

df['Precio'] = df.Fare.map(lambda x: '< 40 $' if x <= 40 else ('< 80 $' if x <= 80 else '> 100 $'))



columnsList = ['Pclass', 'Sex', 'Familia', 'Embarked','Edad', 'Precio', 'Title']



Grafico = plt.figure(figsize =(5, 16)) 

Grafico.patch.set_facecolor('white')



plt.subplots_adjust(hspace = .7)



for i in range(1,len(columnsList) + 1):    

    plt.subplot(len(columnsList), 1, i)

    plot(df, columnsList[i -1])



plt.show()

plt.close()
df = df.drop(['PassengerId'

              , 'Name'

              , 'Cabin'

              , 'SibSp'

              , 'Parch'

              , 'Edad'

              , 'Precio'

              , 'Familia']

             , axis = 'columns')



df.head()
X = df.iloc[:, 1:]

y = df['Survived']



Text = Pipeline(

    steps = [

        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')), 

        ('onehot' , OneHotEncoder(handle_unknown = 'ignore'))])



Pipe = ColumnTransformer(

    transformers = [

        ('cat', Text, X.select_dtypes(include = ['object']).columns)])



Classifier = SVC(kernel = 'linear'

                 , gamma = 'scale'

                 , random_state = 0)



Modelo = Pipeline(steps = [('Prepo', Pipe)

                           , ('Clf', Classifier)])

Modelo.fit(X, y)
kf = KFold(n_splits = 10)



cross_val_score(Modelo, X, y, cv = kf, scoring = 'accuracy').mean()
df = (pd.read_csv(f'{PATH_DATA}test.csv'

                  , sep = ','

                  , header = 0))



df = Actualizar_df(df)
df['Survived'] = Modelo.predict(df[X.columns.tolist()])

df[['PassengerId','Survived']].to_csv('submission.csv', index=False)    