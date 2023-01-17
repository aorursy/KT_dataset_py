from IPython.display import display, Markdown

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import warnings



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn.metrics import confusion_matrix, precision_score, r2_score, recall_score, accuracy_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.exceptions import ConvergenceWarning

from sklearn.neural_network import MLPClassifier

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC



## Eliminamos warnings de sklearn

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore')
def plot(df, col, Var = 'Churn'):

    df_temp = df.groupby([col], as_index = True).mean()[Var].sort_values(ascending = True)

    

    Grafico = df_temp.plot(kind = 'barh'

                       , width = 0.5

                       , color = plt.get_cmap('Blues')(np.linspace(start = 0.15

                                           , stop = 0.85

                                           , num = len(df_temp)))

                       , stacked = True

                       , legend = False

                       , fontsize = 10)

    

    Grafico.set_xlim([0,0.55])

    Grafico.set_ylabel('')

    Grafico.grid(axis='x',alpha=0.25)

    Grafico.set_xticklabels(['{:3.0f} %'.format(x * 100) for x in Grafico.get_xticks()])

    [spine.set_visible(False) for spine in Grafico.spines.values()]

    Grafico.spines['left'].set_visible(True)



    plt.tick_params(left = False, bottom = False)

    plt.title(col)

    df_temp = None

    

def matriz(y, pred, Modelo = ''):

    mc = confusion_matrix(y, pred)

    Total = sum(np.transpose(mc))

    mcr = np.transpose(np.round(np.transpose(mc) / Total * 100,2))

    MC = pd.DataFrame(mc, columns=['No', 'Si'])



    MC['index'] = ['No', 'Si']

    MC['No'] *= -1



    Grafico = MC.set_index('index').sort_values(['index']

                                                , ascending=False).plot(kind = 'barh'

                                                                        , stacked = True

                                                                        , width = 0.75

                                                                        , color=[['#c9c9c9','#e60000'],['#00c800','#c9c9c9']]

                                                                        , legend = False

                                                                        , figsize = (3,1.2))

    

    plt.gca().set_xticks([])

    plt.gca().set_yticks([])

    

    plt.gca().grid(axis='x',alpha=0)

    plt.gca().grid(axis='y',alpha=0)

    plt.gca().set_xlabel('{:.2f} % - '.format(np.mean(y == pred) * 100) + Modelo

                         , fontsize = 12

                         , alpha = 0.6)

    plt.gca().set_ylabel('')

    

    [spine.set_visible(False) for spine in plt.gca().spines.values()]

    

    plt.show()

    plt.close()
def markdown(txt):

    display(Markdown(txt))
le = LabelEncoder() 



df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



df['TotalCharges'].replace(' ', np.nan, inplace=True)

df = df.dropna(subset = ['TotalCharges'])

df = df.sample(frac = 1

               , random_state = 12345)

   

df['Churn'] =  df[['Churn']].apply(le.fit_transform)



df.head(3)
df_Churn = df.groupby(by = ['Churn'])['customerID'].count().reset_index()

df_Churn['Ratio'] = df_Churn.customerID / df_Churn.customerID.sum()

df_Churn
columnsList = df.columns.tolist()

excluir = ['customerID','TotalCharges','tenure','MonthlyCharges','Churn','Churn_b']



for i in excluir:

    if i in columnsList:

        columnsList.remove(i)



fig = plt.figure(figsize =(5, 30)) 

fig.patch.set_facecolor('white')



plt.subplots_adjust(hspace = .7)



for i in range(1,len(columnsList) + 1):    

    plt.subplot(len(columnsList), 1, i)

    plot(df, columnsList[i -1])



plt.show()

plt.close()
Var_Continuas = ['TotalCharges','tenure','MonthlyCharges']



for i in Var_Continuas:

    df[[i]].apply(pd.to_numeric).hist(bins = 50

                                      , grid = False

                                      , color = '#86bf91'

                                      , zorder = 2

                                      , figsize = (6, 3)

                                      , rwidth = 0.8)

    

    [spine.set_visible(False) for spine in plt.gca().spines.values()]

    plt.gca().spines['bottom'].set_visible(True)

    plt.tick_params(left = False, bottom = False)

    plt.show()

    plt.close()
X = df[['tenure']]

y = df['Churn'] 



tree = DecisionTreeClassifier(criterion = 'entropy'

                                 , min_samples_split = 2000

                                 , max_depth = 2 )



tree = tree.fit(X, y)



plt.rcParams['figure.figsize'] = (12, 6)

Grafico = plot_tree(tree

                    , fontsize = 12

                    , max_depth = 7

                    , impurity = True

                    , rounded = True

                    , filled= True

                    , class_names = ['No', 'Yes'] )



#plt.savefig('tree.png', dpi = 210)

plt.show()

plt.close()
df['Tenure'] = df.tenure.map(lambda x: 'bajo' if x <= 17.5 else ('medio' if x <= 59.5 else 'alto'))

columnsList.append('Tenure')



df.groupby(['Tenure'])['Churn'].count().reset_index()
X = df[columnsList]

y = df['Churn']



XE, XT, ye, yt = train_test_split(X

                                  , y

                                  , test_size = 0.25

                                  , random_state = 0)
Params = {'Random Forest': RandomForestClassifier(n_estimators = 250

                                                   , random_state = 0)

           , 'SGD': SGDClassifier(max_iter = 5

                                  , tol = None)

           , 'Perceptron': MLPClassifier(random_state = 0)

           , 'KNN': KNeighborsClassifier()

           , 'Tree Classifier': DecisionTreeClassifier(criterion = 'entropy'

                                                       , random_state = 0)

           , 'Estra Tree Classifier': ExtraTreesClassifier(random_state = 0)

           , 'Gradient Boosting Classifier': GradientBoostingClassifier(random_state = 0)

           , 'SVC': SVC(kernel = 'linear'

                        , random_state = 0

                        , gamma = 'scale' )

           , 'Regresión logística': LogisticRegression(solver = 'lbfgs'

                                                       , multi_class = 'multinomial'

                                                       , class_weight = 'balanced'

                                                       , random_state = 0)}



Nums = Pipeline(

    steps = [

        ('imputer', SimpleImputer(strategy = 'median')), 

        ('scaler' , StandardScaler())])



Text = Pipeline(

    steps = [

        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')), 

        ('onehot' , OneHotEncoder(handle_unknown = 'ignore'))])



Pipe = ColumnTransformer(

    transformers = [

        ('num', Nums, X.select_dtypes(include = ['int64', 'float64']).columns), 

        ('cat', Text, X.select_dtypes(include = ['object']).columns)])
Result, classifiersList = [{},[]]



kf = KFold(n_splits = 10)

           

markdown(f'### Comparativa de los distintos modelos:') 

           

for i in Params:

    Score, lista = [[],[]]

    clf = Pipeline(steps=[('Prepo', Pipe), 

                          ('Modelo', Params[i])])

    clf.fit(XE, ye)

    Score.append(round(np.mean(yt == clf.predict(XT)) * 100, 2))

    Score.append(cross_val_score(clf, XE, ye, cv=kf, scoring = 'accuracy').mean())

    Score.append(round(precision_score(yt, clf.predict(XT), labels = [0, 1], pos_label = 1) * 100, 2))

    Score.append(r2_score(yt, clf.predict(XT)))

    Score.append(recall_score(yt, clf.predict(XT), average = None).round(2))

    Result[i] = Score



    lista.append(i)

    lista.append(clf)

    classifiersList.append(tuple(lista))



Result = (pd.DataFrame(Result, index=['Precisión (accuracy)'

                                      , 'Cross Val' 

                                      , 'Score (True)'

                                      , 'R Cuadrado'

                                      , 'Recall'])

          .transpose()

          .sort_values(by = 'Precisión (accuracy)'

                       , ascending = False)

          .reset_index()

          .rename(columns = {'index':'Modelo'})

         )



markdown(f'* Matríz de confusión:')



Best = Pipeline(steps=[('Prepo', Pipe),

                      ('Modelo', Params[Result['Modelo'][0]])])



Best.fit(XE, ye)



Grafico = matriz(yt, Best.predict(XT), Result['Modelo'][0])        

markdown(f'* Precisión modelos sobre base test:')

print(accuracy_score(yt, Best.predict(XT)))

Result
from sklearn.ensemble import VotingClassifier



clf = VotingClassifier(classifiersList, n_jobs=-1)

clf.fit(XE, ye)

predictions = clf.predict(XT)

print(accuracy_score(yt, predictions))
recall_score(yt, clf.predict(XT), average = None).round(2)