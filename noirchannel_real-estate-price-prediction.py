import numpy as np

import pandas as pd

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt



matplotlib.rcParams.update({'font.size': 14})

plt.style.use('fivethirtyeight')



%matplotlib inline

%config Inlinebackend.figure_format = 'svg'
from sklearn.metrics import r2_score as r2, mean_squared_error as mse

from sklearn.model_selection import train_test_split, KFold, GridSearchCV



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
class Prepeare:

    '''Подготовка данных к анализу'''

    

    def __init__(self, p):

        self.path = p

        

    def df(self):

        '''чтение данных'''

        

        df = pd.read_csv(self.path)

        return df

    

    def index(self):

        '''индексирование данных'''

        

        df = self.df()

        df.set_index('Id', inplace=True)

        return df

    

    def split(self):

        '''разделение данных'''

        

        df = self.index()

        if 'Price' in df:

            X = df.drop('Price', axis=1)

            y = df['Price']

            return X, y

        else:

            X = df

            return X
class Treatment:

    '''Обработка данных'''

    

    def __init__(self, df):

        self.df = df

    

    def FMedian(self):

        '''Вычисление медианы'''

        

        df = self.df

        median = X['Healthcare_1'].median()

        df['Healthcare_1'] = df['Healthcare_1'].fillna(median)

        return df

        

    def FCleaning(self):

        '''Очистка данных'''

        

        df = self.FMedian()

        df[df['KitchenSquare'] >= 250] = X['KitchenSquare'].median()

        return df

    

    def del_columns(self):

        '''Удаление столбцов'''

        

        df = self.FCleaning()

        df = df.drop(['Ecology_2', 'Ecology_3', 'Shops_2', 'LifeSquare'], axis=1)

        return df
def scattering(X, Y):

    '''показывает разброс показателей'''

    

    sns.scatterplot(X, Y)

    plt.title(f'{X.name} and {Y.name}')

    plt.ylabel(f'{Y.name}')

    plt.xlabel(f'{X.name}')

    plt.show()

    print(

        f'"{Y.name}"\n' 

        f'median\t= {Y.median()}\n'

        f'mode\t= {Y.mode()[0]}\n'

        f'mean\t= {Y.mean()}'

    )
def evaluate_preds(true_values, pred_values, save=False):

    """Оценка качества модели и график preds vs true"""

    

    print("R2:\t" + str(round(r2(true_values, pred_values), 3)) + "\n" +

          "RMSE:\t" + str(round(np.sqrt(mse(true_values, pred_values)), 3)) + "\n" +

          "MSE:\t" + str(round(mse(true_values, pred_values), 3))

         )

    

    plt.figure(figsize=(8,8))

    

    sns.scatterplot(x=pred_values, y=true_values)

    plt.plot([0, 500000], [0, 500000], linestyle='--', color='black')  # диагональ, где true_values = pred_values

    

    plt.xlabel('Predicted values')

    plt.ylabel('True values')

    plt.title('True vs Predicted values')

    

    if save == True:

        plt.savefig(REPORTS_FILE_PATH + 'report.png')

    plt.show()
PATH_TO_TRAIN = '../input/train.csv'

PATH_TO_TEST = '../input/test.csv'



X = Prepeare(PATH_TO_TRAIN).split()[0]

y = Prepeare(PATH_TO_TRAIN).split()[1]

X_final = Prepeare(PATH_TO_TEST).split()
scattering(y, X['KitchenSquare']) # просмотр выбрасов

X.info()
X = Treatment(X).del_columns()

scattering(y, X['KitchenSquare']) # просмотр выбрасов

X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)



model = LGBMRegressor(

    criterion='mse',

    max_depth=6,

    n_estimators=200

)

model.fit(X_train, y_train)



y_train_pred = model.predict(X_train)

evaluate_preds(y_train, y_train_pred)
y_test_pred = model.predict(X_test)

evaluate_preds(y_test, y_test_pred)
feature_importances = pd.DataFrame(zip(

    X_train.columns,

    model.feature_importances_

), columns=['feature_name', 'importance'])



feature_importances.sort_values(by='importance', ascending=False, inplace=True)

feature_importances
X_final = Treatment(X_final).del_columns()



y_pred_final = model.predict(X_final)



preds_final = pd.DataFrame()

preds_final['Id'] = X_final.index

preds_final['Price'] = y_pred_final

preds_final.to_csv('predictions.csv', index=False)



preds_final.head()