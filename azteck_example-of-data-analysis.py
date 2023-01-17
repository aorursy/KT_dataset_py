import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None) # позволяет отображать все столбцы

pd.options.display.float_format = '{:,.2f}'.format # задает формат отображения вещественных значений в Pandas, где 2 знака после точки и в качестве разделителя разрядов запятая

import warnings

warnings.filterwarnings('ignore') #отключает предупреждения от пакетов...

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from tqdm import tqdm # progress bar

import sys

print(sys.version)

import os # пакет для работы с файлами

print('Working directory: '+os.getcwd()) # проверим текущую рабочую директорию

path='/kaggle/input'
def df_snapshot (df, vert=False):

    print("Raws:", df.shape[0])

    print("Columns:", df.shape[1])

    if vert == False:

        return df.head(8)

    elif vert == True: # Транспонированная форма представления набора данных

        return df.head(8).T



DATA=pd.read_excel(path+r'/compo (Autosaved).xlsx', index_col=0, dtype={})

df_snapshot (DATA)
def my_describe (df):

    """

    Разделяет описательные статистики набора данных на две последовательные таблицы:

    сначала категориальные признаки, зачет количественные.

    """

    if "object" in list(df.dtypes):

        display(df.describe(include=['object']))

    if 'datetime64' in list(df.dtypes):

        display(df.describe(include=['datetime64']))

    display(df.describe(percentiles=[0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]))

    

my_describe (DATA)
ftrs=dict()

ftrs['L']=["Mill tonnes overall", "Average mill tph", "Mill tonnes overall", "Average mill tph", "Mill tonnes overall", "Mill tonnes overall", "Li2O %", "Fe", "Fe", "Li %", "Ret +  600 µm", "P 80  µm"]

ftrs['R']=["P 80  µm", "P 80  µm", "Ret -   25 µm", "Ret -   25 µm", "Li2O %", "Fe", "P 80  µm", "P 80  µm", "Ret -   25 µm", "Ret -   25 µm", "Ret -   25 µm", "Ret -   25 µm"]

ftrs_table = pd.DataFrame(ftrs)

ftrs_table
clmns = list(set(ftrs['L']+ftrs['R']))

DATA[clmns].dtypes
clmns = list(set(ftrs['L']+ftrs['R'])) #признаки для корреляционного анализа

DATA[clmns].isnull().sum()
for clmn in DATA[clmns].columns:

    DATA[clmn].replace(np.nan, DATA[clmn].mean(),inplace=True)

DATA[clmns].isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



clmns = list(set(ftrs['L']+ftrs['R'])) #признаки для корреляционного анализа



corr = DATA[clmns].corr()

fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, xticklabels=corr.columns,

        yticklabels=corr.columns,cmap='RdBu',

        annot=True, vmin=0, vmax=1, linewidths=.5, ax=ax)
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

lm = LinearRegression()



def feature_rel_1 (df, feature1, feature2):

    '''

    Классический scatter plot наблюдений двух признаков

    '''

    from scipy import stats

    import warnings

    warnings.filterwarnings('ignore') #отключает предупреждения от пакетов...



    display(df[[feature1, feature2]].corr())



    temp=df[[feature1, feature2]].dropna()

    print('Number of observations:', temp.shape[0])

    if temp.shape[0]<100:

        print('Usually for significant correlation we need 100+ - 200+ observations')

    print('______________________________________________________________________________')

    

    pearson_coef, p_value = stats.pearsonr(temp[feature1], temp[feature2])

    print("The Pearson Correlation Coefficient is", round(pearson_coef, 2), " with a P-value of P =", round(p_value, 2))

    print('______________________________________________________________________________')

    if p_value<0.001:

        print('there is STRONG evidence that the correlation is significant')

    elif p_value<0.05:

        print('there is MODERATED evidence that the correlation is significant')

    elif p_value<0.1:

        print('there is WEAK evidence that the correlation is significant')

    elif p_value>=0.1:

        print('there is NO evidence that the correlation is significant')



    sns.regplot(x=feature2, y=feature1, data=df)

    plt.ylim(0,)



def feature_rel_2 (df, feature1, feature2, coef=False):

    '''

    График распределения остатков по парной регрессии между тестируемыми признаками для подтверждения/опровержения линейности связи между признаками

    '''

    sns.residplot(x=feature2, y=feature1, data=df)

    plt.show()

    

    temp=df[[feature1, feature2]].dropna()

    lm.fit(temp[feature2].values.reshape(-1, 1), temp[feature1].values.reshape(-1, 1))

    # Find the R^2

    print('The R-square is: ', lm.score(temp[feature2].values.reshape(-1, 1), temp[feature1].values.reshape(-1, 1)))

    print('We can say that ~ ', round(lm.score(temp[feature2].values.reshape(-1, 1), temp[feature1].values.reshape(-1, 1))*100,2),'% of the variation of the ', feature1, 'is explained by ', feature2)

    if coef == True:

        print(lm.intercept_, lm.coef_)
feature_rel_1 (DATA, "Mill tonnes overall", "P 80  µm")
feature_rel_2 (DATA, "Mill tonnes overall", "P 80  µm")
feature_rel_1 (DATA, "Average mill tph", "P 80  µm")
feature_rel_2 (DATA, "Average mill tph", "P 80  µm")
feature_rel_1 (DATA, "Mill tonnes overall", "Ret -   25 µm")
feature_rel_2 (DATA, "Mill tonnes overall", "Ret -   25 µm")
feature_rel_1 (DATA, "Average mill tph", "Ret -   25 µm")
feature_rel_2 (DATA, "Average mill tph", "Ret -   25 µm")
feature_rel_1 (DATA, "Mill tonnes overall", "Li2O %")
feature_rel_2 (DATA, "Mill tonnes overall", "Li2O %")
feature_rel_1 (DATA, "Mill tonnes overall", "Fe")
feature_rel_2 (DATA, "Mill tonnes overall", "Fe")
feature_rel_1 (DATA, "Li2O %", "P 80  µm")
feature_rel_2 (DATA, "Li2O %", "P 80  µm")
feature_rel_1 (DATA, "Fe", "P 80  µm")
feature_rel_2 (DATA, "Fe", "P 80  µm")
feature_rel_1 (DATA, "Fe", "Ret -   25 µm")
feature_rel_2 (DATA, "Fe", "Ret -   25 µm", coef=True)
feature_rel_1 (DATA[DATA["Li %"]<1.5], "Li %", "Ret -   25 µm")
feature_rel_2 (DATA[DATA["Li %"]<1.5], "Li %", "Ret -   25 µm")
feature_rel_1 (DATA, "Ret +  600 µm", "Ret -   25 µm")
feature_rel_2 (DATA, "Ret +  600 µm", "Ret -   25 µm")
feature_rel_1 (DATA, "P 80  µm", "Ret -   25 µm")
feature_rel_2 (DATA, "P 80  µm", "Ret -   25 µm", coef=True)