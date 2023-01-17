# Подключаем библиотеки

import pandas as pd
import numpy as np

from scipy.stats import shapiro, ttest_rel, chi2_contingency

from sklearn.metrics import f1_score, r2_score, mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
%config Inlinebackend.figure_format = 'svg'

import warnings
warnings.filterwarnings('ignore')
def sampling(df: pd.DataFrame, target: str, columns: list, final=False) -> tuple:
    """Функция для разбиения данных"""
    
    # сплитим на X и y
    df = df[columns]
    
    Xy_df = df[df[target].notna()]
    
    X = Xy_df.drop(target, axis=1)
    y = Xy_df[target]

    # сплитим на train и test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if final:
        final = df[df[target].isna()]
        final = final.drop(target, axis=1)
        return X_train, X_test, y_train, y_test, final
    elif not final:
        return X_train, X_test, y_train, y_test
def pred_feature(df, target, columns, max_depth, learning_rate, n_estimators, random_state=42):
    """Функция для обучения LGBM"""
    
    # сплитим на train и test
    X_train, X_test, y_train, y_test, final = sampling(df=df, target=target, columns=columns, final=True)
        
    # обучаем модель
    xgb_model = LGBMRegressor(max_depth=max_depth, learning_rate=learning_rate, 
                              n_estimators=n_estimators, random_state=random_state)
    xgb_model.fit(X_train, y_train)

    # предсказываем зачения
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)

    # смотрим результат
    print(target)
    evaluation_result(y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred)
        
    df_pred = pd.DataFrame()
    df_pred[target] = xgb_model.predict(final)
    df_pred = df_pred.set_index(final.index)
        
    return df_pred, xgb_model
def balancing(df, target_name):
    """функция для балансировки значений"""

    target_counts = df[target_name].value_counts()

    major_class_name = target_counts.argmax()
    minor_class_name = target_counts.argmin()

    disbalance_coeff = int(target_counts[major_class_name] / target_counts[minor_class_name]) - 1

    for i in range(disbalance_coeff):
        sample = df[df[target_name] == minor_class_name].sample(target_counts[minor_class_name])
        df = df.append(sample, ignore_index=True)

    return df.sample(frac=1)
# Формируем пути до данных

PATH_TO_TRAIN = '../input/course-project/course_project_train.csv'
PATH_TO_TEST = '../input/course-project/course_project_test.csv'

# PATH_TO_TRAIN = 'course_project_train.csv'
# PATH_TO_TEST = 'course_project_test.csv'
def missing_data(df: pd.DataFrame) -> pd.Series:
    """get dataframe and calculate emissions in its data in percentage"""
    
    featur_counts = df.count()
    object_counts = df.shape[0]
    
    calc_procent = (1 - featur_counts/object_counts) * 100
    
    feature_procent = round(calc_procent, 2)
            
    result = {i: f'{v}%' for i, v in feature_procent.items() if v != 0}
    
    return pd.Series(result)
def correlation(
    df: pd.DataFrame, 
    target: str
) -> (print, plt.plot):
    """calculates correlation whith target and plots graph"""
    
    corr_table = df.corr()
    
    result = corr_table.drop([target], axis=0) if target != None else corr_table
    sort_values = result.sort_values(target, ascending=False)*100 if target != None else result
    
    coef = round(sort_values[target], 2)
    
    print(coef)
    
    plt.figure(figsize=(12,4))

    plt.barh(coef.index, coef.values)

    plt.title(f'Correlation with {target}', fontsize=16)
    plt.xlabel('percent', fontsize=14)
    
    plt.grid()
def emission(
    series: pd.Series, 
    val: int = None
) -> print:
    """
    get series and shop emissions in it, 
    if val then can calculate emissions and is percent
    """
    
    print(
        f'min = {series.min()}\n'
        f'max = {series.max()}\n'
        f'mean = {series.mean()}'
    )
    if val != None:
        em = series > val

        print(
            f'Число выбрасов = {em.sum()}\n'
            f'Процент выбрасов = {round(em.sum() / series.shape[0] * 100, 2)}%'
        )
        
    plt.scatter(series.index, series.values)
    plt.plot()
def chi2_test(df, feature, values, target='Credit Default'):
    """функция для проверги гиппотез на синхронность"""
    
    alpha=0.5
    
    df = df[df[feature].notna()][:1000]

    df1 = df[df[feature] == values[0]]
    test1 = df1[target].value_counts()

    df2 = df[df[feature] == values[1]]
    test2 = df2[target].value_counts()
    
    p = chi2_contingency([test1, test2])[1]
    print(f'p = {p}')
    
    if p <= alpha:
        print(f'{values[0]} != {values[1]}, при alpha = {alpha}') 
    elif p > alpha:
        print(f'{values[0]} == {values[1]}, при alpha = {alpha}')
def shapiro_test(feature: pd.Series) -> print:
    """функция для проверги гиппотез на нормальность"""
    
    
    alpha = 0.5
    result = shapiro(feature[:1000])
    
    p = result[1]
    print(f'p = {p}')
    
    if p <= alpha:
        print(f'не имеет нормального распредение при alpha = {alpha}')
    elif p > alpha:
        print(f'имеет нормального распредение при alpha = {alpha}')
def evaluation_result(y_train, y_train_pred, y_test, y_test_pred, show=True):
    """функция для оценки результатов обучения"""
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(
        'R2:\n',
        f'train = {r2_train}\n',
        f'test = {r2_test}'
    )

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = ax

        ax1.scatter(y_train, y_train_pred)
        ax1.set_title(f'r2_train = {round(r2_train, 3)}')
        ax1.set_xlabel('true')
        ax1.set_ylabel('pred')

        ax2.scatter(y_test, y_test_pred)
        ax2.set_title(f'r2_test = {round(r2_test, 3)}')
        ax2.set_xlabel('true')
        ax2.set_ylabel('pred')

        fig.set_size_inches(12, 4)
        plt.subplots_adjust(wspace=0.4, hspace=0.1)
        plt.show()
def evaluation_classification(y_train_true, y_train_pred, y_test_true, y_test_pred):
    
    f1_train = f1_score(y_true=y_train_true, y_pred=y_train_pred)
    f1_test = f1_score(y_true=y_test_true, y_pred=y_test_pred)
    diff = f1_train - f1_test
    
    print(
        f'f1 score:\n',
        f'\ttrain = {round(f1_train, 2)}\n',
        f'\ttest = {round(f1_test, 2)}\n\n',
        f'\tdiff = {round(diff, 2)}'
    )
def plot_outliers(df_train, df_test, col_name, hist = False):
    
    fig, ax = plt.subplots(1,3,figsize=(16,5))
    ax[1].set_title('Train Dataset')
    ax[2].set_title('Test Dataset')
    sns.distplot(df_train[col_name], color='b', ax=ax[0], hist=hist, label='Train')
    sns.distplot(df_test[col_name], color='r', ax=ax[0], hist=hist, label='Test')
    sns.boxplot(df_train[col_name], ax=ax[1])
    sns.boxplot(df_test[col_name], color='r', ax=ax[2])
    plt.show()
Xy_df = pd.read_csv(PATH_TO_TRAIN)
Xy_df.head(2)
# Состояние данных

Xy_df.info()
# Просмотрим их целостность в процентнах:

missing_data(df=Xy_df)
# Просмотрим сводку

Xy_df.describe()
# Посмотрим корреляцию с таргером

correlation(df=Xy_df, target='Credit Default')
X_final = pd.read_csv(PATH_TO_TEST)
X_final.head(2)
# Состояние данных

X_final.info()
# Просмотрим их процентное соотношение:

missing_data(df=X_final)
# Сопоставим данные в train и test
# cont_cols = ['Annual Income', 'Tax Liens', 'Number of Open Accounts', 
#              'Years of Credit History', 'Maximum Open Credit', 'Number of Credit Problems', 'Months since last delinquent', 
#              'Bankruptcies', 'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt', 'Credit Score']

# for col in cont_cols:
#     plot_outliers(Xy_df, X_final, col)
class Preprocessing:
    """Класс для предобработки данных"""
    
    def __init__(self, df, features):
        
        self.features = features
        self.df = df.copy()
        
        self.missMSLD = 'Изначально пуст'
        self.missBankruptcies = 'Изначально пуст'
        self.emmissMOC = 'Изначально пуст'
        self.model = 'Изначально пуст'
        
    def concat_values(self):
        """Объединение значений"""
        
        df = self.df
        
        df.loc[df['Home Ownership'] == 'Have Mortgage', 'Home Ownership'] = 'Home Mortgage'
        df.loc[df['Home Ownership'] == 'Rent', 'Home Ownership'] = 'Own Home'
        
    def mark(self):
        """Пометка ошибок в данных"""
        
        df = self.df
        
        df.loc[df['Current Loan Amount'] < 2e7, 'emissionsCLA'] = 0
        df.loc[df['Current Loan Amount'] > 2e7, 'emissionsCLA'] = 1
        
        df.loc[df['Credit Score'] <= 999, 'errorsCS'] = 0
        df.loc[df['Credit Score'] > 999, 'errorsCS'] = 1
        df.loc[df['Credit Score'].isna(), 'errorsCS'] = 2
        
        df.loc[df['Annual Income'].notna(), 'omissionsAI'] = 0
        df.loc[df['Annual Income'].isna(), 'omissionsAI'] = 1
        
    def retype_in_float(self):
        """Ретипизация данных из строк в категории"""
    
        df = self.df
        
        df['Home Ownership'] = df['Home Ownership'].map({'Home Mortgage': 0, 'Own Home': 1}).astype(float)
        df['Term'] = df['Term'].map({'Short Term': 0, 'Long Term':1}).astype(float)
        
    def missingMSLD(self):
        """заполнение пропусков суммой пропусков"""
        
        df = self.df
        
        self.missMSLD = (df['Months since last delinquent'].isna()).sum()
        
        df.loc[df['Months since last delinquent'].isna(), 'Months since last delinquent'] = self.missMSLD
        
    def missingBankruptcies(self):
        """заполнение пропусков медианой"""
        
        df = self.df
        
        self.missBankruptcies = df['Bankruptcies'].quantile(q=0.5)
        
        df.loc[df['Bankruptcies'].isna(), 'Bankruptcies'] = self.missBankruptcies
        
    def missingAnIn(self):
        """заполнение пропусков при помощи LGBM"""
        
        df = self.df

        target='Annual Income'
        
        columns = ['Annual Income', 'Home Ownership', 'Tax Liens', 'Number of Open Accounts', 'Maximum Open Credit', 
                   'Number of Credit Problems', 'Months since last delinquent', 'Current Loan Amount', 
                   'Current Credit Balance', 'Monthly Debt', 'Credit Score']
        
        pred, self.model = pred_feature(df=df, target=target, columns=columns, 
                                        max_depth=1, learning_rate=0.2, n_estimators=1950)
        df.loc[df[target].isna(), target] = pred
        
    def missingCrSc(self):
        """заполнение пропусков при помощи LGBM"""
        
        df = self.df
        
        target = 'Credit Score'
        
        columns = ['Credit Score', 'Term', 'Current Loan Amount', 'Home Ownership', 'Annual Income', 
                   'Years of Credit History', 'Maximum Open Credit', 'Number of Credit Problems', 
                   'Months since last delinquent', 'Bankruptcies', 'Current Credit Balance', 'Monthly Debt', 
                   'errorsCS', 'Number of Open Accounts']
        
        pred, self.model = pred_feature(df=df, target=target, columns=columns, 
                                        max_depth=1, learning_rate=0.1, n_estimators=1300)
        df.loc[df[target].isna(), target] = pred
        
    def emissionsCS(self):
        """обработка выбросов путем деления на 10"""
        
        df = self.df
        
        df.loc[df['Credit Score']>999, 'Credit Score'] = df.loc[df['Credit Score']>999, 'Credit Score'] / 10
        
    def emissionsMOC(self):
        """Обработка выбросов средним значением"""
        
        df = self.df
        
        self.emmissMOC = df['Maximum Open Credit'].mean()
        
        df.loc[df['Maximum Open Credit'] > 2e8, 'Maximum Open Credit'] = self.emmissMOC
        
    def emissionsCLA(self):
        """заполнение пропусков при помощи LGBM"""
        
        df = self.df
        
        target = 'Current Loan Amount'
        
        columns = ['Current Loan Amount', 'Annual Income', 'Maximum Open Credit', 'Term', 'Current Credit Balance', 
                   'Number of Open Accounts', 'Years of Credit History', 'Credit Score']
        
        df.loc[df[target] > 2e7, target] = np.nan
        
        pred, self.model = pred_feature(df=df, target=target, columns=columns, 
                                        max_depth=1, learning_rate=0.1, n_estimators=490)
        df.loc[df[target].isna(), target] = pred
        
    def genering_features(self):
        """Генерация новых признаков"""
        
        df = self.df
        
        df['Months to Maturity'] = df['Current Credit Balance']/df['Monthly Debt']
        df.loc[df['Months to Maturity'].isna(), 'Months to Maturity'] = 0
        
        df['Monthly Income'] = df['Annual Income'] / 12
        
        new_columns = 'Months to Maturity', 'Monthly Income'
        
        for col in new_columns:
            self.features.append(col)
        
    def standard_data(self):
        """Стандартизация данных"""
        
        df = self.df
        
        scaler = StandardScaler()
        df[self.features] = scaler.fit_transform(df[self.features])

        
    def fit(self):
        """Порядок выполнения"""
        
        self.concat_values()
        self.mark()
        self.retype_in_float()

        self.missingMSLD()
        self.missingBankruptcies()

        self.emissionsCS()
        self.emissionsMOC()

        self.missingAnIn()
        self.missingCrSc()

        self.emissionsCLA()
        
        genering_features()
        
        self.standard_data()
    
    def transform(self):
        pass
# разобъем пригнаки 
TARGET_NAME = 'Credit Default'
FEATURE_NAMES = ['Annual Income', 'Tax Liens', 'Number of Open Accounts', 'Years of Credit History', 'Maximum Open Credit', 
                 'Number of Credit Problems', 'Months since last delinquent', 'Bankruptcies', 'Current Loan Amount', 
                 'Current Credit Balance', 'Monthly Debt', 'Credit Score', 'Term', 'Home Ownership']

# инициализируем класс предобработки
xy_prep = Preprocessing(df=Xy_df, features=FEATURE_NAMES)
annInc = Xy_df[Xy_df['Annual Income'].notna()]['Annual Income']

sns.distplot(annInc)
plt.show()
shapiro_test(feature=annInc)
mountlyD = Xy_df['Monthly Debt']

sns.distplot(mountlyD)
plt.show()
shapiro_test(feature=mountlyD)
sns.countplot(x='Home Ownership', hue='Credit Default', data=Xy_df)
plt.show()
feature = 'Home Ownership'

values = 'Have Mortgage', 'Home Mortgage'
chi2_test(df=Xy_df, feature=feature, values=values)

values = 'Own Home', 'Rent'
chi2_test(df=Xy_df, feature=feature, values=values)
xy_prep.concat_values()
xy_prep.mark() # пропуски и выбросы
xy_prep.retype_in_float() # ретайпы
missing_data(df=Xy_df)
"""Судя по большому количеству пропусков в этом признаке можно сделать вывод, 
что они говорят об отсутствии просрочек по платежу, 
а значит посчитать количество месяцев не возможно.

Тогда заполним пропуски числом, которое будет показывать нам - сколько всего таких пропусков."""

xy_prep.missingMSLD()
"""Здесь ставнительно не много пропусков поэтому заполним их медиальным значением"""

xy_prep.missingBankruptcies()
emission(
    series=Xy_df['Credit Score'], 
    val=999,
)
"""Зная, что максимальный кредитный рейтинг = 999, поделим выбросы на 10"""

xy_prep.emissionsCS()
emission(
    series=Xy_df['Maximum Open Credit'],
    val=2e8
)
# Наблюдается 3 выброса, приведем их к среднему значению

xy_prep.emissionsMOC()
xy_prep.missingAnIn()
xy_prep.missingCrSc()
emission(
    series=Xy_df['Current Loan Amount'],
    val=2e7,
)
xy_prep.emissionsCLA()
xy_prep.genering_features()
correlation(df=Xy_df, target='Credit Default')
Xy_df.columns
# Отбор признаков
columns = ['Home Ownership', 'Annual Income', 'Tax Liens', 'Number of Open Accounts', 'Maximum Open Credit','Term', 
           'Current Credit Balance', 'Monthly Debt', 'Credit Score', 'Current Loan Amount', 
           'Months to Maturity', 'Credit Default', 'Monthly Income']

# columns = ['Home Ownership', 'Annual Income', 'Tax Liens',
#            'Number of Open Accounts', 'Years of Credit History',
#            'Maximum Open Credit',
#            'Term', 'Months to Maturity',
#            'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt',
#            'Credit Score', 'Credit Default']

# xy_prep.standard_data()

Xy = xy_prep.df
Xy = Xy[columns]

# Разбмение на X и y
X = Xy.drop('Credit Default', axis=1)
y = Xy['Credit Default']

# Разбиение на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

# # Балансировка классов
df_to_balansing = pd.concat([X_train, y_train], axis=1)

df_balanced = balancing(df=df_to_balansing, target_name=TARGET_NAME)

X_train = df_balanced.drop(columns=TARGET_NAME)
y_train = df_balanced[TARGET_NAME]
model_lgbm = LGBMClassifier(random_state=42)
model_lgbm.fit(X_train, y_train)

y_train_pred = model_lgbm.predict(X_train)
y_test_pred = model_lgbm.predict(X_test)

evaluation_classification(y_train, y_train_pred, y_test, y_test_pred)
model_xgb = XGBClassifier(random_state=42)
model_xgb.fit(X_train, y_train)

y_train_pred = model_xgb.predict(X_train)
y_test_pred = model_xgb.predict(X_test)

evaluation_classification(y_train, y_train_pred, y_test, y_test_pred)
model_catb = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=1,
                                random_state=42, silent=True)
model_catb.fit(X_train, y_train)

y_train_pred = model_catb.predict(X_train)
y_test_pred = model_catb.predict(X_test)

evaluation_classification(y_train, y_train_pred, y_test, y_test_pred)
feature_importances = pd.DataFrame(zip(
    X_train.columns,
    model_catb.feature_importances_
), columns=['feature_name', 'importance'])

feature_importances.sort_values(by='importance', ascending=False, inplace=True)
feature_importances
Xy