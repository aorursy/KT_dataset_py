import pandas as pd

marketing_pd = pd.read_csv('../input/lights-sales-data/lights_sales_data.csv')

marketing_pd.head()
correl_matrix=marketing_pd.loc[:, [x for x in marketing_pd.columns if x != 'month_number' and x != 'sales']].corr()

correl_matrix
import seaborn as sns

from matplotlib import pyplot as plt



plt.figure(figsize=(8, 4))

plt.title('Multicollinearity of data')

sns.heatmap(correl_matrix, annot=True, cmap='coolwarm')
pd.DataFrame([marketing_pd.loc[:, ['Newspapers', 'Radio', 'TV']].sum(axis = 1), 

              marketing_pd.total_marketing_spendings]).T.corr()
marketing_pd = marketing_pd.loc[:, [x for x in marketing_pd.columns if x != 'total_marketing_spendings' 

                              and x != 'total_marketing_spendings, $']]

marketing_pd.head()
from matplotlib import gridspec

from math import ceil

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from sklearn.linear_model import LinearRegression



ax = [] # Массив для складирования графиков

n_pict_in_line = 2 # Количество картинок в одной строке



gs = gridspec.GridSpec(ceil(len(marketing_pd.columns[1:-1]) / float(n_pict_in_line)), n_pict_in_line) 

# Создаём сетку для графиков

fig = plt.figure(figsize=(n_pict_in_line * 6+3, ceil(len(marketing_pd.columns[1:-1]) / n_pict_in_line) * 4+3))



for i, col in enumerate(marketing_pd.columns[1:-1]): # Строим графики для каждого столбца из датафрейма marketing_pd

    ax.append(plt.subplot(gs[i // n_pict_in_line, i % n_pict_in_line])) # Добавляем subplot в нужную клетку

    marketing_pd.plot(x = col, y = 'sales', kind = 'scatter', ax = ax[i], label = 'Data') # строим график рассеяния значений

    

    # построение графика линейной зависимости результирующего значения каждого из показателей и вычисление коэф детерминации

    model = LinearRegression() #используем такой синтаксис, так как на вход должна подаваться линейная модель

    model.fit(marketing_pd.loc[:, [col]], marketing_pd.sales)

    r_sq = model.score(marketing_pd.loc[:, [col]], marketing_pd.sales)

    ax[i].plot(list(range(int(ax[i].get_xticks()[0]), int(ax[i].get_xticks()[-1]))),

            [model.intercept_ + model.coef_*x for x in list(range(int(ax[i].get_xticks()[0]), int(ax[i].get_xticks()[-1])))]

            , label = 'Linear Prediction', color = 'black')

    

    # формирование вывода результата на графике

    handles, labels = ax[i].get_legend_handles_labels()

    handles.append(mpatches.Patch(color='none', label='$R^2 = ' + str(round(r_sq, 2)) + '$'))

    ax[i].legend(handles = handles)

    

plt.show()
import statsmodels.api as sm



X = sm.add_constant(marketing_pd.loc[:, ['Radio', 'days_of_sales', 'TV']])

 

model = sm.OLS(marketing_pd.sales, X).fit()

predictions = model.predict(X) 

 

print_model = model.summary()

print(print_model)

plt.scatter(marketing_pd.sales, predictions, label = 'predictions vs data')

plt.xlabel('Data')

plt.ylabel('Predictions')

plt.title('Предсказание против исходных данных')

plt.legend()

plt.show()
from statsmodels.graphics.gofplots import qqplot # Импорт части библиотеки для построения qq-plot

qqplot(model.resid  # Ошибки модели

       , line = 's')
from statsmodels.stats.stattools import durbin_watson

durbin_watson(model.resid)
marketing_pd['resid'] = model.resid

marketing_pd.head()
plt.title('Correlation of errors')

sns.heatmap(marketing_pd.loc[:, ['days_of_sales', 'Radio', 'TV', 'resid']].corr(), annot=True, cmap='coolwarm');
# описательная статистика результатов продаж

marketing_pd['sales'].describe()