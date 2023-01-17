#Импортирование библиотек, загрузка файла
import pandas as pd              #Импорт библиотеки Pandas
import numpy as np               #Импорт библиотеки Numpy
import seaborn as sns            #Импорт библиотеки Seaborn
import matplotlib.pyplot as plt  #Импорт библиотеки Matplotlib


data = pd.read_csv('../input/telecom-users/telecom_users.csv')  #Открытие файла с данными
data.head()
#Проверка пропущенных значений
data.info()
# Приведение числовых данных к соответсвующим типам для получения общей статистической информации.
new_type_list = []
for i in data['TotalCharges']:
    try:
        i = float(i)
    except:
        i = 0
    new_type_list.append(i)

data['TotalCharges'] = new_type_list
#Информация о типах данных
data.dtypes
#Получение базовых статистик
data.describe()
# Полчуение статистики по нецифровым данным
data.describe(include=[np.object])
# Посчитаем процент оттока клиентов за данный период
count = data.describe(include=[np.object]).loc['count','Churn']
Churn = count - data.describe(include=[np.object]).loc['freq','Churn']
print(f'Процент оттока клиентов составляет {Churn * 100 // count}%')
# Удаление признака с ID клиента
del data['customerID']
# Для разбиения на классы воспользуемся автоматическим алгоритмом:
# Данный алгоритм можно применять при большом количестве признаков.
def classificator(data_frame):  #Определяем функицю
    for feature in list(data_frame.columns):      # Задаем итератор в рамках названий колонок
        if data_frame[feature].dtype == 'O':      # Условие для признаков соответсвующих типу даных "Objekt"
            data_frame[feature].replace(['Yes', 'No'], [1, 0], inplace=True)   # Замена значений 'Yes' и 'No' на 0 и 1
            for iteration, value in enumerate(list(data_frame[feature].unique())):  # Итератор в рамках уникальных значений признака
                if type(value) == str:         # Условие для замены нецифровых значений
                    if data_frame[feature].nunique() > 2:  # Условие компенсации порядкового номера для 
                        iteration += 1                     # тех признаков, в которых не было значений 'Yes' и 'No'
                    data_frame[feature].replace(value, iteration, inplace=True)  # Замена всех нецифровых значений на соответсвущй порядковый номер 
data_new = data.copy()
classificator(data_new)
data_new
#Посмотрим на корреляционные связи между признаками
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlation of Features', y=1, size=15)
sns.heatmap(data_new.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True, fmt='.2g')
plt.show()
#Поскольку безымянный пердиктор не имеет существенной зависимости с целевой переменной, можем его удалить.
del data['Unnamed: 0']
# Посмтотрим сколько процентов клиентов пользовались услугами безопасности
security_data = data_new[['Churn', 'OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection', 'PaperlessBilling']].groupby('Churn').agg('sum')
security_data.iloc[0,:] = security_data.iloc[0,:]//((data_new.Churn.count()-data_new.Churn.sum())/100)
security_data.iloc[1,:] = security_data.iloc[1,:]//(data_new.Churn.sum()/100)
security_data
# Посмотрим на эти значения на графике
labels = security_data.columns  # Значения подписей по x
percent_0 = security_data.iloc[0,:]  # Список процентов оставшихся клиентов
percent_1 = security_data.iloc[1,:]  # Список процентов ушедших клиентов

x = np.arange(len(labels))  # Список координат столбцов по x
width = 0.3  # Переменная для ширины столбцов

fig, ax = plt.subplots(figsize=(10,5))  # Определение фигуры и осей
rects1 = ax.bar(x - width/2, percent_0, width, label='Оставшиеся клиенты')  # Определение колонок для данных об отсавшихся клиентах
rects2 = ax.bar(x + width/2, percent_1, width, label='Ушедшие клиенты')   # Определение колонок для данных об ушедших клиентах

# Указание подписей для осей, таблицы, легенды
ax.set_ylabel('Проценты')
ax.set_xlabel('Услуги')
ax.set_title('Использование клиентами услуг безопасности, %')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')

# Определение функции для отображения столбцов с аннотациями
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # Задание параметров для аннотаций
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height), # Получение точек координат для текста
                    xytext=(0, 2),    # Высота текста над столбцами
                    # Расположение текста относительно столбцов
                    textcoords="offset points",  
                    ha='center', va='bottom')

autolabel(rects1)  # Выполнение функции для первой группы данных
autolabel(rects2)  # Выполнение функции для второй группы данных

plt.show()
# Посмотрим на график зависимости процента ушедших клиентов от количества месяцев использования услуг
percent_churn = data_new[data_new.Churn==1][['Churn','tenure']].groupby('tenure').sum()
percent_churn.Churn = list(map(lambda x: x / (data_new.Churn.sum() / 100), percent_churn.Churn))
plt.figure(figsize=(8,5))
plt.title('Процент оттока по количеству месяцев пользования услугами', y=1.03, size=15)
plt.xlabel('Месяцы')
plt.ylabel('Проценты')
plt.plot(percent_churn.index, percent_churn.Churn)
plt.show()
# Посмотрим как распределены дынные о месячной плате за связь среди 2 групп клиентов
sns.set(style="whitegrid") 
ax = sns.catplot('Churn', 'MonthlyCharges', data=data_new, aspect=1.1, height=5, kind='box') 
y_0 = data_new.MonthlyCharges[data_new.Churn==0].median()
y_1 = data_new.MonthlyCharges[data_new.Churn==1].median()
ax = sns.lineplot(x=[-0.3,0.3], y=y_0)
ax = sns.lineplot(x=[0.7,1.3], y=y_1)
from sklearn.neighbors import KNeighborsClassifier  #Импорт классификатора "K-ближайших соседей"
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier #Импорт классификатора "Градиентный бустинг" и "Рандомный лес"
from sklearn.linear_model import LogisticRegression  #Импорт классификатора "Логистическая регрессия"
from sklearn.svm import SVC  #Импорт классификатора "Метод опорных векторов"

#Импорт методов автоматической обработки данных
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

#Импорт метрик качества классификации
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import roc_auc_score
# Параметры базовых алгоритмов

knn_params = {'n_neighbors' : np.arange(1, 10, 1)}  # Параметры для классификатора KNeighborsClassifier

gbc_params = {'learning_rate': np.arange(0.1, 0.6, 0.1)}   # Параметры для классификатора GradientBoostingClassifier

rfc_params = {'n_estimators': range(10, 100, 10),  # Параметры для классификатора RandomForestClassifier
              'min_samples_leaf': range(1, 7)}

svc_params = {'kernel': ['linear', 'rbf'], 
'C': np.arange(0.1, 1, 0.2)}                       # Параметры для классификатора SVC

lr_params = {'C': np.arange(0.2, 1, 0.1)} # Параметры для классификатора LogisticRegression

skf = StratifiedKFold(n_splits=8, random_state=17) # Параметры для кросс-валидации
# Разделение данных на тренировочные и тестовые
y = data_new['Churn']
x = data_new[['tenure', 'OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection', 'MonthlyCharges', 'PaperlessBilling']]
# Посмотрим на балансировку целевого признака
print('Положительных значений -', y.sum())
print('Отрицательных значений -', y.count() - y.sum())
# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3,  stratify=y, random_state=17)
# GridSearch для каждой из моделей

knn = KNeighborsClassifier()                       # Определение объекта классификатора KNeighborsClassifier
gbc = GradientBoostingClassifier(random_state=17)  # Определение объекта классификатора GradientBoostingClassifier
rfc = RandomForestClassifier(random_state=17)      # Определение объекта классификатора RandomForestClassifier
svc = SVC(random_state=17, probability=True)       # Определение объекта классификатора SVC
lr = LogisticRegression(random_state=17,
                        #class_weight = {1:5},
                        solver = 'liblinear')      # Определение объекта классификатора LogisticRegression

gscv_knn = GridSearchCV(estimator=knn, param_grid=knn_params, cv=skf)  # Определение объекта кросс-валидации для KNeighborsClassifier
gscv_gbc = GridSearchCV(estimator=gbc, param_grid=gbc_params, cv=skf)  # Определение объекта кросс-валидации для GradientBoostingClassifier
gscv_rfc = GridSearchCV(estimator=rfc, param_grid=rfc_params, cv=skf)  # Определение объекта кросс-валидации для RandomForestClassifier
gscv_svc = GridSearchCV(estimator=svc, param_grid=svc_params, cv=skf)  # Определение объекта кросс-валидации для SVC
gscv_lr = GridSearchCV(estimator=lr, param_grid=lr_params, cv=skf)     # Определение объекта кросс-валидации для LogisticRegression

knn_model = gscv_knn.fit(X_train, y_train)  # Обучение модели KNeighborsClassifier на кросс-валидации
gbc_model = gscv_gbc.fit(X_train, y_train)  # Обучение модели GradientBoostingClassifier на кросс-валидации
rfc_model = gscv_rfc.fit(X_train, y_train)  # Обучение модели RandomForestClassifier на кросс-валидации
svc_model = gscv_svc.fit(X_train, y_train)  # Обучение модели SVC на кросс-валидации
lr_model = gscv_lr.fit(X_train, y_train)    # Обучение модели LogisticRegression на кросс-валидации
# Получение лучших параметров для классификаторов полученных на кросс-валидации
print('Лучшие параметры:')
print(f'KNeighborsClassifier %s \nGradientBoostingClassifier %s \nRandomForestClassifier %s \nSVC %s \nLogisticRegression %s' %(
                                                                gscv_knn.best_params_,
                                                                gscv_gbc.best_params_,
                                                                gscv_rfc.best_params_,
                                                                gscv_svc.best_params_,
                                                                gscv_lr.best_params_))
# Получение прогнозов для каждой модели

knn_predict = knn_model.predict(X_test)
gbc_predict = gbc_model.predict(X_test)
rfc_predict = rfc_model.predict(X_test)
svc_predict = svc_model.predict(X_test)
lr_predict = lr_model.predict(X_test)
# Получение значений полноты моделей классификаторов
metrics_scores = [recall_score, precision_score, accuracy_score]
predicts = [knn_predict, gbc_predict, rfc_predict, svc_predict, lr_predict]
models_names = ['KNeighbors', 'GradientBoosting', 'RandomForest', 'SVC', 'LogisticRegression']
scores_names = ['recall_score', 'precision_score', 'accuracy_score']
values_list = []
for i, score in enumerate(metrics_scores):
    for predict in predicts:
        values_list.append(round(score(y_test, predict),3))    
    
    x = np.arange(len(models_names))  # Список координат столбцов по x
    
    fig, ax = plt.subplots(figsize=(10,5))  # Определение фигуры и осей
    rects = ax.bar(x, values_list, 0.6)  # Определение колонок для данных
    
    # Указание подписей для осей, таблицы, легенды
    ax.set_ylabel('Значение метрики')
    ax.set_xlabel('Модели')
    ax.set_title(f'Значения {scores_names[i]} для моделей классификации')
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    
    # Определение функции для отображения столбцов с аннотациями
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # Задание параметров для аннотаций
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height), # Получение точек координат для текста
                        xytext=(0, 2),    # Высота текста над столбцами
                        # Расположение текста относительно столбцов
                        textcoords="offset points",  
                        ha='center', va='bottom')
    
    autolabel(rects)  # Выполнение функции
    values_list = []

plt.show()
# Получение значений ROC-AUC score для моделей классификаторов

auc_list = []
models = [knn_model, gbc_model, rfc_model, svc_model, lr_model]
for model in models:
    proba = model.predict_proba(X_test)
    auc_list.append(round(roc_auc_score(y_test, proba[:, 1]),3))
    
x = np.arange(len(models_names))  # Список координат столбцов по x

fig, ax = plt.subplots(figsize=(10,5))  # Определение фигуры и осей
rects = ax.bar(x, auc_list, 0.6)  # Определение колонок для данных

# Указание подписей для осей, таблицы, легенды
ax.set_ylabel('Значение метрики')
ax.set_xlabel('Модели')
ax.set_title('Значения roc_auc_score для моделей классификации')
ax.set_xticks(x)
ax.set_xticklabels(models_names)

# Определение функции для отображения столбцов с аннотациями
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # Задание параметров для аннотаций
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height), # Получение точек координат для текста
                    xytext=(0, 2),    # Высота текста над столбцами
                    # Расположение текста относительно столбцов
                    textcoords="offset points",  
                    ha='center', va='bottom')

autolabel(rects)  # Выполнение функции
values_list = []

plt.show()