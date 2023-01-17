# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
sns.set();
import pandas as pd

df = pd.read_csv("../input/vodafone/vodafone-subset-1.csv")
df.head()
plt.scatter(df['AVG_ARPU'], df['SCORING']);
A=df.groupby('SCORING')['AVG_ARPU'].mean().plot(kind='bar') 
plt.ylabel('СРЕДНЯЯ СТОИМОСТЬ ТАРИФА')
plt.xlabel('Доход в месяц')
plt.show();
A=df.groupby('SCORING')['AVG_ARPU'].mean()
A

plt.scatter(df['calls_count_in_weekdays'], df['DATA_VOLUME_WEEKDAYS']);
sns.boxplot(df['calls_count_in_weekdays']);
sns.boxplot(df['DATA_VOLUME_WEEKDAYS']);
from scipy.stats import pearsonr, spearmanr, kendalltau
r = spearmanr(df['DATA_VOLUME_WEEKDAYS'], df['calls_count_in_weekdays'])
print('Spearman correlation:', r[0], 'p-value:', r[1])
# Убираем из блокнота лишние сообщения
import warnings
warnings.filterwarnings('ignore')
df.keys()
df1= df[['SCORING', 'AVG_ARPU', 'ROUM', 'calls_count_in_weekdays', 'DATA_VOLUME_WEEKDAYS', 'car','uklon', 'gender']]
df1.rename(columns={'SCORING': 'Уровень дохода', 
                     'AVG_ARPU': 'Стоимость услуг',
                     'ROUM': 'Факт поездок заграницу',
                     'car': 'Наличие машины',
                     'calls_count_in_weekdays': 'Кол-во использование телефона в будние',
                     'DATA_VOLUME_WEEKDAYS': 'Кол-во использованного трафика',
                     'uklon':'ипользование такси УКЛОН',
                     'gender':'пол' }, inplace=True)
df1
df1['Target'] = df['target']
df1
df1.info()
df2=df1.drop('Target', axis=1)
df2



df1['уровень дохода_инлекс'] = df1['Уровень дохода'].map({'HIGH':6,
                                                           'HIGH_MEDIUM':5,
                                                           'MEDIUM':4,
                                                           'LOW':3,
                                                           'VERY LOW':2,
                                                           '0':1})
df1['Стоимость услуг'] = round(df1['Стоимость услуг'])
df1['Стоимость услуг'] = df1['Стоимость услуг'].astype(int)

df1['уровень дохода_инлекс'] = df1['уровень дохода_инлекс'].astype(int)
df1 = df1.loc[df1['уровень дохода_инлекс'] != 6]

#df1['ипользование такси УКЛОН'] = df1['ипользование такси УКЛОН'].astype(int)
df1['Кол-во использованного трафика'] = df1['Кол-во использованного трафика'].astype(int)
df1['Факт поездок заграницу'] = df1['Факт поездок заграницу'].astype(int)
df1['Кол-во использование телефона в будние'] = df1['Кол-во использование телефона в будние'].astype(int)
df1['Кол-во использование телефона в будние'] = df1['Кол-во использование телефона в будние'].astype(int)
df1
sns.pairplot(df2)

#построение корреляионную таблицу всех признаков 
sns.heatmap(df2.corr(method = 'spearman'), annot = True);
nepr=df1[['Стоимость услуг', 'Наличие машины']]
discr=df1[['Кол-во использованного трафика']]
rang=df1[['уровень дохода_инлекс', 'Target']]
categorial=df1[['Наличие машины', 'пол']]
count_nepr_discr=nepr.merge(discr, left_index=True, right_index=True)
count_nepr_rang=nepr.merge(rang, left_index=True, right_index=True)
count_nepr_cat=nepr.merge(categorial, left_index=True, right_index=True)
count_discr_rang=discr.merge(rang, left_index=True, right_index=True)
count_discr_cat=discr.merge(categorial, left_index=True, right_index=True)
rang_cat=rang.merge(categorial, left_index=True, right_index=True)
sns.heatmap(nepr.corr(method = 'spearman'), annot = True);
sns.heatmap(rang.corr(method = 'spearman'), annot = True);
sns.heatmap(count_nepr_discr.corr(method = 'pearson'), annot = True);
sns.heatmap(count_nepr_rang.corr(method = 'pearson'), annot = True);
sns.heatmap(count_nepr_cat.corr(method = 'pearson'), annot = True);
sns.heatmap(count_discr_rang.corr(method = 'spearman'), annot = True);
sns.heatmap(count_discr_cat.corr(method = 'spearman'), annot = True);
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=1)
# Создаём представителя класса модели, задаём необходимые гиперпараметры
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
df5=df1.drop('Уровень дохода', axis=1)

df5
#df4=df2.drop('Уровень дохода', axis=1)
#df4
#knn.fit(X_train, y_train)
df6=df5.dropna()
df6
X=df6.drop('Target',axis=1)
y = df6['Target']
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1502)

# Создаём представителя класса модели, задаём необходимые гиперпараметры
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

#X_train, X_valid, y_train, y_valid = train_test_split(df6, 
                                                      #df6['Target'], 
                                                      #test_size=0.25, 
                                                      #random_state=1546)
# Строим предсказания на основе обученной модели
y_pred = knn.predict(X_valid)
y_pred
# Вычисляем метрику (меру) качества
a=knn.score(X_valid, y_valid)
a
#кросс-валлидация
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf=KFold(n_splits=5, shuffle=True, random_state=42)
knn=KNeighborsClassifier(n_neighbors=1)
scores=cross_val_score(knn, X, y, cv=kf, scoring = 'accuracy')
print(scores)
mean_score=scores.mean()
print(mean_score)
#Настройка гиперпараметров
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)} #число соседей -от 1 до 50
knn_grid = GridSearchCV(knn, 
                        knn_params, 
                        scoring='accuracy',
                        cv=5)# или cv=kf

knn_grid.fit(X_train, y_train)



knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
results_df=pd.DataFrame(knn_grid.cv_results_)
results_df
import matplotlib.pyplot as plt
plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])
plt.show()


from sklearn.metrics import accuracy_score
accuracy_score(y_valid, y_pred)


[X_train.shape, X_valid.shape, y_train.shape, y_valid.shape]
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2, random_state=2019)
tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot')
print(open('tree.dot').read())
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "data:image/svg+xml;charset=utf-8,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%20standalone%3D%22no%22%3F%3E%3C!DOCTYPE%20svg%20PUBLIC%20%22-%2F%2FW3C%2F%2FDTD%20SVG%201.1%2F%2FEN%22%20%22http%3A%2F%2Fwww.w3.org%2FGraphics%2FSVG%2F1.1%2FDTD%2Fsvg11.dtd%22%3E%3C!--%20Generated%20by%20graphviz%20version%202.40.1%20(20161225.0304)%0A%20--%3E%3C!--%20Title%3A%20Tree%20Pages%3A%201%20--%3E%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%20width%3D%22924pt%22%20height%3D%22289pt%22%20viewBox%3D%220.00%200.00%20924.04%20288.80%22%3E%0A%3Cg%20id%3D%22graph0%22%20class%3D%22graph%22%20transform%3D%22scale(1%201)%20rotate(0)%20translate(4%20284.8)%22%3E%0A%3Ctitle%3ETree%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22%23ffffff%22%20stroke%3D%22transparent%22%20points%3D%22-4%2C4%20-4%2C-284.8%20920.0356%2C-284.8%20920.0356%2C4%20-4%2C4%22%2F%3E%0A%3C!--%200%20--%3E%0A%3Cg%20id%3D%22node1%22%20class%3D%22node%22%3E%0A%3Ctitle%3E0%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22582.3038%2C-280.9003%20323.2318%2C-280.9003%20323.2318%2C-205.4997%20582.3038%2C-205.4997%20582.3038%2C-280.9003%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-264.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EX%5B3%5D%20%26lt%3B%3D%201.5%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-247.4%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.774%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-230.6%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%206255%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-213.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B174%2C%20343%2C%201089%2C%201770%2C%201792%2C%201087%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%201%20--%3E%0A%3Cg%20id%3D%22node2%22%20class%3D%22node%22%3E%0A%3Ctitle%3E1%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22439.8039%2C-169.7003%20215.7317%2C-169.7003%20215.7317%2C-94.2997%20439.8039%2C-94.2997%20439.8039%2C-169.7003%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-153%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EX%5B7%5D%20%26lt%3B%3D%202.5%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-136.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.743%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-119.4%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%203241%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-102.6%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B50%2C%2072%2C%20385%2C%20757%2C%201081%2C%20896%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%200%26%2345%3B%26gt%3B1%20--%3E%0A%3Cg%20id%3D%22edge1%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E0-%26gt%3B1%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M410.4485%2C-205.5528C399.9617%2C-196.2237%20388.6257%2C-186.1392%20377.8087%2C-176.5164%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22379.8413%2C-173.6401%20370.0435%2C-169.6085%20375.1887%2C-178.8701%20379.8413%2C-173.6401%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22371.5017%22%20y%3D%22-190.3659%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3ETrue%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%204%20--%3E%0A%3Cg%20id%3D%22node5%22%20class%3D%22node%22%3E%0A%3Ctitle%3E4%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22696.8039%2C-169.7003%20458.7317%2C-169.7003%20458.7317%2C-94.2997%20696.8039%2C-94.2997%20696.8039%2C-169.7003%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-153%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EX%5B3%5D%20%26lt%3B%3D%20101.5%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-136.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.763%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-119.4%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%203014%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-102.6%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B124%2C%20271%2C%20704%2C%201013%2C%20711%2C%20191%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%200%26%2345%3B%26gt%3B4%20--%3E%0A%3Cg%20id%3D%22edge4%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E0-%26gt%3B4%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M495.0871%2C-205.5528C505.5739%2C-196.2237%20516.9099%2C-186.1392%20527.7269%2C-176.5164%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22530.3469%2C-178.8701%20535.4921%2C-169.6085%20525.6943%2C-173.6401%20530.3469%2C-178.8701%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22534.0339%22%20y%3D%22-190.3659%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EFalse%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%202%20--%3E%0A%3Cg%20id%3D%22node3%22%20class%3D%22node%22%3E%0A%3Ctitle%3E2%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22203.3039%2C-58.6014%20.2317%2C-58.6014%20.2317%2C.2014%20203.3039%2C.2014%20203.3039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22101.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.69%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22101.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%20887%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22101.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B8%2C%2013%2C%2067%2C%20153%2C%20262%2C%20384%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%201%26%2345%3B%26gt%3B2%20--%3E%0A%3Cg%20id%3D%22edge2%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E1-%26gt%3B2%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M244.897%2C-94.3048C222.1897%2C-83.976%20197.7312%2C-72.8506%20175.5689%2C-62.7697%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22176.7749%2C-59.4732%20166.2232%2C-58.5186%20173.8765%2C-65.845%20176.7749%2C-59.4732%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%203%20--%3E%0A%3Cg%20id%3D%22node4%22%20class%3D%22node%22%3E%0A%3Ctitle%3E3%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22439.3039%2C-58.6014%20222.2317%2C-58.6014%20222.2317%2C.2014%20439.3039%2C.2014%20439.3039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22330.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.747%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22330.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%202354%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22330.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B42%2C%2059%2C%20318%2C%20604%2C%20819%2C%20512%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%201%26%2345%3B%26gt%3B3%20--%3E%0A%3Cg%20id%3D%22edge3%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E1-%26gt%3B3%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M328.8679%2C-94.3048C329.1128%2C-85.9126%20329.373%2C-76.9946%20329.6198%2C-68.5382%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22333.1189%2C-68.6165%20329.9122%2C-58.5186%20326.1219%2C-68.4122%20333.1189%2C-68.6165%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%205%20--%3E%0A%3Cg%20id%3D%22node6%22%20class%3D%22node%22%3E%0A%3Ctitle%3E5%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22686.8039%2C-58.6014%20462.7317%2C-58.6014%20462.7317%2C.2014%20686.8039%2C.2014%20686.8039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22574.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.744%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22574.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%202329%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22574.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B70%2C%20125%2C%20478%2C%20832%2C%20646%2C%20178%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%204%26%2345%3B%26gt%3B5%20--%3E%0A%3Cg%20id%3D%22edge5%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E4-%26gt%3B5%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M576.6677%2C-94.3048C576.4228%2C-85.9126%20576.1626%2C-76.9946%20575.9158%2C-68.5382%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22579.4137%2C-68.4122%20575.6234%2C-58.5186%20572.4167%2C-68.6165%20579.4137%2C-68.4122%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%206%20--%3E%0A%3Cg%20id%3D%22node7%22%20class%3D%22node%22%3E%0A%3Ctitle%3E6%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22915.8039%2C-58.6014%20705.7317%2C-58.6014%20705.7317%2C.2014%20915.8039%2C.2014%20915.8039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22810.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.76%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22810.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%20685%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22810.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B54%2C%20146%2C%20226%2C%20181%2C%2065%2C%2013%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%204%26%2345%3B%26gt%3B6%20--%3E%0A%3Cg%20id%3D%22edge6%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E4-%26gt%3B6%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M663.2054%2C-94.3048C686.7206%2C-83.9298%20712.0573%2C-72.7512%20734.9867%2C-62.6348%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22736.5798%2C-65.7575%20744.316%2C-58.5186%20733.7541%2C-59.3531%20736.5798%2C-65.7575%22%2F%3E%0A%3C%2Fg%3E%0A%3C%2Fg%3E%0A%3C%2Fsvg%3E")
#from sklearn.metrics import accuracy_score
#accuracy_score(y_valid, y_pred)
X.columns
# Предсказания для валидационного множества

y_pred=tree.predict(X_valid)
accuracy_score(y_valid, y_pred)
# Кросс-валидация и подбор гиперпараметров
from sklearn.model_selection import GridSearchCV

tree_params={'max_depth': np.arange(2, 11),
               'min_samples_leaf': np.arange(2, 11)}

tree_grid=GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
tree_grid.best_params_
tree_params_d = {'max_depth': np.arange(1, 11)}

tree_grid_d = GridSearchCV(tree, tree_params_d, cv=5, scoring='accuracy')
tree_grid_d.fit(X_train, y_train)

tree_params_s={'min_samples_leaf': np.arange(1, 11)}

tree_grid_s=GridSearchCV(tree, tree_params_s, cv=5, scoring='accuracy') 
tree_grid_s.fit(X_train, y_train)
#отрсиовка графиков 
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True) # 2 графика рядом с одинаковым масштабом по оси Оу

ax[0].plot(tree_params_d['max_depth'], tree_grid_d.cv_results_['mean_test_score']) # accuracy vs max_depth
ax[0].set_xlabel('max_depth')
ax[0].set_ylabel('Mean accuracy on test set')

ax[1].plot(tree_params_s['min_samples_leaf'], tree_grid_s.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf
ax[1].set_xlabel('min_samples_leaf')
ax[1].set_ylabel('Mean accuracy on test set');
# Выбор и отрисовка наилучшего дерева

pd.DataFrame(tree_grid.cv_results_).head().T

best_tree=tree_grid.best_estimator_
y_pred = best_tree.predict(X_valid)
accuracy_score(y_valid, y_pred)
export_graphviz(best_tree, out_file='best_tree.dot')
print(open('best_tree.dot').read()) 
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "data:image/svg+xml;charset=utf-8,%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%20standalone%3D%22no%22%3F%3E%3C!DOCTYPE%20svg%20PUBLIC%20%22-%2F%2FW3C%2F%2FDTD%20SVG%201.1%2F%2FEN%22%20%22http%3A%2F%2Fwww.w3.org%2FGraphics%2FSVG%2F1.1%2FDTD%2Fsvg11.dtd%22%3E%3C!--%20Generated%20by%20graphviz%20version%202.40.1%20(20161225.0304)%0A%20--%3E%3C!--%20Title%3A%20Tree%20Pages%3A%201%20--%3E%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%20width%3D%22924pt%22%20height%3D%22289pt%22%20viewBox%3D%220.00%200.00%20924.04%20288.80%22%3E%0A%3Cg%20id%3D%22graph0%22%20class%3D%22graph%22%20transform%3D%22scale(1%201)%20rotate(0)%20translate(4%20284.8)%22%3E%0A%3Ctitle%3ETree%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22%23ffffff%22%20stroke%3D%22transparent%22%20points%3D%22-4%2C4%20-4%2C-284.8%20920.0356%2C-284.8%20920.0356%2C4%20-4%2C4%22%2F%3E%0A%3C!--%200%20--%3E%0A%3Cg%20id%3D%22node1%22%20class%3D%22node%22%3E%0A%3Ctitle%3E0%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22582.3038%2C-280.9003%20323.2318%2C-280.9003%20323.2318%2C-205.4997%20582.3038%2C-205.4997%20582.3038%2C-280.9003%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-264.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EX%5B3%5D%20%26lt%3B%3D%201.5%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-247.4%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.774%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-230.6%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%206255%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22452.7678%22%20y%3D%22-213.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B174%2C%20343%2C%201089%2C%201770%2C%201792%2C%201087%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%201%20--%3E%0A%3Cg%20id%3D%22node2%22%20class%3D%22node%22%3E%0A%3Ctitle%3E1%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22439.8039%2C-169.7003%20215.7317%2C-169.7003%20215.7317%2C-94.2997%20439.8039%2C-94.2997%20439.8039%2C-169.7003%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-153%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EX%5B7%5D%20%26lt%3B%3D%202.5%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-136.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.743%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-119.4%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%203241%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22327.7678%22%20y%3D%22-102.6%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B50%2C%2072%2C%20385%2C%20757%2C%201081%2C%20896%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%200%26%2345%3B%26gt%3B1%20--%3E%0A%3Cg%20id%3D%22edge1%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E0-%26gt%3B1%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M410.4485%2C-205.5528C399.9617%2C-196.2237%20388.6257%2C-186.1392%20377.8087%2C-176.5164%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22379.8413%2C-173.6401%20370.0435%2C-169.6085%20375.1887%2C-178.8701%20379.8413%2C-173.6401%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22371.5017%22%20y%3D%22-190.3659%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3ETrue%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%204%20--%3E%0A%3Cg%20id%3D%22node5%22%20class%3D%22node%22%3E%0A%3Ctitle%3E4%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22696.8039%2C-169.7003%20458.7317%2C-169.7003%20458.7317%2C-94.2997%20696.8039%2C-94.2997%20696.8039%2C-169.7003%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-153%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EX%5B3%5D%20%26lt%3B%3D%20101.5%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-136.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.763%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-119.4%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%203014%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22577.7678%22%20y%3D%22-102.6%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B124%2C%20271%2C%20704%2C%201013%2C%20711%2C%20191%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%200%26%2345%3B%26gt%3B4%20--%3E%0A%3Cg%20id%3D%22edge4%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E0-%26gt%3B4%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M495.0871%2C-205.5528C505.5739%2C-196.2237%20516.9099%2C-186.1392%20527.7269%2C-176.5164%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22530.3469%2C-178.8701%20535.4921%2C-169.6085%20525.6943%2C-173.6401%20530.3469%2C-178.8701%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22534.0339%22%20y%3D%22-190.3659%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3EFalse%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%202%20--%3E%0A%3Cg%20id%3D%22node3%22%20class%3D%22node%22%3E%0A%3Ctitle%3E2%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22203.3039%2C-58.6014%20.2317%2C-58.6014%20.2317%2C.2014%20203.3039%2C.2014%20203.3039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22101.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.69%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22101.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%20887%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22101.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B8%2C%2013%2C%2067%2C%20153%2C%20262%2C%20384%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%201%26%2345%3B%26gt%3B2%20--%3E%0A%3Cg%20id%3D%22edge2%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E1-%26gt%3B2%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M244.897%2C-94.3048C222.1897%2C-83.976%20197.7312%2C-72.8506%20175.5689%2C-62.7697%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22176.7749%2C-59.4732%20166.2232%2C-58.5186%20173.8765%2C-65.845%20176.7749%2C-59.4732%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%203%20--%3E%0A%3Cg%20id%3D%22node4%22%20class%3D%22node%22%3E%0A%3Ctitle%3E3%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22439.3039%2C-58.6014%20222.2317%2C-58.6014%20222.2317%2C.2014%20439.3039%2C.2014%20439.3039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22330.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.747%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22330.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%202354%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22330.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B42%2C%2059%2C%20318%2C%20604%2C%20819%2C%20512%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%201%26%2345%3B%26gt%3B3%20--%3E%0A%3Cg%20id%3D%22edge3%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E1-%26gt%3B3%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M328.8679%2C-94.3048C329.1128%2C-85.9126%20329.373%2C-76.9946%20329.6198%2C-68.5382%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22333.1189%2C-68.6165%20329.9122%2C-58.5186%20326.1219%2C-68.4122%20333.1189%2C-68.6165%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%205%20--%3E%0A%3Cg%20id%3D%22node6%22%20class%3D%22node%22%3E%0A%3Ctitle%3E5%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22686.8039%2C-58.6014%20462.7317%2C-58.6014%20462.7317%2C.2014%20686.8039%2C.2014%20686.8039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22574.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.744%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22574.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%202329%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22574.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B70%2C%20125%2C%20478%2C%20832%2C%20646%2C%20178%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%204%26%2345%3B%26gt%3B5%20--%3E%0A%3Cg%20id%3D%22edge5%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E4-%26gt%3B5%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M576.6677%2C-94.3048C576.4228%2C-85.9126%20576.1626%2C-76.9946%20575.9158%2C-68.5382%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22579.4137%2C-68.4122%20575.6234%2C-58.5186%20572.4167%2C-68.6165%20579.4137%2C-68.4122%22%2F%3E%0A%3C%2Fg%3E%0A%3C!--%206%20--%3E%0A%3Cg%20id%3D%22node7%22%20class%3D%22node%22%3E%0A%3Ctitle%3E6%3C%2Ftitle%3E%0A%3Cpolygon%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20points%3D%22915.8039%2C-58.6014%20705.7317%2C-58.6014%20705.7317%2C.2014%20915.8039%2C.2014%20915.8039%2C-58.6014%22%2F%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22810.7678%22%20y%3D%22-41.8%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Egini%20%3D%200.76%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22810.7678%22%20y%3D%22-25%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Esamples%20%3D%20685%3C%2Ftext%3E%0A%3Ctext%20text-anchor%3D%22middle%22%20x%3D%22810.7678%22%20y%3D%22-8.2%22%20font-family%3D%22Times%2Cserif%22%20font-size%3D%2214.00%22%20fill%3D%22%23000000%22%3Evalue%20%3D%20%5B54%2C%20146%2C%20226%2C%20181%2C%2065%2C%2013%5D%3C%2Ftext%3E%0A%3C%2Fg%3E%0A%3C!--%204%26%2345%3B%26gt%3B6%20--%3E%0A%3Cg%20id%3D%22edge6%22%20class%3D%22edge%22%3E%0A%3Ctitle%3E4-%26gt%3B6%3C%2Ftitle%3E%0A%3Cpath%20fill%3D%22none%22%20stroke%3D%22%23000000%22%20d%3D%22M663.2054%2C-94.3048C686.7206%2C-83.9298%20712.0573%2C-72.7512%20734.9867%2C-62.6348%22%2F%3E%0A%3Cpolygon%20fill%3D%22%23000000%22%20stroke%3D%22%23000000%22%20points%3D%22736.5798%2C-65.7575%20744.316%2C-58.5186%20733.7541%2C-59.3531%20736.5798%2C-65.7575%22%2F%3E%0A%3C%2Fg%3E%0A%3C%2Fg%3E%0A%3C%2Fsvg%3E")
best_tree