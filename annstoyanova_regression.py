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
from stop_words import get_stop_words

def print_scores(y_valid, y_pred):

    '''

    Функция для быстрого вывода четырёх метрик для регрессии.

    

    y_valid --- истинные значения

    y_pred --- предсказанные моделью значения

    '''

    print('MSE:', mean_squared_error(y_valid, y_pred))

    print('MAE:', mean_absolute_error(y_valid, y_pred))

    print('MedAE:', median_absolute_error(y_valid, y_pred))

    print('R2:', r2_score(y_valid, y_pred))
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import warnings

warnings.filterwarnings('ignore')

import os

import re

import numpy as np

import pandas as pd

import matplotlib



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.datasets import fetch_20newsgroups, load_files

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.pipeline import Pipeline





import pandas as pd

from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("/kaggle/input/vodafone-subset-1.csv")

df.head()
df1=df.drop('target', axis = 1)

#.select_dtypes(exclude=['object'])

target=df[['target']]
df1= df[['SCORING', 'AVG_ARPU', 'ROUM', 'calls_count_in_weekdays', 'DATA_VOLUME_WEEKDAYS', 'car','uklon', 'gender','Oblast_post_HOME','Raion_post_HOME','City_post_HOME','banks_sms_count','telegram_count','linkedin_count','skype_count','sim_count','intagram_count','whatsapp_volume']]

df1.rename(columns={'SCORING': 'Уровень дохода', 

                     'AVG_ARPU': 'Стоимость услуг',

                     'ROUM': 'Факт поездок заграницу',

                     'car': 'Наличие машины',

                     'calls_count_in_weekdays': 'Кол-во использование телефона в будние',

                     'DATA_VOLUME_WEEKDAYS': 'Кол-во использованного трафика',

                     'uklon':'ипользование такси УКЛОН',

                     'gender':'пол',

                   'Oblast_post_HOME':'Область_дом',

                   'Raion_post_HOME':'Район_дом',

                    'City_post_HOME':'Город_дом',

                    'banks_sms_count':'Количество входящих сообщений от всех банков',

                    'telegram_count':'кол-во трафика телеграм',

                    'linkedin_count':'кол-во трафика Линкдн',

                    'skype_count':'кол-во трафика Скайп',

                    'sim_count':'кол-во симок',

                    'intagram_count':'кол-во трафика инстаграм',

                    'whatsapp_volume':'кол-во трафика вотсап'

                    

                   }, inplace=True)

df1
label_encoder = LabelEncoder()



obl = pd.Series(label_encoder.fit_transform(df1['Область_дом']))

obl.value_counts().plot.barh()

print(dict(enumerate(label_encoder.classes_)))
label_encoder = LabelEncoder()

rayon=pd.Series(label_encoder.fit_transform(df1['Район_дом']))

rayon.value_counts().plot.barh()

print(dict(enumerate(label_encoder.classes_)))

label_encoder = LabelEncoder()

gor=pd.Series(label_encoder.fit_transform(df1['Город_дом']))

gor
categorical_columns = df1.columns[df1.dtypes == 'object'].union(['Район_дом'])

for column in categorical_columns:

    df[column] = label_encoder.fit_transform(df1[column])

df1.head()
df1['Область_дом'] = obl

df1['Район_дом'] = rayon

df1['Город_дом'] = gor
df1['индекс уровня дохода'] = df1['Уровень дохода'].map({'HIGH':6,

                                               'HIGH_MEDIUM':5,

                                               'MEDIUM':4,

                                               'LOW':3,

                                               'VERY LOW':2,

                                               '0':1})
df1=df1.drop('Уровень дохода',axis=1)

df1.head()
df2 = pd.get_dummies(df1, columns=['Область_дом', 'Район_дом', "Город_дом",'индекс уровня дохода'])
df2.shape

X=df2.drop('ипользование такси УКЛОН',axis=1)

y=target
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X,

                                                      y,

                                                      test_size=0.3,

                                                      random_state=2020)


log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_valid)



print(accuracy_score(y_valid, y_pred))
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

print(confusion_matrix(y_valid, y_pred))
plot_confusion_matrix(log_reg, X_valid, y_valid,values_format='5g')

plt.show()

from sklearn.model_selection import GridSearchCV



log_params={'C': np.logspace(-3, 3, 10),'penalty': ['l2']} # словарь параметров (ключ: набор возможных значений)



log_grid = GridSearchCV(log_reg, log_params, cv=5, scoring='f1_macro') # кросс-валидация по 5 блокам

log_grid.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)
accuracy_score(y_valid, y_pred)
log_grid.best_params_
log_grid.best_score_
print(log_grid.best_estimator_)
pd.DataFrame(log_grid.cv_results_).T
results_df=pd.DataFrame(log_grid.cv_results_)

plt.plot(results_df['param_C'], results_df['mean_test_score'])



# Подписываем оси и график

plt.xlabel('C')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
target
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

print(confusion_matrix(y_valid, y_pred))
plot_confusion_matrix(log_reg, X_valid, y_valid, values_format='5g')

plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision:', precision_score(y_valid, y_pred,average='macro'))

print('Recall:', recall_score(y_valid, y_pred,average='macro'))

print('F1 score:', f1_score(y_valid, y_pred,average='macro'))
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_valid)

log_reg = LogisticRegression(solver='liblinear', penalty='l1')

C_values = {'C': np.logspace(-3, 3, 10)}

logreg_grid = GridSearchCV(log_reg, C_values, cv=5, scoring='f1_macro')

logreg_grid.fit(X_train, y_train)
print(logreg_grid.best_params_)

print(logreg_grid.best_score_)
results_df=pd.DataFrame(logreg_grid.cv_results_)

plt.plot(results_df['param_C'], results_df['mean_test_score'])



# Подписываем оси и график

plt.xlabel('C')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()