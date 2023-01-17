# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style





import datetime # для работы со времнем





from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

import catboost as catb





%matplotlib inline



TEST_DATASET_FILE = '../input/credit-default/test.csv'

TRAIN_DATASET_FILE = '../input/credit-default/train.csv'



test_all_data = pd.read_csv(TEST_DATASET_FILE)

train_data = pd.read_csv(TRAIN_DATASET_FILE)





def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred):

    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))

    print('TEST\n\n' + classification_report(y_test_true, y_test_pred))

    print('CONFUSION MATRIX\n')

    print(pd.crosstab(y_test_true, y_test_pred))

    

def show_feature_importances(feature_names, feature_importances, get_top=None):

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})

    feature_importances = feature_importances.sort_values('importance', ascending=False)

       

    plt.figure(figsize = (20, len(feature_importances) * 0.355))

    

    sns.barplot(feature_importances['importance'], feature_importances['feature'])

    

    plt.xlabel('Importance')

    plt.title('Importance of features')

    plt.show()

    

    if get_top is not None:

        return feature_importances['feature'][:get_top].tolist()



#Для удаления строк с отсутствующими данными

def to_del_fail_lines(df, features_list):

    for feature in feature_list:

        pass



TARGET_COLUMN = 'Credit Default'

FEATURES_COLUMNS = train_data.columns.drop(TARGET_COLUMN).tolist()
LIST_LIB_FILE = './list_lib_version.txt'

with open(LIST_LIB_FILE, 'w') as lib_listfile:

    pipe = os.popen('pip freeze')

    str = pipe.read()

    lib_listfile.write(str)
#Удаляем все строчки с пропусками

#train_data = train_data.dropna()



def prepare_usage(data, train = True):

    

    #Немножечко медиан в значения которых нет.

    data.loc[(data['Annual Income'].isna()), ['Annual Income']] = data[~((data['Annual Income'].isna()) | (data['Annual Income'] == 0))]['Annual Income'].mean()

    data.loc[(data['Credit Score'].isna()), ['Credit Score']] = data[~((data['Credit Score'].isna()) | (data['Credit Score'] == 0))]['Credit Score'].mean()

    data.loc[(data['Months since last delinquent'].isna()), ['Months since last delinquent']] = data[~((data['Months since last delinquent'].isna()) | (data['Months since last delinquent'] == 0))]['Months since last delinquent'].mean()

    

    #Удаляем строки где целевая переменная = 0 и есть в колонке Nan

    #columns_to_drop_NAN = ['Annual Income', 'Years in current job', 'Months since last delinquent', 'Bankruptcies', 'Credit Score',]

    columns_to_drop_NAN = ['Years in current job', 'Bankruptcies',]

    for col in columns_to_drop_NAN:

        if train:

            data = data[~((data[col].isna()) & (data[TARGET_COLUMN] == 0))]

        else:

            pass

    

    #По остатку заполняем все Nan->0

    data = data.fillna(0)

    

    # Переводим на цифровые рельсы перечисленные колонки 

    columns_to_digit = ['Home Ownership', 'Years in current job', 'Purpose', 'Term', ]

    

    for name_colunm in columns_to_digit:

        groups_data = data.groupby(name_colunm)[[name_colunm]].sum()

        for index, val in enumerate(groups_data.axes[0]):

            data.loc[data[name_colunm] == val, name_colunm] = index



    return data



train_data = prepare_usage(train_data)



train_data.head(5)
train_data.isna().sum()
train_data.count()
X = train_data[FEATURES_COLUMNS]

y = train_data[TARGET_COLUMN]





plt.figure(figsize=(25, 5))



sns.countplot(x = TARGET_COLUMN, data = train_data)



plt.title('Target variable distribution')

plt.show()
%%time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



final_model = catb.CatBoostClassifier(

    auto_class_weights='Balanced',

    silent=True,

    depth=3,

    iterations=70,

    random_state=42)





#final_model = catb.CatBoostClassifier(n_estimators=500, max_depth=3, l2_leaf_reg=10,

#                                      silent=True, random_state=42)

final_model.fit(X_train, y_train)



y_train_pred = final_model.predict(X_train)

y_test_pred = final_model.predict(X_test)



get_classification_report(y_train, y_train_pred, y_test.values, y_test_pred)
important_features_top = show_feature_importances(X_train.columns, final_model.feature_importances_, get_top=15)
test_all_data.head(5)
test_all_data = prepare_usage(test_all_data, False)

test_all_data.head(5)
y_pred_final = final_model.predict(test_all_data)

# test DATA Learn





preds_final = pd.DataFrame()

preds_final = pd.DataFrame({'Id': np.arange(0,y_pred_final.shape[0]), 'Credit Default': y_pred_final})



preds_final.to_csv('predictions.csv', index=False, encoding='utf-8', sep=',')



preds_final.head(10)