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
import pandas as pd

import numpy as np



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, learning_curve

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb, catboost as catb



import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
test = pd.read_csv("../input/test_kaggle.csv")

df = pd.read_csv("../input/train_kaggle.csv")
def data_clean(data):

    data = pd.concat([data, pd.get_dummies(data['Home Ownership'])], axis=1)

    data = data.drop('Home Ownership', axis=1)

    data = data.drop('Have Mortgage', axis=1)

    years_map = {

    '< 1 year': 0,

    '1 year': 1,

    '2 years': 2,

    '3 years': 3,

    '4 years': 4,

    '5 years': 5,

    '6 years': 6,

    '7 years': 7,

    '8 years': 8,

    '9 years': 9,

    '10+ years': 10,

    'nan': 10,

}

    Years_j_mode = data['Years in current job'].mode()

    Years_j_mode_test = data['Years in current job'].mode()

    data['Years in current job'] = data['Years in current job'].fillna('nan')

    data['Years in current job'] = data['Years in current job'].map(lambda v: years_map[v])

    data['Tax Liens'] = data['Tax Liens'].fillna(1.0)

    data['Tax Liens'] = data['Tax Liens'].map({0.0:0, 1.0:1, 2.0:1, 3.0:1, 4.0:1, 5.0:1, 6.0:1, 7.0:1})

    data = pd.concat([data, pd.get_dummies(data['Tax Liens'])], axis=1)

    data = data.drop('Tax Liens', axis=1)

    data.rename(columns={0: 'Tax Liens_0', 1: 'Tax Liense_1'}, inplace=True)

    data['Months since last delinquent'] = data['Months since last delinquent'].fillna(0)

    data['Months since last delinquent'].fillna(0)

    def get_delinquent_category(months):

        if np.isnan(months):

            return 'no problem'

        elif months < 18:

            return 'last year problem'

        else:

            return 'old problem'

    data['Months since last delinquent category'] = data['Months since last delinquent'].map(get_delinquent_category)

    data['Months since last delinquent category'] = data['Months since last delinquent category'].astype('category')

    data = data.drop('Months since last delinquent', axis=1)

    data = pd.concat([data, pd.get_dummies(data['Months since last delinquent category'])], axis=1)

    data = data.drop('Months since last delinquent category', axis=1)

    data['Bankruptcies'] = data['Bankruptcies'].fillna(0.0)

    data['Bankruptcies'] = data['Bankruptcies'].map({0.0:0, 1.0:1, 2.0:1, 3.0:1, 4.0:1})

    data = pd.concat([data, pd.get_dummies(data['Bankruptcies'])], axis=1)

    data = data.drop('Bankruptcies', axis=1)

    data.rename(columns={0: 'Bankruptcies_0', 1: 'Bankruptcies_1'}, inplace=True)

    data = data.drop('Purpose', axis=1)

    data = pd.concat([data, pd.get_dummies(data['Term'])], axis=1)

    data = data.drop('Term', axis=1) 

    return data
df = data_clean(df)

test = data_clean(test)
plt.figure(figsize = (25,20))



sns.set(font_scale=1.4)

sns.heatmap(df.corr().round(3), annot=True, linewidths=.5, cmap='GnBu')



plt.title('Correlation matrix')

plt.show()
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
important_features_top = show_feature_importances(X_train.columns, model.feature_importances_, get_top=21)
df = df.loc[df['Annual Income'].notna(), :]
df.columns
ft = (['Annual Income', 'Years in current job',

       'Number of Open Accounts', 'Years of Credit History',

       'Maximum Open Credit', 'Number of Credit Problems',

       'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt',

       'Credit Score', 'Home Mortgage', 'Own Home', 'Rent',

       'Tax Liens_0', 'Tax Liense_1', 'last year problem',

       'Bankruptcies_0', 'Bankruptcies_1', 'Long Term', 'Short Term'])
target = 'Credit Default'
X = df[ft]

y = df[target]



X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=211)
def grid_search_catboost():

    catmodel = catb.CatBoostClassifier(

        random_state=21,

        silent=True,

    #     loss_function='Logloss',

        eval_metric='F1',

    )



    parameters = {

        'max_depth': [6, 8, 10],

    #     'learning_rate': [0.01, 0.05, 0.1],

        'iterations': [30, 50, 100],

    #     'n_estimators': [450, 479, 500],

    #     'class_weights': [1.5, 3.5],

        'border_count': [30, 35, 40],

        'bagging_temperature': [0, 10, 20, 50],

    }



    cgrid = GridSearchCV(estimator=catmodel, param_grid=parameters, cv=2)

    cgrid.fit(X_train, y_train)



    print(cgrid.best_score_)

    print(cgrid.best_params_)



grid_search_catboost()
model = catb.CatBoostClassifier(

            n_estimators=500,

            max_depth=6,

            class_weights=[1.5, 3.5], 

            learning_rate=0.04,  

            silent=True, 

            random_state=21)



model.fit(X_train, y_train)



y_pred = model.predict(X_test)



acc = accuracy_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print(f'acc: {acc}, recall: {rec}, precision: {prec}, f1: {f1}')

print(classification_report(y_test, y_pred))
pred_test = model.predict(test)
test['Credit Default'] = pred_test
test['Credit Default'] = test['Credit Default'].astype(int)