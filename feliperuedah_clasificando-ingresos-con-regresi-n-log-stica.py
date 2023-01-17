import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

%matplotlib inline
census = pd.read_csv('../input/adult-census-income/adult.csv')

census.head()
census.info()
census.replace('?', np.nan, inplace=True)

census.isna().sum()
null_rows = census.isnull().any(axis=1).sum()

null_perc = 100 * null_rows / len(census)

print('Hay {c} registros con valores nulos. Esto representa un {p:.2f}% del total de registros'.format(c=null_rows, p=null_perc))
census = census.dropna()

census.isnull().sum()
census['income'].value_counts()
census['income'] = census['income'].apply(lambda x: 0 if x=='<=50K' else 1)

census['income'].value_counts()
numeric_cols = census.select_dtypes(include=['int', 'float']).columns

numeric_cols = numeric_cols.drop('income')

numeric_cols
census[numeric_cols] = (census[numeric_cols] - census[numeric_cols].mean()) / census[numeric_cols].std()

census.head()
# Columnas con datos categoricos

categorical_features = ['workclass', 'education', 'marital.status', 

                        'occupation', 'relationship', 'race', 

                        'sex', 'native.country']



for feature in categorical_features:

    # One hot encoding

    dummy = pd.get_dummies(census[feature], drop_first=True)

    

    # Juntar a la tabla

    census = pd.concat([census, dummy], axis=1)

    

    # Eliminar columna pasada

    census = census.drop(feature, axis=1)

    

census.head()
# Features para el modelo

X = census.drop(['income'], axis=1)



# Columna target

y = census['income']



# Distribución de la columna target

y.value_counts(normalize=True) * 100
# Data split

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
y_train.value_counts(normalize=True) * 100
y_test.value_counts(normalize=True) * 100
from sklearn.metrics import accuracy_score



def train_test(features):

    # Entrenar modelo

    model = LogisticRegression()

    model.fit(X_train[features], y_train)



    # Hacer predicciones

    train_pred = model.predict(X_train[features])

    test_pred = model.predict(X_test[features])

    

    # Train accuracy

    train_acc = accuracy_score(y_train, train_pred)

    

    # Test accuracy

    test_acc = accuracy_score(y_test, test_pred)

    

    return(train_acc, test_acc)
t = pd.DataFrame(columns=['features', 'train_acc', 'test_acc'], index=range(1, 11))

t
best_features = []



for i in range(1, 11):

    to_check = list(set(all_features) - set(best_features))

    

    train_accuracies = pd.Series(index=to_check)

    test_accuracies = pd.Series(index=to_check)

    

    for feature in to_check:

        features = best_features + [feature]



        train_acc, test_acc = train_test(features)

        train_accuracies[feature] = train_acc

        test_accuracies[feature] = test_acc

        

    train_accuracies = train_accuracies.sort_values(ascending=False)

    best_feature = train_accuracies.index[0]

    best_features.append(best_feature)

    

    t.loc[i, 'features'] = list(best_features)

    t.loc[i, 'train_acc'] = train_accuracies[best_feature]

    t.loc[i, 'test_acc'] = test_accuracies[best_feature]

    

t
plt.plot(t['test_acc'], color='blue', label='Test')

plt.plot(t['train_acc'], color='orange', label='Train')

plt.xlabel('Número de features')

plt.ylabel('Accuracy')

plt.title('Accuracy por modelo')

plt.legend()
t.sort_values('test_acc', ascending=False).head(3)
print(t.loc[9, 'features'])

print('Test accuracy: ', t.loc[9, 'test_acc']) 