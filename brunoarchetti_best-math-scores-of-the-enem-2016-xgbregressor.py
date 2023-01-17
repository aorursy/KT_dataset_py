import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor #ML

# Gráficos
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
%matplotlib inline

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# Loading datasets
train = pd.read_csv("/kaggle/input/train.csv")
test = pd.read_csv("/kaggle/input/test.csv")
print(train.shape,test.shape)
train.head()
#Checking missing values
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)
# Age x Score Math
df_counts = train.groupby(['NU_NOTA_MT', 'NU_IDADE']).size().reset_index(name='counts')

# Stripplot
fig, ax = plt.subplots(figsize=(16,10), dpi=80)    
sns.stripplot(df_counts.NU_IDADE, df_counts.NU_NOTA_MT, size=df_counts.counts*2, ax=ax)


plt.title('Relação entre Idade e Nota em Matemática', fontsize=22)
plt.show()
# Pizza Graphic for gender
df = train.groupby('TP_SEXO').size()

df.plot(kind='pie', subplots=True, figsize=(8, 4), autopct='%1.2f%%')
plt.title("Genero")
plt.ylabel("")
plt.show()
# Checking target
train['NU_NOTA_MT'].describe()
#Features more correlation
aux = train.copy()
aux2 = train.copy()

aux = aux.loc[:, test.columns]
aux['NU_NOTA_MT'] = aux2.NU_NOTA_MT

c = aux.corr()
c.NU_NOTA_MT.sort_values()
#Creating model
new_vector_training = [
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5',
    'NU_NOTA_COMP3',
    'NU_NOTA_REDACAO',
    'NU_NOTA_LC',
    'NU_NOTA_CH',
    'NU_NOTA_CN',
    'NU_NOTA_MT'
]

train_final = train.copy()
train_final = train_final.loc[:, new_vector_training]
train_final.dropna(subset=['NU_NOTA_MT'], inplace=True)
train_final.head()
#Creating predict
new_vector_test = [
    'NU_INSCRICAO',
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5',
    'NU_NOTA_COMP3',
    'NU_NOTA_REDACAO',
    'NU_NOTA_LC',
    'NU_NOTA_CH',
    'NU_NOTA_CN'
]

test_final = test.copy()
test_final1 = test_final.loc[:, new_vector_test]
test_final2 = test_final.loc[:, new_vector_test]

test_final1.drop(['NU_INSCRICAO'], axis=1, inplace=True)
y = train_final.NU_NOTA_MT
X = train_final.drop(['NU_NOTA_MT'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# Model XGBRegressor
model = XGBRegressor(n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)

# Accuracy
acc = round(model.score(X_train, y_train) * 100, 2)
print('Accuracy model:',acc, "\n")
test_final1.head()
Y_pred = model.predict(test_final1)
result = pd.DataFrame({'NU_INSCRICAO': test_final2['NU_INSCRICAO'], 'NU_NOTA_MT': Y_pred})
result.isnull().sum()
result.head()
result['NU_NOTA_MT'].describe()
submission = result.loc[: , ['NU_INSCRICAO', 'NU_NOTA_MT']]
submission.to_csv('answer.csv', index=False)