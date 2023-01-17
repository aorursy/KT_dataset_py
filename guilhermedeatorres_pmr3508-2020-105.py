import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.ma import MaskedArray
import sklearn.utils.fixes
from sklearn.preprocessing import LabelEncoder


sklearn.utils.fixes.MaskedArray = MaskedArray

%matplotlib inline
plt.style.use('seaborn')

#Importa as bibliotecas
df_train = pd.read_csv("train_data.csv", na_values = '?')
df_train.set_index('Id',inplace=True)
df_train.head()


df = pd.read_csv("train_data.csv", index_col=['Id'], na_values="?")

#Realiza as leituras do excel
df.head()

#Verifica os dados
df.describe()

#Função para verificar o comportamento estatístico
df_analysis = df.copy()
encoder = LabelEncoder()
df_analysis['income'] = encoder.fit_transform(df_analysis['income'])
df_analysis['income']
mask = np.triu(np.ones_like(df_analysis.corr(), dtype=np.bool))

plt.figure(figsize=(10,10))

sns.heatmap(df_analysis.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='autumn')
plt.show()

#Gráficos para verificar o comportamento de cada variável em função da renda
sns.distplot(df_analysis['age']);
sns.catplot(x="income", y="hours.per.week", data=df_analysis);
sns.catplot(x="income", y="hours.per.week", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="education.num", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="age", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.gain", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.loss", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.gain", data=df_analysis);
sns.catplot(y="sex", x="income", kind="bar", data=df_analysis);
sns.catplot(y="workclass", x="income", kind="bar", data=df_analysis);
sns.catplot(y="marital.status", x="income", kind="bar", data=df_analysis);
sns.catplot(y="occupation", x="income", kind="bar", data=df_analysis);
sns.catplot(y="native.country", x="income", kind="bar", data=df_analysis);
df = df.drop(['fnlwgt'], axis=1)
df.head()
Y = df.pop('income')
X = df
Y.head()
cols = list(X.select_dtypes(include = [np.number]).columns.values)
print (cols)
cols.remove('capital.gain')
cols.remove('capital.loss')
print (cols)
sparse_cols = ['capital.gain', 'capital.loss']
categorical = list(X.select_dtypes(exclude=[np.number]).columns.values)
print (categorical)
from sklearn.preprocessing import OneHotEncoder


one_hot = OneHotEncoder(sparse=False)


