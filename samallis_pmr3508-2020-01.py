# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #
import seaborn as sns

%matplotlib inline
plt.style.use('seaborn')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from subprocess import check_output
print(check_output(["ls", "../input/adult-pmr3508"]).decode("utf8"))
df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", 
                      na_values = '?',
                      sep=r'\s*,\s*',
                      engine='python')
print ('Tamanho do DataFrame: ', df_train.shape)
df_train.head()
total = df_train.isnull().sum().sort_values(ascending = False)
percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])
missing_data.head()
print('occupation:\n')
print(df_train['occupation'].describe())

print('\n\nworkclass:\n')
print(df_train['workclass'].describe())

print('\n\nnative.country:\n')
print(df_train['native.country'].describe())
print ('% de aparição dos dados do atributo "occupation" \n')
print ((df_train['workclass'].value_counts()/df_train['workclass'].count())*100)
print ('\n')
print ('% de aparição dos dados do atributo "workclass"\n')
print ((df_train['native.country'].value_counts()/df_train['native.country'].count())*100)
print ('\n')
print ('% de aparição dos dados do atributo "native.country" \n')
print ((df_train['occupation'].value_counts()/df_train['occupation'].count())*100)
value = df_train['workclass'].describe().top
df_train['workclass'] = df_train['workclass'].fillna(value)

value = df_train['native.country'].describe().top
df_train['native.country'] = df_train['native.country'].fillna(value)

value = df_train['occupation'].describe().top
df_train['occupation'] = df_train['occupation'].fillna(value)
total = df_train.isnull().sum().sort_values(ascending = False)
percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])
missing_data.head()
sns.pairplot(df_train,diag_kws={'bw':"1.0"}, hue="income")
df_train.describe()
plt.figure(figsize=(13, 7))
df_train['capital.gain'].hist(color = 'blue')
plt.xlabel('Capital Gain')
plt.ylabel('Quantidade')
plt.title('Capital Gain')
plt.figure(figsize=(13, 7))
df_train['capital.loss'].hist(color = 'red')
plt.xlabel('Capital Loss')
plt.ylabel('Quantidade')
plt.title('Capital Loss')
plt.figure(figsize=(13, 7))
df_train['age'].hist(color = 'coral')
plt.xlabel('Idade')
plt.ylabel('Quantidade')
plt.title('Idade')
sns.set()
plt.figure(figsize=(13,7))
sns.distplot(df_train['age'], color = 'coral', bins = 70)
plt.xlabel('Idade')
plt.ylabel('Quantidade')
plt.title('Distribuição de Idades')
plt.figure(figsize=(13, 7))
df_train['hours.per.week'].hist(color = 'purple')
plt.xlabel('h/semana')
plt.ylabel('Quantidade')
plt.title('Horas / Semana')
super_work = df_train[df_train['hours.per.week'] > 40]
plt.figure(figsize=(13, 7))
super_work['hours.per.week'].hist(color = 'brown', bins = 5)
plt.xlabel('h/semana')
plt.ylabel('Quantidade')
plt.title('Trabalhadores com mais de 40h semanais')
mean = super_work['hours.per.week'].describe()['mean']
print('{0} horas por semana ({1} horas por dia com finais de semana livre).'.format(int(mean), int(mean/5)))
sns.catplot(y="hours.per.week", x="age", kind="bar", data=df_train, aspect=3, height=5)
plt.xlabel('Idade')
plt.ylabel('Horas/Semana')
plt.title('Quantidade de Trabalhadores e Horas Semanais')
def compare_histogram(df, obj_var, test_var, obj_labels = None, alpha = 0.7):
    
    if obj_labels is None:
        obj_labels = df[obj_var].unique()
    
    #obj_var = 'income'
    #obj_labels = ['>50K', '<=50K']
    #test_var = 'age' (for example)
    
    temp = []
    n_labels = len(obj_labels)
    for i in range(n_labels):
        temp.append(df[df[obj_var] == obj_labels[i]])
        temp[i] = np.array(temp[i][test_var]).reshape(-1,1)

    fig = plt.figure(figsize= (13,7))
    
    for i in range(n_labels):
        plt.hist(temp[i], alpha = alpha)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    plt.title('Histogram over \'' + test_var + '\' filtered by \'' + obj_var + '\'')
    plt.legend(obj_labels)
compare_histogram(df_train, 'income', 'age')
plt.figure(figsize=(13, 7))
df_train['sex'].value_counts().plot(kind = 'pie')
plt.title('Percentual de Homens e mulheres')
# Verificando a quantidade de homens
masc = df_train[df_train['sex'] == 'Male'].count()[0]

# Verificando a quantidade de mulheres
fema = df_train.shape[0] - masc
print("Há {0} homens e {1} mulheres nesta empresa, ou seja, apenas {2:3.2f}% são mulheres.".format(male, female, female*100/(female+male)))
compare_histogram(df_train, 'income', 'sex')
plt.xlabel('Sexo')
plt.ylabel('Quantidade')
plt.title('Homens e Mulheres e a Distribuição de Salários')
#Homens = [qtd > 50K, qtd <= 50K]
male_income = []
temp = df_train[df_train['sex'] == 'Male']
male_income.append(temp[temp['income'] == '>50K'].count()[0])
male_income.append(male-male_income[0])

# Mulheres = [qtd > 50K, qtd <= 50K]
female_income = []
temp = df_train[df_train['sex'] == 'Female']
female_income.append(temp[temp['income'] == '>50K'].count()[0])
female_income.append(female-female_income[0])

# % homens que recebem mais que 50k:
male_over = male_income[0]/male

# % de mulheres que recebem mais de 50k:
female_over = female_income[0]/female
print('Há {0:1.2f}% homens que possuem renda anual superior a 50.000, já entre as mulheres, há apenas {1:2.2f}% que possuem essa mesma renda.'.format(male_over*100, female_over*100))
compare_histogram(df_train, 'sex', 'hours.per.week')
plt.xlabel('Sexo')
plt.ylabel('Quantidade')
plt.title('Homens e Mulheres - Horas Semanais')
female = df_train[df_train['sex'] == 'Female']
male = df_train[df_train['sex'] == 'Male']
plt.figure(figsize=(13, 7))
male['occupation'].value_counts().plot(kind = 'bar', color = 'yellow')
plt.ylabel('Quantidade')
plt.title('Homens: Ocupação de cargos')

plt.figure(figsize=(13, 7))
female['occupation'].value_counts().plot(kind = 'bar', color = 'silver')
plt.ylabel('Quantidade')
plt.title('Mulheres: Ocupação de cargos')
plt.figure(figsize=(13, 7))
df_train['race'].value_counts().plot(kind = 'pie')
plt.title('Percentual Étnico')
compare_histogram(df_train, 'income', 'race')
plt.xlabel('Etnias')
plt.ylabel('Quantidade')
plt.title('Etnias - Horas Semanais')
# Etnias
white = df_train[df_train['race'] == 'White'].count()[0]
black = df_train[df_train['race'] == 'Black'].count()[0]
amer = df_train[df_train['race'] == 'Amer-Indian-Eskimo'].count()[0]
other = df_train[df_train['race'] == 'Other'].count()[0]
asian = df_train[df_train['race'] == 'Asian-Pac-Islander'].count()[0]

# Etnias e seus perceituais de salários anuais com mais ou menos que 50k
white_income = []
temp = df_train[df_train['race'] == 'White']
white_income.append(temp[temp['income'] == '>50K'].count()[0])
white_income.append(white-white_income[0])

black_income = []
temp = df_train[df_train['race'] == 'Black']
black_income.append(temp[temp['income'] == '>50K'].count()[0])
black_income.append(black-black_income[0])

amer_income = []
temp = df_train[df_train['race'] == 'Amer-Indian-Eskimo']
amer_income.append(temp[temp['income'] == '>50K'].count()[0])
amer_income.append(amer-amer_income[0])

asian_income = []
temp = df_train[df_train['race'] == 'Asian-Pac-Islander']
asian_income.append(temp[temp['income'] == '>50K'].count()[0])
asian_income.append(asian-asian_income[0])

other_income = []
temp = df_train[df_train['race'] == 'Other']
other_income.append(temp[temp['income'] == '>50K'].count()[0])
other_income.append(other-other_income[0])
print('Brancos:\n \n  {0:1.2f}% recebem mais de 50k anuais\n'.format(white_income[0]*100/white))
print('Negros:\n \n  {0:1.2f}% recebem mais de 50k anuais\n'.format(black_income[0]*100/black))
print('Amer-Indian-Eskimo:\n \n  {0:1.2f}% recebem mais de 50k anuais\n'.format(amer_income[0]*100/amer))
print('Asian-Pac-Islander:\n \n  {0:1.2f}% recebem mais de 50k anuais\n'.format(asian_income[0]*100/asian))
print('Outros:\n \n  {0:1.2f}% recebem mais de 50k anuais'.format(other_income[0]*100/other))
white = df_train[df_train['race'] == 'White']
black = df_train[df_train['race'] == 'Black']
plt.figure(figsize=(13, 7))
white['occupation'].value_counts().plot(kind = 'bar', color = 'blue')
plt.ylabel('Quantidade')
plt.title('Brancos: Ocupações - Histograma')

plt.figure(figsize=(13, 7))
black['occupation'].value_counts().plot(kind = 'bar', color = 'red')
plt.ylabel('Quantidade')
plt.title('Negros: Ocupações - Histograma')
var1 = 'race'
var2 = 'education.num'

data = pd.concat([df_train[var2], df_train[var1]], axis=1)

f, ax = plt.subplots(figsize=(15, 7))

sns.boxplot(x=var1, y=var2, data=data, notch = True)
plt.ylabel('Quantidade')
plt.xlabel('Etnia')
plt.title('Educação e Etnia')