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
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.shape
df.head()
df.isnull().values.any()
df = df.rename(columns={'Outcome': 'Diabetes'})
df.head()
def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr (df)
df.corr()
num_true = len(df.loc[df['Diabetes'] == True])
num_false = len(df.loc[df['Diabetes'] == False])
print('Número de Casos Verdadeiros: {0} ({1:2.2f}%)'.format(num_true, (num_true/ (num_true + num_false)) *100))
print('Número de Casos Falsos     : {0} ({1:2.2f}%)'.format(num_false, (num_false/ (num_false + num_true)) *100))

from sklearn.model_selection import train_test_split
df.columns
variaveis = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI',
       'DiabetesPedigreeFunction', 'Age']
target = ['Diabetes']
x = df[variaveis].values
y = df[target].values
split_test_size = 0.30
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size = split_test_size)
print('{0:0.2f}% nos dados de treino'.format((len(x_treino) /len(df.index)) *100))
print('{0:0.2f}% nos dados de teste'.format((len(x_teste) /len(df.index)) *100))
df.columns
print('# Linhas no df {0}'.format(len(df)))
print('# Linhas missing Glucose: {0}'.format(len(df.loc[df['Glucose'] ==0])))
print('# Linhas missing BloodPressure: {0}'.format(len(df.loc[df['BloodPressure'] ==0])))
print('# Linhas missing SkinThickness: {0}'.format(len(df.loc[df['SkinThickness'] ==0])))
print('# Linhas missing Insulin: {0}'.format(len(df.loc[df['Insulin'] ==0])))
print('# Linhas missing BMI: {0}'.format(len(df.loc[df['BMI'] ==0])))
print('# Linhas missing DiabetesPedigreeFunction: {0}'.format(len(df.loc[df['DiabetesPedigreeFunction'] ==0])))
print('# Linhas missing Age: {0}'.format(len(df.loc[df['Age'] ==0])))
from sklearn.impute import SimpleImputer
preenche_0 = SimpleImputer(missing_values = 0, strategy = 'mean')
x_treino = preenche_0.fit_transform(x_treino)
x_teste = preenche_0.fit_transform(x_teste)
x_treino
from sklearn.naive_bayes import GaussianNB
modelo_v1 = GaussianNB()
modelo_v1.fit(x_treino, y_treino.ravel())
from sklearn import metrics
predict_target = modelo_v1.predict(x_treino)
print ('Exatidão (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_treino, predict_target) *100))
print()

predict_test = modelo_v1.predict(x_teste)
print ('Exatidão (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_teste, predict_test) *100))
print()
print('Confusion Matrix')

print('{0}'.format(metrics.confusion_matrix(y_teste, predict_test, labels = [1,0])))
print('')

print('Classification Report')
print(metrics.classification_report(y_teste, predict_test, labels = [1,0]))
from sklearn.ensemble import RandomForestClassifier
modelo_v2 = RandomForestClassifier()
modelo_v2.fit(x_treino, y_treino.ravel())
rf_predict_treino = modelo_v2.predict(x_treino)
print ('Exatidão (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_treino, rf_predict_treino) *100))
print()
rf_predict_test = modelo_v2.predict(x_teste)
print ('Exatidão (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_teste, rf_predict_test) *100))
print()
print('Confusion Matrix')

print('{0}'.format(metrics.confusion_matrix(y_teste, rf_predict_test, labels = [1,0])))
print('')

print('Classification Report')
print(metrics.classification_report(y_teste, rf_predict_test, labels = [1,0]))

from sklearn.linear_model import LogisticRegression
modelo_v3 = LogisticRegression()
modelo_v3.fit(x_treino, y_treino.ravel())
lm_predict_treino = modelo_v3.predict(x_treino)
print ('Exatidão (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_treino, lm_predict_treino) *100))
print()
lm_predict_test = modelo_v2.predict(x_teste)
print ('Exatidão (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_teste, rf_predict_test) *100))
print()
print('Confusion Matrix')

print('{0}'.format(metrics.confusion_matrix(y_teste, lm_predict_test, labels = [1,0])))
print('')

print('Classification Report')
print(metrics.classification_report(y_teste, lm_predict_test, labels = [1,0]))
print('Resumo:')
print ('Exatidão do GaussianNB (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_teste, predict_test) *100))
print ('Exatidão do RandomForestClassifier (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_teste, rf_predict_test) *100))
print ('Exatidão do LogisticRegression (Accuracy): {0:.2f}%'.format(metrics.accuracy_score(y_teste, lm_predict_test) *100))
print()