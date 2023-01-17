import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')

print("Setup Complete")
import pandas as pd

data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
data.info()
data['SeniorCitizen'] = data['SeniorCitizen'].astype('object')

data['Churn'] = data['Churn'].replace({'No': 0, "Yes": 1}).astype('int64')

data['TotalCharges'] = data['TotalCharges'].replace(' ', 0).astype('float64')

data.info()
plt.figure(figsize=(16,9))

data['Churn'].value_counts().plot(kind='pie', autopct='%.2f%%' )

#plt.axis('equal')
sns.catplot(x='gender', hue = 'Churn', data=data, kind='count', orient='h')
df = data.copy()

plt.figure(figsize=(30,10))

sns.lineplot(x='tenure', y='MonthlyCharges', hue='Churn', data=data)
#g = sns.FacetGrid(df, col = 'Churn', height = 5, aspect = 5)

#ax = g.map(sns.barplot,'tenure', 'MonthlyCharges', palette='Blues_d')
df_zero = df.loc[(data.Churn == 0), :]

plt.figure(figsize=(30,8))

sns.barplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df)
plt.figure(figsize=(30,8))

sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df)
#def cat_graphs(column):

   

 #   data[column].value_counts().plot(kind='pie', autopct='%.2f%%')

# plt.axis('equal');

    

  #  sns.catplot(x=column, hue='Churn', data=data, kind='count')
categoricals = [c for c in df.columns if df[c].dtype == 'object']

df_categoricals = df[categoricals].copy()

df_categoricals = df_categoricals.drop(columns='customerID', axis = 1)

df_categoricals.info()
for col in df_categoricals.columns:

    

    df_categoricals = pd.concat([df_categoricals, pd.get_dummies(df_categoricals[col], prefix = col)], axis = 1)

    df_categoricals = df_categoricals.drop(columns = col, axis = 1)
df_categoricals.head()
numericals = [c for c in df.columns if df[c].dtypes == 'int64' or df[c].dtypes == 'float64' ]



df_numericals = df[numericals].copy()

df_numericals.info()
df = pd.concat([df_numericals, df_categoricals], axis = 1)
df.head()
y = df['Churn']

x = df.drop(columns='Churn')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

#from sklearn.linear_model import LogisticRegression



clf = RandomForestClassifier(n_estimators = 10000 ,max_features='sqrt', max_depth = 8, random_state=42)

clf.fit(x_train, y_train)



y_pred = clf.predict(x_test)



print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test,y_pred))

print("Recall:", metrics.recall_score(y_test,y_pred))
# importando as bibliotecas dos modelos

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



# criando uma lista com todos os modelos

classifiers = [

    KNeighborsClassifier(3),

    GaussianNB(),

    LogisticRegression(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier()]



# criando uma funçào para rodas o pipeline 

for clf in classifiers:

    

    # ajustando o modelo

    clf.fit(x_train, y_train)

    # armazenando o nome do modelo

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados

    print('****Results****')

    # fazendo predições

    # calculando as métricas

    y_pred = clf.predict(x_test)

    # imprimindo as métricas

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print("Precision:", metrics.precision_score(y_test, y_pred))

    print("Recall:", metrics.recall_score(y_test, y_pred))
from sklearn.model_selection import GridSearchCV



parameters = {

             'criterion': ('gini', 'entropy'),

             'max_depth' : range(1,20,2),

             'min_samples_split': range(10, 500, 20)

             }



clf_tree = DecisionTreeClassifier()



clf = GridSearchCV(clf_tree, parameters, verbose=1)



clf.fit(x,y)







print('Best parameters:' + str(clf.best_params_))



print('****Results****')



y_pred = clf.predict(x_test)



print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

print('Precision:', metrics.precision_score(y_test, y_pred))

print('Recall:', metrics.recall_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

#from sklearn.linear_model import LogisticRegression



clf = RandomForestClassifier(n_estimators = 10000 , max_depth = 9, random_state=42, min_samples_split = 270)

clf.fit(x_train, y_train)



y_pred = clf.predict(x_test)



print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test,y_pred))

print("Recall:", metrics.recall_score(y_test,y_pred))
imp = pd.Series(data=clf.feature_importances_, index=x.columns).sort_values(ascending=False)

plt.figure(figsize=(10,12))

plt.title("Feature importance")

ax = sns.barplot(y=imp.index, x=imp.values, palette="Blues_d", orient='h')