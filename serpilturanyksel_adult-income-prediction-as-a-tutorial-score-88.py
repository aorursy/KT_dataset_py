

import pandas as pd

df = pd.read_csv('../input/adult-income/adult11.csv', sep=',')

df.head()
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 

              'occupation', 'relationship','race', 'gender', 'capital-gain', 'capital-loss', 

              'hours-per-week', 'native-country', 'salary']

df.head()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline 
df.info()
df.isnull().sum()
df.corr()
ax=sns.heatmap(df.corr(),annot=True, cmap="spring")

bottom, top = ax.get_ylim()
#Let's have a look at histograms.

df.hist(figsize=(18,8), color = "yellow")

plt.show()
fig,axarr = plt.subplots(nrows=1,ncols=2,figsize=(11,6))

sns.countplot(x="salary", data=df, palette='spring',ax=axarr[0])

df.salary.value_counts().plot.pie(autopct ="%1.1f%%",ax=axarr[1])

plt.title('Salary Karşılaştırması')

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,7))

sns.set(style="whitegrid")

sns.countplot(df['education'], order = df['education'].value_counts().index, palette='spring')

plt.xticks(rotation=70)

df['education'].value_counts()
plt.figure(figsize=(12,6))

sns.barplot(x="education", y="hours-per-week", data=df, hue="salary", palette='spring')

plt.xticks(rotation=70)
plt.figure(figsize=(12,8))

sns.pointplot(x="education", y="hours-per-week", hue='gender', palette="spring", linestyles=["-", "-."], data=df)

plt.xticks(rotation=80)
df.select_dtypes(exclude = 'category').plot(kind = 'box', figsize = (20,8))
df=df.drop("fnlwgt",axis=1)
plt.figure(figsize=(19,12))

num_feat = df.select_dtypes(include=['int64']).columns

for i in range(5):

    plt.subplot(2,3,i+1)

    plt.boxplot(df[num_feat[i]])

    plt.title(num_feat[i],color="g",fontsize=20)

    plt.yticks(fontsize=14)

    plt.xticks(fontsize=14)

plt.show()
#I wanted to look at the data against Capital Gain.

df.describe()
df[['education', 'education-num']].groupby(['education'], as_index=False).mean().sort_values(by='education-num', ascending=False)
df.drop(columns=['education'], inplace=True)
df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')

df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')



df.head(10)
X = df.drop(['salary'], axis=1)

y = df['salary']

X.select_dtypes(include='object').tail(20)
categorical_columns = [c for c in X.columns  if X[c].dtype.name == 'object']

for c in categorical_columns:

  X[c] = np.where(X[c] == ' ?', X[c].mode(), df[c])

X.select_dtypes(include='object').tail(20)
X = pd.concat([X, pd.get_dummies(X.select_dtypes(include='object'))], axis=1)

X = X.drop(['workclass', 'marital-status', 'occupation',

       'relationship', 'race', 'gender', 'native-country'], axis=1)

X.head()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix as cm

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

d_tree1 = DecisionTreeClassifier(max_depth = 3, random_state=42)

d_tree1.fit(X_train, y_train)

print ("Train data set size : ", X_train.shape)

print ("Test data set size : ", X_test.shape)
predictions = d_tree1.predict(X_test)

score = round(accuracy_score(y_test, predictions), 3)

cm1 = cm(y_test, predictions)

ax=sns.heatmap(cm1, annot=True, fmt=".0f", center=0, cmap="spring", linewidths="0.5"  )

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Accuracy Score: {0}'.format(score), size = 16)

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, target_names=['50Kdan düşük', '50Kdan yüksek']))
from sklearn.tree import export_graphviz

# resim göstermesi için

from IPython.display import Image

# bu çıktıyı yazdırmak yerine string olarak kaydetmek için 

from sklearn.externals.six import StringIO

# burasının konuyla alakası yok, çıktı için bir nesne yaratılıyor

dt_data =StringIO()

# ağacı dışarıya aktarıyoruz

export_graphviz(d_tree1, out_file=dt_data,filled=True,rounded=True,impurity=False,

               feature_names=X.columns, class_names=['<=50K','>50K'])

# pydotplus kütüphanesini çağıralım ve grafik yaratalım

import pydotplus

graph = pydotplus.graph_from_dot_data(dt_data.getvalue())

# bu grafiği gösterelim

Image(graph.create_png())
plt.figure(figsize=(16, 9))



from sklearn import ensemble



d_tree2 = DecisionTreeClassifier(max_depth = 8, random_state=42)

d_tree2.fit(X_train, y_train)

ranking = d_tree2.feature_importances_

features = np.argsort(ranking)[::-1][:10]

columns = X.columns



plt.title("Attribute Importance Ranking by Decision Tree", y = 1.03, size = 18)

plt.bar(range(len(features)), ranking[features], color="yellow", align="center")

plt.xticks(range(len(features)), columns[features], rotation=80)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix as cm

lr = LogisticRegression(random_state=42)

lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

score = round(accuracy_score(y_test, predictions), 3)

cm1 = cm(y_test, predictions)

ax=sns.heatmap(cm1, annot=True, fmt=".0f", center=0, cmap="spring", linewidths="0.5")

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Accuracy Score: {0}'.format(score), size = 15)

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))
from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(loss='exponential',learning_rate=0.01,n_estimators=200)

GB.fit(X_train,y_train)
predictions = GB.predict(X_test)

score = round(accuracy_score(y_test, predictions), 3)

cm1 = cm(y_test, predictions)

ax=sns.heatmap(cm1, annot=True, fmt=".0f", center=0, cmap="spring", linewidths="0.5")

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Accuracy Score: {0}'.format(score), size = 15)

plt.show()
y_pred = GB.predict(X_test)

print('Classification Report:')

print('\n')

print(classification_report(y_test,y_pred))
import xgboost as xgb

Xgb = xgb.XGBClassifier(learning_rate=0.1,n_estimators=500,max_depth=5,min_child_weight=4,random_state=42 )

Xgb.fit(X_train, y_train)

predictions = Xgb.predict(X_test)

XGBA = accuracy_score(y_test, predictions)
score = round(accuracy_score(y_test, predictions), 3)

cm1 = cm(y_test, predictions)

ax=sns.heatmap(cm1, annot=True, fmt=".0f", center=0, cmap="spring", linewidths="0.5")

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Accuracy Score: {0}'.format(score), size = 15)

plt.show()


Comparison = pd.DataFrame({d_tree1.score(X_test, y_test),

                              lr.score(X_test, y_test),

                       Xgb.score(X_test, y_test),

                       GB.score(X_test, y_test)})



Comparison.index = ['Decission Tree Score','Logistik Regression Score',

                       'XGBoosting Classification Score',

                       'Gradient Boosting Score']



Comparison
lastTable = pd.DataFrame({ "Real Salary": y_test[0:10],

                        "Decission Tree Score": d_tree1.predict(X_test)[0:10],

                        "Logistik Regression Score": lr.predict(X_test)[0:10],

                          "Gradient Boosting Score": GB.predict(X_test)[0:10],

                        "XGBoosting Classification Score": Xgb.predict(X_test)[0:10]}); lastTable