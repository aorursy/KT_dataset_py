# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Data display coustomization

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
wine = pd.read_csv(r"/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

wine.head()
wine_dub = wine.copy()



# Checking for duplicates and dropping the entire duplicate row if any

wine_dub.drop_duplicates(subset=None, inplace=True)

wine_dub.shape
wine.shape
#Duplicate Found 
wine=wine_dub.copy()

del wine_dub

wine.shape
wine.shape
wine.info()
wine.describe()
wine.isnull().sum()
wine.isnull().sum(axis=1)
sns.violinplot('fixed acidity',data=wine)

plt.show()
percentiles = wine['fixed acidity'].quantile([0.05,0.95]).values

wine['fixed acidity'][wine['fixed acidity'] <= percentiles[0]] = percentiles[0]

wine['fixed acidity'][wine['fixed acidity'] >= percentiles[1]] = percentiles[1]
sns.violinplot('fixed acidity',data=wine)

plt.show()
wine.quality.nunique()
sns.violinplot(y='fixed acidity',x='quality',data=wine)

plt.show()
sns.violinplot('volatile acidity',data=wine)

plt.show()
percentiles = wine['volatile acidity'].quantile([0.05,0.95]).values

wine['volatile acidity'][wine['volatile acidity'] <= percentiles[0]] = percentiles[0]

wine['volatile acidity'][wine['volatile acidity'] >= percentiles[1]] = percentiles[1]
sns.violinplot('volatile acidity',data=wine)

plt.show()
sns.violinplot(y='volatile acidity',x='quality',data=wine)

plt.show()
sns.violinplot('citric acid',data=wine)

plt.show()
percentiles = wine['citric acid'].quantile([0.01,0.99]).values

wine['citric acid'][wine['citric acid'] <= percentiles[0]] = percentiles[0]

wine['citric acid'][wine['citric acid'] >= percentiles[1]] = percentiles[1]
sns.violinplot('citric acid',data=wine)

plt.show()
sns.violinplot(y='citric acid',x='quality',data=wine)

plt.show()
sns.violinplot('residual sugar',data=wine)

plt.show()
percentiles = wine['residual sugar'].quantile([0.1,0.9]).values

wine['residual sugar'][wine['residual sugar'] <= percentiles[0]] = percentiles[0]

wine['residual sugar'][wine['residual sugar'] >= percentiles[1]] = percentiles[1]
sns.violinplot('residual sugar',data=wine)

plt.show()
sns.violinplot(y='residual sugar',x='quality',data=wine)

plt.show()
sns.violinplot('chlorides',data=wine)

plt.show()
percentiles = wine['chlorides'].quantile([0.1,0.9]).values

wine['chlorides'][wine['chlorides'] <= percentiles[0]] = percentiles[0]

wine['chlorides'][wine['chlorides'] >= percentiles[1]] = percentiles[1]
sns.violinplot('chlorides',data=wine)

plt.show()
sns.violinplot(y='chlorides',x='quality',data=wine)

plt.show()
sns.violinplot('free sulfur dioxide',data=wine)

plt.show()
percentiles = wine['free sulfur dioxide'].quantile([0.05,0.95]).values

wine['free sulfur dioxide'][wine['free sulfur dioxide'] <= percentiles[0]] = percentiles[0]

wine['free sulfur dioxide'][wine['free sulfur dioxide'] >= percentiles[1]] = percentiles[1]
sns.violinplot('free sulfur dioxide',data=wine)

plt.show()
sns.violinplot(y='free sulfur dioxide',x='quality',data=wine)

plt.show()
sns.violinplot('total sulfur dioxide',data=wine)

plt.show()
percentiles = wine['total sulfur dioxide'].quantile([0.05,0.95]).values

wine['total sulfur dioxide'][wine['total sulfur dioxide'] <= percentiles[0]] = percentiles[0]

wine['total sulfur dioxide'][wine['total sulfur dioxide'] >= percentiles[1]] = percentiles[1]
sns.violinplot('total sulfur dioxide',data=wine)

plt.show()
sns.violinplot(y='total sulfur dioxide',x='quality',data=wine)

plt.show()
sns.violinplot('density',data=wine)

plt.show()
percentiles = wine['density'].quantile([0.01,0.99]).values

wine['density'][wine['density'] <= percentiles[0]] = percentiles[0]

wine['density'][wine['density'] >= percentiles[1]] = percentiles[1]
sns.violinplot('density',data=wine)

plt.show()
sns.violinplot(y='density',x='quality',data=wine)

plt.show()
sns.violinplot('pH',data=wine)

plt.show()
percentiles = wine['pH'].quantile([0.05,0.95]).values

wine['pH'][wine['pH'] <= percentiles[0]] = percentiles[0]

wine['pH'][wine['pH'] >= percentiles[1]] = percentiles[1]
sns.violinplot('pH',data=wine)

plt.show()
sns.violinplot(y='pH',x='quality',data=wine)

plt.show()
sns.violinplot('sulphates',data=wine)

plt.show()
percentiles = wine['sulphates'].quantile([0.05,0.95]).values

wine['sulphates'][wine['sulphates'] <= percentiles[0]] = percentiles[0]

wine['sulphates'][wine['sulphates'] >= percentiles[1]] = percentiles[1]
sns.violinplot('sulphates',data=wine)

plt.show()
sns.violinplot(y='sulphates',x='quality',data=wine)

plt.show()
sns.violinplot('alcohol',data=wine)

plt.show()
percentiles = wine['alcohol'].quantile([0.01,0.99]).values

wine['alcohol'][wine['alcohol'] <= percentiles[0]] = percentiles[0]

wine['alcohol'][wine['alcohol'] >= percentiles[1]] = percentiles[1]
sns.violinplot('alcohol',data=wine)

plt.show()
sns.violinplot(y='alcohol',x='quality',data=wine)

plt.show()
ax=sns.countplot('quality',data=wine)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()
wine.describe()
plt.figure(figsize = (10,5))

sns.heatmap(wine.corr(), annot = True, cmap="rainbow")

plt.show()
from sklearn.model_selection import train_test_split

train,test = train_test_split(wine, train_size=0.7, random_state=1)

train.shape,test.shape
X_train=train.drop('quality',axis=1)

X_test=test.drop('quality',axis=1)

y_train=train['quality']

y_test=test['quality']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[:]=scaler.fit_transform(X_train[:])

X_test[:]=scaler.transform(X_test[:])
from catboost import CatBoostClassifier

model=CatBoostClassifier()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

categorical_features_indices
model.fit(X_train,y_train,cat_features=([]))
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score,roc_auc_score

score_cbc=model.score(X_test,y_test)

print('Score :',score_cbc)
roc_auc_score_cbc=roc_auc_score(y_test,model.predict_proba(X_test),multi_class='ovr')

print('Compute Area Under the Receiver Operating Characteristic Curve',roc_auc_score_cbc)
confusion_matrix(y_test,model.predict(X_test))
print(classification_report(y_test,model.predict(X_test)))
cohen_kappa_score_cbc=cohen_kappa_score(model.predict(X_test),y_test)

print('Cohen’s kappa :',cohen_kappa_score_cbc)
print('Score for CatBoostClassifier:                                                        ',score_cbc)

print('Compute Area Under the Receiver Operating Characteristic Curve for CatBoostClassifier',roc_auc_score_cbc)

print('Cohen’s kappa for CatBoostClassifier:                                                ',cohen_kappa_score_cbc)



    
print('Confusion Matrix for CatBoostClassifier:')

print(confusion_matrix(y_test,model.predict(X_test)))
print('           Classification Report for CatBoostClassifier:')

print(classification_report(y_test,model.predict(X_test))) 