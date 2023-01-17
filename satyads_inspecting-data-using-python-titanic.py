import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import seaborn as sns
from sklearn import preprocessing
from yellowbrick.features import PCA
from yellowbrick.target import FeatureCorrelation
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
continious_variables = df[['Age','Fare','Survived']]
sns.pairplot(continious_variables,hue='Survived')
plt.show()
sns.lmplot(x="Age", y="Fare", hue="Survived", data=continious_variables)
plt.show()
continious_variables.corr()

continious_cat = df[['Age','Fare','Pclass','Sex','SibSp','Parch','Embarked','Survived']]
fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='Pclass', y='Age', hue='Survived',data=continious_cat, ax=axs[0])
sns.boxplot(x='Pclass', y='Fare', hue='Survived',data=continious_cat, ax=axs[1])
plt.show()
model = smf.ols(formula='Age ~ C(Pclass)', data=continious_cat)
res = model.fit()
print('ANOVA - Age~Class')
print(res.summary())
model = smf.ols(formula='Fare ~ C(Pclass)', data=continious_cat)
res = model.fit()
print('ANOVA - Fare~Class')
print(res.summary())
fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='Sex',y='Age', hue='Survived',data=continious_cat, ax=axs[0])
sns.boxplot(x='Sex', y='Fare', hue='Survived',data=continious_cat, ax=axs[1])
plt.show()
model = smf.ols(formula='Age ~ C(Sex)', data=continious_cat)
res = model.fit()
print('ANOVA - Age~Sex')
print(res.summary())
model = smf.ols(formula='Fare ~ C(Sex)', data=continious_cat)
res = model.fit()
print('ANOVA - Fare~Sex')
print(res.summary())
fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='SibSp', y='Age', hue='Survived',data=continious_cat, ax=axs[0])
sns.boxplot(x='SibSp',y='Fare', hue='Survived',data=continious_cat, ax=axs[1])
plt.show()
model = smf.ols(formula='Age ~ C(SibSp)', data=continious_cat)
res = model.fit()
print('ANOVA - Age~SibSp')
print(res.summary())
model = smf.ols(formula='Fare ~ C(SibSp)', data=continious_cat)
res = model.fit()
print('ANOVA - Fare~SibSp')
print(res.summary())
fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='Parch', y='Age', hue='Survived',data=continious_cat, ax=axs[0])
sns.boxplot(x='Parch', y='Fare', hue='Survived',data=continious_cat, ax=axs[1])
plt.show()
fig, axs = plt.subplots(ncols=2)
sns.boxplot(x='Embarked',y='Age', hue='Survived',data=continious_cat, ax=axs[0])
sns.boxplot(x='Embarked',y='Fare', hue='Survived',data=continious_cat, ax=axs[1])
plt.show()
categorical_data = df[['Pclass','Sex','SibSp','Parch','Embarked','Survived']] 


fig, axs = plt.subplots(ncols=2)
sns.countplot(x='Pclass', hue='Survived',data=categorical_data, ax=axs[0])
sns.countplot(x='Sex', hue='Survived',data=categorical_data, ax=axs[1])
plt.show()
fig, axs = plt.subplots(ncols=2)
sns.countplot(x='SibSp', hue='Survived',data=categorical_data, ax=axs[0])
sns.countplot(x='Parch', hue='Survived',data=categorical_data, ax=axs[1])
plt.show()
fig, axs = plt.subplots(ncols=1)
sns.countplot(x='Embarked', hue='Survived',data=categorical_data, ax=axs)
plt.show()
fig, axs = plt.subplots(ncols=3)
sns.countplot(x='Pclass', hue='Sex',data=categorical_data, ax=axs[0])
sns.countplot(x='Pclass', hue='Parch',data=categorical_data, ax=axs[1])
sns.countplot(x='Pclass', hue='Embarked',data=categorical_data, ax=axs[2])
plt.show()
kendall_corr,p_val = kendalltau(x=df.Pclass,y=df.Sex)
print(kendall_corr,p_val)
df[['Pclass','Sex','SibSp','Parch','Embarked','Survived']].corr(method='kendall')
print(df[['Age','Fare','Pclass','Sex','SibSp','Parch','Embarked','Survived']].shape,
      df[['Age','Fare','Pclass','Sex','SibSp','Parch','Embarked','Survived']].dropna().shape)
df = df[['Age','Fare','Pclass','Sex','SibSp','Parch','Embarked','Survived']]
df = df.dropna()
df.Sex = pd.get_dummies(df.Sex,drop_first=True)
df[['Embarked_Q','Embarked_S']] = pd.get_dummies(df.Embarked,drop_first=True)
df = df.drop(['Embarked'],axis=1)
df.head()
label_binarizer = preprocessing.LabelBinarizer()
X,y = df.loc[:, df.columns != 'Survived'],label_binarizer.fit_transform(df[['Survived']]).flatten()
visualizer = PCA(scale=True,projection=3)
visualizer.fit_transform(X, y)
visualizer.show()
visualizer = PCA(scale=True,proj_features=True,projection=2)
visualizer.fit_transform(X, y)
visualizer.show()
visualizer = FeatureCorrelation(method='mutual_info-regression', labels=X.columns)

visualizer.fit(X, y, random_state=0)
visualizer.show()










