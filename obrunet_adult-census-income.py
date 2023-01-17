import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV
df = pd.read_csv('../input/adult.csv')

df.head()
df.shape
df.info()
df.duplicated().sum()
df = df.drop_duplicates()

df.shape
cat_feat = df.select_dtypes(include=['object']).columns

cat_feat
print('% of missing values :')

for c in cat_feat:

    perc = len(df[df[c] == '?']) / df.shape[0] * 100

    print(c, f'{perc:.1f} %')
df.describe()
# Taking a look at the target (income) without distinction of sex

print(f"Ratio above 50k : {(df['income'] == '>50K').astype('int').sum() / df.shape[0] * 100 :.2f}%")
num_feat = df.select_dtypes(include=['int64']).columns

num_feat
plt.figure(1, figsize=(16,10))

sns.pairplot(data=df, hue='sex')

plt.show()
plt.figure(figsize=(18,10))

plt.subplot(231)



i=0

for c in num_feat:

    plt.subplot(2, 3, i+1)

    i += 1

    sns.kdeplot(df[df['sex'] == 'Female'][c], shade=True, )

    sns.kdeplot(df[df['sex'] == 'Male'][c], shade=False)

    plt.title(c)



plt.show()
plt.figure(figsize=(18,25))

plt.subplot(521)



i=0

for c in cat_feat:

    plt.subplot(5, 2, i+1)

    i += 1

    sns.countplot(x=c, data=df, hue='sex')

    plt.title(c)



plt.show()
# nb of female / male

nb_female = (df.sex == 'Female').astype('int').sum()

nb_male = (df.sex == 'Male').astype('int').sum()

nb_female, nb_male
# nb of people earning more or less than 50k per gender

nb_male_above = len(df[(df.income == '>50K') & (df.sex == 'Male')])

nb_male_below = len(df[(df.income == '<=50K') & (df.sex == 'Male')])

nb_female_above = len(df[(df.income == '>50K') & (df.sex == 'Female')])

nb_female_below = len(df[(df.income == '<=50K') & (df.sex == 'Female')])

nb_male_above, nb_male_below, nb_female_above, nb_female_below
print(f'Among Males   : {nb_male_above/nb_male*100:.0f}% earn >50K // {nb_male_below/nb_male*100:.0f}% earn <=50K')

print(f'Among Females : {nb_female_above/nb_female*100:.0f}% earn >50K // {nb_female_below/nb_female*100:.0f}% earn <=50K')
# normalization

nb_male_above /= nb_male 

nb_male_below /= nb_male

nb_female_above /= nb_female

nb_female_below /= nb_female

nb_male_above, nb_male_below, nb_female_above, nb_female_below
print(f'Among people earning >50K  : {nb_male_above / (nb_male_above + nb_female_above) *100 :.0f}% are Females and {nb_female_above / (nb_male_above + nb_female_above) *100 :.0f}% are Males')

print(f'Among people earning =<50K : {nb_male_below / (nb_male_below + nb_female_below) *100 :.0f}% are Females and {nb_female_below / (nb_male_below + nb_female_below) *100 :.0f}% are Males')
df['US native'] = (df['native.country'] == 'United-States').astype('int')

plt.figure(figsize=(6,4))

sns.countplot(x='US native', data=df, hue='sex')

plt.show()
plt.figure(figsize=(6,4))

sns.countplot(x='income', data=df, hue='US native')

plt.show()
# nb of people earning more or less than 50k per origin

nb_native_above = len(df[(df.income == '>50K') & (df['US native'] == 1)])

nb_native_below = len(df[(df.income == '<=50K') & (df['US native'] == 1)])

nb_foreign_above = len(df[(df.income == '>50K') & (df['US native'] == 0)])

nb_foreign_below = len(df[(df.income == '<=50K') & (df['US native'] == 0)])

nb_native_above, nb_native_below, nb_foreign_above, nb_foreign_below
nb_native = (df['US native'] == 1).astype('int').sum()

nb_foreign = df.shape[0] - nb_native

nb_native, nb_foreign
print(f'Among natives    : {nb_native_above/nb_native*100:.0f}% earn >50K // {nb_native_below/nb_native*100:.0f}% earn <=50K')

print(f'Among foreigners : {nb_foreign_above/nb_foreign*100:.0f}% earn >50K // {nb_foreign_below/nb_foreign*100:.0f}% earn <=50K')
# normalization

nb_native_above /= nb_native

nb_native_below /= nb_native

nb_foreign_above /= nb_foreign

nb_foreign_below /= nb_foreign

nb_native_above, nb_native_below, nb_foreign_above, nb_foreign_below
print(f'Among people earning >50K  : {nb_native_above / (nb_native_above + nb_foreign_above) *100 :.0f}% are natives and {nb_foreign_above / (nb_native_above + nb_foreign_above) *100 :.0f}% are foreigners')

print(f'Among people earning =<50K : {nb_native_below / (nb_native_below + nb_foreign_below) *100 :.0f}% are natives and {nb_foreign_below / (nb_native_below + nb_foreign_below) *100 :.0f}% are foreigners')
num_feat = df.select_dtypes(include=['float', 'int']).columns

num_feat
sns.set(style="white")



# Compute the correlation matrix

corr = df[num_feat].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(7, 6))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
df['income'] = pd.get_dummies(df['income'], prefix='income', drop_first=True)
y = df.income

df = df.drop(columns=['income'])
print(f'Ratio above 50k:  {y.sum()/len(y)*100:.2f}%')
#cat_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
df.head()
cols = list(df.columns)

cols
selected_feat = cols.copy()

selected_feat.remove('US native')

selected_feat
df_final = df[selected_feat]
cat_feat = df_final.select_dtypes(include=['object']).columns

X = pd.get_dummies(df_final[cat_feat], drop_first=True)
#X = pd.concat([df_final[continuous_columns], df_dummies], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
def print_score(model, name):

    model.fit(X_train, y_train)

    print('Accuracy score of the', name, f': on train = {model.score(X_train, y_train)*100:.2f}%, on test = {model.score(X_test, y_test)*100:.2f}%')
print_score(LogisticRegression(), 'LogisticReg')
print_score(DecisionTreeClassifier(), 'DecisionTreeClf')
rf = RandomForestClassifier().fit(X_train, y_train)

print(f'Accuracy score of the RandomForrest: on train = {rf.score(X_train, y_train)*100:.2f}%, on test = {rf.score(X_test, y_test)*100:.2f}%')
# fit an Extra Tree model to the data

print_score(DecisionTreeClassifier(), 'ExtraTreesClf')
rfc = RandomForestClassifier()

param_grid = { 

    'n_estimators': [50, 100, 150, 200, 250],

    'max_features': [1, 2, 3, 4, 5],

    'max_depth' : [4, 6, 8]

}
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

rfc_cv.fit(X_train, y_train)
rfc_cv.best_params_
rfc_best = RandomForestClassifier(max_depth=8, max_features=5, n_estimators=250).fit(X_train, y_train)

print(f'Accuracy score of the RandomForrest: on train = {rfc_best.score(X_train, y_train)*100:.2f}%, on test = {rfc_best.score(X_test, y_test)*100:.2f}%')
# indexes of columns which are the most important

np.argsort(rf.feature_importances_)[-16:]
# most important features

[list(X.columns)[i] for i in np.argsort(rf.feature_importances_)[-16:]][::-1]
# Feature importances

features = X.columns

importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

num_features = len(importances)



# Plot the feature importances of the tree

plt.figure(figsize=(16, 4))

plt.title("Feature importances")

plt.bar(range(num_features), importances[indices], color="g", align="center")

plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')

plt.xlim([-1, num_features])

plt.show()



# Print values

for i in indices:

    print ("{0} - {1:.3f}".format(features[i], importances[i]))
(pd.Series(rf.feature_importances_, index=X_train.columns)

   .nlargest(15)

   .plot(kind='barh'))
extree = ExtraTreesClassifier().fit(X_train, y_train)

(pd.Series(extree.feature_importances_, index=X_train.columns)

   .nlargest(15)

   .plot(kind='barh'))