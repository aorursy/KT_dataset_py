# import packages

from sklearn import model_selection
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

%matplotlib inline
# Read train and test datasets

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Show dataset

train.head()
train.shape
train.dtypes
train.describe()
# Transform category columns

# Train
train['Sex'] = np.where(train.Sex == 'male', 1, 0)

label, unique = pd.factorize(train['Embarked'])
label = pd.DataFrame(label)
train['Embarked'] = label

# Test
test['Sex'] = np.where(test.Sex == 'male', 1, 0)

label, unique = pd.factorize(test['Embarked'])
label = pd.DataFrame(label)
test['Embarked'] = label

# Plot correlations

def plot_correlation_map( df ):
    corr = df.corr()
    ay, ax = plt.subplots(figsize =(10, 8))
    cmap = sns.diverging_palette(240, 10, as_cmap = True)
    ay = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

plot_correlation_map(train)
# Drop columns unimport (bad features)

train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Tranform Missing values

# Set Fare
test.loc[test.Fare.isnull() & test.Pclass==1, 'Fare'] = np.nanmedian(test.Fare[test.Pclass==1])
test.loc[test.Fare.isnull() & test.Pclass==2, 'Fare'] = np.nanmedian(test.Fare[test.Pclass==2])
test.loc[test.Fare.isnull() & test.Pclass==3, 'Fare'] = np.nanmedian(test.Fare[test.Pclass==3])

# Age with median
median_class1 = np.nanmedian(train.Age[train.Pclass==1])
median_class2 = np.nanmedian(train.Age[train.Pclass==2])
median_class3 = np.nanmedian(train.Age[train.Pclass==3])

# Set Age
# Train dataset
train.loc[(train.Pclass==1) & (train.Age.isnull()), 'Age']  = median_class1
train.loc[(train.Pclass==2) & (train.Age.isnull()), 'Age']  = median_class2
train.loc[(train.Pclass==3) & (train.Age.isnull()), 'Age']  = median_class3

# Test dataset
test.loc[(test.Pclass==1) & (test.Age.isnull()), 'Age']  = median_class1
test.loc[(test.Pclass==2) & (test.Age.isnull()), 'Age']  = median_class2
test.loc[(test.Pclass==3) & (test.Age.isnull()), 'Age']  = median_class3
# Check missing values again

test.isnull().sum().sort_values(ascending=False).head(15)
# Split Target and Features

x = train.drop('Survived', axis=1)
y = train['Survived']
# Features importances

from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier()
modelo.fit(x, y)

# Print results
print(x.columns)
print(modelo.feature_importances_)
# Solution with Decision Tree Classifier

# Define folds
num_folds = 10
num_instances = len(x)
seed = 7

# folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)

# Create model
modelo = DecisionTreeClassifier(max_depth=3, random_state=0)

# fit
modelo.fit(x, y)

# result
resultado = model_selection.cross_val_score(modelo, x, y, cv = kfold, scoring = 'accuracy')

# score
print(modelo.score(x, y))

# accuracy
print("Acurácia: %.3f (%.3f)" % (resultado.mean(), resultado.std()))

# 0.827160493827
# accuracy: 0.818 (0.055)
# The best solution - Gradient Boost Classifier


# Definindo os valores para o número de folds
num_folds = 15
num_instances = len(x)
seed = 7

# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)

# Create model
modelo = GradientBoostingClassifier()

# fit 
modelo.fit(x, y)

resultado = model_selection.cross_val_score(modelo, x, y, cv = kfold, scoring = 'accuracy')

print(modelo.score(x, y))
print("Acurácia: %.3f (%.3f)" % (resultado.mean(), resultado.std()))

# 0.901234567901
# accuracy: 0.828 (0.059)
# Save result

df_result = pd.DataFrame()
df_result['PassengerId'] = test['PassengerId']

test_predict = test.drop('PassengerId', axis=1)

df_result['Survived'] = modelo.predict(test_predict)
df_result.to_csv('result_v7.csv', index=False)