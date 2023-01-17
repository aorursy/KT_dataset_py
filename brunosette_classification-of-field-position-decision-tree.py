import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn import tree

from datetime import datetime

import os



def load_dataset():

    fifa_filepath = '../input/data.csv'

    data = pd.read_csv(fifa_filepath)

    data.head()

    

    # Seleciona apenas algumas features de interesse

    df2 = data.loc[:, 'Crossing':'Release Clause']

    df1 = data[['Age', 'Overall', 'Value', 'Wage', 'Preferred Foot', 'Skill Moves', 'Position', 'Height', 'Weight']]

    df = pd.concat([df1, df2], axis=1)

    # Excluit todos os exemplos que possuem features ausentes

    df = df.dropna()

    



    # REaliza alguns procedimentos para a padronização de algumas features

    def value_to_int(df_value):

        try:

            value = float(df_value[1:-1])

            suffix = df_value[-1:]

    

            if suffix == 'M':

                value = value * 1000000

            elif suffix == 'K':

                value = value * 1000

        except ValueError:

            value = 0

        return value

    # Realiza alguns procedimentos para a padronização de algumas features

    df['Value_float'] = df['Value'].apply(value_to_int)

    df['Wage_float'] = df['Wage'].apply(value_to_int)

    df['Release_Clause_float'] = df['Release Clause'].apply(lambda m: value_to_int(m))

    

    def weight_to_int(df_weight):

        value = df_weight[:-3]

        return value

      

    df['Weight_int'] = df['Weight'].apply(weight_to_int)

    df['Weight_int'] = df['Weight_int'].apply(lambda x: int(x))

    

    def height_to_int(df_height):

        try:

            feet = int(df_height[0])

            dlm = df_height[-2]

            if dlm == "'":

                height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)

            elif dlm != "'":

                height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)

        except ValueError:

            height = 0

        return height

    

    df['Height_int'] = df['Height'].apply(height_to_int)

    

    df = df.drop(['Value', 'Wage', 'Release Clause', 'Weight', 'Height'], axis=1)

    

    # Label encoder na feature Preferred Foot

    le_foot = preprocessing.LabelEncoder()

    df["Preferred Foot"] = le_foot.fit_transform(df["Preferred Foot"].values)

    

    # Transforma o problema em um problema de 3 classes, separadas por setor do campo

    for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:

      df.loc[df.Position == i , 'Pos'] = 'Strikers' 

    

    for i in ['CAM', 'CDM', 'LCM', 'CM', 'LAM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM']:

      df.loc[df.Position == i , 'Pos'] = 'Midfielder' 

    

    for i in ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB','GK']:

      df.loc[df.Position == i , 'Pos'] = 'Defender' 

    

    return df
df = load_dataset()

df.head()
df.describe()
df.info()
plt.figure(figsize=(12, 8))

plt.title("Quantidade de jogadores por posição")

plt.xlabel("Área de atuaçãos")

plt.ylabel("Quantidade de jogadores")

fig = sns.countplot(x = 'Pos', data =df)
# Set up the matplotlib figure

f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False)

sns.despine(left=True)

sns.boxplot('Pos', 'Overall', data = df, ax=axes[0, 0])

sns.boxplot('Pos', 'HeadingAccuracy', data = df, ax=axes[0, 1])

sns.boxplot('Pos', 'ShortPassing', data = df, ax=axes[1, 1])

sns.boxplot('Pos', 'Weight_int', data = df, ax=axes[1, 0])

mean_value_per_age = df.groupby('Age')['Value_float'].mean()

p = sns.barplot(x = mean_value_per_age.index, y = mean_value_per_age.values)

p = plt.xticks(rotation=90)
mean_wage_per_age = df.groupby('Age')['Wage_float'].mean()

p = sns.barplot(x = mean_wage_per_age.index, y = mean_wage_per_age.values)

p = plt.xticks(rotation=90)


sns.jointplot(x='Age', y="Overall", data=df, kind="kde")

sns.jointplot(x='Value_float', y="Overall", data=df)
sns.lineplot(x='Value_float', y="Overall", data=df)
ax = sns.scatterplot(x="ShortPassing", y="Finishing", hue="Pos",data=df)
df_new = df[['Overall',

'BallControl',

'Acceleration',

'LongShots',

'Aggression',

'Pos']]



sns.pairplot(df_new, kind="scatter", hue="Pos") 

plt.show()

df_new = df[['Height_int',

'ShortPassing',

'Finishing',

'Volleys',

'HeadingAccuracy',

'Pos']]



sns.pairplot(df_new, kind="scatter", hue="Pos") 

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split





target_names = df["Pos"].unique()

le_class = preprocessing.LabelEncoder()

df['Pos'] = le_class.fit_transform(df['Pos'])



y = df["Pos"]



df.drop(columns=["Position","Pos"],inplace=True)





X_train_dev, X_test, y_train_dev, y_test = train_test_split(df, y, 

                                                    test_size=0.20, 

                                                    random_state=42 )

print(X_train_dev.shape)

print(X_test.shape)



print(X_train_dev.info())



def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
def plot_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    

    Arguments

    ---------

    confusion_matrix: numpy.ndarray

        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 

        Similarly constructed ndarrays can also be used.

    class_names: list

        An ordered list of class names, in the order they index the given confusion matrix.

    figsize: tuple

        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,

        the second determining the vertical size. Defaults to (10,7).

    fontsize: int

        Font size for axes labels. Defaults to 14.

        

    Returns

    -------

    matplotlib.figure.Figure

        The resulting confusion matrix figure

    """

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    sns.set(font_scale=1.4)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16})

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig
tr_acc = []

mln_set = range(75,90) #Setando os valores em que iremos testar os nós folha, inicio e fim											

# min_impurity = 82/100000                                #Anterior



for minImp in mln_set:

    clf = tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=minImp/100000) #controlando o valor máximo de nós folha

    scores = cross_val_score(clf, X_train_dev, y_train_dev, cv=10)

    tr_acc.append(scores.mean())



best_mln = mln_set[np.argmax(tr_acc)]

print(best_mln)
from sklearn import tree

start_time = timer(None)

clf = tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=best_mln/100000)

clf = clf.fit(X_train_dev, y_train_dev)

timer(start_time)

preds = clf.predict(X_test)

cf = confusion_matrix(y_test,preds)



print(plot_confusion_matrix(cf, class_names=target_names))



print(" Acc: ",accuracy_score(y_test, preds))

from sklearn.ensemble import BaggingClassifier



start_time = timer(None)

clf = BaggingClassifier(tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=best_mln/100000))#

clf = clf.fit(X_train_dev, y_train_dev)

timer(start_time)

preds = clf.predict(X_test)



cf = confusion_matrix(y_test,preds)



print(plot_confusion_matrix(cf, class_names=target_names))



print(" Acc: ",accuracy_score(y_test, preds))

dtrain = xgb.DMatrix(X_train_dev, label=y_train_dev)



dtest = xgb.DMatrix(X_test,label=y_test)



param = {

    'max_depth': 3,  # the maximum depth of each tree

    'eta': 0.3,  # the training step for each iteration

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'num_class': 3}  # the number of classes that exist in this datset

num_round = 50  # the number of training iterations

start_time = timer(None)

bst = xgb.train(param, dtrain, num_round)

bst.dump_model('dump.raw.txt')

preds = bst.predict(dtest)

best_preds = np.asarray([np.argmax(line) for line in preds])

timer(start_time)

cf = confusion_matrix(y_test, best_preds)



print(plot_confusion_matrix(cf, class_names=target_names))

print(" Acc: ",accuracy_score(y_test, best_preds))


