import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random as rnd

import os

import warnings



from operator import add



MEDIUM_SIZE = 10

SMALL_SIZE = 8

MEDIUM_SIZE = 10

BIGGER_SIZE = 12



%matplotlib inline

warnings.filterwarnings('ignore')



print(os.listdir("../input"))

os.chdir("../input")
df = pd.read_csv('train.csv')

df.head()
df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)

print(f'The dataset contains {df.shape[0]} samples, {df.shape[1] - 1} independent features 1 target binary variable.')
print(df.info())

missing_values = (df.isnull().sum() / len(df)) * 100

print("\nFeatures with missing values (in %): \n", missing_values[missing_values > 0])
df.drop(['Cabin'], axis = 1, inplace = True)

df.describe()
print(np.char.center(" Unique values ", 60, fillchar = "*"))

print("\nPclass: ", sorted(df.Pclass.unique()))

print("SibSp: ", sorted(df.SibSp.unique()))

print("Parch: ", sorted(df.Parch.unique()))

print("Sex: ", sorted(df.Sex.unique()))

print("Embarked: ", df.Embarked.unique())
df.Age.fillna(df.dropna().Age.mean(), inplace = True)
def draw_semi_pie_chart(data, column, fig, title, renamed_index_dict = None):

    default_colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666']

    rnd.shuffle(default_colors)

    

    if renamed_index_dict is None:

        ax = df[column].value_counts()

    else:

        ax = df[column].value_counts().rename(index = renamed_index_dict)

    

    ax = ax.plot.pie(colors = default_colors, autopct='%1.1f%%', startangle=90, title = title, pctdistance=0.85)

    ax.set_ylabel('')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):

        item.set_fontsize(20)

    centre_circle = plt.Circle((0,0), 0.70, fc='white')

    fig.gca().add_artist(centre_circle)
fig = plt.gcf()

fig.set_size_inches(18, 12)

grid_rows = 2

grid_cols = 3



# Draw Pclass Pie chart

plt.subplot(grid_rows, grid_cols, 1)

draw_semi_pie_chart(df, 'Pclass', fig, 'Passenger Ticket Class', {1: '1st', 2: '2nd', 3: '3rd'})



# Draw Sex chart

plt.subplot(grid_rows, grid_cols, 2)

draw_semi_pie_chart(df, 'Sex', fig, title = 'Sex')



# Draw Embarked  chart

plt.subplot(grid_rows, grid_cols, 3)

draw_semi_pie_chart(df, 'Embarked', fig, 'Port of Embarkation', {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})



# Draw SibSp chart

plt.subplot(grid_rows, grid_cols, 4)

df.SibSp.value_counts().plot(kind = 'bar', title = 'No of Siblings/Spouse')



# Draw Parch chart

plt.subplot(grid_rows, grid_cols, 5)

df.Parch.value_counts().plot(kind = 'bar', title = 'No of Parents/Children')



fig.tight_layout()

plt.show()
def create_percent_stacked_barchart(data, title = None, ylabel = None, xlabel = None):

    default_colors = ['#219AD8', '#019600', '#3C5F5A']

    # From raw value to percentage

    totals = data.sum(axis=1)

    bars = ((data.T / totals) * 100).T

    r = list(range(data.index.size))



    # Plot

    barWidth = 0.95

    names = data.index.tolist()

    bottom = [0] * bars.shape[0]



    # Create bars

    color_index = 0

    plots = []

    for bar in bars.columns:

        plots.append(plt.bar(r, bars[bar], bottom=bottom, color=default_colors[color_index], edgecolor='white', width=barWidth))

        bottom = list(map(add, bottom, bars[bar]))

        color_index = 0 if color_index >= len(default_colors) else color_index + 1



    # Custom x axis

    plt.title(title)

    plt.xticks(r, names)

    plt.xlabel(data.index.name if xlabel is None else xlabel)

    plt.ylabel(data.columns.name if ylabel is None else ylabel)

    ax = plt.gca()

        

    y_labels = ax.get_yticks()

    ax.set_yticklabels([str(y) + '%' for y in y_labels])



    flat_list = [item for sublist in data.T.values for item in sublist]

    for i, d in zip(ax.patches, flat_list):

        data_label = str(d) + " (" + str(round(i.get_height(), 2)) + "%)"

        ax.text(i.get_x() + 0.45, i.get_y() + 5, data_label, horizontalalignment='center', verticalalignment='center', fontdict = dict(color = 'white', size = 15))



    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):

        item.set_fontsize(20)

    

    legend = ax.legend(plots, bars.columns.tolist(), fancybox=True)

    plt.setp(legend.get_texts(), fontsize='large')
fig = plt.gcf()

fig.set_size_inches(25, 25)

grid_rows = 3

grid_cols = 2



# Draw Survival vs Pclass

plt.subplot(grid_rows, grid_cols, 1)

temp = df[['Pclass','Survived']].groupby(['Pclass','Survived']).size().unstack('Survived')

temp.rename(index={1: '1st', 2: '2nd', 3: '3rd'}, columns={0:'Not Survived', 1:'Survived'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Survival vs Ticket Class', ylabel = 'Passengers', xlabel = 'Ticket Class')



# Draw Survival vs SibSp

plt.subplot(grid_rows, grid_cols, 2)

temp = df[['SibSp','Survived']].groupby(['SibSp','Survived']).size().unstack('Survived')

temp.rename(columns={0:'Not Survived', 1:'Survived'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Survival vs No of Siblings/Spouse', ylabel = 'Passengers', xlabel = 'No of Siblings/Spouse')



# Draw Survival vs Parch

plt.subplot(grid_rows, grid_cols, 3)

temp = df[['Parch','Survived']].groupby(['Parch','Survived']).size().unstack('Survived')

temp.rename(columns={0:'Not Survived', 1:'Survived'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Survival vs No of Parents/Children', ylabel = 'Passengers', xlabel = 'No of Parents/Children')



# Draw Survival vs Sex

plt.subplot(grid_rows, grid_cols, 4)

temp = df[['Sex','Survived']].groupby(['Sex','Survived']).size().unstack('Survived')

temp.rename(columns={0:'Not Survived', 1:'Survived'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Survival vs Sex', ylabel = 'Passengers')



# Draw Survival vs Embarked

plt.subplot(grid_rows, grid_cols, 5)

temp = df[['Embarked','Survived']].groupby(['Embarked','Survived']).size().unstack('Survived')

temp.rename(index = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}, columns={0:'Not Survived', 1:'Survived'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Survival vs Embarked', ylabel = 'Passengers')



fig.tight_layout()

plt.show()
fig = plt.gcf()

fig.set_size_inches(10, 5)

sns.heatmap(df.corr(), annot = True)

plt.show()
continuous_features = ['Fare', 'Parch', 'SibSp', 'Age']

number_of_columns = len(continuous_features)

number_of_rows = 5

plt.figure(figsize=(23, 18))



for i, f in enumerate(continuous_features):

    plt.subplot(number_of_rows + 1, number_of_columns, i + 1)

    sns.distplot(df[f], kde=True)
sns.pairplot(df, hue = 'Survived', markers=["o", "s"], vars = ['Fare', 'Parch', 'SibSp', 'Age'], palette = sns.color_palette("bright", 10))
nominal_features = ['Sex', 'Embarked', 'Pclass']

x = pd.get_dummies(df.drop(['Survived'], axis = 1), columns = nominal_features, drop_first=True).values

y = df.Survived.values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
print("Shape of X before Dimensionlity Reduction: ", x_train.shape)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()

x_train = lda.fit_transform(x_train, y_train)

x_test = lda.transform(x_test)



print("Shape of X after Dimensionlity Reduction: ", x_train.shape)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

rfclassifier = RandomForestClassifier(bootstrap = True, max_depth = 2, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 7)

rfclassifier.fit(x_train, y_train)

y_pred_rf = rfclassifier.predict(x_test)

print("Trained")
from sklearn.metrics import confusion_matrix



print("Random Forest Confusion Matrix")

cm = confusion_matrix(y_test, y_pred_rf)

print(cm)
from sklearn.metrics import accuracy_score

print("Accuracy: ", accuracy_score(y_test, y_pred_rf))
df_test = pd.read_csv('test.csv')

df_test.head()



df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

df_test.Age.fillna(df_test.dropna().Age.mean(), inplace = True)

df_test.Fare.fillna(df_test.dropna().Fare.mean(), inplace = True)



x_t = pd.get_dummies(df_test, columns = nominal_features, drop_first=True).values



x_t = sc.transform(x_t)

x_t = lda.transform(x_t)

y_pred_t = rfclassifier.predict(x_t)



df_test = pd.read_csv('test.csv').PassengerId

df_test['Survived'] = pd.Series(df_test)
df_test.to_csv('/submission.csv', index = False)

df_test.head()