# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#read training set

train_set= pd.read_csv("../input/train.csv")

test_set= pd.read_csv("../input/test.csv")



# Store our test passenger IDs for easy access

PassengerId = test_set['PassengerId']


display(train_set.head())

display(train_set.describe())

display(train_set.info())
train_set.describe(include=['O'])

#description des types "Objets"

#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html

train_set[['Age', 'Fare']].describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
display(train_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

train_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).count().sort_values(by='Survived', ascending=False)
%matplotlib inline

import seaborn

seaborn.set() 



#-------------------Survived/Died by Class -------------------------------------

survived_class = train_set[train_set['Survived']==1]['Pclass'].value_counts()

dead_class = train_set[train_set['Survived']==0]['Pclass'].value_counts()

df_class = pd.DataFrame([survived_class,dead_class])

df_class.index = ['Survived','Died']

df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Class")



Class1_survived= df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100

Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100

Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100

print("Percentage of Class 1 that survived:" ,round(Class1_survived),"%")

print("Percentage of Class 2 that survived:" ,round(Class2_survived), "%")

print("Percentage of Class 3 that survived:" ,round(Class3_survived), "%")



# display table

display(df_class)

display(survived_class)
# ou plus simplement

train_set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# feature correlation

#drop not number/category features

correl_train = train_set.drop(['PassengerId', 'Name', 'Ticket', 'Cabin' ], axis=1)

correl_train['Sex01']=(correl_train['Sex']=='male')



corr=correl_train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 2, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



correl_train.head()
with sns.axes_style("white"):

    sns.jointplot(x=np.minimum(train_set['Fare'] , 75) , y=train_set['Age'], kind="hex", color="k");

            

with sns.axes_style("white"):

    sns.jointplot(x=np.minimum(train_set['Fare'] , 75) , y=train_set['Pclass'], kind="hex", color="k");

#En visualisant la variance:

sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_set);
#Copy to ensure we don't poluate initial set

easy_correl_train = correl_train.copy()

easy_correl_train['Age'][easy_correl_train['Age'].isnull()] = -10

#correl_train.dropna()

easy_correl_train.loc[:,'Fare']= np.minimum(easy_correl_train['Fare'] , 100)

display(easy_correl_train.describe())

display(correl_train.describe())
#correl_train[['Fare', 'Pclass']]

#correl_train.dropna()

sns.pairplot(easy_correl_train)
#Approximation gaussiennes

g = sns.PairGrid(easy_correl_train)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);
# easy start : 3 variables : sex, age, class

#Bucket / categorise age

# replace sex by binary number for easier modeling (2 categories -> no need for dummyfication)

# splits & categories can be parametrised to optimise split

def simplify_ages(df, bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120), group_names = [10, 0, 1, 2, 3, 4, 5, 6]):

                  #['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']):

    #remove NA & void

    idx=df['Age'].isnull()

    df['Age'][idx] = -0.5

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_set(df):

    df=simplify_ages(df)

    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    return df



featured_set=simplify_set(train_set[["Survived","Age","Sex","Pclass"]].copy())

simple_test_set=simplify_set(test_set[["Age","Sex","Pclass"]].copy())



sns.barplot(x="Age", y="Survived", hue="Sex", data=featured_set);
grid = sns.FacetGrid(featured_set, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()

#plt.hist2d(easy_correl_train['Pclass'], easy_correl_train['Age'])

#x=easy_correl_train['Pclass'], y=easy_correl_train['Age'], c=easy_correl_train['Survived'])

#simple_set=easy_correl_train[['Pclass'],['Age']].groupby(['Pclass','Age']).count()

#simple_set.head()
#from https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont





cv = KFold(n_splits=10)            # Desired number of Cross Validation folds

accuracies = list()

max_attributes = len(list(featured_set))

depth_range = range(1, max_attributes + 1)



# Testing max_depths from 1 to max attributes

# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:

    fold_accuracy = []

    tree_model = tree.DecisionTreeClassifier(max_depth = depth)

    print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(featured_set):

        f_train = featured_set.loc[train_fold] # Extract train data with cv indices

        f_valid = featured_set.loc[valid_fold] # Extract valid data with cv indices



        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 

                               y = f_train["Survived"]) # We fit the model with the fold train data

        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 

                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    print("Accuracy per fold: ", fold_accuracy, "\n")

    # print("Average accuracy: ", avg)

    print("\n")

    

# Just to show results conveniently

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models

y_train = featured_set['Survived']

x_train = featured_set.drop(['Survived'], axis=1).values 

x_test = simple_test_set.values



depth_target=4



# Create Decision Tree with max_depth

decision_tree = tree.DecisionTreeClassifier(max_depth = depth_target)

decision_tree.fit(x_train, y_train)



# Predicting results for test dataset

y_pred = decision_tree.predict(x_test)

submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)



# Export our trained model as a .dot file

with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(decision_tree,

                              out_file=f,

                              max_depth = depth_target,

                              impurity = True,

                              feature_names = list(featured_set.drop(['Survived'], axis=1)),

                              class_names = ['Died', 'Survived'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)

draw.text((10, 0), # Drawing offset (position)

          'tree title', # Text to draw

          (0,0,255), # RGB desired color

          font=font) # ImageFont object with desired font

img.save('sample-out.png')

PImage("sample-out.png")



# Code to check available fonts and respective paths

# import matplotlib.font_manager

# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')