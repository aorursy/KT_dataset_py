import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import xlrd



pd.set_option("display.max.columns", None)

sns.set_style("whitegrid")

 

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head()
def woman_child_or_man(passenger):

    age, sex = passenger

    if age < 16:

        return "child"

    else:

        return dict(male="man", female="woman")[sex]
df_train["Class"] = df_train.Pclass.map({1: "First", 2: "Second", 3: "Third"})

df_train["Who"] = df_train[["Age", "Sex"]].apply(woman_child_or_man, axis=1)

df_train["Adult_male"] = df_train.Who == "man"

df_train["Deck"] = df_train.Cabin.str[0].map(lambda s: "np.nan" if s == "T" else s)

df_train["Embark_town"] = df_train.Embarked.map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})

df_train["Alive"] = df_train.Survived.map({0: "no", 1: "yes"})

df_train["Alone"] = ~(df_train.Parch + df_train.SibSp).astype(bool)



#df_train = df_train.drop(["Name", "Ticket", "Cabin"], axis=1)
df_test["Class"] = df_test.Pclass.map({1: "First", 2: "Second", 3: "Third"})

df_test["Who"] = df_test[["Age", "Sex"]].apply(woman_child_or_man, axis=1)

df_test["Adult_male"] = df_test.Who == "man"

df_test["Deck"] = df_test.Cabin.str[0].map(lambda s: "np.nan" if s == "T" else s)

df_test["Embark_town"] = df_test.Embarked.map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})

df_test["Alone"] = ~(df_test.Parch + df_train.SibSp).astype(bool)



#df_train = df_train.drop(["Name", "Ticket", "Cabin"], axis=1)
df_train.sample(n=5).head(5)
columns = ["Survived", "Class", "Who", "Alive", "Embark_town", "Deck"]



for column in columns:

           df_train[column] = df_train[column].astype('category')
matrix_train_pps = pps.matrix(train_pps)[['x','y','ppscore']].pivot(columns='x',index='y', values='ppscore')
colorMap = sns.color_palette("plasma", n_colors=5)

plt.figure(figsize = (16,5))

sns.heatmap(matrix_train_pps, vmin=0, vmax=1,cmap=colorMap ,linewidths=0.5,annot=True)
col = df_train.columns

print(df_train[col[0:11]])



pps.score(df_train, "Class","Alive")



train_pps = df_train[col[0:11]]

pps.matrix(train_pps)
ax = sns.catplot(data=df_train,

            x="Class", 

            col = 'Alive',

            sharex=False, sharey=True,

            hue="Who", 

            kind = 'count', 

            palette='colorblind',

            col_order = ["yes","no"])

ax.fig.suptitle('Who survived?', y=1.05)

plt.show()
pd.crosstab(index=[df_train.Alive,df_train.Class],

                  columns=df_train.Who,

                  values=df_train.Who,

                  normalize = False,

                  margins = False,

                  aggfunc = "count").fillna(0)
print("Percentage of Who survived, grouped by class, survival and gender.")

pd.crosstab(index=[df_train.Alive,df_train.Class],

                  columns=df_train.Who,

                  values=df_train.Who,

                  normalize = True,

                  margins = True, 

                  aggfunc = "count").round(4)*100
df_train[["Alive","Class","Who"]].groupby(["Alive","Class"]).count()
pd.pivot_table(df_train,values = "Survived",index = ["Pclass","Who"],columns = ["Alive"],aggfunc="count",margins=True)
df_train.dtypes
df_train.describe(include = "category")
df_train.describe(include = "all")
for column in df_train.columns:

    print(column, len(df_train[column].dropna().unique()))
print("df_test shape: ")

print(df_test.shape)



print("df_train shape: ")

print(df_train.shape)
df_train.columns
df_test.columns
# Create a list of test and train

df_list = [df_test, df_train.drop(columns=['Survived'])]





# Concatenate

df_combined = pd.concat(df_list, axis = 0)
df_combined.shape[0]



survived = 1502

all_passengers = 2224

died = all_passengers - survived



historical_data_list = [[0, survived], [1,died]]

    

historical_data = pd.DataFrame(historical_data_list, columns =['Survived','Total']) 



#print(historical_data)



print("Amount of passengers on Titanic: %d" % (sum(historical_data["Total"])))

print("Amount of passengers in Datasets: %d" % (df_train.shape[0] + df_test.shape[0]))



print("\nThe difference between the real amount of passengers compared \nto the amount of passenger in the test and training set is %d" % (all_passengers-df_combined.shape[0])) 
print("\n" + "Info of Train Data" + "\n" + "_"*80)

df_train.info()



print("\n" + "Any missing value?" + "\n" + "_"*80)



print(df_train.isnull().values.any())



print("\n" + "Where are the missing values?" + "\n" + "_"*80)

print(df_train.isnull().sum())
print("\n" + "Info of Test Data" + "\n" + "_"*80)

df_test.info()



print("\n" + "Any missing value?" + "\n" + "_"*80)



print(df_test.isnull().values.any())



print("\n" + "Where are the missing values?" + "\n" + "_"*80)

print(df_test.isnull().sum())
#df_train = df_train.drop(["Name", "Ticket", "Cabin"], axis=1)
df_train.columns
pal = "Colorblind"
g = sns.catplot("Who", col="Pclass", col_wrap=3,

                data=df_train,

                kind="count", height=5, aspect=.5, palette = "colorblind")
sns.catplot(x = "Adult_male",kind = "count",hue = "Survived", data=df_train, palette="colorblind",col="Class");
import plotly.express as px



fig = px.histogram(df_train, x="Who", color="Who", facet_row="Class", facet_col="Alive",

                  category_orders={"Alive": ["yes", "no"], "Class": ["First", "Second", "Third"]})

fig.show()
g = sns.FacetGrid(df_train, col='Class',row="Alive")

g = g.map(sns.distplot, "Age",bins=5)
fig = px.histogram(df_train, x="Class", y="Age", color="Who",

                   marginal="violin", # or violin, rug

                   hover_data=df_train.columns)

fig.show()
sns.catplot(x = "Who",kind = "count",hue = "Survived", data=df_train, palette="plasma",col="Pclass");
g = sns.FacetGrid(df_train, row='Class',col="Survived")

g = g.map(sns.distplot, "Age",bins=5)
sns.violinplot(x="Class", y="Age", hue="Sex", data=df_train, split=True, inner = "quartile").set_title("Age Distribution in class",y=1.07)

plt.hlines([0,16], xmin=-1, xmax=3, linestyles="dotted")

sns.violinplot(x="Class", y="Fare", hue="Sex", data=df_train, split=True, inner = "quartile").set_title("Fare Distribution in class",y=1.07)

plt.hlines([0,16], xmin=-1, xmax=3, linestyles="dotted")

g = sns.catplot("Alive", col="Deck",col_wrap=4,

                data=df_train[df_train.Deck.notnull()],

                kind="count", height=2.5, aspect=.8)