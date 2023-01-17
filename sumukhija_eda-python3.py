import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv("../input/UCI_Credit_Card.csv")
df.shape
df.head(1)
df.isnull().sum()
df = df.rename(columns={'default.payment.next.month': 'IS_DEFAULT', 

                        'PAY_0': 'PAY_1'})

df.head(1)
list_of_cols = list(df)

for col in list_of_cols:

    if col.startswith('LIMIT') or col.startswith('BILL') or col.startswith('PAY_A'):

        pass

    else:

        print(str(col) + ": "+ str(df[col].unique()))
df.loc[df.EDUCATION >= 4, 'EDUCATION'] = 0

df.loc[df.MARRIAGE == 3, 'MARRIAGE'] = 0
fig = sns.countplot(x = 'IS_DEFAULT', data = df)

fig.set_xticklabels(["No Default", "Default"])
number_of_defaulters = len(df[df.IS_DEFAULT == 1]) 

number_of_non_defaulters = len(df) - number_of_defaulters

percentage_of_defaulters = number_of_defaulters/number_of_non_defaulters * 100

round(percentage_of_defaulters, 2) #28.4%

df['LIMIT_BAL'].describe()
bins = [0, 200000, 400000, 600000, 800000, 1000000]

df['LIMIT_GROUP'] = pd.cut(df['LIMIT_BAL'], bins,include_lowest=True)
# df_2 = df.LIMIT_GROUP.groupby(df.IS_DEFAULT)

# df_2.IS_DEFAULT

# # axis = df_2.LIMIT_GROUP.value_counts(sort = False).plot.bar(rot=0, color="r", figsize=(6,4))
#Computing percentage

number_of_male_card_holders = (df.SEX == 1).sum() #11,888

number_of_female_card_holders = (df.SEX == 2).sum() #18,112



number_of_male_defaulters = (df[df.SEX == 1].IS_DEFAULT == 1).sum() #2,873

number_of_female_defaulters = (df[df.SEX == 2].IS_DEFAULT == 1).sum() #3,763



percentage_of_male_def = round((number_of_male_defaulters/number_of_male_card_holders) * 100,2) #24.17%

percentage_of_female_def = round((number_of_female_defaulters/number_of_female_card_holders) * 100,2) #20.78%

temp_df = pd.DataFrame({"non-defaulters":{"male":100 - percentage_of_male_def, "female":100 - percentage_of_female_def},"defaulters":{"male":percentage_of_male_def, "female":percentage_of_female_def}})



#Plotting chart

fig = temp_df.plot(kind = 'bar')

fig.set_title("Percentage of male and female non-defaulters vs defaulters")

fig.set_ylabel("Percentage")
#Computing percentage

number_of_unknown_edu_card_holders = (df.EDUCATION == 0).sum() #468

number_of_grad_edu_card_holders = (df.EDUCATION == 1).sum() #10,585

number_of_uni_card_holders = (df.EDUCATION == 2).sum() #14,030

number_of_high_school_card_holders = (df.EDUCATION == 3).sum() #4,917



number_of_unknown_edu_defaulters = (df[(df.EDUCATION == 0)].IS_DEFAULT == 1).sum() #33

number_of_grad_defaulters = (df[(df.EDUCATION == 1)].IS_DEFAULT == 1).sum() #2036

number_of_uni_defaulters = (df[(df.EDUCATION == 2)].IS_DEFAULT == 1).sum() #3330

number_of_high_school_defaulters = (df[(df.EDUCATION == 3)].IS_DEFAULT == 1).sum() #1237



percentage_of_unknown_def = round((number_of_unknown_edu_defaulters/number_of_unknown_edu_card_holders) * 100,2) #7.05

percentage_of_grad_def = round((number_of_grad_defaulters/number_of_grad_edu_card_holders) * 100,2) #19.23

percentage_of_uni_def = round((number_of_uni_defaulters/number_of_uni_card_holders) * 100,2) #23.73

percentage_of_high_school_def = round((number_of_high_school_defaulters/number_of_high_school_card_holders) * 100,2) #25.16

temp_df = pd.DataFrame({"non-defaulters":{"Unknown":100 - percentage_of_unknown_def, "Graduates":100 - percentage_of_grad_def, "University":100 - percentage_of_uni_def, "High school":100 - percentage_of_high_school_def},"defaulters":{"Unknown": percentage_of_unknown_def, "Graduates": percentage_of_grad_def, "University": percentage_of_uni_def, "High school":percentage_of_high_school_def}})



#Plotting chart

fig = temp_df.plot(kind = 'bar')

fig.set_title("Percentage of non-defaulters & defaulters based on education level")

fig.set_ylabel("Percentage")
number_of_others_card_holders = (df.MARRIAGE == 0).sum() #377

number_of_married_card_holders = (df.MARRIAGE == 1).sum() #13,659

number_of_unmarried_card_holders = (df.MARRIAGE == 2).sum() #15,964



number_of_others_def = (df[(df.MARRIAGE == 0)].IS_DEFAULT == 1).sum() #89

number_of_married_def = (df[(df.MARRIAGE == 1)].IS_DEFAULT == 1).sum() #3,206

number_of_ummarried_def = (df[(df.MARRIAGE == 2)].IS_DEFAULT == 1).sum() #3,341



percentage_of_others_def = round(number_of_others_def/number_of_others_card_holders * 100,2) #23.61

percentage_of_married_def = round(number_of_married_def/number_of_married_card_holders * 100,2) #23.47

percentage_of_ummarried_def = round(number_of_ummarried_def/number_of_unmarried_card_holders * 100,2) #20.93





temp_df = pd.DataFrame({"non-defaulters":{"Unknown":100 - percentage_of_others_def, "Married":100 - percentage_of_married_def, "Unmarried":100 - percentage_of_ummarried_def},

                        "defaulters":{"Unknown":percentage_of_others_def, "Married":percentage_of_married_def, "Unmarried": percentage_of_ummarried_def}})

fig = temp_df.plot(kind = 'barh')

fig.set_title("Percentage of non-defaulters & defaulters based on education level")

fig.set_xlabel("Percentage")
sns.set(rc={'figure.figsize':(12,5)})

fig = sns.countplot(x = 'AGE', data = df, hue = 'IS_DEFAULT')

fig.legend(title='Is Default?', loc='upper right', labels=["Not Default", "Default"])

fig.set_title("Defaulters based on education level")
sns.set(rc={'figure.figsize':(25,8)})

sns.set_context("talk", font_scale=0.7)

sns.heatmap(df.iloc[:,1:].corr(), cmap='Greens', annot=True)
X_train = df.iloc[:,[0,2,5,6,7,8,9,10]]

Y_train = df.iloc[:,[23]]
from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()

model = clf.fit(X_train, Y_train)

acc = round(model.score(X_train, Y_train) * 100, 2)

train_pred = model_selection.cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs = -1)

c_acc = round(metrics.accuracy_score(Y_train, train_pred) * 100, 2)

print(acc)

print(c_acc)