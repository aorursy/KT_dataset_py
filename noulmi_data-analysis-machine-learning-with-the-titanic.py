import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#read the datasets

train = pd.read_csv("../input/train.csv")

holdout = pd.read_csv("../input/test.csv")
#lets describe our train df...



columns = ['SibSp','Parch','Fare','Cabin','Embarked']

train[columns].describe(include='all',percentiles=[])
#and holdout df



holdout[columns].describe()
chance_survive = len(train[train["Survived"] == 1]) / len(train["Survived"])
def plot_survival(df, index, color = "blue", use_index = True, num_xticks = 0, xticks = "", position = 0.5, legend = 

                  ["General probabilty of survival"]):

    df_pivot = df.pivot_table(index=index,values="Survived")

    df_pivot.plot.bar(ylim = [0,1], color = color, use_index = use_index, position = position)

    plt.axhline(chance_survive, color = "red", linewidth = 1)

    if num_xticks>0:

        plt.xticks(range(num_xticks), xticks)

    plt.legend(legend)

    plt.title("Plotting the survival probability by "+index+"\n")
plot_survival(train, "Sex", color = ["pink", "blue"], use_index = False, num_xticks = 2, xticks = ['Female', 'Male'])
plot_survival(train,"Pclass", color = "blue", use_index = True)
plot_survival(train,'Parch', "red", True , position = 0.3)

plot_survival(train,'SibSp', "blue", True)
cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]



def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df
train = process_age(train,cut_points,label_names)

holdout = process_age(holdout,cut_points,label_names)
plot_survival(train,"Age_categories", use_index = False, num_xticks = len(train["Age_categories"].unique()), 

              xticks = train["Age_categories"].unique().sort_values())
def process_fare(df, cut_points, label_names):

    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels = label_names)

    return df
train = process_fare(train, [0,12,50,100,1000], ["0-12$","12-50$","50-100$","100+$"])
holdout = process_fare(holdout, [0,12,50,100,1000], ["0-12$","12-50$","50-100$","100+$"])
plot_survival(train,"Fare_categories", use_index = False, num_xticks = len(train["Fare_categories"].unique())-1, 

              xticks = train["Fare_categories"].unique().sort_values())
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df
#create dummies for the Age_cat, Class, Sex, Embarked features & Fare



for col in ["Age_categories", "Pclass", "Sex", "Embarked", "Fare_categories"]:

    train = create_dummies(train, col)

    holdout = create_dummies(holdout, col)
# the holdout DF has one missing value for the Fare feature. We can easliy replace it by the mean of that column.

# Same for the embarked column. We can easily replace it by the most common value, "S"



holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())

train["Embarked"] = train["Embarked"].fillna("S")

holdout["Embarked"] = holdout["Embarked"].fillna("S")
from sklearn.preprocessing import minmax_scale



cols = ["SibSp", "Parch", "Fare"]

new_cols = ["SibSp_scaled", "Parch_scaled", "Fare_scaled"]



for col, new_col in zip(cols, new_cols):

    train[new_col] = minmax_scale(train[col])

    holdout[new_col] = minmax_scale(holdout[col])
# our SibSp and Parch got converted to floats when scaled, we should keep that in mind 



dtypes = train[cols + new_cols].dtypes
columns = ['Age_categories_Missing', 'Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',

       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',

       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']



columns_not_scaled = ['Age_categories_Missing', 'Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',

       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',

       'SibSp', 'Parch', 'Fare']
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



lr.fit(train[columns], train["Survived"])

coefficients = lr.coef_

feature_importance = pd.Series(coefficients[0], index = train[columns].columns)



lr.fit(train[columns_not_scaled], train["Survived"])

coefficients2 = lr.coef_

feature_importance_2 = pd.Series(coefficients2[0], index = train[columns_not_scaled].columns)
ordered_feature_importance = feature_importance.abs().sort_values()

#ordered_feature_importance_2 = feature_importance_2.abs().sort_values()

ordered_feature_importance.plot.barh(color = "blue")

#ordered_feature_importance_2.plot.barh(color = "red")

#plt.legend(['Scaling', 'No scaling'], loc = 4)
def get_titles(train, test):

    titles_train = train["Name"].str.split(".", expand = True)

    titles_test = test["Name"].str.split(".", expand = True)

    titles_train = titles_train[0].str.split(",", expand = True)

    titles_test = titles_test[0].str.split(",", expand = True)

    titles_train = titles_train[1]

    titles_test = titles_test[1]

    train["titles"] = titles_train.astype("category")

    test["titles"] = titles_test.astype("category")

    #train = train.drop("Name", axis = 1)

    #test = test.drop("Name", axis = 1)

    return train, holdout
train, holdout = get_titles(train,holdout)
import numpy as np
plot_survival(train,"titles", use_index = False, num_xticks = len(train["titles"].unique())-1, 

              xticks = train["titles"].unique().sort_values())
train = create_dummies(train,"titles")

holdout = create_dummies(holdout,"titles")
print("number of null values :",train["Cabin"].isnull().sum())

print("number of non_null values :",train["Cabin"].notnull().sum())
train["Cabin"] = train["Cabin"].fillna("unknown")

holdout["Cabin"] = holdout["Cabin"].fillna("unknown")
cabins = train["Cabin"].tolist()

cabins_h = holdout["Cabin"].tolist()
cabins_type = []

for i in cabins:

    cabins_type.append(i[0:1])

    

cabins_type_holdout = []

for i in cabins_h:

    cabins_type_holdout.append(i[0:1])
train["Cabin"] = cabins_type

holdout["Cabin"] = cabins_type_holdout
train = create_dummies(train, "Cabin")

holdout = create_dummies(holdout, "Cabin")
plot_survival(train,"Cabin", use_index = True)
print("We have now ",len(train.columns), "columns as predictors to fit")
columns_cabins_titles = ['Age_categories_Missing', 'Age_categories_Infant',

'Age_categories_Child', 'Age_categories_Teenager',

'Age_categories_Young Adult', 'Age_categories_Adult',

'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',

'Sex_female', 'SibSp_scaled', 'Parch_scaled',

'Fare_categories_0-12$', 'Fare_categories_12-50$',

'Fare_categories_50-100$', 'Fare_categories_100+$', 'titles_ Capt', 'titles_ Col', 'titles_ Don',

'titles_ Dr', 'titles_ Jonkheer', 'titles_ Lady', 'titles_ Major',

'titles_ Master', 'titles_ Miss', 'titles_ Mlle', 'titles_ Mme',

'titles_ Mr', 'titles_ Mrs', 'titles_ Ms', 'titles_ Rev', 'titles_ Sir',

'titles_ the Countess', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D',

'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_u']



other_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3']
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()





logreg.fit(train[columns_cabins_titles], train["Survived"])

feature_importance_2 = logreg.coef_
feature_importance_2 = pd.Series(feature_importance_2[0], index = train[columns_cabins_titles].columns)

ordered_feature_importance = feature_importance_2.abs().sort_values()
ordered_feature_importance.plot.barh(color = "blue", figsize = (10,10))
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior', "titles_ Col", "titles_ Don", "titles_ Dr", "titles_ Master", "titles_ Miss", 

        "titles_ Mr", "titles_ Mrs", "titles_ Ms", "titles_ Rev"]
columns_cabins_titles = ['Age_categories_Missing', 'Age_categories_Infant',

'Age_categories_Child', 'Age_categories_Teenager',

'Age_categories_Young Adult', 'Age_categories_Adult',

'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',

'Sex_female', 'SibSp_scaled', 'Parch_scaled',

'Fare_categories_0-12$', 'Fare_categories_12-50$',

'Fare_categories_50-100$', 'Fare_categories_100+$', 'titles_ Capt', 'titles_ Col', 'titles_ Don',

'titles_ Dr', 'titles_ Jonkheer', 'titles_ Lady', 'titles_ Major',

'titles_ Master', 'titles_ Miss', 'titles_ Mlle', 'titles_ Mme',

'titles_ Mr', 'titles_ Mrs', 'titles_ Ms', 'titles_ Rev', 'titles_ Sir',

'titles_ the Countess', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D',

'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T']
import seaborn as sns
def plot_correlation_heatmap(df):

    corr = df.corr()

    

    sns.set(style="white")

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(520, 10, as_cmap=True)





    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
corrs = train[columns_cabins_titles].corr()
plot_correlation_heatmap(corrs.corr())
from sklearn.feature_selection import RFECV

lr = LogisticRegression()

selector = RFECV(lr,cv=10)

selector.fit(train[columns_cabins_titles],train["Survived"])
optimized_columns = train[columns_cabins_titles].columns[selector.support_]
print("We only have", len(optimized_columns), "columns left for our model. Let's test it")
lr = LogisticRegression()

scores = model_selection.cross_val_score(lr, train[optimized_columns], train["Survived"], cv=10)

accuracy_optimized = scores.mean()
lr = LogisticRegression()

scores = model_selection.cross_val_score(lr, train[columns_cabins_titles], train["Survived"], cv=10)

accuracy_n_optimized = scores.mean()
print("We have an accuracy of ", accuracy_optimized, "with optimized predictors", "vs ", accuracy_n_optimized, "before")
#Before fitting our model to our real holdout dataset we should remove the "Captain" column since it does not exist in the 

#holdout.



optimized_columns = optimized_columns.drop("titles_ Capt")
lr = LogisticRegression()

lr.fit(train[optimized_columns],train["Survived"])

holdout_predictions = lr.predict(holdout[optimized_columns])
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)



submission.to_csv("submission_2.csv",index=False)