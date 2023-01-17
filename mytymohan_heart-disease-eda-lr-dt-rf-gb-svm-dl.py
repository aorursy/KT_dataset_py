import warnings

warnings.filterwarnings('ignore')
!ls ../input/
import pandas as pd

ht_dt = pd.read_csv("../input/heart.csv", header = 'infer')
print("The heart dataset has {0} rows and {1} columns".format(ht_dt.shape[0], ht_dt.shape[1]))
ht_dt.head()
import seaborn as sns

sns.countplot(ht_dt['target'],label="Count")
#Function to calculate missing value

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(ht_dt)
#Missing values with respect to each column in the dataset

import seaborn as sns

sns.heatmap(ht_dt.isnull(), cbar=False)
#correlation matrix

import matplotlib.pyplot as plt

%matplotlib inline

ht_dt_ft = ht_dt.drop('target', axis=1)

fig=plt.gcf()

fig.set_size_inches(15,7)

fig=sns.heatmap(ht_dt_ft.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',

                square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
from scipy.stats import spearmanr

import numpy as np

labels = []

values = []

for col in ht_dt.columns:

    if col not in ["target"]:

        labels.append(col)

        values.append(spearmanr(ht_dt[col].values, ht_dt["target"].values)[0])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')

 

ind = np.arange(corr_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,30))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='g')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

plt.show()
#scatterplot - set 1

set1 = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'target']

set1_dt = ht_dt[set1]

sns.pairplot(set1_dt, hue="target")
#scatterplot - set 2

set2 = ['thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

set2_dt = ht_dt[set2]

sns.pairplot(set2_dt, hue="target")
cm_surv = ["darkgrey" , "lightgreen"]

age_uniq = ht_dt.age.nunique()

print("Number of unique values in age is {}".format(age_uniq))

sns.catplot(x="sex", y="age", hue="target", inner="quart", kind="violin", palette=cm_surv, split=True, data=ht_dt)
trestbps_uniq = ht_dt.trestbps.nunique()

print("Number of unique values in trestbps is {}".format(trestbps_uniq))

sns.catplot(x="sex", y="trestbps", hue="target", kind="violin", inner="quart", palette=cm_surv, split=True, data=ht_dt)
thalach_uniq = ht_dt.thalach.nunique()

print("Number of unique values in thalach is {}".format(thalach_uniq))

sns.catplot(x="sex", y="thalach", hue="target", inner="quart", kind="violin", palette=cm_surv, split=True, data=ht_dt)
chol_uniq = ht_dt.chol.nunique()

print("Number of unique values in chol is {}".format(chol_uniq))

sns.catplot(x="sex", y="chol", hue="target", inner="quart", kind="violin", palette=cm_surv, split=True, data=ht_dt)
oldpeak_uniq = ht_dt.oldpeak.nunique()

print("Number of unique values in oldpeak is {}".format(oldpeak_uniq))

sns.catplot(x="sex", y="oldpeak", hue="target", kind="violin", inner="quart", palette=cm_surv, split=True, data=ht_dt)
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())



sex_q = """

select sex, target, count(*) as cnt

From ht_dt

GROUP BY sex, target;

"""



sex_df = pysqldf(sex_q)



sex_df_0 = sex_df[sex_df.target == 0]

sex_df_1 = sex_df[sex_df.target == 1]



import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



fig = {

  "data": [

    {

      "values": sex_df_0.cnt,

      "labels": sex_df_0.sex,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": sex_df_1.cnt,

      "labels": sex_df_1.sex,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Sex Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.16,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
cp_q = """

select cp, target, count(*) as cnt

From ht_dt

GROUP BY cp, target;

"""



cp_df = pysqldf(cp_q)



cp_df_0 = cp_df[cp_df.target == 0]

cp_df_1 = cp_df[cp_df.target == 1]



fig = {

  "data": [

    {

      "values": cp_df_0.cnt,

      "labels": cp_df_0.cp,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": cp_df_1.cnt,

      "labels": cp_df_1.cp,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"cp Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
fbs_q = """

select fbs, target, count(*) as cnt

From ht_dt

GROUP BY fbs, target;

"""



fbs_df = pysqldf(fbs_q)



fbs_df_0 = fbs_df[fbs_df.target == 0]

fbs_df_1 = fbs_df[fbs_df.target == 1]



fig = {

  "data": [

    {

      "values": fbs_df_0.cnt,

      "labels": fbs_df_0.fbs,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": fbs_df_1.cnt,

      "labels": fbs_df_1.fbs,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Fbs Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
restecg_q = """

select restecg, target, count(*) as cnt

From ht_dt

GROUP BY restecg, target;

"""



restecg_df = pysqldf(restecg_q)



restecg_df_0 = restecg_df[restecg_df.target == 0]

restecg_df_1 = restecg_df[restecg_df.target == 1]



fig = {

  "data": [

    {

      "values": restecg_df_0.cnt,

      "labels": restecg_df_0.restecg,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": restecg_df_1.cnt,

      "labels": restecg_df_1.restecg,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"restecg Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
exang_q = """

select exang, target, count(*) as cnt

From ht_dt

GROUP BY exang, target;

"""



exang_df = pysqldf(exang_q)



exang_df_0 = exang_df[exang_df.target == 0]

exang_df_1 = exang_df[exang_df.target == 1]



fig = {

  "data": [

    {

      "values": exang_df_0.cnt,

      "labels": exang_df_0.exang,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": exang_df_1.cnt,

      "labels": exang_df_1.exang,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"exang Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
sl_q = """

select slope, target, count(*) as cnt

From ht_dt

GROUP BY slope, target;

"""



sl_df = pysqldf(sl_q)



sl_df_0 = sl_df[sl_df.target == 0]

sl_df_1 = sl_df[sl_df.target == 1]



fig = {

  "data": [

    {

      "values": sl_df_0.cnt,

      "labels": sl_df_0.slope,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": sl_df_1.cnt,

      "labels": sl_df_1.slope,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Slope Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
ca_q = """

select ca, target, count(*) as cnt

From ht_dt

GROUP BY ca, target;

"""



ca_df = pysqldf(ca_q)



ca_df_0 = ca_df[ca_df.target == 0]

ca_df_1 = ca_df[ca_df.target == 1]



fig = {

  "data": [

    {

      "values": ca_df_0.cnt,

      "labels": ca_df_0.ca,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": ca_df_1.cnt,

      "labels": ca_df_1.ca,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Ca Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
thal_q = """

select thal, target, count(*) as cnt

From ht_dt

GROUP BY thal, target;

"""



thal_df = pysqldf(thal_q)



thal_df_0 = thal_df[thal_df.target == 0]

thal_df_1 = thal_df[thal_df.target == 1]



fig = {

  "data": [

    {

      "values": thal_df_0.cnt,

      "labels": thal_df_0.thal,

      "domain": {"x": [0, .48]},

      "name": "No Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": thal_df_1.cnt,

      "labels": thal_df_1.thal,

      "domain": {"x": [.52, 1]},

      "name": "With Heart Disease",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"thal Vs Target",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "No Disease",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "With Disease",

                "x": 0.85,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(ht_dt.loc[:, ht_dt.columns != 'target'], 

                                                    ht_dt['target'], stratify=ht_dt['target'], 

                                                    random_state=66)



print("Training features have {0} records and Testing features have {1} records.".\

      format(X_train.shape[0], X_test.shape[0]))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression().fit(X_train, y_train)

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))

print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
tree = DecisionTreeClassifier(max_depth=3, random_state=0)

tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
print("Feature importances:\n{}".format(tree.feature_importances_))
dis_ft = [x for i,x in enumerate(ht_dt.columns) if i!=8]

def plot_feature_importances_diabetes(model):

    plt.figure(figsize=(8,6))

    n_features = 13

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), dis_ft)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)

plot_feature_importances_diabetes(tree)

plt.savefig('feature_importance')
#Random forest with 100 trees

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)

rf.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))
rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)

rf1.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))
plot_feature_importances_diabetes(rf1)
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)

gb.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))
#GB after pruning

gb1 = GradientBoostingClassifier(random_state=0, max_depth=1)

gb1.fit(X_train, y_train)

print("****Gradient Boosting after Pruning using Max_depth****")

print("Accuracy on training set: {:.3f}".format(gb1.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb1.score(X_test, y_test)))
#GB after tuning learning rate

gb2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)

gb2.fit(X_train, y_train)

print("****Gradient Boosting after tuning Learning rate****")

print("Accuracy on training set: {:.3f}".format(gb2.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb2.score(X_test, y_test)))
plot_feature_importances_diabetes(gb2)
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)

svc = SVC()

svc.fit(X_train_scaled, y_train)

print("****Results after scaling****")

print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42)

mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)

mlp = MLPClassifier(random_state=0)

mlp.fit(X_train_scaled, y_train)

print("****Results after scaling****")

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
#Tuning the iteration

mlp = MLPClassifier(max_iter=1000, random_state=0)

mlp.fit(X_train_scaled, y_train)

print("****Results after tuning iteration****")

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
mlp = MLPClassifier(max_iter=100, alpha=1, random_state=0)

mlp.fit(X_train_scaled, y_train)

print("****Results after tuning alpha & regularizing the weights****")

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
plt.figure(figsize=(20, 5))

plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')

plt.yticks(range(13), dis_ft)

plt.xlabel("Columns in weight matrix")

plt.ylabel("Input feature")

plt.colorbar()