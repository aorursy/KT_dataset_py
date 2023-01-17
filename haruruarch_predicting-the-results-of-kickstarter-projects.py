# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

print(keras.__version__)



import sklearn as sk

print(sk.__version__)



import matplotlib

import matplotlib.pyplot as plt

print(matplotlib.__version__)

%matplotlib inline



import seaborn as sns

print(sns.__version__)



import warnings

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

ks_2018 = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", header = 0)

ks_2018.head(100)
ks_2018.describe()
#Selecting the variables to use for prediction

ks_2018_modified = ks_2018[["category", "name", "state", "usd_goal_real", "country", "usd_pledged_real", "deadline", "currency"]]



#Creating a new objective variable: "percentage"

ks_2018_modified["percentage"] = (ks_2018["usd_pledged_real"] / ks_2018["usd_goal_real"]) * 100



#Combining "country" and "currency" into one variavle

ks_2018_modified["currency_country"] = list(zip(ks_2018_modified.currency, ks_2018_modified.country))

#Creating a new variable: "name_length"

ks_2018_modified["name_length"]      = [len(str(i)) for i in ks_2018["name"]]

#Creating a new variable: "has_special_characters"

def count_special(characters):

    if type(characters) != str:

        return 0

    count = 0

    special_characters = ["!", "?", "*", ":", "&", "'", "-", "|", ",", ".", "(", ")"]

    for c in characters:

        if c in special_characters:

            count += 1

    return count

ks_2018_modified["has_special_characters"] = ks_2018["name"].apply(count_special)
ks_2018_modified.head()
#Checking the modified data

ks_2018_modified.describe()
print(ks_2018_modified.isnull().any())

print(ks_2018_modified.isnull().sum())
#Converting "launched" & "deadline" into date

ks_2018_modified["launched"] = pd.to_datetime(ks_2018["launched"]).dt.date

ks_2018_modified["deadline"] = pd.to_datetime(ks_2018["deadline"]).dt.date

#Creating a new variable: "period"

ks_2018_modified["period"] = ks_2018_modified["deadline"] - ks_2018_modified["launched"]

ks_2018_modified["period"] = ks_2018_modified["period"] / np.timedelta64(1,'D')

#Creating new variables: "launched_d" & "deadline_d" (number of days from 2009-01-01)

ks_2018_modified["launched_d"] = [(pd.to_datetime(i) - pd.to_datetime("2009-01-01")) / np.timedelta64(1, 'D') for i in ks_2018_modified["launched"]]

ks_2018_modified["deadline_d"] = [(pd.to_datetime(i) - pd.to_datetime("2009-01-01")) / np.timedelta64(1, 'D') for i in ks_2018_modified["deadline"]]

#Creating new variables: "deadline_month" & "deadline_dayofweek" 

ks_2018_modified["deadline_month"]     = pd.to_datetime(ks_2018_modified["deadline"]).dt.month

ks_2018_modified["deadline_dayofweek"] = pd.to_datetime(ks_2018_modified["deadline"]).dt.dayofweek
#observing categorical variables

states = ks_2018_modified["state"].unique()

categories = ks_2018_modified["category"].unique()

num_categories = len(categories)

print(states)

print(categories)

print(num_categories, "categoies")
print(ks_2018_modified["category"].value_counts())

print("\n Grouping... \n")

category_dict = ks_2018_modified["category"].value_counts(normalize = True).to_dict()

ks_2018_modified["category"] = [i if category_dict[i] > 0.01 else "rare" for i in ks_2018_modified["category"]]

print(ks_2018_modified["category"].value_counts())
indices = [i != float('inf') for i in ks_2018_modified['percentage']]

ks_2018_modified = ks_2018_modified[indices].copy()
#Drawing graphs to visually examine the correlation between variables and the result

pd.plotting.scatter_matrix(ks_2018_modified, figsize=(10, 10))

plt.show()
#Trying to see if there is any useful/important variables

ks_2018_modified.corr()
#Checking if there is country-specific trend ver.1

sns.catplot(x = "country", y = "percentage", data = ks_2018_modified,  height = 10)

plt.show()
#Checking if there is country-specific trend ver.2

plt.figure(figsize = (15, 5))

v = sns.violinplot(x = "country", y = "percentage", data = ks_2018_modified,  width = 1.5, bw = "silverman")

v.set(ylim = (0, 400))

plt.show()
# from pandas.plotting import register_matplotlib_converters

# register_matplotlib_converters()

# plt.figure(figsize = (15, 10))

# s = sns.distplot(ks_2018_modified["launched_d"], y = "percentage", data = ks_2018_modified, hue = "currency")

# plt.show()

# #開始時期によってKickstarterがバズったりすることで成功率が上がっているというような傾向がないかさぐってみたかった、あと通貨による影響があるか

# #開始時期がおかしいやつがあったので、時期を指定した
# ax = sns.jointplot((pd.to_datetime(ks_2018_modified["launched"] - pd.to_datetime("2009-01-01")).dt.date, ks_2018_modified["percentage"], xlim = (0, 3500), ylim = (0, 15000), color = "g")



# #開始時期（2009-01-01からの日数）と達成率
# ax = sns.kdeplot(ks_2018_modified["usd_goal_real"], ks_2018_modified["percentage"], n_levels = 10,

#                  xlim = (0, 500000), ylim = (0, 25000), cmap="Blues", shade=True, shade_lowest=False)

# #目標金額と達成率、kde重すぎて動かない(;_;)



# import itertools

# li_combi = list(itertools.combinations(df_wine.columns[1:], 2))

# for X,Y in li_combi:

#     print("X=%s"%X,"Y=%s"%Y)

#     df_wine.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10,c="Class label",colormap="winter")#散布図の作成

#     plt.xlabel(X)

#     plt.ylabel(Y)

#     plt.tight_layout()

#     plt.show()#グラフをここで描画させるための行
# # categoryごとのstateの出現頻度を確認

# # データ内のcategoryを抽出しcategoryに格納

# category=df_cloudfound_sct.groupby('category')

# # stateを相対的な頻度に変換

# category=category['state'].value_counts(normalize=True).unstack() 

# # successfulの降順ソート

# category=category.sort_values(by=['successful'],ascending=False)

# # 縦棒グラフ（積み上げ）でグラフ作成

# category[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
#importing stuff

!pip install pydotplus

!pip install optuna

import optuna 

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, LassoCV

from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, f1_score, r2_score

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import pydotplus

from IPython.display import Image

from sklearn.externals.six import StringIO

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential, Model, load_model

from keras.layers import Dropout, Dense

from keras.optimizers import SGD, RMSprop, Adam

from keras.callbacks import History

from scipy import stats
#Distribution of "percentage"

sns.distplot(ks_2018_modified["usd_goal_real"], hist = False, color = "g", kde_kws = {"shade": True})
#Logariyhmic Transformation

ks_2018_modified["percentage"] = [np.log(i) if i > 0 else 0 for i in ks_2018_modified["percentage"]]

sns.distplot(ks_2018_modified["percentage"], color = "pink")

#Standardization

stdsc_percentage = StandardScaler()

ks_2018_modified["std_percentage"] = stdsc_percentage.fit_transform(ks_2018_modified[["percentage"]].values)

sns.distplot(ks_2018_modified["std_percentage"], color = "skyblue")

#Box-Cox Transformation

# ks_2018_modified["percentage"] = [i if i > 0 else 0 for i in ks_2018_modified["percentage"]]

# ks_2018_modified["bc_percentage"] = stats.boxcox(ks_2018_modified["percentage"])

# sns.distplot(ks_2018_modified["bc_percentage"], color = "purple")

#Standardization + Box-Cox Transformation

# ks_2018_modified["sbc_percentage"] = stats.boxcox(ks_2018_modified["std_percentage"])

# sns.distplot(ks_2018_modified["sbc_percentage"], color = "yellow")
# plt.figure(figsize = (15, 10))

# p = sns.scatterplot(x = "period", y = "percentage", data = ks_2018_modified, hue = "category")

# p.set(ylim = (0, 1000))

# p.set(xlim = (0, 100))

# plt.show()

# #カテゴリーごとの期間と達成率

# #下にオレンジが多くて、上に緑が多い？
ks_2018_modified["has_special_characters"] = ks_2018_modified["name_length"] / ks_2018_modified["has_special_characters"]
#dealing with categorical variables

X = pd.get_dummies(ks_2018_modified[["category", "name_length", "has_special_characters", "deadline_month", "deadline_dayofweek", "launched_d", "usd_goal_real", "currency_country", "period"]], columns = ["category", "currency_country"])



X.head()
X.describe()
#目標達成率（回帰）

Y1 = ks_2018_modified["percentage"]

Y1 = pd.DataFrame(Y1, columns = ["percentage"])

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.3, random_state = 0)





#成功するかどうか（分類）

#Selecting only successful/failed projects and ignoring others(canceled, live, etc.)

# indices = ks_2018_modified["state"].isin(["failed", "successful"])

# X = X[indices].copy()

X = ks_2018_modified.query('state == ["failed", "successful"]')



Y2 = [1 if i == "successful" else 0 for i in X["state"]]



X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.3, random_state = 0)



X2_train.head(10)
#Training Data

#目標達成率（回帰）

#periodの標準化

stdsc_period_1 = StandardScaler()

X1_train["period"] = stdsc_period_1.fit_transform(X1_train[["period"]] .values)

#goalの標準化

stdsc_goal_1 = StandardScaler()

X1_train["usd_goal_real"] = stdsc_goal_1.fit_transform(X1_train[["usd_goal_real"]] .values)

#launched_dの標準化

stdsc_launched_1 = StandardScaler()

X1_train["launched_d"] = stdsc_launched_1.fit_transform(X1_train[["launched_d"]] .values)

#launchedとperiodの白色化

cov = np.cov(X1_train[["launched_d", "period"]], rowvar=0) # 分散・共分散を求める

_, S1 = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

launched_period = np.dot(S1.T, X1_train[["launched_d", "period"]].T).T #データを無相関化

stdsc_ld_1 = StandardScaler()

launched_period  = stdsc_ld_1.fit_transform(launched_period) # 無相関化したデータに対して、さらに標準化



#成功するかどうか（分類）

#periodの標準化

stdsc_period_2 = StandardScaler()

X2_train["period"] = stdsc_period_2.fit_transform(X2_train[["period"]] .values)

#goalの標準化

stdsc_goal_2 = StandardScaler()

X2_train["usd_goal_real"] = stdsc_goal_2.fit_transform(X2_train[["usd_goal_real"]] .values)

#launched_dの標準化

stdsc_launched_2 = StandardScaler()

X2_train["launched_d"] = stdsc_launched_2.fit_transform(X2_train[["launched_d"]] .values)

#launchedとperiodの白色化

cov = np.cov(X2_train[["launched_d", "period"]], rowvar=0) # 分散・共分散を求める

_, S2 = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

launched_period = np.dot(S2.T, X2_train[["launched_d", "period"]].T).T #データを無相関化

stdsc_ld_2 = StandardScaler()

launched_period  = stdsc_ld_2.fit_transform(launched_period) # 無相関化したデータに対して、さらに標準化
#Checking if there is any correlated variables remaining 

X1_train.corr().style.background_gradient().format('{:.2f}')
#Testing Data



#目標達成率（回帰）

#periodの標準化

X1_test["period"] = stdsc_period_1.transform(X1_test.copy()[["period"]].values)

#goalの標準化

X1_test["usd_goal_real"] = stdsc_goal_1.transform(X1_test.copy()[["usd_goal_real"]].values)

#launched_dの標準化

X1_test["launched_d"] = stdsc_launched_1.transform(X1_test.copy()[["launched_d"]].values)

#launchedとdeadlineの白色化

launched_period = np.dot(S1.T, X1_test[["launched_d", "period"]].T).T #データを無相関化

launched_period  = stdsc_ld_1.transform(launched_period) # 無相関化したデータに対して、さらに標準化



#成功するかどうか（分類）

#periodの標準化

X2_test["period"] = stdsc_period_2.transform(X2_test.copy()[["period"]].values)

#goalの標準化

X2_test["usd_goal_real"] = stdsc_goal_2.transform(X2_test.copy()[["usd_goal_real"]].values)

#launched_dの標準化

X2_test["launched_d"] = stdsc_launched_2.transform(X2_test.copy()[["launched_d"]].values)

#launchedとdeadlineの白色化

launched_period = np.dot(S2.T, X2_test[["launched_d", "period"]].T).T #データを無相関化

launched_period  = stdsc_ld_2.transform(launched_period) # 無相関化したデータに対して、さらに標準化
#Feature Selection

estimator = LassoCV(normalize = True, cv = 20)

sfm = SelectFromModel(estimator, threshold = 1e-5)



sfm.fit(X1_train.values, Y1_train.values)

# print(sfm.get_support())

# print(~sfm.get_support())

new_index = sfm.get_support()

print(X1_train.columns[new_index])

new_X1_train = X1_train[X1_train.columns[new_index]]

new_X1_test  = X1_test[X1_train.columns[new_index]]
#目標達成率（回帰）



#NN

#optunaによるハイパーパラメータオプティマイゼーション

def create_model(num_layer, num_shape, use_dropout, activation):

    model = Sequential()

    model.add(Dense(num_shape, input_dim = len(X1_train.columns), activation = activation))

    for layer in range (2, num_layer):

        model.add(Dense(num_shape, input_dim = num_shape, activation = activation))

        if use_dropout == "true":

            model.add(Dropout(0.2))

    model.add(Dense(1, activation = activation))

#     model.summary()

#     plot_model(model)

#     SVG(model_to_dot(model).create(prog='dot', format='svg'))

    return model



def train(model, optimizer, learning_rate, num_shape, num_layer, use_dropout, activation, trial):

    epochs = 32

    kf = KFold(n_splits = 5, shuffle = True, random_state = 0)

    for train, valid in kf.split(X1_train, Y1_train):

        if optimizer == "sgd":

            model.compile(loss = 'mean_squared_error', optimizer = SGD())

        elif optimizer == "rmsprop":

            model.compile(loss = 'mean_squared_error', optimizer = RMSprop())

        elif optimizer == "adam":

            model.compile(loss = 'mean_squared_error', optimizer = Adam())

        history = History()

        model.fit(X1_train.values[train], Y1_train.values[train], epochs = epochs, callbacks = [history], verbose=0)

        scores = model.evaluate(X1_train.values[valid], Y1_train.values[valid], verbose=0)   

        model.save("model_{}.h5".format(trial.number))

        del model

        return history



def objective(trial):

    optimizer = trial.suggest_categorical("optimizer", ["sgd", "rmsprop", "adam"])

    activation = trial.suggest_categorical("activation", ["relu", "tanh", "softplus"])

    use_dropout = trial.suggest_categorical("use_dropout", ["true", "false"])

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e0)

    num_layer = trial.suggest_int("num_layer", 1, 10)

    num_shape = trial.suggest_int("num_shape", 10, 100)

    model = create_model(num_layer, num_shape, use_dropout, activation)

    hist = train(model, optimizer, learning_rate, num_shape, num_layer, use_dropout, activation, trial)

    

    return np.min(hist.history["loss"])



study = optuna.create_study()

study.optimize(objective, n_trials = 5)

hist_df = study.trials_dataframe()

hist_df.to_csv("optuna_results.csv")

print("Best Parameters: ", study.best_params)

print("Best Value: ", study.best_value)

print("Best Trial", study.best_trial)



#成功するかどうか（分類）

#Decision Tree

clf = DecisionTreeClassifier(criterion = "gini", max_depth = None, min_samples_leaf = 3, random_state = 0)

clf = clf.fit(X2_train, Y2_train)

print("training score(Decision Tree)= ", clf.score(X2_train, Y2_train))

#Drawing trees

dot_data = StringIO() #dotファイル情報の格納先

export_graphviz(clf, out_file = dot_data,  

                     feature_names = X2_train.columns,  

                     class_names = ["successful", "failed"],  

                     filled = True, rounded = True,  

                     special_characters = True) 

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

# Image(graph.create_png())



#Random Forest

n_estimators = 300

rf_clf = RandomForestClassifier(n_estimators = n_estimators, random_state = 0)

rf_clf.fit(X2_train, Y2_train)

print("training score(Random Forest)= ", rf_clf.score(X2_train, Y2_train))

print(rf_clf.feature_importances_)

pd.DataFrame(rf_clf.feature_importances_, index = X2_train.columns).plot.bar(figsize = (15, 5))

plt.ylabel("Importance")

plt.xlabel("Features")

plt.show()

for i, est in enumerate(rf_clf.estimators_):

    print(i)

    # 決定木の描画

    dot_data = StringIO() #dotファイル情報の格納先

    export_graphviz(est, out_file=dot_data,  

                         feature_names = X2_train.columns,  

                         class_names = ["successful", "failed"],  

                         filled=True, rounded=True,  

                         special_characters=True) 

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

    display(Image(graph.create_png()))

    

#AdaBoost

ab_clf = AdaBoostClassifier(random_state = 0, n_estimators = n_estimators)

ab_clf.fit(X2_train, Y2_train)

print("training score(Adaboost)= ", ab_clf.score(X2_train, Y2_train))

print(ab_clf.feature_importances_)

pd.DataFrame(ab_clf.feature_importances_, index = X2_train.columns).plot.bar(figsize = (15, 5))

plt.ylabel("Importance")

plt.xlabel("Features")

plt.show()

for i, est in enumerate(ab_clf.estimators_):

    print(i)

    # 決定木の描画

    dot_data = StringIO() #dotファイル情報の格納先

    export_graphviz(est, out_file=dot_data,  

                         feature_names = X2_train.columns,  

                         class_names = ["successful", "failed"],    

                         filled=True, rounded=True,  

                         special_characters=True) 

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

    display(Image(graph.create_png()))
"""

予測精度または識別精度を確認する

    回帰問題の場合は、MSE、RMSE、MAEを求める

    分類問題の場合は、混同行列を作成し、Accuracy、Recall、Precisionを求める

"""

#目標達成率（回帰）

# Y1_estimated = lr1.predict(X1_test)





best_model = load_model("model_{}.h5".format(study.best_trial.number))

Y1_pred = best_model.predict(X1_test)

print("R^2(test): " , r2_score(Y1_test, Y1_pred))

mape = np.mean(np.abs((Y1_test - Y1_pred) / Y1_test)) * 100

print("Mean Absolute Percentage Error(test): ", mape, "%")

mse = mean_squared_error(Y1_test, Y1_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(Y1_test, Y1_pred)

print("MSE: {}, RMSE: {}, MAE: {}".format(mse, rmse, mae))

print(Y1_pred[:20], "\n", Y1_test[:20])
#成功するかどうか（分類）-- Decicion Tree, Random Forest, Adaboost

print("Evaluation on Test Dataset:")

accuracy6 = accuracy_score(clf.predict(X2_test), Y2_test)

print("Accuracy(Decision Tree): ",             accuracy6)

accuracy7 = accuracy_score(rf_clf.predict(X2_test), Y2_test)

print("Accuracy(RandomForest): ",        accuracy7)

accuracy8 = accuracy_score(ab_clf.predict(X2_test), Y2_test)

print("Accuracy(AdaBoost): ",            accuracy8)
Y2_estimated = ab_clf.predict(X2_test)

c_matrix = pd.DataFrame(confusion_matrix(Y2_test, Y2_estimated), 

                        index=['正解 = 成功', '正解 = 失敗'], 

                        columns=['予測 = 成功', '予測 = 失敗'])

#Recall（実際に正しいもののうち、正であると予測された割合）

recall = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = 失敗"][0])

#Precision（正と予測したもののうち、どれくらい正しかったか）

precision = c_matrix["予測 = 成功"][0] / (c_matrix["予測 = 成功"][0] + c_matrix["予測 = 成功"][1])

#F1

f1 = f1_score(Y2_test, Y2_estimated)

print("AdaBoost\n Recall: {}, Precioin: {}, F1: {}".format(recall, precision, f1))

c_matrix