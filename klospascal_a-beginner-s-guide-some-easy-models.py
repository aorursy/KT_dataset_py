import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/avocado.csv")
df.head()
# Split Date into D,M,Y
date = df["Date"].apply(lambda x: x.split("-"))
df["Date_Year"] = date.apply(lambda date: date[0]).astype(int)
df["Date_Month"] = date.apply(lambda date: date[1]).astype(int)
df["Date_Day"] = date.apply(lambda date: date[2]).astype(int)
df.drop("Date", axis=1, inplace = True)

#Check if year and Date_Year are matching
match = df["Date_Year"] == df["year"]
for i in match:
    if i == False:
        print("No match")
df.drop("year", axis=1, inplace = True)

df.head()
#Get an Overview
print(df["Unnamed: 0"].max())
print(df.index.max())

df.drop("Unnamed: 0", axis=1, inplace = True)

print(df.type.unique())
print(df.region.unique())
#Maybe cluster to N, W, S, E
df.describe()
#Check for Outliers (Min/Max)
print('Columns with NAN:\n', df.isnull().sum())
sns.boxplot(x=df['AveragePrice'])
sns.boxplot(x=df['Total Volume'])
sns.boxplot(x=df['4046'])
sns.boxplot(x=df['4225'])
sns.boxplot(x=df['4770'])
sns.boxplot(x=df['Total Bags'])
sns.boxplot(x=df['Small Bags'])
sns.boxplot(x=df['Large Bags'])
sns.boxplot(x=df['XLarge Bags'])
sns.boxplot(x=df['Date_Year'])
sns.boxplot(x=df['Date_Month'])
sns.boxplot(x=df['Date_Day'])
sns.pairplot(df)
#Delete Outliers with ZScore
not_to_check = ["type","region"]

from scipy import stats
 
z = np.abs(stats.zscore(df.drop(not_to_check, axis=1)))
threshold = 3
row_to_drop = np.where(z > threshold)[0]
df = df.drop(df.index[row_to_drop])

print(len(row_to_drop), "Outliers removed")
sns.pairplot(df)
#Heatmap

corr = df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
input_list = ["Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags", "Large Bags", "XLarge Bags"]

def log_var(input_var):
    df[input_var] = np.log(1 + df[input_var])

log_var(input_list)

df.head()
sns.pairplot(df)
df["type"] = df["type"].replace("conventional",0).replace("organic",1)
df.drop("Date_Day", axis=1, inplace=True)
df.head()
df_region = df
df = pd.get_dummies(df, columns = ["region"])

df.head()
df_price = df

mean = df_price["AveragePrice"].mean()

df_price.loc[df_price["AveragePrice"] > mean, 'Price'] = 1
df_price.loc[df_price["AveragePrice"] < mean, 'Price'] = 0

df_price = df_price.drop("AveragePrice", axis=1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = df.drop(["type"], axis=1)
y = df["type"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
print(model.score(x_train, y_train))
df["type"].value_counts()
from sklearn.metrics import roc_curve, auc, precision_recall_curve

#ROC Curve
fpr_model, tpr_model, thresholds_model = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.plot(fpr_model, tpr_model)
plt.xlabel("P(FP)")
plt.ylabel("P(TP)")
#Recall Curve
precision_model, recall_model, thresholds_model = precision_recall_curve(y_test, model.predict_proba(x_test)[:,1])
plt.plot(precision_model, recall_model)
plt.xlabel("Precision")
plt.ylabel("Recall")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = df_region.drop(["region"], axis=1)
y = df_region["region"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x = df_region.drop(["region"], axis=1)
y = df_region["region"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(x_train, y_train)

print(knn.score(x_test, y_test))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x = df_region.drop(["region"], axis=1)
y = df_region["region"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

rfc = RandomForestClassifier(criterion = "entropy", n_estimators = 300)
rfc.fit(x_train, y_train)

print(rfc.score(x_test, y_test))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

x = df_region.drop(["region"], axis=1)
y = df_region["region"]

scores = cross_val_score(LogisticRegression(solver="liblinear"), x, y, cv = RepeatedKFold(n_repeats = 2))

print(np.mean(scores))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

x = df_region.drop(["region"], axis=1)
y = df_region["region"]

knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(x,y)
scores = cross_val_score(knn, x, y, cv = RepeatedKFold(n_repeats = 2))

print(np.mean(scores))
df_price.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
print(model.score(x_train, y_train))
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(x_train, y_train)

print(knn.score(x_test, y_test))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

rfc = RandomForestClassifier(criterion = "entropy", n_estimators = 300)
rfc.fit(x_train, y_train)

print(rfc.score(x_test, y_test))
from sklearn.model_selection import cross_val_score

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

scores = cross_val_score(LogisticRegression(solver="liblinear"), x, y, cv = RepeatedKFold(n_repeats = 2))

print(np.mean(scores))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

knn = KNeighborsClassifier(n_neighbors=5)
knn = knn.fit(x,y)
scores = cross_val_score(knn, x, y, cv = RepeatedKFold(n_repeats = 2))

print(np.mean(scores))
df_price["Price"].value_counts()
#ROC Curve
fpr_model, tpr_model, thresholds_model = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.plot(fpr_model, tpr_model, label = "LogisticRegression")

fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn.predict_proba(x_test)[:,1])
plt.plot(fpr_knn, tpr_knn, label = "KNN")

fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_test, rfc.predict_proba(x_test)[:,1])
plt.plot(fpr_rfc, tpr_rfc, label = "RFC")
plt.xlabel("P(FP)")
plt.ylabel("P(TP)")
plt.legend(loc = "best")
#Recall Curve
precision_model, recall_model, thresholds_model = precision_recall_curve(y_test, model.predict_proba(x_test)[:,1])
plt.plot(precision_model, recall_model, label = "LogisticRegression")

precision_knn, recall_knn, thresholds_knn = precision_recall_curve(y_test, knn.predict_proba(x_test)[:,1])
plt.plot(precision_knn, recall_knn, label = "KNN")

precision_rfc, recall_rfc, thresholds_rfc = precision_recall_curve(y_test, rfc.predict_proba(x_test)[:,1])
plt.plot(precision_rfc, recall_rfc, label = "RFC")
#Validation Curve
from sklearn.model_selection import validation_curve
param_range = np.array([40, 30, 20, 15, 10, 8, 7, 6, 5, 4, 3, 2, 1])

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

train_scores, test_scores = validation_curve(
    KNeighborsClassifier(), 
    x,
    y,
    param_name = "n_neighbors",
    param_range=param_range)

plt.plot(param_range, np.mean(train_scores, axis = 1))
plt.plot(param_range, np.mean(test_scores, axis = 1))
#Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x, y = shuffle(x, y)

train_sizes_abs, train_scores, test_scores = learning_curve(LogisticRegression(solver="liblinear"), x, y)
plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))
x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x, y = shuffle(x, y)

train_sizes_abs, train_scores, test_scores = learning_curve(KNeighborsClassifier(n_neighbors=5), x, y)
plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))
x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x, y = shuffle(x, y)

train_sizes_abs, train_scores, test_scores = learning_curve(RandomForestClassifier(criterion = "entropy", n_estimators = 300), x, y)
plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x_train, x_test, y_train, y_test = train_test_split(x,y)

# Choose the type of classifier. 
clf = KNeighborsClassifier()

# Choose some parameter combinations to try
parameters = {'n_neighbors': [1,3,5,7,9,11,13,15], 
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(x_train, y_train)

# Set the clf to the best combination of parameters
print(grid_obj.best_estimator_)
print(grid_obj.best_score_)
models = ["Logistic Regression", "Random Forest Classifier", "KNeighbors Classifier"]
scores = []
predictions_list = []

x = df_price.drop(["Price"], axis=1)
y = df_price["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

lr = LogisticRegression()
rfc = RandomForestClassifier(criterion = "entropy", n_estimators = 300)
knn = KNeighborsClassifier(n_neighbors = 5)

Logistic_Regression = lr.fit(x_train,y_train)
RFC = rfc.fit(x_train, y_train)
KNN = knn.fit(x_train,y_train)

scores.append(Logistic_Regression.score(x_train, y_train))
scores.append(RFC.score(x, y))
scores.append(KNN.score(x_train, y_train))

predictions_LR = Logistic_Regression.predict(x_test)
predictions_list.append(accuracy_score(y_test, predictions_LR))
predictions_RFC = RFC.predict(x_test)
predictions_list.append(accuracy_score(y_test, predictions_RFC))
predictions_KNN = KNN.predict(x_test)
predictions_list.append(accuracy_score(y_test, predictions_KNN))
sns.set_color_codes("muted")
sns.barplot(x=predictions_list, y=models, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()
