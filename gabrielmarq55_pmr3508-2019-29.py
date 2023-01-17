import pandas as pd
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
# import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
adult = pd.read_csv("Databases/train_data.csv",
        na_values="?")
adult.shape
adult = adult.drop('Id', axis=1)
adult.head()
fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(adult.shape[1]) / cols)
for i, column in enumerate(adult.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if adult.dtypes[column] == np.object:
        adult[column].value_counts().plot(kind="bar", axes=ax)
    else:
        adult[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
nadult = adult.dropna()
# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

# Calculate the correlation and plot it
encoded_data, _ = number_encode_features(nadult)
sns.heatmap(encoded_data.corr(), square=True)
plt.show()

nadult_l50 = nadult.loc[adult['income'] == '<=50K']
nadult_h50 = nadult.loc[adult['income'] == '>50K']
fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(nadult.shape[1]) / cols)
for i, column in enumerate(nadult.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.hist([nadult_h50[column].dropna(),nadult_l50[column].dropna()],stacked=True)
    ax.set_title(column)
    ax.legend(['>50','<=50'],loc=1, prop={'size': 10})
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=1, wspace=0.2)
testAdult = pd.read_csv("Databases/test_data.csv",
        na_values="?")
testAdult.shape
testAdult = testAdult.drop('Id', axis=1)
testAdult.head()
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult.income
XtestAdult = testAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
best_meanScore = 0

for n in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(Xadult, Yadult) 
    for folds in range(2,10):
        score_rf = cross_val_score(knn, Xadult, Yadult, cv=folds, scoring='accuracy').mean()
        if score_rf > best_meanScore:
            best_meanScore = score_rf
            best_pair = [n, folds] #best pair of n and cv
    
print(best_pair)
best_meanScore
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(Xadult, Yadult) 
score_rf = cross_val_score(knn, Xadult, Yadult, cv=9, scoring='accuracy')
score_rf
Ypred = knn.predict(XtestAdult)
savepath = "predictions.csv" 
prev = pd.DataFrame(Ypred, columns = ["income"]) 
prev.to_csv(savepath, index_label="Id") 
prev