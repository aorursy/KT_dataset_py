# This Python 3 environment 
import pandas as pd
import numpy as np  
import os
import json
import copy
from sklearn.externals import joblib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))


# Any results you write to the current directory are saved as output.
df = pd.DataFrame()
files_directory = sorted([f for f in os.listdir("../input/train_simplified")]) # Sort the directory first
training_categories = len(files_directory) #340
category_samples = 60
test_sample_cats = 112199 # Number of rows to load from the test set
downsize = 25 # Resolution of images, max is 255 but please don't do this unless you've got time to kill / compuational power
zero_lst = np.zeros(downsize**2, dtype=int)

df_from_each_file = [pd.read_csv("../input/train_simplified/"+f, nrows=category_samples) for f in files_directory]
df = pd.concat(df_from_each_file, ignore_index=True)

print("From {a} to {z}..".format(a=files_directory[0], z=files_directory[training_categories-1]))
print("-------")


df = df[df["recognized"]==True]
df['drawing'] = df['drawing'].apply(json.loads)
df = df[["drawing","word"]].reset_index(drop=True)
print(df.info())
## Gather all the points
def pointList(draw_list):
    point_list = []
    for n in range(0,len(draw_list)):
        for x,y in list(zip(draw_list[n][0],draw_list[n][1])):
            point_list.append((x,y))
    return point_list
## Bound the points, and scale
def transform(pt_list):
    xlist = []
    ylist = []
    scl_list = []
    for x,y in pt_list:
        xlist.append(x)
        ylist.append(y)
    xmx = max(xlist)
    xmn = min(xlist)
    ymx = max(ylist)
    ymn = min(ylist)
    for x,y in pt_list:
        try:
            x_scl = round((x-xmn)*downsize/(xmx - xmn))
            y_scl = round((y-ymn)*downsize/(ymx - ymn))
        except ZeroDivisionError:
            x_scl = round((x-xmn)*downsize/(xmx - xmn + 0.0001))
            y_scl = round((y-ymn)*downsize/(ymx - ymn + 0.0001))
        scl_list.append((x_scl, y_scl))
    return scl_list
def ConvertTo1D(scl_pt_list):
    indx_lst = copy.copy(zero_lst)
    for x,y in scl_pt_list:
            i = x + downsize*(y-1)
            indx_lst[i-1] = 1
    return indx_lst
print("Preparing df_vec..")
df_vec = df["drawing"].apply(pointList
                     ).apply(transform
                     ).apply(ConvertTo1D
                     ).apply(pd.Series)
print("Learning..")
X = df_vec
y = df["word"]
print("{} accuracy would be better than guessing".format(round(1/training_categories, 4)))
# Let's get learning!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

classifiers = {#"K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=2),
               "Random Forest" : RandomForestClassifier(n_estimators=10, random_state=0),
               #"Support Vector Clf" : LinearSVC(penalty="l2", random_state=0),
               "Logistic Regression" : LogisticRegression(penalty="l2", random_state=0),
               #"Perceptron" : Perceptron(penalty="l2", random_state=0),
               #"Naive Bayes" : GaussianNB(),
               #"Decision Tree" : DecisionTreeClassifier(random_state=0)
               }
# Find best Test_Train test accuracy

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

splits=2
rounding_prec = '.4f'
clfy_acc = []
for name,clfy in classifiers.items():
    res = format(cross_val_score(clfy, X, y, cv=splits).mean(), rounding_prec)
    print("{s}-fold split: {n} test accuracy = {r}".format(s=splits, n=name, r=res))
    clfy_acc.append(res)
  
# Save the best trained model
print("..Saving..")
clf_acc_finder = list(zip(classifiers.keys(), clfy_acc))
for cl,ac in clf_acc_finder:
    if ac == max(clfy_acc):
        fitted_clf = classifiers[cl].fit(X, y)
#        joblib.dump(classifiers[cl], 
#                            "{c}_Trained_{cat}_by_{sam}.joblib".format(c=str(cl),
#                                                                       cat=training_categories,
#                                                                       sam=category_samples)) 
        print("Model Selected! {c} with {a} testing accuracy:".format(c=cl, a=ac))
print("..Loading & transforming test data..")
test_df = pd.read_csv("../input/test_simplified.csv", nrows=test_sample_cats)
X_test = test_df["drawing"].apply(json.loads
                          ).apply(pointList
                          ).apply(transform
                          ).apply(ConvertTo1D
                          ).apply(pd.Series)

X_test_keys = test_df["key_id"]

## Load in the model

# model_loaded = os.listdir("../input/training-vector-representations-of-google-q-d/")[-2] # Picks the model out
# clf_loaded = joblib.load("../input/training-vector-representations-of-google-q-d/"+str(model_loaded))
# print("..{} model selected!".format(str(model_loaded)[0:-24]))
print("..Predicting..")

pred = fitted_clf.predict(X_test)
pred = [p.replace(" ", "_") for p in pred]
df_pred = pd.DataFrame.from_dict({"key_id":X_test_keys, 
                                  "word":["{pr}".format(pr=p) for p in pred]}, orient="columns")

print("Submission shape: {s}".format(s=df_pred.shape))
print("Number of possible classes: {n}".format(n=training_categories))

df_pred.to_csv("SubmissionMG.csv", index=False)
print("Sucessfully created file") 