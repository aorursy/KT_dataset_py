%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from nose.tools import *



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix, recall_score

from sklearn.feature_selection import RFE



import pickle



np.random.seed(13)
mushrooms_data = pd.read_csv("../input/mushroom-classification/mushrooms.csv", na_values = ["?"])

mushrooms_data.shape
mushrooms_data.info()
mushrooms_data.sample(10)
mushrooms_data["class"].unique()
mushrooms_data.apply(pd.Series.nunique)
mushrooms_data.drop("veil-type", axis = 1, inplace=True)

mushrooms_data.shape
mushrooms_data.isnull().sum()
mushrooms_data.drop("stalk-root", axis = 1, inplace=True)

mushrooms_data.shape
def plot_attribute_class_bar(attribute):

    """Function to plot classification bar of given attribute"""

    plot_df = mushrooms_data[[attribute, "class"]]

    plot_df = plot_df.groupby([attribute, "class"]).size().unstack(fill_value=0)

    labels = plot_df.index

    edible_freq = plot_df.e

    poisonous_freq = plot_df.p

    

    x = np.arange(len(labels))  # the label locations

    width = 0.4  # the width of the bars

    

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, edible_freq, width, label="Edible", color="g")

    rects2 = ax.bar(x + width/2, poisonous_freq, width, label="Poisonous", color="r")

    

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_ylabel("Frequency")

    ax.set_title("Classification bar of %s"%(attribute))

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.legend()

    

    

    def autolabel(rects):

        """Attach a text label above each bar in *rects*, displaying its height."""

        for rect in rects:

            height = rect.get_height()

            ax.annotate("{}".format(height),

                        xy=(rect.get_x() + rect.get_width() / 2, height),

                        xytext=(0, 0),

                        textcoords="offset points",

                        ha="center", va="bottom")

    

    

    autolabel(rects1)

    autolabel(rects2)

    

    fig.tight_layout()

    

    plt.show()
for column in mushrooms_data.columns[1:]:

    plot_attribute_class_bar(column)
mushrooms_data.groupby("class").size() / len(mushrooms_data)
mushrooms_data_attributes = mushrooms_data.drop("class", axis = 1)

mushrooms_data_attributes.shape
mushrooms_data_labels = mushrooms_data["class"]

mushrooms_data_labels.shape
mushrooms_data_attributes = pd.get_dummies(mushrooms_data_attributes)

mushrooms_data_attributes.shape
mushrooms_data_attributes.head(10)
attributes_train, attributes_test, labels_train, labels_test = train_test_split(

    mushrooms_data_attributes, mushrooms_data_labels, train_size = 0.75, stratify = mushrooms_data_labels)
attributes_train, attributes_val, labels_train, labels_val = train_test_split(

    attributes_train, labels_train, train_size = 0.80, stratify = labels_train)
attributes_train.shape, attributes_val.shape, attributes_test.shape
labels_train.shape, labels_val.shape, labels_test.shape
mushroom_model = LogisticRegression(solver = "lbfgs")

mushroom_model.fit(attributes_train, labels_train)
mushroom_model.score(attributes_train, labels_train)
mushroom_model.score(attributes_val, labels_val)
params = {

    "C": [1e-7, 1e-5, 1e-4, 1e-3, 0.01, 0.1],

    "fit_intercept": [True, False]

}

skf = StratifiedKFold(n_splits=10, random_state=None)

k_fold = skf.split(attributes_train, labels_train)

grid_search = GridSearchCV(LogisticRegression(solver="lbfgs"), params, make_scorer(f1_score, pos_label = "e"), cv = k_fold)

grid_search.fit(attributes_train, labels_train)
grid_search.best_params_
best_estimator = grid_search.best_estimator_

best_estimator.fit(attributes_train, labels_train)
best_estimator.score(attributes_train, labels_train)
best_estimator.score(attributes_val, labels_val)
grid_search.cv_results_
print(classification_report(labels_val, best_estimator.predict(attributes_val)))
model = LogisticRegression(solver="lbfgs")

rfe = RFE(model, 6)

X_rfe = rfe.fit_transform(attributes_train,labels_train)

model.fit(X_rfe,labels_train)

print(rfe.support_)

print(rfe.ranking_)
#no of features

nof_list=np.arange(1,20)            

high_score=0

#Variable to store the optimum features

nof=0           

score_list =[]

for n in range(len(nof_list)):

    model = LogisticRegression(C = 1e9, solver="lbfgs")

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(attributes_train,labels_train)

    X_test_rfe = rfe.transform(attributes_val)

    model.fit(X_train_rfe,labels_train)

    score = model.score(X_test_rfe,labels_val)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))
cols = list(attributes_train.columns)

model = LogisticRegression(solver="lbfgs")

#Initializing RFE model

rfe = RFE(model, nof)             

#Transforming data using RFE

X_rfe = rfe.fit_transform(attributes_train,labels_train)  

#Fitting the data to model

model.fit(X_rfe,labels_train)              

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)
opt_attributes_train = attributes_train[selected_features_rfe]

opt_attributes_val = attributes_val[selected_features_rfe]

opt_attributes_test = attributes_test[selected_features_rfe]
opt_attributes_train.shape, opt_attributes_val.shape, opt_attributes_test.shape
opt_mushroom_model = LogisticRegression(solver = "lbfgs")

opt_mushroom_model.fit(opt_attributes_train, labels_train)
opt_mushroom_model.score(opt_attributes_train, labels_train)
opt_mushroom_model.score(opt_attributes_val, labels_val)
opt_mushroom_model.score(opt_attributes_test, labels_test)
print(classification_report(labels_test, opt_mushroom_model.predict(opt_attributes_test)))
confusion_matrix(labels_test, opt_mushroom_model.predict(opt_attributes_test))
recall_score(labels_test, opt_mushroom_model.predict(opt_attributes_test), pos_label = "p")
recall_score(labels_test, opt_mushroom_model.predict(opt_attributes_test), pos_label = "e")
## Save the model to disk

#filename = "../input/models/final_model.p"

## Store data (serialize)

#with open(filename, "wb") as handle:

#    pickle.dump(opt_mushroom_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 

## Some time later...

# 

## Load the model from disk

## Load data (deserialize)

#with open(filename, "rb") as handle:

#    loaded_model = pickle.load(handle)

#

#assert(loaded_model.score(opt_attributes_test, labels_test) == opt_mushroom_model.score(opt_attributes_test, labels_test))