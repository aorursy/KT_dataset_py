import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
import matplotlib.image as mpimg       # reading images to numpy arrays

import scipy.ndimage as ndi            # to determine shape centrality



# reading an image file using matplotlib into a numpy array

# good ones: 11, 19, 23, 27, 48, 53, 78, 218

img = mpimg.imread('../input/images/78.jpg')



# using image processing module of scipy to find the center of the leaf

cy, cx = ndi.center_of_mass(img)



plt.imshow(img, cmap='Set3')  # show me the leaf

plt.scatter(cx, cy)           # show me its center

plt.show()
train.head()
#test_id will be used later, so save it

test_ids = test.id



# Drop id 

train.drop(['id'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)
# find the sets of margin, shape and texture columns 

margin_cols = [col for col in train.columns if 'margin' in col]

shape_cols = [col for col in train.columns if 'shape' in col] 

texture_cols = [col for col in train.columns if 'texture' in col]
# correlation matrix for margin features

corr = train[margin_cols].corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(8, 6))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap)
train.columns
X = train.drop('species', axis=1)

y = train[["species"]]



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder().fit(y)

y=le.transform(y)
sss = StratifiedShuffleSplit (n_splits = 10, test_size=0.2, random_state=123)



scores = []

k = 0



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]
# https://www.kaggle.com/jeffd23/leaf-classification/10-classifier-showdown-in-scikit-learn

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=1000),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()

]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("+"*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("+"*30)
#sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log)



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



#sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log)



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()
from sklearn.preprocessing import StandardScaler

# Standardize features by removing the mean and scaling to unit variance

scaler = StandardScaler().fit(X_train)

X_train_scale = scaler.transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



params = {'C':[1000, 2000], 'tol': [0.001, 0.0001]}

#params = {'C':[2000], 'tol': [0.0001]}

log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial',class_weight='balanced',max_iter=400)

grid_search_lgr = GridSearchCV(log_reg, params, scoring='neg_log_loss', refit='True', n_jobs=-1, cv=5)



grid_search_lgr.fit(X_train_scale, y_train)



print('Best score: {}'.format(grid_search_lgr.best_score_))

print('Best parameters: {}'.format(grid_search_lgr.best_params_))
#RandomForestClassifier(n_estimators=1000)

# Test this 
X_test_scale = scaler.transform(test)



y_pred_prob = grid_search_lgr.predict_proba(X_test_scale)

print (y_pred_prob.shape)



# some manipulation of output

#y_pred_prob <- y_pred_prob^5

#for(x in seq_len(nrow(y_pred_prob))){

#  y_pred_prob[x,] <- y_pred_prob[x,]/sum(y_pred_prob[x,]) 

#}
submission = pd.DataFrame(y_pred_prob, columns=list(le.classes_))

submission.insert(0, 'id', test_ids)

submission.reset_index()



#submission = pd.DataFrame(y_pred_prob, index=test_ids, columns=le.classes_)



submission.head()
submission.to_csv('submission.csv', index=False)