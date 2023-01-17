# import all the tools we need

# regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# modles of sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#  model evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve 

df = pd.read_csv('heart-disease.csv')
df.head()
# lets calc. how many of each classes there are
df['target'].value_counts()
df['target'].value_counts().plot(kind = 'bar', color = ['salmon', 'lightblue']);
df.info()
df.isna().sum()
df.sex.value_counts()
# compare target column with sex column
pd.crosstab(df.target, df.sex)
# create a plot of crosstab
pd.crosstab(df.target, df.sex).plot(kind = 'bar', 
                                    figsize = (10, 6), 
                                    color =['salmon', 'lightblue'])
plt.title('heart disease frequency for sex')
plt.xlabel('0 = no disease , 1 = disease')
plt.ylabel('amount')
plt.legend(['feamale', 'male'])
plt.xticks(rotation =0)
# creating an another figure
plt.figure(figsize =(10, 6))

# scatter with positive cases
plt.scatter(df.age[df.target ==1],
            df.thalach[df.target == 1],
            c = 'salmon')

# scatter for negative cases
plt.scatter(df.age[df.target ==0],
            df.thalach[df.target == 0],
            c = 'blue')


plt.title('age vs. max heart rate for heart disease')
plt.xlabel('age')
plt.ylabel('max heart rate')
plt.legend(['disease', 'not disease'])
# check distribution of age with histagram
df.age.plot.hist()
df.cp.value_counts()
pd.crosstab(df.target, df.cp)
pd.crosstab(df.cp, df.target).plot(kind = 'bar',
                                   figsize=(10,6),
                                   color = ['salmon', 'pink'])

plt.title("heart disease frequency per chest pain type")
plt.xlabel('chest pain')
plt.ylabel('amount')
plt.legend(['not disease', 'disease'])
df.head()
df.corr()
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot = True,
                 linewidths= 0.5,
                 fmt = '.2f',
                 cmap = 'YlGnBu')
df.head()
# split data into x and y
x = df.drop('target', axis =1)
y = df.target

x.head()
y.head()
# split data into train and test
np.random.seed(42)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)


# put models in a dictionary
clf = { 'logistic regressor' : LogisticRegression(),
         'k-n neighbour' : KNeighborsClassifier(),
         'random forest' : RandomForestClassifier()}

# create a function to fit and score models
def fit_and_score(clf, xtrain, xtest, ytrain, ytest):
    """
    fits and evaluates machine learning models
    """
    # set random seed
    
    np.random.seed(42)
    # make dictionary to store model scores
    model_scores = {}
    # loop through models
    for name, model in clf.items():
        # fit the model to the data
        model.fit(xtrain, ytrain)
        #evaluate the model and append it's score into model_scores
        model_scores[name] = model.score(xtest, ytest)
    return model_scores

model_scores = fit_and_score(clf = clf,
                             xtrain = xtrain,
                             xtest = xtest, 
                             ytrain = ytrain, ytest=ytest)
model_scores
model_compare = pd.DataFrame(model_scores, index =['accuracy'])
model_compare.T.plot.bar();
# let's tune KNN

train_scores = []
test_scores = []

# create a list of different values of n_neighbours
neighbours = range(1, 21)

# set up KNN instance
KNN = KNeighborsClassifier()

# loop through different n_neighbours
for i in neighbours:
    KNN.set_params(n_neighbors = i)

    # fit the algorithom
    KNN.fit(xtrain, ytrain)
    
    # update the training score list
    train_scores.append(KNN.score(xtrain, ytrain))
    
    # update the test score list
    test_scores.append(KNN.score(xtest, ytest))
train_scores
test_scores
plt.plot(neighbours, train_scores, test_scores)
plt.xlabel('no. of neighbours')
plt.ylabel('model score')
plt.xticks(np.arange(1, 21, 1))
plt.legend(['train scores', 'test scores'])

print(f'max knn score on the test data : {max(test_scores) * 100 :.2f}%')
# create a hyperperameter grid for logistic regression
lr_grid = {'C' : np.logspace(-4, 4, 20),
            'solver': ['liblinear']}

# create a hyperperameter grid for random forest
rf_grid = {'n_estimators' : np.arange(10, 1000, 50),
           'max_depth': [None, 3, 5, 10],
           'min_samples_split' : np.arange(2, 20, 2),
           'min_samples_leaf' : np.arange(1, 20, 2)}
# Tune LogisticRegression

np.random.seed(42)

#set up random hyperperameter search for LogisticResgression
rs_lr = RandomizedSearchCV(LogisticRegression(),
                           param_distributions= lr_grid,
                           cv = 5, 
                            n_iter= 20,
                            verbose=True)

#fit random hyperperameter search for LogisticRegression
rs_lr.fit(xtrain, ytrain)

rs_lr.best_params_
rs_lr.score(xtest, ytest)
# tune random forest classifier
np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv = 5, 
                           n_iter = 20, 
                           verbose=True)

rs_rf.fit(xtrain, ytrain)
rs_rf.score(xtest, ytest)
# creating grid for LogisticResgession Model
lr_grid= {'C' : np.logspace(-4, 4, 30),
          'solver' :['liblinear']}

# setup grid hyperperameter search for logistic regressor
lr_gs = GridSearchCV(LogisticRegression(),
                     param_grid= lr_grid,
                     cv = 5, 
                     verbose=True)

lr_gs.fit(xtrain, ytrain)
lr_gs.best_params_
lr_gs.score(xtest, ytest)
# let's make predictions with tuned model
y_preds = lr_gs.predict(xtest)
y_preds
ytest
# plot rOC curve and calculate AUC metrics
plot_roc_curve(lr_gs, xtest, ytest)
# confusion metrics
print(confusion_matrix(y_preds, ytest))
sns.set(font_scale = 1.5)

def plot_conf_metrics(ytest, y_preds):
    """
    this function plots confusion matrics using seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize = (3,3))
    ax = sns.heatmap(confusion_matrix(ytest, y_preds),
                     annot = True,
                     cbar = False)
    plt.xlabel('true labels')
    plt.ylabel('predicted labels')
    
plot_conf_metrics(ytest, y_preds)
    
    
print(classification_report(ytest, y_preds))
# check best parameters
lr_gs.best_params_
# create new classifier with best params
clf = LogisticRegression(C = 0.20433597178569418, solver = 'liblinear')
# cross validated accuracy
cv_acc = cross_val_score(clf, x, y, cv = 5, scoring ='accuracy')
cv_acc = np.mean(cv_acc)
cv_acc
# cross validated precision
cv_pre = cross_val_score(clf, x, y, scoring = 'precision')
cv_pre = np.mean(cv_pre)
cv_pre
# cross validated recall
cv_rec = cross_val_score(clf, x, y, scoring = 'recall')
cv_rec = np.mean(cv_rec)
cv_rec
# cross validated f1 score
cv_f1 = cross_val_score(clf, x, y, scoring = 'f1')
cv_f1 = np.mean(cv_f1)
cv_f1
# visualise cross validated matrics
cv_matrics = pd.DataFrame({'accuracy' : cv_acc,
              'precision': cv_pre,
              'recall': cv_rec,
               'f1 score': cv_f1},
                         index = [0])

cv_matrics.T.plot.bar(title ='cross validated matrics',
                      legend = False);
# fitting
clf.fit(xtrain, ytrain)
# check coef_
clf.coef_
# match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict
# visualize feature importance
feature_df = pd.DataFrame(feature_dict, index =[0])
feature_df.T.plot.bar(title = 'feature importance', legend = False)

pd.crosstab(df.sex, df.target) 
pd.crosstab(df.slope, df.target)
import pickle
# save model to disk
filename = 'heart-disease-final-model.sav'
pickle.dump(clf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(xtest, ytest)
print(result)
