#Import all the tools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn


#plots to appear inside the notebook
%matplotlib inline

#Models from scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve

heart_disease_data  =  pd.read_csv("../input/heart-disease.csv")
#Size of data (Rows,Column)
heart_disease_data.shape
#Top 5 results
heart_disease_data.head()
# 1 - > Having heart disease
heart_disease_data["target"].value_counts()
heart_disease_data["target"].value_counts().plot(kind="bar",color=["brown","green"]);
#heart_disease_data.info()
#heart_disease_data.describe()
#Check missing values
heart_disease_data.isna().sum()
heart_disease_data.sex.value_counts()
# Compare target column with sex column
pd.crosstab(heart_disease_data.target,heart_disease_data.sex)

# Create a plot of crosstab

pd.crosstab(heart_disease_data.target,heart_disease_data.sex).plot(kind="bar",figsize=(10,6),
                    color=["brown","green"])
plt.title("Heart disease frequency  for sex")
plt.xlabel("0 = No disease 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])
plt.xticks(rotation=0)
#Graph b/w Age and Max Heart Rate for Heart Disease

plt.figure(figsize=(10,6))

#sbn.set_style("darkgrid")

#plt.style.use("dark_background")

#Scatter with positive
plt.scatter(heart_disease_data.age[heart_disease_data.target == 1],
        heart_disease_data.thalach[heart_disease_data.target==1],
           color="orange")

#Scatter with negative
plt.scatter(heart_disease_data.age[heart_disease_data.target == 0],
        heart_disease_data.thalach[heart_disease_data.target== 0],
           color="blue");

plt.title("Hear disease scatter plot for Age vs Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max heart rate")
plt.legend(["Disease","No-Disease"])
#Check the distribution of age with histogram 

heart_disease_data.age.plot.hist()

pd.crosstab(heart_disease_data.cp,heart_disease_data.target).plot(kind="bar",color=["yellow","blue"],figsize=(10,6))
plt.title("Hear disease frequency per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["Disease","No-Disease"])
plt.xticks(rotation=0)

pd.crosstab(heart_disease_data.cp,heart_disease_data.target)
# Find the correlation between our independent variables
corr_heart_matrix = heart_disease_data.corr()
corr_heart_matrix

#Negative Correlation = a relationship b/w two varibles in which one variable increases  as the other decreases
'''It also means that with the decrease of variable X, variable Y should increase instead. '''
corr_heart_matrix = heart_disease_data.corr()
plt.figure(figsize=(15, 10))

fig, ax = plt.subplots(figsize=(15,10))
ax = sbn.heatmap(corr_heart_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
#Split data into X and y
X = heart_disease_data.drop("target",axis=1)
y = heart_disease_data["target"]
# Split data into train and test sets
np.random.seed(42)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# Create a new models in a dictionary
models = {"Logistic Regrssion": LogisticRegression(),
         "KNN":KNeighborsClassifier(),
         "Random Forest":RandomForestClassifier()}

#Function for Fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores
model_compare_heart = pd.DataFrame(model_scores,index=["accuracy"])
#model_compare_heart.plot.bar()
model_compare_heart.T.plot.bar()
# Create a list of train and test scores
train_scores = []
test_scores = []

# create a list of different values for N_neighbors
neighbors= range(1,25)

knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors = i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    
plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 25, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")
# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Different RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
#Tune LogisticsRegression
np.random.seed(42)

# cv - >cross validation 5+5+5+5+5+5=20
#Setup random hyperparameter
hyper_log_lgres = RandomizedSearchCV(LogisticRegression(),
                                    param_distributions=log_reg_grid,
                                    cv= 5,n_iter=20,verbose=True)
hyper_log_lgres.fit(X_train,y_train)
#Best Hyperparameter

hyper_log_lgres.best_params_
#Evalauate the LogisticRegression model 
hyper_log_lgres.score(X_test,y_test)
#Tune RandomForestClassifier
np.random.seed(42)

# cv - >cross validation 5+5+5+5+5+5=20
#Setup RandomForestClassifier
hyper_log_rdmFostCls = RandomizedSearchCV(RandomForestClassifier(),
                                    param_distributions=rf_grid,
                                    cv= 5,n_iter=20,verbose=True)
hyper_log_rdmFostCls.fit(X_train,y_train)
#Best Hyperparameter
hyper_log_rdmFostCls.best_params_
#Evalauate the RandomForestClassifier model 
hyper_log_rdmFostCls.score(X_test,y_test)
#setup for LogisticRegression
log_reg_grid = {"C":np.logspace(-4,4,30),
               "solver":["liblinear"]}
gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                         cv= 5,
                         verbose=True)
#Fit grid hyperparameter search model
gs_log_reg.fit(X_train,y_train)

#Check best hyperparameter
gs_log_reg.best_params_
#Evaluate the grid search LogisticRegression model
gs_log_reg.score(X_test,y_test)
# Make preidctions on test data
y_preds = gs_log_reg.predict(X_test)
y_preds
# Plot ROC curve and calculate AUC metric
from sklearn.metrics import plot_roc_curve


#roc_curve(gs_log_reg, X_test)
plot_roc_curve(gs_log_reg,X_test,y_test)
#Confusion matrix
print(confusion_matrix(y_test,y_preds))
# Import Seaborn
import seaborn as sbn
sbn.set(font_scale=1.5) # Increase font size

def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sbn.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, 
                     cbar=True)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
plot_conf_mat(y_test, y_preds)
#Classification report
print(classification_report(y_test,y_preds))
#Check Best hyperparameter
gs_log_reg.best_params_
# New classifier with best parameter
clf = LogisticRegression(C=0.20433597178569418,solver='liblinear')
clf
#Crosss-validated accuracy
crss_acc = cross_val_score(clf,X,y,cv=5,scoring="accuracy")
#crss_acc
cv_acc_mean = np.mean(crss_acc)
cv_acc_mean
#Crosss-validated precision
crss_prec = cross_val_score(clf,X,y,cv=5,scoring="precision")
cv_prec_mean = np.mean(crss_prec)
cv_prec_mean
#Crosss-validated recall
crss_rec = cross_val_score(clf,X,y,cv=5,scoring="recall")
cv_rec_mean = np.mean(crss_rec)
cv_rec_mean
#Crosss-validated F1
crss_f1 = cross_val_score(clf,X,y,cv=5,scoring="f1")
crss_f1_mean = np.mean(crss_f1)
crss_f1_mean
# Visualizing cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc_mean,
                            "Precision": cv_prec_mean,
                            "Recall": cv_rec_mean,
                            "F1": crss_f1_mean},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);
#gs_log_reg.best_params_
# Fit an instance of Logistic Regression
clf  = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")
clf.fit(X_train,y_train)
#Check coeff
clf.coef_
# Match features to columns
features_dict = dict(zip(heart_disease_data.columns, list(clf.coef_[0])))
features_dict
# Visualize feature importance
features_heart_disease = pd.DataFrame(features_dict, index=[0])
features_heart_disease.T.plot.bar(title="Feature Importance", legend=False, color="blue");
pd.crosstab(heart_disease_data["sex"], heart_disease_data["target"])
### Slope (positive coefficient) with target
pd.crosstab(heart_disease_data["slope"], heart_disease_data["target"])


