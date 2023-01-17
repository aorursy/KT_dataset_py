# Configuration

# EDA / Plotting

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Print the plots in the notebook

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation of models' performance function
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

plt.style.use("seaborn-talk")
heart_disease_raw = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart_disease_raw  # Show the first 5 and last 5 samples
print(f"Our dataset contains {heart_disease_raw.shape[0]} samples.\n")
print(f"Each sample contains {heart_disease_raw.shape[1]} features.")
print('This also can be seen in the shape of the Matrix above.')
# And there's no missing values! That's nice
heart_disease_raw.isna().sum()
new_labels = {   # Change the encoded version of the feature names
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type", # Check Extra Info Section
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholestoral (mg/dl)",
    "fbs": "FBS > 120 dg/dl",
    'restecg': "Resting Electrocardiographic Results", 
    'thalach': "Maximum Heart Rate Achieved",
    'exang': "Exercise Induced Angina", 
    'oldpeak': "ST Depression", 
    'slope': "Slope", 
    'ca': "Number of Major Vessels", 
    'thal': "THAL", 
    'target': "Target"    
}

sex_class = {
    1: "Male",
    0: "Female"
}

exercise_angina = {
    1: "Yes",
    0: "No"
}

fbs_values = {
    1: "Yes",
    0: "No"
}

target_label = {
    1: "Positive",
    0: "Negative"
}

#Replacements
heart_disease = heart_disease_raw.rename(columns = new_labels) # Return the copy
heart_disease["Sex"].replace(sex_class, inplace=True)
heart_disease["Exercise Induced Angina"].replace(exercise_angina, inplace=True)
heart_disease["FBS > 120 dg/dl"].replace(fbs_values, inplace=True)
heart_disease["Target"].replace(target_label, inplace=True)
heart_disease.head()

#heart_disease.to_csv('data/heart-disease-hf.csv')  # Save for future use
# hf stands fot human friendly
heart_disease.info()    # Same Memory Space, new data type added
def basic_metrics(label, print_label):
    
    """
    Basic Function to print central tendency metrics from any label passed.
    The heart_disease name for the data is assumed.
    
        @params
            label: Name of the column to evaluate
            print_label: Name of the column, formated to print.
    """
    print("Average", print_label, "of the Patients:", heart_disease[label].mean().astype(np.float16))
    print("Most Frequently", print_label, "of Patients:", heart_disease[label].median())
    print("Variance of the samples:", heart_disease[label].var().astype(np.float16))
    print("Standar Deviation of Samples from the Mean:", heart_disease[label].std().astype(np.float16))
    print()
    print("Minimun", print_label, ":", heart_disease[label].min())
    print("Max", print_label, ":", heart_disease[label].max())
basic_metrics("Age", "Age")
basic_metrics("Resting Blood Pressure (mm Hg)", "Blood Pressure")
basic_metrics("Serum Cholestoral (mg/dl)", "Serum Cholestoral")
heart_disease['Target'].value_counts()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6), sharey=True)

fig.suptitle("Distribution of Data", fontsize=18)

heart_disease["Age"].hist(ax=ax[0]);
ax[0].set_xlabel("Age")
ax[0].set_ylabel("Frequency")

heart_disease["Resting Blood Pressure (mm Hg)"].hist(ax=ax[1])
ax[1].set_xlabel("Blood Pressure (mm Hg)")

heart_disease["Serum Cholestoral (mg/dl)"].hist(ax=ax[2]);
ax[2].set_xlabel("Serum Cholestoral (mg/dl)");
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

fig.text(0.07, 0.5, "Frequency", rotation="vertical", fontsize=16)
fig.text(0, 0.9, "Distribution of the Features in the Dataset", fontsize=18)
fig.subplots_adjust(hspace=0.5)

pd.value_counts(heart_disease["Sex"]).plot(ax=ax[0,0], kind="pie", autopct='%.2f', 
                                          fontsize='x-large');
ax[0, 0].set_xlabel("Sex")
ax[0, 0].set_ylabel("")

pd.value_counts(heart_disease["Chest Pain Type"]).plot(ax=ax[0,1], kind="bar", rot=0);
ax[0, 1].set_xlabel("Chest Pain Type")

pd.value_counts(heart_disease["FBS > 120 dg/dl"]).plot(ax=ax[0,2], kind="pie",
                                                      autopct='%.2f', fontsize='x-large');
ax[0, 2].set_xlabel("FBS > 120 dg/dl")
ax[0, 2].set_ylabel("")

pd.value_counts(heart_disease["Exercise Induced Angina"]).plot(ax=ax[1,0], kind="bar", rot=0);
ax[1, 0].set_xlabel("Exercise Induced Angina")

pd.value_counts(heart_disease["THAL"]).plot(ax=ax[1,1], kind="bar", rot=0);
ax[1, 1].set_xlabel("THAL")

pd.value_counts(heart_disease["Number of Major Vessels"]).plot(ax=ax[1,2], kind="bar", rot=0)
ax[1, 2].set_xlabel("Number of Major Vessels");
target_column = heart_disease['Target']

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

fig.text(0.07, 0.5, "Frequency", rotation="vertical", fontsize='xx-large')

pd.crosstab(heart_disease["Sex"], target_column).plot(ax=ax[0,0], kind="bar", rot=0, legend=None);
ax[0, 0].set_xlabel("Sex")

pd.crosstab(heart_disease["Chest Pain Type"], target_column).plot(ax=ax[0,1], kind="bar", rot=0, legend=None);
ax[0, 1].set_xlabel("Chest Pain Type")

pd.crosstab(heart_disease["FBS > 120 dg/dl"], target_column).plot(ax=ax[0,2], kind="bar", rot=0, legend=None);
ax[0, 2].set_xlabel("FBS > 120 dg/dl")


pd.crosstab(heart_disease["Exercise Induced Angina"], target_column).plot(ax=ax[1,0], kind="bar", rot=0, legend=None);
ax[1, 0].set_xlabel("Exercise Induced Angina")

pd.crosstab(heart_disease["THAL"], target_column).plot(ax=ax[1,1], kind="bar", rot=0, legend=None);
ax[1, 1].set_xlabel("THAL")

pd.crosstab(heart_disease["Number of Major Vessels"], target_column).plot(ax=ax[1,2], kind="bar", rot=0, legend=None)
ax[1, 2].set_xlabel("Number of Major Vessels");

handles, label = ax[1, 2].get_legend_handles_labels()
fig.legend(handles, label, loc='lower left', fontsize='xx-large');
heart_disease['Target'].value_counts()
pd.crosstab(heart_disease["Sex"], target_column)
pd.crosstab(heart_disease["FBS > 120 dg/dl"], target_column)
fig, ax = plt.subplots(figsize=(10, 8))

correlation_map = heart_disease_raw.corr()

mask = np.zeros_like(correlation_map)
mask[np.triu_indices_from(mask)] = True
mask[np.diag_indices_from(mask)] = False # Mask to hide to upper triangle

ax = sns.heatmap(correlation_map, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask); 
mask = heart_disease['Target'] == 'Yes'

fig, ax = plt.subplots(nrows=2, figsize=(12, 15))
ax[0].scatter(x=heart_disease_raw['thalach'], y=heart_disease_raw['age'],
              c=heart_disease_raw['target'], cmap='Spectral', alpha=0.8, s=40)


ax[1].scatter(x=heart_disease_raw['trestbps'], y=heart_disease_raw['age'],
             c=heart_disease_raw['target'], cmap='Spectral');
X, y = heart_disease_raw.drop('target', axis=1), heart_disease_raw['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.head()
y_train.head()
def train_n_score(models, X_train, X_test, y_train, y_test):
    """
    Train (fit) and Evaluate a set of machine learning models, passed inside a dict.
    The Key of the Dict will be the name of the given machine learning model, and thus
    referenced with that name in the output of this function.
    
    The model will be scored with the build in score function that comes inside the model.
    i.e:
    `model.score()`
    
    return:
        A dictionary, with the score of each model passed. Model referenced using the name
        passed in the dictionary.
    """
    
    np.random.seed(42)
    
    model_store = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train each model
        
        model_store[name] = model.score(X_test, y_test)
    
    return model_store # Return the dictionary with the score of the models
models = {'Logistic Regression': LogisticRegression(max_iter=1000), # Avoid Converge Warning
          'KNN': KNeighborsClassifier(),
          'Random Forest': RandomForestClassifier()}
models_score = pd.DataFrame(train_n_score(models, X_train, X_test, y_train, y_test),
                           index=['Accuracy'])

ax = models_score.T.plot.bar(rot=0, xlabel='Models', ylabel='Accuracy (Max 1)',
                           colormap='Dark2', yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                           title='Accuracy of the Models', legend=False);
models_score
# Baseline Hyperparameters

log_model_hyperparam = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}

random_forest_hyperparam = {
    'n_estimators': np.arange(10, 1000, 50),
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': np.arange(2, 20, 2),
    'min_samples_leaf': np.arange(1, 20, 2)
}
# Tunning LogisticRegression model

np.random.seed(42)

rs_logistic_model = RandomizedSearchCV(LogisticRegression(), log_model_hyperparam,
                                      n_iter=20,# Iterate 20 times over the combinations
                                      verbose=True)

rs_logistic_model.fit(X_train, y_train) 
# Train the model, with cross validation and hyperparam
# Check the best combination of hyperparams
print('Best combination of params to the model:', rs_logistic_model.best_params_)
print(f'With the given params, the model scored: {rs_logistic_model.best_score_:.3f}')
# Tunnig the RandomForest model

np.random.seed(42)

rs_randomforest_model = RandomizedSearchCV(RandomForestClassifier(), 
                                           random_forest_hyperparam,
                                      n_iter=20,# Iterate 20 times over the combinations
                                      verbose=True)

rs_randomforest_model.fit(X_train, y_train) 
# Check the best combination of hyperparams
print('Best combination of params to the model:', rs_randomforest_model.best_params_)
print()
print(f'With the given params, the model scored: {rs_randomforest_model.best_score_:.3f}')
models_score
# Exhaustive combination of parameters

# Hyperparametrs v2
log_model_hyperparamv2 = {
    'C': np.logspace(-4, 4, 30),
    'solver': ['liblinear']
}

grid_logistic_model = GridSearchCV(LogisticRegression(), 
                                  param_grid=log_model_hyperparamv2,
                                  verbose=True)

grid_logistic_model.fit(X_train, y_train)
# Check the best combination of hyperparams
print('Best combination of params to the model:', grid_logistic_model.best_params_)
print(f'With the given params, the model scored: {grid_logistic_model.best_score_:.3f}')

# Didn't got any improvement :(
# First make predictions to have what to compare

y_predicts = grid_logistic_model.predict(X_test)
y_predicts
def plot_confusion_matrix(ax, y_test, y_preds):
    """
    Plot a Confusion Matrix using Seaborn's heatmap and Scikit Learn's confusion_matrix
    """
    sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False, ax=ax,
               cmap='bone')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

fig.subplots_adjust(wspace=0.4)

ax[0].set_title('ROC Curve')
plot_roc_curve(grid_logistic_model, X_test, y_test, ax=ax[0])
plot_confusion_matrix(ax[1], y_test, y_predicts);
accuracy = pd.DataFrame(cross_val_score(grid_logistic_model, X, y, 
                                        verbose=False, scoring='accuracy')).T
accuracy['Mean'] = np.mean(accuracy.T)
accuracy
recall = pd.DataFrame(cross_val_score(grid_logistic_model, X, y,
                                      verbose=False, scoring='recall')).T
recall['Mean'] = np.mean(recall.T)
recall
precision = pd.DataFrame(cross_val_score(grid_logistic_model, X, y, 
                                        verbose=False, scoring='precision')).T
precision['Mean'] = np.mean(precision.T)
precision
f1 = pd.DataFrame(cross_val_score(grid_logistic_model, X, y, 
                                        verbose=False, scoring='f1')).T
f1['Mean'] = np.mean(f1.T)
f1
pd.DataFrame({'Accuracy': accuracy['Mean'], 'F1': f1['Mean'], 'Recall': recall['Mean'],
         'Precision': precision['Mean']}, index=[0]).T.plot.bar(rot=0, legend=False,
                                                               title='Cross Validated Metrics',
                                                               figsize=(12,7));
grid_logistic_model.best_estimator_.coef_
feature_coefficients = dict(zip(heart_disease_raw.columns, 
                                list(grid_logistic_model.best_estimator_.coef_[0]))) # The number of features
feature_coefficients
pd.DataFrame(feature_coefficients, index=[0]).T.plot.bar(title='Feature Importance',
                                                         legend=False);