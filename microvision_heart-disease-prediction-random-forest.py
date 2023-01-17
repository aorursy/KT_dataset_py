import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import plotly.graph_objs as go
import plotly.offline as py

# For model Building
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

# For model evaluation
from sklearn.metrics import plot_roc_curve, auc
from sklearn.metrics import confusion_matrix
import eli5
from eli5.sklearn import PermutationImportance
import shap
from pdpbox import pdp, get_dataset, info_plots
# Import the dataset as CSV file
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
# Preview the dataset
df.head()
# Descriptive statistics
df.describe()
# Summary of the dataset
df.info()
# Check missing values
df.isnull().sum()
def histograms(data, features, rows, columns):
    """
    Plot histograms for each feature
    """
    fig = plt.figure(figsize=(15,15))
    for idx, feature in enumerate(features):
        ax = plt.subplot(rows, columns, idx+1)
        plt.hist(data[feature], bins='auto')
        ax.set_title(data.columns[idx], color='Red')
    plt.tight_layout()
    plt.show()
    return plt

histograms(df, df.columns, 5, 3);
def pi_chart(data):
    """
    The ratio of the target results
    """
    results = data['target'].value_counts()
    values = [results[0], results[1]]
    labels = ['Not Heart Disease', 'Heart Disease']
    fig_pie = go.Pie(labels=labels, values=values)
    py.iplot([fig_pie])
    return py
    
pi_chart(df);
def age_groups(ages):
    """
    Create age groups
    """
    age_group_list = []
    for age in ages:        
        if age // 10 == 2:
            age_group_list.append('20 - 29')  
        elif age // 10 == 3:
            age_group_list.append('30 - 39')       
        elif age // 10 == 4:
            age_group_list.append('40 - 49')
        elif age // 10 == 5:
            age_group_list.append('50 - 59')
        elif age // 10 == 6:
            age_group_list.append('60 - 69')
        elif age // 10 == 7:
            age_group_list.append('70 - 79')
        elif age // 10 == 8:
            age_group_list.append('80 - 89')
    return age_group_list    


def age_groups_for_sex(data):
    """
    Count age groups for sex:
    - sex: 0 = female; 1 = male
    """
    plt.figure(figsize=(10, 7))
    age_group = age_groups(df['age'])
    data_age = sorted(age_group)
    sns.countplot(data_age, hue='sex', data=data) 
    plt.title('Age Groups for Sex', fontsize=20)
    plt.xlabel('age groups', fontsize=15)
    plt.ylabel('count', fontsize=15)
    plt.legend(title='sex', fontsize=15)
    plt.tight_layout()
    plt.show()

age_groups_for_sex(df);
def HD_freq_sex_and_age(data):
    """
    Count heart disease and not heart disease frequency for sex
    Heart disease and not heart disease distribution for age
    - sex: 0 = female; 1 = male
    """
    plt.figure(figsize=(20, 8))
    # Heart disease frequency for sex
    plt.subplot(1, 2, 1)
    sns.countplot(x='target', hue='sex', data=data)
    plt.title('Heart Disease Frequency for Sex', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', fontsize=15)
    plt.ylabel('count', fontsize=15)
    plt.legend(title='sex', loc='best')
    
    # Heart disease distribution for age
    plt.subplot(1, 2, 2)
    sns.violinplot(x='target', y='age', hue='sex', data=data)
    plt.title('Heart Disease Distribution for Age', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    plt.ylabel('age', fontsize=15)
    plt.legend(title='sex', loc='best')
    plt.show()
    
HD_freq_sex_and_age(df);
def heat_map(data):
    """
    Correlation Heat Map of the dataset
    """
    plt.figure(figsize=(15, 7))
    sns.heatmap(data.corr(), annot=True, linewidth=0.2, 
                fmt='.2f', cmap='Reds')
    plt.title('Heatmap for the Dataset', fontsize=30)
    plt.show()
    
heat_map(df);
def symptoms_cp_exang(data):
    """
    Count cp (chest pain types) and exang (exercise-induced angina)
    """
    plt.figure(figsize=(20, 8))
    # cp
    plt.subplot(1, 2, 1)
    sns.countplot(x='target', hue='cp', data=data)
    plt.title('Chest Pain Type for target variable', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    plt.ylabel('Count', fontsize=15)
    
    # exang
    plt.subplot(1, 2, 2)
    sns.countplot(x='target', hue='exang', data=data)
    plt.title('Exercise-induced angina for target variable', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.show()

symptoms_cp_exang(df);
def risk_factors(data):
    """
    Show figures of risk factors for developing heart disease
    """
    fig = plt.figure(figsize=(20, 20))
    # trestbps: resting blood pressure
    plt.subplot(2, 2, 1)
    sns.violinplot(x='target', y='trestbps', data=data)
    plt.title('trestbps: Resting blood pressure (mmHg)', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    plt.ylabel('trestbps', fontsize=15)
       
    # chol: cholesterol 
    plt.subplot(2, 2, 2)
    sns.violinplot(x='target',y='chol', data=data)
    plt.title('chol: Serum cholesterol (mg/dl)', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    plt.ylabel('chol', fontsize=15)

    # fbs: fasting blood sugar
    plt.subplot(2, 2, 3)
    sns.countplot(x='target', hue='fbs', data=data)
    plt.title('fbs: Fasting blood sugar (> 120 mg/dl)', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    plt.ylabel('count', fontsize=15)
    plt.show()

    
risk_factors(df);
def heart_functions(data):
    """
    Show figures of heart functions
    """
    fig = plt.figure(figsize=(20, 25))
    # restecg: resting electrocardiographic results
    plt.subplot(3, 2, 1)
    sns.countplot(x='target', hue='restecg', data=data)
    plt.title('restecg: Resting electrocardiographic results', 
              fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)    

    # slope: the slope of the peak exercise ST segment
    plt.subplot(3, 2, 2)
    sns.countplot(x='target', hue='slope', data=data)
    plt.title('slope: The slope of the peak exercise ST segment', 
              fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    
    # thalach: maximum heart rate achieved
    plt.subplot(3, 2, 3)
    sns.violinplot(x='target', y='thalach', data=data)
    plt.title('thalach: Maximum heart rate achieved', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)    

    # oldpeak: ST depression induced by exercise relative to rest
    plt.subplot(3, 2, 4)
    sns.violinplot(x='target', y='oldpeak', data=data)
    plt.title('oldpeak: ST depression induced by exercise relative to rest', 
              fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    
    # ca: number of major vessels colored by fluoroscopy
    plt.subplot(3, 2, 5)
    sns.countplot(x='target', hue='ca', data=data)
    plt.title('ca: Number of major vessels colored by fluoroscopy', 
              fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)  
    
    # thal: thallium stress test   
    plt.subplot(3, 2, 6)
    sns.countplot(x='target', hue='thal', data=data)
    plt.title('thal: Thallium stress test ', fontsize=20)
    plt.xlabel('target (0 = Not Heart Disease; 1 = Heart Disease)', 
               fontsize=15)
    plt.ylabel('count', fontsize=15)
    plt.show()
    
heart_functions(df);
def features_with_age(data): 
    """
    Effect of features for heart disease prediction based on age
    """
    fig = plt.figure(figsize=(20, 15))
    # trestbps: resting blood pressure (mmHg) 
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='age', y='trestbps', hue='target', data=data)
    plt.title('trestbps: Resting blood pressure (mmHg)', fontsize=20)
    plt.xlabel('age', fontsize=15)
    plt.ylabel('trestbps', fontsize=15)
        
    # chol: cholesterol (mg/dl) 
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='age',y='chol', hue='target', data=data)
    plt.title('chol: Serum cholesterol (mg/dl)', fontsize=20)
    plt.xlabel('age', fontsize=15)
    plt.ylabel('chol', fontsize=15)
 
    # thalach: maximum heart rate achieved (bpm)
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='age', y='thalach', hue='target', data=data)
    plt.title('thalach: Maximum heart rate achieved (bpm)', fontsize=20)
    plt.xlabel('age',fontsize=15)
    plt.ylabel('thalach', fontsize=15)
       
    # oldpeak: ST depression induced by exercise relative to rest
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='age', y='oldpeak', hue='target', data=data)
    plt.title('oldpeak: ST depression induced by exercise relative to rest',
              fontsize=20)
    plt.xlabel('age',fontsize=15)
    plt.ylabel('oldpeak', fontsize=15)
    plt.show()


features_with_age(df);
def get_train_test_split(data):
    """
    Split into train and test set:
    - X = independent variables.
    - y = dependent variable.
    - Setup train_size, 80%, and test_size, 20%, of the dataset.
    """
    X = data.drop(['target'], axis=1)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        random_state=42)
    print('Shape of X_train', X_train.shape)
    print('Shape of X_test', X_test.shape)
    print('Shape of y_train', y_train.shape)
    print('Shape of y_test', y_test.shape)
    return X_train, X_test, y_train, y_test    
 
    
X_train, X_test, y_train, y_test = get_train_test_split(df);    
# Build and fit random forest model without hyperparameter tuning
rf_model = RandomForestClassifier().fit(X_train, y_train)
def get_test_score(model, X_test, y_test):
    """
    Evaluate random forest model on test dataset
    """
    model_score = model.score(X_test, y_test)*100
    return model_score
# Test score for rf_model
rfmodel_score = get_test_score(rf_model, X_test, y_test)
print('RandomForestClassifier Test Score:', rfmodel_score)
# Predict the target value
y_pred = rf_model.predict(X_test)

# Classification Report of rf_model
print(classification_report(y_pred, y_test))
def get_GridSearchCV_params(model, X_train, y_train):
    """
    Get the best hyperparameters of the model:
    - Set different GridSearchCV hyperparameters
    - Implement and fit GridSearchCV model
    """
    params_grid = {"max_depth": [2, 3, 4, 5],
                   "max_features": ['auto', 'sqrt', 'log2'],
                   "n_estimators":[0, 10, 50],
                   "random_state": [0, 10, 42]}

    grid_search = GridSearchCV(model, params_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params


# Apply GridSearchCV
best_param_dicts = get_GridSearchCV_params(rf_model, X_train, y_train)
grid_rf_model = RandomForestClassifier(max_depth=best_param_dicts['max_depth'],
                                       max_features=best_param_dicts['max_features'],
                                       n_estimators=best_param_dicts['n_estimators'],
                                       random_state=best_param_dicts['random_state']).fit(X_train, y_train)

# Test score for GridSearchCV model
grid_score = get_test_score(grid_rf_model, X_test, y_test)
print('GridSearchCV Score:', grid_score)
# Make prediction on test dataset
y_grid_pred = grid_rf_model.predict(X_test)

# Classification Report of GridSearchCV model
print(classification_report(y_grid_pred, y_test))
def plot_learning_curve(model, title, X, y, ylim=None, cv=5, n_jobs=4, 
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Draw the training and GridSearchCV testing learning curves
    """
    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=20)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Number of training samples', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.tick_params(labelsize=14)
    
    # Get training and test scores along with train_sizes
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, 
                                                            cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    
    # Calculate mean and standard deviation of training and test data
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(color='gray',linestyle='-')
    
    # Plot the learning curves
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color='g')
   
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')
    plt.legend(loc='best')
    
    return plt


title = 'Learning Curves (Random Forest Model)'
plot_learning_curve(grid_rf_model, title, X_train, y_train, ylim=None, 
                    cv=5, n_jobs=4, train_sizes=np.linspace(0.1, 1.0, 10));
def conf_matrix(y_predict, y_test):
    """
    Plot a confusion matrix of RandomForestClassifier with GridsearchCV
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_predict), 
                annot=True, fmt='d')
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    
conf_matrix(y_grid_pred, y_test)
# Plot ROC curve and calculalte AUC
plot_roc_curve(grid_rf_model, X_test, y_test);
def elis5_values(model, X_test, y_test):
    """
    Calculate and show permutation importance
    """
    perm_impt = PermutationImportance(model).fit(X_test, y_test)
    return eli5.show_weights(perm_impt, feature_names=X_test.columns.tolist())

elis5_values(grid_rf_model, X_test, y_test)
def partial_dependence_plots(model, X_test):
    """
    Calculater and show partial dependence plots of Top 3 important featurs:
    - thal: thallium stress test
    - cp: chest pain type
    - ca: number of major vessels colored by fluoroscopy
    """
    top5_features = X_test.loc[:,['thal', 'cp', 'ca']] 
    top5_feature_list = top5_features.columns.values.tolist()
    for i in top5_feature_list:
        pdp_fig = pdp.pdp_isolate(model=model, dataset=X_test,
                                  model_features=X_test.columns,
                                  feature=i)
        pdp.pdp_plot(pdp_fig, i)
        plt.show()
        
partial_dependence_plots(grid_rf_model, X_test)
def get_shap_values(model, X_test):
    """
    Calculate and show Shap values
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1],
                           shap_values[1], X_test)
    
get_shap_values(grid_rf_model, X_test)
def sum_shap_values(model, X_test):
    """
    Summary of SHAP value plots:
    - Average impact on random forest model output magnitude.
    - Impact (plus or minus) on random forest model output.
    """
    plt.figure(figsize=(10, 7))
    explainer2 = shap.TreeExplainer(model)
    shap_values2 = explainer2.shap_values(X_test)
    shap.summary_plot(shap_values2[1], X_test, plot_type='bar')
    shap.summary_plot(shap_values2[1], X_test)
    
sum_shap_values(grid_rf_model, X_test);