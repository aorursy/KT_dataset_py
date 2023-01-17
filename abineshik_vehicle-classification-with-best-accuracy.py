import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
df = pd.read_csv('vehicle.csv')
df.head()
df.shape
df.info()
df.isna().sum()
col = df.columns[df.isnull().any()]
df.describe().T
sns.countplot(df['class'])
fig, ax = plt.subplots(figsize = [20, 8])
corr = df.corr() #Finding correlation of all the features
sns.heatmap(corr, annot = True)
fig, ax = plt.subplots(figsize = [15, 6])
corr_pos = corr.abs() # Making all the values postive
mask = corr_pos < 0.8 #Mask the correlation less than 0.8
sns.heatmap(corr, annot = True, mask = mask)
selected_columns = [ 'radius_ratio','pr.axis_aspect_ratio', 'max.length_aspect_ratio', 
                    'scatter_ratio','skewness_about', 'skewness_about.1',
                    'hollows_ratio']
i_median = SimpleImputer(strategy = 'median')
df_median = df.copy() #Making a copy to impute the dataset
df_median[col] = i_median.fit_transform(df[col]) # Imputing with median for missing values
corr_median = df_median.corr() # finding the correlation for imputed dataframe
diff = corr - corr_median
print(diff.max()) #printing only the maximum correlation of each features
fix, ax = plt.subplots(nrows = 2, ncols = 4, figsize = [20, 7])

for col, axes in zip(selected_columns, ax.flatten()):
  sns.distplot(df[col], ax = axes)
  mean = df[col].mean()
  median = df[col].median()
  axes.axvline(mean, color = 'r', linestyle = '--') # Vertical line along axis to indicate the mean
  axes.axvline(median, color = 'b', linestyle = '--') # Vertical line to indicate the median
  axes.legend({'Mean': mean, 'Median': median})
fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = [20, 7])

for col, axes in zip(selected_columns, ax.flatten()):
  sns.boxplot(df[col], ax = axes)
fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = [20, 7])

for col, axes in zip(selected_columns, ax.flatten()):
  sns.boxplot(x = col, y = 'class', data = df, ax = axes)
def outlier_removal(ar): # Function to replace outliers with 1.5*IQR in both lower and higher side
  for i in range(7):
    p = np.percentile(ar[:, i], [25, 75])
    iqr = p[1] - p[0]
    q1 = p[0]- 1.5*iqr
    q3 = p[1]+ 1.5*iqr
    ar[:, i][ar[:, i]<q1] = q1
    ar[:, i][ar[:, i]>q3] = q3
  return ar
seed = 3
pipeline = Pipeline(
    [('impute', SimpleImputer(strategy = 'median')),
    ('outlier', FunctionTransformer(outlier_removal)),
    ('scale', StandardScaler()),
    ('model', SVC(random_state = seed))
       ])
x = df[selected_columns]
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.3, random_state = seed)
performance = pd.DataFrame(columns = ['Type', 'Train Accuracy', 'Test Accuracy', 'Mean cross validation score', 'Deviation'])
normal_model = pipeline.fit(x_train, y_train)
cv = StratifiedKFold()
normal_score = cross_val_score( normal_model, X=x, y=y, scoring = 'accuracy', n_jobs = -1, cv = cv)
performance = performance.append({'Type': 'Model without PCA and tuning',
                                 'Train Accuracy': normal_model.score(x_train, y_train)*100,
                                 'Test Accuracy': normal_model.score(x_test, y_test)*100,
                                 'Mean cross validation score': normal_score.mean()*100,
                                 'Deviation': 2*100*normal_score.std()}, ignore_index = True)

print("--------------------------Model without PCA and tuning-----------------------------------\n")
print("Train accuracy : {:2.2f}%".format(normal_model.score(x_train, y_train)*100))
print("Test accuracy : {:2.2f}%".format(normal_model.score(x_test, y_test)*100))
print("Cross validation mean score: {:2.2f}% with deviation (+/-{:2.2f}%)" .format(normal_score.mean()*100, 2*100*normal_score.std()))
params_svm = {
    'model__C': [i for i in range(1, 15, 1)],
    #'model__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'model__kernel' : ['rbf'], # Checked with all kernaels but RBF performed well, so using RBF in all the upcoming iterations
    'model__gamma' : np.logspace(-5, -1, 30)
            }


model_tuning = GridSearchCV(pipeline, params_svm, scoring = 'accuracy')
model_tuning.fit(x_train, y_train)
tuned_score = cross_val_score( model_tuning.best_estimator_, X=x, y=y, scoring = 'accuracy', n_jobs = -1, cv = cv)
performance = performance.append({'Type': 'Model without PCA and with tuning',
                                 'Train Accuracy': model_tuning.score(x_train, y_train)*100,
                                 'Test Accuracy': model_tuning.score(x_test, y_test)*100,
                                 'Mean cross validation score': tuned_score.mean()*100,
                                 'Deviation': 2*100*tuned_score.std()}, ignore_index = True)
print("--------------------------Model without PCA and with tuning-----------------------------------\n")
print("Train accuracy : {:2.2f}%".format(model_tuning.score(x_train, y_train)*100))
print("Test accuracy : {:2.2f}%".format(model_tuning.score(x_test, y_test)*100))
print("Cross validation mean score: {:2.2f}% with deviation (+/-{:2.2f}%)" .format(tuned_score.mean()*100, 2*100*tuned_score.std()))
def outlier_removal_pca(ar):
  for i in range(18):
    p = np.percentile(ar[:, i], [25, 75])
    iqr = p[1] - p[0]
    q1 = p[0]- 1.5*iqr
    q3 = p[1]+ 1.5*iqr
    ar[:, i][ar[:, i]<q1] = q1
    ar[:, i][ar[:, i]>q3] = q3
  return ar
from sklearn.decomposition import PCA
pipeline_pca = Pipeline(
    [('impute', SimpleImputer(strategy = 'median')),
    ('outlier', FunctionTransformer(outlier_removal_pca)),
    ('scale', StandardScaler()),
    ('pca', PCA(random_state = seed)),
    ('model', SVC( random_state = seed))
       ])
x_pca = df.drop('class', axis = 1)
y_pca = df['class']
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, y_pca, stratify = y_pca, test_size = 0.3, random_state = seed)
pipeline_pca.fit(x_train_pca, y_train_pca)
normal_score_pca = cross_val_score( pipeline_pca, X=x_pca, y=y_pca, scoring = 'accuracy', n_jobs = -1, cv = cv)
performance = performance.append({'Type': 'Model with PCA and without tuning',
                                 'Train Accuracy': pipeline_pca.score(x_train_pca, y_train_pca)*100,
                                 'Test Accuracy': pipeline_pca.score(x_test_pca, y_test_pca)*100,
                                 'Mean cross validation score': normal_score_pca.mean()*100,
                                 'Deviation': 2*100*normal_score_pca.std()}, ignore_index = True)
print("--------------------------Model with PCA and without tuning-----------------------------------\n")
print("Train accuracy : {:2.2f}%".format(pipeline_pca.score(x_train_pca, y_train_pca)*100))
print("Test accuracy : {:2.2f}%".format(pipeline_pca.score(x_test_pca, y_test_pca)*100))
print("Cross validation mean score: {:2.2f}% with deviation (+/-{:2.2f}%)" .format(normal_score_pca.mean()*100, 2*100*normal_score_pca.std()))
params_svm = {
    'model__C': [i for i in range(1, 15, 1)],
    #'model__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'model__kernel' : ['rbf'],
    'model__gamma' : np.logspace(-5, -1, 30)
            }
model_tuning_pca = GridSearchCV(pipeline_pca, params_svm, scoring = 'accuracy')
model_tuning_pca.fit(x_train_pca, y_train_pca)
tuned_score_pca = cross_val_score( model_tuning_pca.best_estimator_, X=x_pca, y=y_pca, scoring = 'accuracy', n_jobs = -1, cv = cv)
performance = performance.append({'Type': 'Model with PCA and tuning',
                                 'Train Accuracy': model_tuning_pca.score(x_train_pca, y_train_pca)*100,
                                 'Test Accuracy': model_tuning_pca.score(x_test_pca, y_test_pca)*100,
                                 'Mean cross validation score': tuned_score_pca.mean()*100,
                                 'Deviation': 2*100*tuned_score_pca.std()}, ignore_index = True)
print("--------------------------Model with PCA and tuning-----------------------------------\n")
print("Train accuracy : {:2.2f}%".format(model_tuning_pca.score(x_train_pca, y_train_pca)*100))
print("Test accuracy : {:2.2f}%".format(model_tuning_pca.score(x_test_pca, y_test_pca)*100))
print("Cross validation mean score: {:2.2f}% with deviation (+/-{:2.2f}%)" .format(tuned_score_pca.mean()*100, 2*100*tuned_score_pca.std()))
model_tuning_pca.best_params_
n_features = np.arange(2, 19)
model = pipeline_pca
pca_analysis = pd.DataFrame(columns = ['n_features', 'Explained_variation', 'Accuracy'])
for i in n_features:
  #Using the best paramters estimated by the model in hyper parameter tuning
  model.set_params(pca__n_components=  i, model__C = 14, model__gamma = 0.0386, model__kernel = 'rbf') 
  model.fit(x_train_pca, y_train_pca)
  pca_analysis = pca_analysis.append({'n_features': i,
                       'Explained_variation': sum(model.named_steps['pca'].explained_variance_ratio_)*100,
                      'Accuracy': model.score(x_test_pca, y_test_pca)*100 }, ignore_index = True)
  
fig, ax = plt.subplots(figsize = [10,7])
sns.lineplot(x= 'n_features', y ='value' , hue = 'variable', data = pd.melt(pca_analysis, 'n_features'))
plt.grid()
model.set_params(pca__n_components=  8, model__C = 14, model__gamma = 0.0386, model__kernel = 'rbf') 
model.fit(x_train_pca, y_train_pca)
final_score_pca = cross_val_score( model, X=x_pca, y=y_pca, scoring = 'accuracy', n_jobs = -1, cv = cv)
performance = performance.append({'Type': 'Model with PCA, tuning and dimension reduction',
                                 'Train Accuracy': model.score(x_train_pca, y_train_pca)*100,
                                 'Test Accuracy': model.score(x_test_pca, y_test_pca)*100,
                                 'Mean cross validation score': final_score_pca.mean()*100,
                                 'Deviation': 2*100*final_score_pca.std()}, ignore_index = True)
print("--------------------------Model with PCA and Dimension reduction-----------------------------------\n")
print("Train accuracy : {:2.2f}%".format(model.score(x_train_pca, y_train_pca)*100))
print("Test accuracy : {:2.2f}%".format(model.score(x_test_pca, y_test_pca)*100))
print("Cross validation mean score: {:2.2f}% with deviation (+/-{:2.2f}%)" .format(final_score_pca.mean()*100, 2*100*final_score_pca.std()))
performance