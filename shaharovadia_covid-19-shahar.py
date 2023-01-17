import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
#allow displaying large tables:
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#remove warnings for too many figures:
plt.rcParams.update({'figure.max_open_warning': 0})

dataset = pd.read_excel('../input/covid19/dataset.xlsx', index_col=0)
dataset['Urine - pH'].replace('Não Realizado', np.nan, inplace=True)
dataset['Urine - pH'] = dataset['Urine - pH'].astype('float64')
dataset.replace('not_done', np.nan, inplace=True)
dataset['Urine - Leukocytes'].replace('<1000', '999', inplace=True)
dataset['Urine - Leukocytes'] = dataset['Urine - Leukocytes'].astype('float64')
dataset.replace('not_detected', 0, inplace=True)
dataset.replace('detected', 0, inplace=True)
dataset.replace('negative', 0, inplace=True)
dataset.replace('positive', 1, inplace=True)
dataset.replace('absent', 0, inplace=True)
dataset.replace('present', 1, inplace=True)
df_temp = dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']].astype("str").apply(LabelEncoder().fit_transform)
dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']] = df_temp.where(~dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']].isna(), dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']])
dataset['Urine - Aspect'] = dataset['Urine - Aspect'].astype("float64")
dataset['Urine - Urobilinogen'] = dataset['Urine - Urobilinogen'].astype("float64")
dataset['Urine - Crystals'] = dataset['Urine - Crystals'].astype("float64")
dataset['Urine - Color'] = dataset['Urine - Color'].astype("float64")
covidOnly = dataset.loc[dataset['SARS-Cov-2 exam result'] == 1]
print("Array size with covid patient only ",covidOnly.shape)
# .notnull() return new array with True\False for non-null elements
# .sum(axis=0) summing over the columns (this is the default of .sum())
# >0  return new True\False for every column with non-null elements
# we use .loc[:,...] to return all rows, and non-null columns
features_to_remove = covidOnly.notnull().sum(axis=0) == 0

#features_to_remove is a mask
#features_to_remove[features_to_remove] returns a reduced array of the True elements only
print("features with no observations at all:",features_to_remove[features_to_remove].index.values)


covidOnly = covidOnly.loc[:,covidOnly.notnull().sum(axis=0) > 0]
print("Array size with covid patient only and non-empty examinations", covidOnly.shape)
corrMatrix = covidOnly.select_dtypes(exclude='object').corr(min_periods=15)
#get column name where the sum of not NA elements is 0 
columnsWithoutCorr = corrMatrix.columns[corrMatrix.notna().sum() == 0]

print("features to remove: ", columnsWithoutCorr)

#remove all rows\columns with these features
corrMatrix=corrMatrix.drop(columnsWithoutCorr, axis=0).drop(columnsWithoutCorr,axis=1)

print("remaining corr-matrix size: ",corrMatrix.shape)
matrix = np.triu(corrMatrix)
plt.figure(figsize = (15, 15))
sns.heatmap(corrMatrix.abs(), cmap = plt.cm.RdYlBu_r, vmin = -0.2, annot = False, vmax = 0.8, mask=matrix)

sortedCorrMatrix = corrMatrix.abs().unstack().reset_index().dropna().sort_values(by=0, ascending=False)
sortedCorrMatrix = sortedCorrMatrix[sortedCorrMatrix['level_0'] != sortedCorrMatrix['level_1']]
sortedCorrMatrix = sortedCorrMatrix[::2]
sortedCorrMatrix[sortedCorrMatrix[0] > 0.5]
def plot_joint(x_name,y_name):
    fig, ax =plt.subplots(1,3)
    g1 = sns.jointplot(x=x_name,y=y_name, data=dataset, kind="kde", ax=ax[0])
    g1.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+", ax=ax[0])
    g1.ax_joint.collections[0].set_alpha(0)
    
    g2 = sns.jointplot(x=x_name,y=y_name, data=corrMatrix, kind="kde", ax=ax[2])
    g2.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+", ax=ax[0])
    g2.ax_joint.collections[0].set_alpha(0)
    
#for index, row in sortedCorrMatrix[0:1].iterrows():
#    plot_joint(row['level_0'],row['level_1'])
 
from scipy.stats import pearsonr

def corrfunc(x,y, ax=None, label=0, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    # Unicode for lowercase rho (ρ)
    rho = '\u03C1' + str(label)
    ax.annotate(f'{rho} = {r:.2f}', xy=(.1, .9 - (0.1*label)), xycoords=ax.transAxes)
    
#for index, row in sortedCorrMatrix[0:5].iterrows():
#    jointplot_w_hue(data=dataset, x=row['level_0'],y=row['level_1'], hue='SARS-Cov-2 exam result')
#    g = sns.pairplot(dataset, hue="SARS-Cov-2 exam result", vars=[row['level_1'], row['level_0']]) 
#    g.map_upper(corrfunc)
def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df,
        height=8
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        rho = '\u03C1'
        commonNonEmptyIndex =  ~(np.isnan(df_group[col_x]) | np.isnan(df_group[col_y]) ) 
        p,r= pearsonr(df_group[col_x][commonNonEmptyIndex].values, df_group[col_y][commonNonEmptyIndex].values)
       # print(("No COVID" if name == 0 else "COVID") + f'\n{rho} = {p:.2f}, r={r:.2f}' + " N ="+str(commonNonEmptyIndex.sum()))
        legends.append(("No COVID" if name == 0 else "COVID") + f'\n{rho} = {p:.2f}' + " N="+str(commonNonEmptyIndex.sum()))
        
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,            
            vertical=True
        )
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.legend(legends, framealpha=0.5)

relevantMat = sortedCorrMatrix[sortedCorrMatrix[0]>0.5]
patient_care_setting=['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)','Patient addmited to intensive care unit (1=yes, 0=no)']
for index, row in relevantMat.iterrows():
#    plot_joint(row['level_0'],row['level_1'])
#    print(row['level_0'],row['level_1'],dataset[row['level_0']].dtype,dataset[row['level_1']].dtype)
    if (row['level_0'] in patient_care_setting or row['level_1'] in patient_care_setting): 
        continue
    multivariateGrid(row['level_0'], row['level_1'], 'SARS-Cov-2 exam result', df=dataset)
# Correlation with output variable
cor_target = corrmat["SARS-Cov-2 exam result"]
# Selecting highly correlated features
relevant_features = cor_target[cor_target>0.15].index.tolist()
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(abs(dataset[relevant_features].corr().iloc[0:1, :]), yticklabels=[relevant_features[0]], xticklabels=relevant_features, vmin = 0.0, square=True, annot=True, vmax=1.0, cmap='RdPu')
nof_positive_cases = len(dataset_positive.index)
nof_negative_cases = len(dataset_negative.index)
fig1, ax1 = plt.subplots()
ax1.pie([nof_positive_cases, nof_negative_cases], labels=['Positive cases', 'Negative cases'], autopct='%1.1f%%', startangle=90, colors=['#c0ffd5', '#ffc0cb'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
columns_to_exclude = missing_data_positive.index[missing_data_positive['Percent'] > 0.998].tolist()
dataset.drop(columns=columns_to_exclude, inplace=True)
print(columns_to_exclude)
# Redefine dataset positive and negative
dataset_negative = dataset[dataset['SARS-Cov-2 exam result'] == 0]
dataset_positive = dataset[dataset['SARS-Cov-2 exam result'] == 1]
dataset_negative = dataset_negative.dropna(axis=0, thresh=20)
X = pd.concat([dataset_negative, dataset_positive])
nof_positive_cases = len(dataset_positive.index)
nof_negative_cases = len(dataset_negative.index)
fig1, ax1 = plt.subplots()
ax1.pie([nof_positive_cases, nof_negative_cases], labels=['Positive cases', 'Negative cases'], autopct='%1.1f%%', startangle=90, colors=['#c0ffd5', '#ffc0cb'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
corrmat = abs(X.corr())
# Correlation with output variable
cor_target = corrmat["SARS-Cov-2 exam result"]
# Selecting highly correlated features
relevant_features = cor_target[cor_target>0.15].index.tolist()
f, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(abs(X[relevant_features].corr().iloc[0:1, :]), yticklabels=[relevant_features[0]], xticklabels=relevant_features, vmin = 0.0, square=True, annot=True, vmax=1.0, cmap='RdPu')
X_with_relevant_features = X[relevant_features]
y_with_relevant_features = X_with_relevant_features['SARS-Cov-2 exam result']
X_with_relevant_features.drop(columns=['SARS-Cov-2 exam result'], inplace=True)
y = X['SARS-Cov-2 exam result']
X.drop(columns=['SARS-Cov-2 exam result'], inplace=True)
def print_scores(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%% " % (precision * 100))
    print("Recall: %.2f%% " % (recall * 100))
    print("AUC: %.2f%% " % (roc * 100))
def plot_confusion_matrix(y_test, y_pred):
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, cmap='RdPu')
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_with_relevant_features, y_with_relevant_features, test_size=0.2, random_state=42)
print("Number of samples in train set: %d" % y_train_rf.shape)
print("Number of positive samples in train set: %d" % (y_train_rf == 1).sum(axis=0))
print("Number of negative samples in train set: %d" % (y_train_rf == 0).sum(axis=0))
print()
print("Number of samples in test set: %d" % y_test_rf.shape)
print("Number of positive samples in test set: %d" % (y_test_rf == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (y_test_rf == 0).sum(axis=0))
imp = SimpleImputer(strategy='median')
imp = imp.fit(X_with_relevant_features)
rfc = RandomForestClassifier()

# Define parameters and grid search
n_estimators = [100, 300, 500, 800, 1000]
max_depth = [5, 8, 15, 25, 30]
grid = dict(n_estimators=n_estimators, max_depth=max_depth)
grid_search = GridSearchCV(estimator=rfc, param_grid=grid, n_jobs=-1, cv=10, scoring='recall', error_score=0)
grid_result = grid_search.fit(imp.transform(X_train_rf), y_train_rf)
print("Best recall: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
rfc.n_estimators = grid_result.best_params_['n_estimators']
rfc.max_depth = grid_result.best_params_['max_depth']
                                   
model_rfc = rfc.fit(imp.transform(X_train_rf), y_train_rf)
y_pred_rf = model_rfc.predict(imp.transform(X_test_rf))
print_scores(y_test_rf, y_pred_rf)
plot_confusion_matrix(y_test_rf, y_pred_rf)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, test_size=0.3, random_state=42)
X_test_xgb, X_validation_xgb, y_test_xgb, y_validation_xgb = train_test_split(X_test_xgb, y_test_xgb, test_size=0.5, random_state=42)
print("Number of samples in train set: %d" % y_train_xgb.shape)
print("Number of positive samples in train set: %d" % (y_train_xgb == 1).sum(axis=0))
print("Number of negative samples in train set: %d" % (y_train_xgb == 0).sum(axis=0))
print()
print("Number of samples in validation set: %d" % y_validation_xgb.shape)
print("Number of positive samples in validation set: %d" % (y_validation_xgb == 1).sum(axis=0))
print("Number of negative samples in validation set: %d" % (y_validation_xgb == 0).sum(axis=0))
print()
print("Number of samples in test set: %d" % y_test_rf.shape)
print("Number of positive samples in test set: %d" % (y_test_xgb == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (y_test_xgb == 0).sum(axis=0))
model_xgb = XGBClassifier()

# Define parameters and grid search
n_estimators = [100, 300, 500, 700]
subsample = [0.5, 0.7, 1.0]
max_depth = [6, 7, 9]
grid = dict(n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
grid_search = GridSearchCV(estimator=model_xgb, param_grid=grid, n_jobs=-1, cv=10, scoring='roc_auc', error_score=0)
grid_result = grid_search.fit(X_train_xgb, y_train_xgb)
print("Best AUC: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
model_xgb.n_estimators = grid_result.best_params_['n_estimators']
model_xgb.subsample = grid_result.best_params_['subsample']
model_xgb.max_depth = grid_result.best_params_['max_depth']
model_xgb.fit(X_train_xgb, y_train_xgb, eval_metric='auc', eval_set=[(X_train_xgb, y_train_xgb), (X_validation_xgb, y_validation_xgb)], verbose=False)
val_predictions_xgb = model_xgb.predict(X_validation_xgb)
print_scores(y_validation_xgb, val_predictions_xgb)
predictions_xgb = model_xgb.predict(X_test_xgb)
print_scores(y_test_xgb, predictions_xgb)
plot_confusion_matrix(y_test_xgb, predictions_xgb)
feature_importances = model_xgb.get_booster().get_fscore()
feature_importances_df = pd.DataFrame({'Feature Score': list(feature_importances.values()), 'Features': list(feature_importances.keys())})
feature_importances_df.sort_values(by='Feature Score', ascending=False, inplace=True)
feature_importances_df = feature_importances_df.head(15)
f, ax = plt.subplots(figsize=(7, 7))
plt.title('Top 15 Feature Importances', fontsize=14)
sns.barplot(x=feature_importances_df['Feature Score'], y=feature_importances_df['Features'])
f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
plt.title('ROC Curve', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
fpr, tpr, thresholds = roc_curve(y_test_xgb, model_xgb.predict_proba(X_test_xgb)[:,1]) 
sns.lineplot(x=fpr, y=tpr, color=sns.color_palette("husl", 8)[-2], linewidth=2, label="AUC = 95.41%")