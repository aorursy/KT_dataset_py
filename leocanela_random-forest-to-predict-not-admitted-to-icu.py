import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, make_scorer
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

raw_data = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
demo_lst = [i for i in raw_data.columns if "AGE_" in i]
demo_lst.append("GENDER")
vitalSigns_lst = raw_data.iloc[:,193:-2].columns.tolist()

raw_data = raw_data[['PATIENT_VISIT_IDENTIFIER']+demo_lst+vitalSigns_lst+['WINDOW','ICU']]
raw_data.head(5)
df_ICU_list = raw_data.groupby("PATIENT_VISIT_IDENTIFIER").agg({"ICU":(list)})
raw_data['ICU_list'] = raw_data.apply(lambda row: df_ICU_list.loc[row['PATIENT_VISIT_IDENTIFIER']]['ICU'], axis=1)
raw_data['VALID_WINDOW'] = raw_data.apply(lambda row: row['ICU_list'].index(1)-1 if 1 in row['ICU_list'] else 4, axis=1)
for index, row in raw_data.iterrows():
    if index%5 > row['VALID_WINDOW']:
        raw_data.loc[index, vitalSigns_lst] = np.nan
(raw_data[raw_data.isnull().any(axis=1)].isna().sum() / len(raw_data)).mean()
def agg_function(column):
    if column.name in demo_lst: return min(column)
    elif column.name in vitalSigns_lst: return list(column.dropna())[0] if len(column.dropna()) > 0 else np.nan
    elif column.name == 'ICU': return max(column)
    else: return column
        
agg_data = raw_data.groupby("PATIENT_VISIT_IDENTIFIER", as_index=False).agg(agg_function)
(agg_data[agg_data.isnull().any(axis=1)].isna().sum() / len(agg_data)).max()
agg_data.dropna(inplace=True)
agg_data.dtypes[:10]
agg_data['AGE_PERCENTIL'] = agg_data['AGE_PERCENTIL'].apply(lambda row: 9 if row[0] == 'A' else int(row[0]) )
data = agg_data.copy(deep=True)
low_variances = data.var()
low_variances = low_variances.loc[low_variances < 0.01]
low_variances.sort_values()
data.shape
model_data = data.drop(low_variances.index, axis=1)
model_data.shape
width=8
height=6
fig = plt.figure(figsize=(width,height))
sns.heatmap(model_data.iloc[:, 1:-1].corr(), cmap='bwr')
corr_values = model_data.iloc[:, 1:-1].corr(method='pearson').abs()
corrdf = corr_values.unstack().sort_values(ascending=False).to_frame()
corrdf.reset_index(inplace=True)
corrdf.columns=['Feature 1', 'Feature 2', 'Pearsons R']
corrdf['SAME_FEAT'] = corrdf.apply(lambda row: 1 if row['Feature 1'] == row['Feature 2'] else 0, axis=1)
corrdf = corrdf.loc[corrdf['SAME_FEAT'] == 0]
corrdf.drop('SAME_FEAT', axis=1, inplace=True)
multicol_df = corrdf[corrdf['Pearsons R'] > 0.8].iloc[::2]
multicol_df['Feat1 Target Corr'] = multicol_df.apply(lambda row: model_data.corr().abs()['ICU'][row['Feature 1']], axis=1)
multicol_df['Feat2 Target Corr'] = multicol_df.apply(lambda row: model_data.corr().abs()['ICU'][row['Feature 2']], axis=1)
multicol_df.reset_index(drop=True, inplace=True)
multicol_df
multicol_columns_to_drop = multicol_df.apply(lambda x: x['Feature 1'] if x['Feat1 Target Corr'] < x['Feat2 Target Corr'] else x['Feature 2'], axis=1).unique()
multicol_columns_to_drop
model_data.drop(multicol_columns_to_drop, axis=1, inplace=True)
width=8
height=6
fig = plt.figure(figsize=(width,height))
sns.heatmap(model_data.iloc[:, 1:-1].corr(), cmap='bwr')
model_data.columns
tnr = make_scorer(recall_score,pos_label=0)
X = model_data.iloc[:,1:-1]
y = model_data['ICU']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
X_train.shape
clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
print('True Negative Rate: ', tnr(clf, X_test, y_test))
importance = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=["Importance"])
importance.sort_values(by='Importance').plot(kind='barh');
model_data_fe = model_data.copy(deep=True)
model_data_fe['SpO2_DIFF'] = (data['OXYGEN_SATURATION_MAX'] - data['OXYGEN_SATURATION_MIN'])
model_data_fe['SpO2_MEAN'] = (data['OXYGEN_SATURATION_MAX'] + data['OXYGEN_SATURATION_MIN'])/2
scaler = MinMaxScaler((1,3))
BPSMean = scaler.fit_transform(data['BLOODPRESSURE_SISTOLIC_MEAN'].to_numpy().reshape(-1,1))
BPDMean = scaler.fit_transform(data['BLOODPRESSURE_DIASTOLIC_MEAN'].to_numpy().reshape(-1,1))
PAMMean = ((BPDMean*2) / BPSMean)/3
model_data_fe['PAM_MEAN'] = PAMMean
model_data_fe['PULSE_PRESSURE'] = BPSMean - BPDMean
X = model_data_fe.iloc[:,1:].drop(['ICU'], axis=1)
y = model_data_fe['ICU']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
X_train.shape
clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
print('True Negative Rate: ', tnr(clf, X_test, y_test))

importance = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=["Importance"])
importance.sort_values(by='Importance').plot(kind='barh');
X_train.shape
param_grid = { 
    'n_estimators': [100, 500, 1000, 2000],
    'max_features' : [3, 4, 6, 8],
    'max_leaf_nodes' : [2, 5, 10],
    'criterion' :['gini', 'entropy'],
}
gs_rfc = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42), 
    param_grid=param_grid, 
    cv=5, 
    scoring=tnr)
gs_rfc.fit(X_train, y_train)
gs_rfc.best_params_
final_clf = RandomForestClassifier(**gs_rfc.best_params_)
final_clf.fit(X_train, y_train)
importance = pd.DataFrame(final_clf.feature_importances_, index=X.columns, columns=["Importance"])
importance.sort_values(by='Importance').plot(kind='barh');
def score(pred, real):
    accuracy  = accuracy_score(pred, real)
    roc_auc   = roc_auc_score(pred, real)
    recall    = recall_score(pred, real)
    specificity = tnr(final_clf, X_test, y_test)
    return pd.DataFrame.from_dict(
        {'True Negative Rate': specificity, 'Accuracy': accuracy, 'ROC_AUC': roc_auc, 'Recall': recall},
        orient='index',
        columns=['Score']
    )

score(final_clf.predict(X_test), y_test)