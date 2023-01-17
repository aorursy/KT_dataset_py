import pandas as pd

import numpy as np



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import seaborn as sns



from imblearn.over_sampling import SMOTE



from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif, chi2

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')



data.head()
data.describe()
fig_age = go.Figure()



fig_age.add_trace(go.Histogram(x=data['age'],

                               marker_color='#6a6fff'))



fig_age.update_layout(

    title_text='Age Distribution',

    xaxis_title_text='Age',

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_age.show()
normal = data[data['anaemia']==0]



anemia = data[data['anaemia']==1]
colors= ['#7eff5e', '#ff5e79']



labels = ['Normal', 'Anemia']



values = [len(normal[normal['DEATH_EVENT'] == 1]), 

          len(anemia[anemia['DEATH_EVENT'] == 1])]



fig_anemia = go.Figure()



fig_anemia.add_trace(go.Pie(labels=labels, values=values,

                            hole=.4, marker_colors=colors))



fig_anemia.update_layout(

    title_text='Total number of deaths - Anemia',

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_anemia.show()
normal_cpk_level = data[(data['creatinine_phosphokinase'] >= 10) & 

                        (data['creatinine_phosphokinase'] <= 120)]



abnormal_cpk_level = data[(data['creatinine_phosphokinase'] < 10) | 

                          (data['creatinine_phosphokinase'] > 120)]
fig_creatinine = go.Figure()



fig_creatinine.add_trace(go.Histogram(x=data['creatinine_phosphokinase'],

                                      marker_color='#6a6fff'))



fig_creatinine.update_layout(

    title_text='Creatinine Phosphokinase Distribution',

    xaxis_title_text='Creatinine Phosphokinase (mcg/L)',

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_creatinine.show()
fig_creatinine = go.Figure()



fig_creatinine.add_trace(go.Box(y=data['creatinine_phosphokinase'], 

                                name='Box', marker_color='#6a6fff'))



fig_creatinine.update_layout(

    title_text='Creatinine Phosphokinase BoxPlot',

    yaxis_title_text='Creatinine Phosphokinase (mcg/L)', 

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_creatinine.show()
colors = ['#7eff5e', '#ff5e79']



labels = ['CPK Normal Level', 'CPK Abnormal Level']



values = [len(normal_cpk_level[normal_cpk_level['DEATH_EVENT'] == 1]),

          len(abnormal_cpk_level[abnormal_cpk_level['DEATH_EVENT'] == 1])]



fig_creatinine = go.Figure()



fig_creatinine.add_trace(go.Pie(labels=labels, values=values, 

                                hole=.4, marker_colors=colors))



fig_creatinine.update_layout(

    title_text='Total number of deaths - CPK',

    template = 'plotly_dark',

    width=750, 

    height=600

)
normal = data[data['diabetes']==0]



diabetes = data[data['diabetes']==1]
colors = ['#7eff5e', '#ff5e79']



labels = ['Normal', 'Diabetes']



values = [len(normal[normal['DEATH_EVENT'] == 1]), 

          len(diabetes[diabetes['DEATH_EVENT'] == 1])]



fig_diabetes = go.Figure()



fig_diabetes.add_trace(go.Pie(labels=labels, values=values,

                              hole=.4, marker_colors=colors))



fig_diabetes.update_layout(

    title_text='Total number of deaths - Diabetes',

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, height=600)

normal_ejection_fract = data[data['ejection_fraction'] >= 55]



reduced_ejection_fract = data[data['ejection_fraction'] <= 50]



borderline_ejection_fract = data[(data['ejection_fraction'] < 55) & 

                                 (data['ejection_fraction'] > 50)]
fig_eject_fract = go.Figure()



fig_eject_fract.add_trace(go.Histogram(x=data['ejection_fraction'],

                                      marker_color='#6a6fff'))



fig_eject_fract.update_layout(

    title_text='Ejection Fraction Distribution',

    xaxis_title_text='Ejection fraction (%)',

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, height=600

)



fig_eject_fract.show()
colors = ['#7eff5e', '#ff5e79', '#fddb3a']



labels = ['Normal Ejection Fraction', 'Reduced Ejection Fraction', 

          'Borderline Ejection Fraction ']



values = [len(normal_ejection_fract[normal_ejection_fract['DEATH_EVENT']==1]),

          len(reduced_ejection_fract[reduced_ejection_fract['DEATH_EVENT']==1]),

          len(borderline_ejection_fract[borderline_ejection_fract['DEATH_EVENT']==1])]



fig_eject_fract = go.Figure()



fig_eject_fract.add_trace(go.Pie(labels=labels, values=values,

                         hole=.4, marker_colors=colors))



fig_eject_fract.update_layout(

    title_text='Total number of deaths - Ejection Fraction',

    template = 'plotly_dark',

    width=750, 

    height=600

)
normal_blood_pressure = data[data['high_blood_pressure'] == 0]



high_blood_pressure = data[data['high_blood_pressure'] == 1]
color = ['#7eff5e', '#ff5e79']



labels = ['Normal Blood Pressure', 'High Blood Pressure']



values = [len(normal_blood_pressure[normal_blood_pressure['DEATH_EVENT'] == 1]), 

          len(high_blood_pressure[high_blood_pressure['DEATH_EVENT'] == 1])]



fig_pressure = go.Figure()



fig_pressure.add_trace(go.Pie(labels=labels, values=values,

                             hole=.4, marker_colors=colors))



fig_pressure.update_layout(

    title_text='Total number of deaths - Blood Pressure',

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, 

    height=600

)
normal_platelets_level = data[(data['platelets'] >= 150000) & (data['platelets'] <= 450000)]



abnormal_platelets_level = data[(data['platelets'] < 150000) | (data['platelets'] > 450000)]
fig_platelets = go.Figure()



fig_platelets.add_trace(go.Histogram(x=data['platelets'], 

                                      marker_color='#6a6fff'))



fig_platelets.update_layout(

    title_text='Platelets Distribution',

    xaxis_title_text='Platelets (kiloplatelets/mL)',

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, height=600

)



fig_platelets.show()
colors = ['#7eff5e', '#ff5e79']



labels = ['Normal Platelets Level', 'Abnormal Platelets Level']



values = [len(normal_platelets_level[normal_platelets_level['DEATH_EVENT']==1]),

          len(abnormal_platelets_level[abnormal_platelets_level['DEATH_EVENT']==1])]



fig_platelets = go.Figure()



fig_platelets.add_trace(go.Pie(labels=labels, values=values, 

                         hole=.4, marker_colors=colors))



fig_platelets.update_layout(

    title_text='Total number of deaths - Platelets',

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_platelets.show()
normal_range_creatinine = data[(data['serum_creatinine'] >= 0.7) & (data['serum_creatinine'] <= 1.2)]



out_range_creatinine = data[(data['serum_creatinine'] < 0.7) | (data['serum_creatinine'] > 1.2)]
fig_creatinine = go.Figure()



fig_creatinine.add_trace(go.Histogram(x=data['serum_creatinine'], 

                                      marker_color='#6a6fff'))



fig_creatinine.update_layout(

    title_text='Serum Creatinine Distribution',

    xaxis_title_text='Serum Creatinine(mg/dL)',

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, height=600

)



fig_creatinine.show()
colors = ['#7eff5e', '#ff5e79']



labels = ['Normal Creatinine Level', 'Abnormal Creatinine Level']



values = [len(normal_range_creatinine[normal_range_creatinine['DEATH_EVENT']==1]),

          len(out_range_creatinine[out_range_creatinine['DEATH_EVENT']==1])]



fig_creatinine = go.Figure()



fig_creatinine.add_trace(go.Pie(labels=labels, values=values, 

                         hole=.4, marker_colors=colors))



fig_creatinine.update_layout(

    title_text='Total number of deaths - Creatinine',

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_creatinine.show()
normal_sodium_level = data[(data['serum_sodium'] >= 135) & (data['serum_sodium'] <= 145)]

abnormal_sodium_level = data[(data['serum_sodium'] < 135) | (data['serum_sodium'] > 145)]
fig_sodium = go.Figure()



fig_sodium.add_trace(go.Histogram(x=data['serum_sodium'], 

                                  marker_color='#6a6fff'))



fig_sodium.update_layout(

    title_text='Serum Sodium Distribution',

    xaxis_title_text='Serum Sodium (mEq/L)',

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, height=600

)



fig_sodium.show()
colors = ['#7eff5e', '#ff5e79']



labels = ['Normal Sodium Level', 'Abnormal Sodium Level']



values = [len(normal_sodium_level[normal_sodium_level['DEATH_EVENT']==1]),

          len(abnormal_sodium_level[abnormal_sodium_level['DEATH_EVENT']==1])]



fig_sodium = go.Figure()



fig_sodium.add_trace(go.Pie(labels=labels, values=values, 

                         hole=.4, marker_colors=colors))



fig_sodium.update_layout(

    title_text='Total number of deaths - Sodium',

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_sodium.show()
colors = ['#013766', '#bc4558']



labels = ['Male', 'Female']



values = [len(data[(data['DEATH_EVENT'] == 1) & (data['sex'] == 1)]), 

          len(data[(data['DEATH_EVENT'] == 1) & (data['sex'] == 0)])]



fig_sex = go.Figure()



fig_sex.add_trace(go.Pie(labels=labels, values=values, 

                         hole=.4, marker_colors=colors))



fig_sex.update_layout(

    title_text='Total number of deaths - Sex',

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_sex.show()
labels = ['Smokers', 'No smokers']



values = [len(data[(data['DEATH_EVENT'] == 1) & (data['smoking'] == 1)]), 

          len(data[(data['DEATH_EVENT'] == 1) & (data['smoking'] == 0)])]



fig_smoking = go.Figure()



fig_smoking.add_trace(go.Pie(labels=labels, values=values,

                            hole=.4))



fig_smoking.update_layout(

    title_text='Total number of deaths - Smoking',

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_smoking.show()
fig_time = go.Figure()



fig_time.add_trace(go.Histogram(x=data['time'], 

                                marker_color='#6a6fff'))



fig_time.update_layout(

    title_text='Time Distribution',

    xaxis_title_text='Time (days)',

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_time.show()
survived = data[data['DEATH_EVENT'] == 0]



dead = data[data['DEATH_EVENT'] == 1]
fig_target = go.Figure()



fig_target.add_trace(go.Histogram(x=survived['DEATH_EVENT'], 

                                  name='Survived'))



fig_target.add_trace(go.Histogram(x=dead['DEATH_EVENT'], 

                                  name='No Survived'))



fig_target.update_layout(

    yaxis_title_text='Count', 

    bargap=0.05, 

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_target.show()
sns.pairplot(data, hue='DEATH_EVENT')
fig_corr = px.imshow(data.corr(), color_continuous_scale='peach')



fig_corr.update_layout(

    title={

        'text': "Features correlation",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'}, 

    template = 'plotly_dark',

    width=750, 

    height=600

)



fig_corr.show()
numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction',

                      'platelets', 'serum_creatinine', 'serum_sodium',

                      'time']



categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure',

                        'sex', 'smoking']



numerical_selector = SelectKBest(f_classif, k=4)



categorical_selector =  SelectKBest(chi2, k=1)



X_numerical = numerical_selector.fit_transform(data[numerical_features], 

                                                  data['DEATH_EVENT'])



X_categorical = categorical_selector.fit_transform(data[categorical_features],

                                                    data['DEATH_EVENT'])



print('Numerical features selected:', data[numerical_features].columns[numerical_selector.get_support()].to_list())



print('Categorical features selected:', data[categorical_features].columns[categorical_selector.get_support()].to_list())
X_selected = data[['age', 'ejection_fraction', 'serum_creatinine', 'time', 

                   'high_blood_pressure']]



y = data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, 

                                                    test_size = 0.2, 

                                                    stratify = y)
scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)



X_test_scaled = scaler.transform(X_test)
rfc_parameters = {'n_estimators' : [10, 20, 50, 100],

                  'criterion' : ['gini', 'entropy'],

                  'max_depth' : [3, 5, 7, 9, 10]

                 }



grid_search_rfc = GridSearchCV(estimator = RandomForestClassifier(), 

                           param_grid = rfc_parameters,

                           cv = 10,

                           n_jobs = -1)



grid_search_rfc.fit(X_train_scaled, y_train)



rfc = grid_search_rfc.best_estimator_



y_pred_rfc = rfc.predict(X_test_scaled)



rfc_accuracy = accuracy_score(y_test, y_pred_rfc)



rfc_cv_score = cross_val_score(rfc, X_selected, y, cv=10).mean()
print(classification_report(y_test, y_pred_rfc))
knn_parameters = {'n_neighbors' : [i for i in range(1, 40)]}



grid_search_knn = GridSearchCV(estimator = KNeighborsClassifier(), 

                           param_grid = knn_parameters,

                           cv = 10,

                           n_jobs = -1)



grid_search_knn.fit(X_train_scaled, y_train)



knn = grid_search_knn.best_estimator_



y_pred_knn = knn.predict(X_test_scaled)



knn_accuracy = accuracy_score(y_test, y_pred_knn)



knn_cv_score = cross_val_score(knn, X_selected, y, cv=10).mean()
print(classification_report(y_test, y_pred_knn))
no_skill = [0 for _ in range(len(y_test))]



rfc_probs = rfc.predict_proba(X_test_scaled)



rfc_probs = rfc_probs[:, 1]



knn_probs = knn.predict_proba(X_test_scaled)



knn_probs = knn_probs[:, 1]
rfc_auc = roc_auc_score(y_test, rfc_probs)



knn_auc = roc_auc_score(y_test, knn_probs)



print('(RFC) ROC AUC score:', rfc_auc)



print('(KNN) ROC AUC score:', knn_auc)
ns_fpr, ns_tpr, a =  roc_curve(y_test, no_skill)



rfr_fpr, rfr_tpr, a =  roc_curve(y_test, rfc_probs)



knn_fpr, knn_tpr, a =  roc_curve(y_test, knn_probs)
fig_auc = go.Figure()



fig_auc.add_trace(go.Scatter(x=ns_fpr, y=ns_tpr, mode='lines',line_dash='dot', 

                             name = 'No Skill (AUC = 0.5)'))



fig_auc.add_trace(go.Scatter(x=rfr_fpr, y=rfr_tpr, mode='lines', 

                             name=('RFC (AUC = %f)' %rfc_auc)))



fig_auc.add_trace(go.Scatter(x=knn_fpr, y=knn_tpr, mode='lines', 

                             name=('KNN (AUC = %f)' %knn_auc)))



fig_auc.update_layout(xaxis_title = 'False Positive Rate', 

                      yaxis_title='True Positive Rate', 

                      width=700, height=500)



fig_auc.show()
models = [('RFC', rfc_accuracy, rfc_cv_score), 

          ('KNN', knn_accuracy, knn_cv_score)]



model_comparasion = pd.DataFrame(models, columns=['Model', 'Accuracy Score', 'CV Score'])



model_comparasion.head()