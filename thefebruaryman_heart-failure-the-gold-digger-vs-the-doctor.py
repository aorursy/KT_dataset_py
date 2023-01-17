import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#Suppressing all warnings
warnings.filterwarnings("ignore")

%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, plot_roc_curve

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from pandas_profiling import ProfileReport
def loadData():
    return pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

data = loadData()
report = ProfileReport(data,progress_bar=False)
report.to_notebook_iframe()
data_corr = data.corr(method='pearson')

data_corr_columns = data_corr.columns
data_corr_index = data_corr.index

data_corr = data_corr.to_numpy()

np.fill_diagonal(data_corr, np.nan)

data_corr = pd.DataFrame(data_corr, columns=data_corr_columns, index=data_corr_index)
plt.subplots(figsize=(15,10))
sns.heatmap(data=data_corr, cmap='YlGnBu', annot=True).set_title('Correlation Heat Map');
# dropping target value from featues
features = data.drop(['DEATH_EVENT'],axis=1)
# extracting DEATH_EVENT as an array
target = data['DEATH_EVENT']

# Spliting the data into train test sets
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=30,
                                                    stratify=target)
# lists of columns that will go through the pipeline
scaled_cols = ['age',
               'creatinine_phosphokinase',
               'ejection_fraction',
               'platelets',
               'serum_creatinine',
               'serum_sodium']
nothing_cols = ['anaemia',
                'diabetes',
                'high_blood_pressure',
                'sex',
                'smoking',
                'time']
scaler = StandardScaler()

numeric_transform = Pipeline(steps=[
    ('scaler', StandardScaler())])

nothing_transform = Pipeline(steps=[
    ('nothing', None)])


preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, scaled_cols),
        ('nothing', nothing_transform, nothing_cols)
    ])


rf = Pipeline(steps=[('preprocessor', preprocess),
#                     ('classifier', RandomForestClassifier())
                    ])
# transforming the training and testing data
trans_train_data = rf.fit_transform(X_train)

trans_test_data = rf.transform(X_test)
# using the Random Forest Classifier with it default parameters
rfc = RandomForestClassifier()
# training the model with our transformed data
rfc.fit(trans_train_data, y_train)
# getting predictions to our test data
predictions = rfc.predict(trans_test_data)
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)

cm = pd.DataFrame(cm,
                  columns=['Predicted Negative',
                           'Predicted Positive'],
                  index=['Actual Negative',
                         'Actual Positive'])

cm.style.background_gradient(cmap='viridis')
# Lets first get the probability given to each prediction
y_pred_proba = rfc.predict_proba(X_test)

# next make a dataframe of the outcomes
train_df = pd.DataFrame(pd.Series(y_test))

# reset the index so we can concatenate the right rows together
train_df.reset_index(inplace=True)

# crate a dataframe of the predictions
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])

# from the proba we are taking the second column and making it a dataframe
proba_df = pd.DataFrame(y_pred_proba[:,1], columns=['proba'])
# concatenating all the dataframes together
precision_recall = pd.concat([train_df, predictions_df, proba_df], axis=1, ignore_index=False).set_index('index')
# increasing max columns displayed by pandad
pd.options.display.max_columns = 30

precision_recall.sample(30).sort_values('proba').T
gold_digger_df = precision_recall.copy()
doctor_df = precision_recall.copy()
gold_digger_df.loc[(gold_digger_df.proba >= 0.7), ('Predictions')] = 1
gold_digger_df.loc[(gold_digger_df.proba < 0.7), ('Predictions')] = 0

doctor_df.loc[(doctor_df.proba >= 0.4), ('Predictions')] = 1
doctor_df.loc[(doctor_df.proba < 0.4), ('Predictions')] = 0
print(classification_report(gold_digger_df.DEATH_EVENT, gold_digger_df.Predictions))
gb_cm = confusion_matrix(gold_digger_df.DEATH_EVENT, gold_digger_df.Predictions)
gb_cm = pd.DataFrame(gb_cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
gb_cm.style.background_gradient(cmap='viridis')
print(classification_report(doctor_df.DEATH_EVENT, doctor_df.Predictions))
dr_cm = confusion_matrix(doctor_df.DEATH_EVENT, doctor_df.Predictions)
dr_cm = pd.DataFrame(dr_cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
dr_cm.style.background_gradient(cmap='viridis')
plot_roc_curve(rfc, trans_test_data, y_test);
