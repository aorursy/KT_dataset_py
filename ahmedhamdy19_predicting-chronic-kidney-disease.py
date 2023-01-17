import pandas as pd

import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
kidney_main_df = pd.read_csv('https://assets.datacamp.com/production/repositories/943/datasets/82c231cd41f92325cf33b78aaa360824e6b599b9/chronic_kidney_disease.csv',header=None)
kidney_main_df.head()
kidney_main_df.columns=['Age','Blood Pressure','Specific Gravity','Albumin','Sugar','Red Blood Cells','Pus Cell', 'Pus Cell clumps', 'Bacteria', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium', 'Potassium', 'Hemoglobin',  'Packed Cell Volume', 'White Blood Cell Count', 'Red Blood Cell Count', 'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia', 'Class']
kidney_main_df.replace('?',np.nan,inplace=True)
kidney_main_df.describe()
kidney_main_df.rename(columns={'Class': 'class'}, inplace=True)

kidney_main_df['class'].replace({'notckd':0,'ckd':1}, inplace=True)

kidney_main_df['class'].value_counts()
shuffled_kidney_dt = kidney_main_df.iloc[np.random.permutation(len(kidney_main_df))]

kidney_dt_new=shuffled_kidney_dt.reset_index(drop=True)
kidney_dt_new_target = kidney_dt_new['class'].values

kidney_dt_new_features = kidney_dt_new.drop(['class'], axis=1)

kidney_dt_new_features.head()
kidney_dt_new_features.dtypes
# Fill missing values with 0

kidney_dt_without_missing_values = kidney_dt_new_features.fillna(-999)



# Convert df into a dictionary: df_dict

kidney_dt_new_dict = kidney_dt_without_missing_values.to_dict("records")



# Create the DictVectorizer object: dv

dv = DictVectorizer(sparse=False)



# Apply dv on df: df_encoded

kidney_dt_new_encoded = dv.fit_transform(kidney_dt_new_dict)



# Print the resulting first five rows

print(kidney_dt_new_encoded)



# Print the vocabulary

print(dv.vocabulary_)
np.isnan(kidney_dt_new_encoded).any()
kidney_dt_new_encoded[0].size
training_indices, validation_indices = training_indices, testing_indices = train_test_split(kidney_dt_new.index, stratify = kidney_dt_new_target, train_size=0.75, test_size=0.25,random_state=42)

training_indices.size, validation_indices.size
exported_pipeline = make_pipeline(

    MinMaxScaler(),

    LinearSVC(C=1.0, dual=True, loss="hinge", penalty="l2", tol=0.001)

)



exported_pipeline.fit(kidney_dt_new_encoded[training_indices],  kidney_dt_new_target[training_indices])
exported_pipeline.score(kidney_dt_new_encoded[validation_indices], kidney_dt_new.loc[validation_indices, 'class'].values)
pred = exported_pipeline.predict(kidney_dt_new_encoded[validation_indices])



#Print out the accuracy score on the test set

accuracy_score(pred, kidney_dt_new.loc[validation_indices, 'class'].values)
#Print out the confusion matrix on the test set

confusion_matrix(pred,kidney_dt_new.loc[validation_indices, 'class'].values)/len(kidney_dt_new.loc[validation_indices, 'class'].values)
MSE_CV_scores = -cross_val_score(exported_pipeline,kidney_dt_new_encoded[training_indices], kidney_dt_new_target[training_indices], cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



# Compute the 10-folds CV RMSE

RMSE_CV = (MSE_CV_scores.mean())**(1/2)



# Print RMSE_CV

print('CV RMSE: {:.2f}'.format(RMSE_CV))
classification_report(pred,kidney_dt_new.loc[validation_indices, 'class'].values)