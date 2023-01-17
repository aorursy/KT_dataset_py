# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from IPython.display import clear_output



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_data = pd.read_excel(os.path.join(dirname, 'dataset.xlsx'))

df_data.columns = [x.lower().strip().replace(' ','_').replace(',', '') for x in df_data.columns]

df_data.columns = [re.sub('[^A-Za-z0-9\\_\\-]+', '', x) for x in df_data.columns]

print(df_data.columns)
print(df_data.head())

print(df_data.shape)
df_data = df_data.drop(np.where(df_data.isna().sum(axis=1) >= df_data.shape[1]-6)[0], axis=0)

df_data.reset_index()
import matplotlib.pyplot as plt



data_to_plot = ['hematocrit', 'hemoglobin', 'platelets', 'mean_platelet_volume', 'red_blood_cells', 'lymphocytes', \

 'mean_corpuscular_hemoglobin_concentrationmchc', 'leukocytes', 'basophils', 'mean_corpuscular_hemoglobin_mch', 'eosinophils', \

 'mean_corpuscular_volume_mcv', 'monocytes', 'red_blood_cell_distribution_width_rdw', 'serum_glucose', 'neutrophils', 'urea', \

 'proteina_c_reativa_mgdl', 'creatinine', 'potassium', 'sodium', 'alanine_transaminase', 'aspartate_transaminase', \

 'gamma-glutamyltransferase', 'total_bilirubin', 'direct_bilirubin', 'indirect_bilirubin', 'alkaline_phosphatase', 'ionized_calcium', \

 'magnesium', 'pco2_venous_blood_gas_analysis', 'hb_saturation_venous_blood_gas_analysis', 'base_excess_venous_blood_gas_analysis', \

 'po2_venous_blood_gas_analysis', 'fio2_venous_blood_gas_analysis', 'total_co2_venous_blood_gas_analysis', 'ph_venous_blood_gas_analysis', \

 'hco3_venous_blood_gas_analysis', 'rods_', 'segmented', 'promyelocytes', 'metamyelocytes', 'myelocytes', 'myeloblasts', 'urine_-_density', \

 'urine_-_red_blood_cells', 'relationship_patientnormal', 'international_normalized_ratio_inr', 'lactic_dehydrogenase', 'vitamin_b12', 'creatine_phosphokinasecpk', \

 'ferritin', 'arterial_lactic_acid', 'lipase_dosage', 'albumin', 'hb_saturation_arterial_blood_gases', 'pco2_arterial_blood_gas_analysis', \

 'base_excess_arterial_blood_gas_analysis', 'ph_arterial_blood_gas_analysis', 'total_co2_arterial_blood_gas_analysis', 'hco3_arterial_blood_gas_analysis', \

 'po2_arterial_blood_gas_analysis', 'arteiral_fio2', 'phosphor', 'cto2_arterial_blood_gas_analysis']



for column in data_to_plot:

  fig, axs = plt.subplots(1, 3)

  fig.set_size_inches(18, 5)

  axs[0].set_title(column + ' mean value')

  df_data.groupby('sars-cov-2_exam_result')[column].mean().plot(kind='barh', ax=axs[0])

  axs[1].set_title(column + '- who tested positive for COVID-19')

  df_data[df_data['sars-cov-2_exam_result'] == 'positive'][column].hist(ax=axs[1])

  axs[2].set_title(column + '- who tested negative for COVID-19')

  df_data[df_data['sars-cov-2_exam_result'] == 'negative'][column].hist(ax=axs[2])
import tensorflow as tf

print(tf.version)
df_data = df_data.dropna(axis=1, how='all')  # Remove columns that does not have data

try:

  df_data.pop('urine_-_leukocytes') # Remove column that does not present consistent data

except:

  pass



df_positive = df_data[df_data['sars-cov-2_exam_result'] == 'positive']

df_negative = df_data[df_data['sars-cov-2_exam_result'] == 'negative']



df_train_positive = df_positive.sample(frac=0.8, random_state=12576)

df_test_positive = df_positive.drop(df_train_positive.index)



df_train_negative = df_negative.sample(frac=0.8, random_state=9658)

df_test_negative = df_negative.drop(df_train_negative.index)



df_train = pd.concat([df_train_positive, df_train_negative], axis=0).reset_index()

df_train_patient = df_train.pop('patient_id')

y_train = df_train.pop('sars-cov-2_exam_result').replace('negative', 0).replace('positive',1)

df_test = pd.concat([df_test_positive, df_test_negative], axis=0).reset_index()

df_test_patient = df_test.pop('patient_id')

y_test = df_test.pop('sars-cov-2_exam_result').replace('negative', 0).replace('positive',1)



df_train.pop('index')

df_test.pop('index')
CATEGORICAL_COLUMNS = ['patient_addmited_to_regular_ward_1yes_0no', 'patient_addmited_to_semi-intensive_unit_1yes_0no', 'patient_addmited_to_intensive_care_unit_1yes_0no', \

 'respiratory_syncytial_virus', 'influenza_a', 'influenza_b', 'parainfluenza_1', 'coronavirusnl63', 'rhinovirusenterovirus', 'coronavirus_hku1', \

 'parainfluenza_3', 'chlamydophila_pneumoniae', 'adenovirus', 'parainfluenza_4', 'coronavirus229e', 'coronavirusoc43', 'inf_a_h1n1_2009', 'bordetella_pertussis', \

 'metapneumovirus', 'parainfluenza_2', 'influenza_b_rapid_test', 'influenza_a_rapid_test', 'strepto_a',  'urine_-_esterase', 'urine_-_aspect', \

 'urine_-_ph', 'urine_-_hemoglobin',  'urine_-_bile_pigments', 'urine_-_ketone_bodies', 'urine_-_nitrite', 'urine_-_urobilinogen', 'urine_-_protein', \

 'urine_-_crystals', 'urine_-_hyaline_cylinders', 'urine_-_granular_cylinders', 'urine_-_yeasts', 'urine_-_color']



NUMERIC_COLUMNS = ['patient_age_quantile', 'hematocrit', 'hemoglobin', 'platelets', 'mean_platelet_volume', 'red_blood_cells', 'lymphocytes', \

 'mean_corpuscular_hemoglobin_concentrationmchc', 'leukocytes', 'basophils', 'mean_corpuscular_hemoglobin_mch', 'eosinophils', \

 'mean_corpuscular_volume_mcv', 'monocytes', 'red_blood_cell_distribution_width_rdw', 'serum_glucose', 'neutrophils', 'urea', \

 'proteina_c_reativa_mgdl', 'creatinine', 'potassium', 'sodium', 'alanine_transaminase', 'aspartate_transaminase', \

 'gamma-glutamyltransferase', 'total_bilirubin', 'direct_bilirubin', 'indirect_bilirubin', 'alkaline_phosphatase', 'ionized_calcium', \

 'magnesium', 'pco2_venous_blood_gas_analysis', 'hb_saturation_venous_blood_gas_analysis', 'base_excess_venous_blood_gas_analysis', \

 'po2_venous_blood_gas_analysis', 'fio2_venous_blood_gas_analysis', 'total_co2_venous_blood_gas_analysis', 'ph_venous_blood_gas_analysis', \

 'hco3_venous_blood_gas_analysis', 'rods_', 'segmented', 'promyelocytes', 'metamyelocytes', 'myelocytes', 'myeloblasts', 'urine_-_density', \

 'urine_-_red_blood_cells', 'relationship_patientnormal', 'international_normalized_ratio_inr', 'lactic_dehydrogenase', 'vitamin_b12', 'creatine_phosphokinasecpk', \

 'ferritin', 'arterial_lactic_acid', 'lipase_dosage', 'albumin', 'hb_saturation_arterial_blood_gases', 'pco2_arterial_blood_gas_analysis', \

 'base_excess_arterial_blood_gas_analysis', 'ph_arterial_blood_gas_analysis', 'total_co2_arterial_blood_gas_analysis', 'hco3_arterial_blood_gas_analysis', \

 'po2_arterial_blood_gas_analysis', 'arteiral_fio2', 'phosphor', 'cto2_arterial_blood_gas_analysis']





for category in CATEGORICAL_COLUMNS:

  df_train[category] = df_train[category].apply(lambda x: str(x))

  df_test[category] = df_test[category].apply(lambda x: str(x))

  df_positive[category] = df_positive[category].apply(lambda x: str(x))

  df_negative[category] = df_negative[category].apply(lambda x: str(x))



for numeric in NUMERIC_COLUMNS:

  df_train[numeric] = pd.to_numeric(df_train[numeric])

  df_train[numeric].fillna(0, inplace=True)

  df_test[numeric] = pd.to_numeric(df_test[numeric])

  df_test[numeric].fillna(0, inplace=True)

  df_positive[numeric] = pd.to_numeric(df_positive[numeric])

  df_positive[numeric].fillna(0, inplace=True)

  df_negative[numeric] = pd.to_numeric(df_negative[numeric])

  df_negative[numeric].fillna(0, inplace=True)



df_train = df_train.replace('nan', 'None')

df_test = df_test.replace('nan', 'None')

df_positive = df_positive.replace('nan', 'None')

df_negative = df_negative.replace('nan', 'None')





feature_columns = []



for column in df_data.columns:

  if column in CATEGORICAL_COLUMNS:

    vocabulary = df_train[column].unique() 

    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(column, vocabulary))

  elif column in NUMERIC_COLUMNS:

    feature_columns.append(tf.feature_column.numeric_column(column, dtype=tf.float32))



print(len(CATEGORICAL_COLUMNS))

print(len(NUMERIC_COLUMNS))

print(len(feature_columns))
# Create input function



def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

  def input_function():

    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

    if shuffle:

      ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds

  return input_function



train_input_fn = make_input_fn(df_train, y_train, num_epochs=50, shuffle=True, batch_size=124)

test_input_fn = make_input_fn(df_test, y_test, num_epochs=1, shuffle=False, batch_size=len(y_test))



linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn, max_steps=30000)
result = linear_est.evaluate(test_input_fn)



clear_output()

print(result)
pred_dicts = list(linear_est.predict(test_input_fn))

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])



probs.plot(kind='hist', bins=20, title='predicted probabilities')
from sklearn.metrics import roc_curve



fpr, tpr, _ = roc_curve(y_test, probs)

plt.plot(fpr, tpr)

plt.title('ROC curve')

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')

plt.xlim(0,)

plt.ylim(0,)
from sklearn.metrics import confusion_matrix

import seaborn as sns



out = probs.copy()

out.iloc[np.where(probs >= 0.1)] = 1

out.iloc[np.where(probs < 0.1)] = 0



cm = confusion_matrix(y_test, out)

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Predicted labels');

ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);