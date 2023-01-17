# from google.colab import drive

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

from sklearn import metrics

import statsmodels.api as sm

import matplotlib.pyplot as plt

from datetime import datetime, date

import numpy as np



# drive import

# drive.mount('/content/drive')

dataset = pd.read_excel("/kaggle/input/updated/dataset.xlsx")

dataset_blood = pd.read_excel("/kaggle/input/complement/dataset_blood.xlsx")

test = pd.read_excel("/kaggle/input/complement/teste_covid.xlsx")  



dataset = dataset.replace(np.nan, 0, regex=True)

dataset_blood = dataset_blood.replace(np.nan, 0, regex=True)

test = test.replace(np.nan, 0, regex=True)

dataset.head()
dataset_blood.head()
dataset['SARS-Cov-2 exam result'].value_counts()
dataset.corr()
for n,i in enumerate(dataset['SARS-Cov-2 exam result']):

  if i == 'positive':

     dataset['SARS-Cov-2 exam result'][n] = 1

  else:

    dataset['SARS-Cov-2 exam result'][n] = 0

dataset['SARS-Cov-2 exam result'].value_counts()



for n,i in enumerate(dataset_blood['SARS-Cov-2 exam result']):

  if i == 'positive':

     dataset_blood['SARS-Cov-2 exam result'][n] = 1

  else:

    dataset_blood['SARS-Cov-2 exam result'][n] = 0

dataset_blood['SARS-Cov-2 exam result'].value_counts()



dataset['SARS-Cov-2 exam result'] = dataset['SARS-Cov-2 exam result'].astype('int')

dataset_blood['SARS-Cov-2 exam result'] = dataset_blood['SARS-Cov-2 exam result'].astype('int')
kviz_age = KNeighborsClassifier(n_neighbors=2)

kviz = KNeighborsClassifier(n_neighbors=2)



kviz.fit(dataset_blood[['Patient age quantile','Hematocrit',	'Hemoglobin',	'Platelets',	'Mean platelet volume ',	'Red blood Cells',	'Lymphocytes',	'Mean corpuscular hemoglobin concentration (MCHC)',	'Leukocytes',	'Basophils',	'Mean corpuscular hemoglobin (MCH)',	'Eosinophils',	'Mean corpuscular volume (MCV)',	'Monocytes',	'Red blood cell distribution width (RDW)',	'Serum Glucose',	'Neutrophils',	'Urea',	'Proteina C reativa mg/dL',	'Creatinine',	'Potassium',	'Sodium',	'Influenza B, rapid test',	'Influenza A, rapid test',	'Alanine transaminase',	'Aspartate transaminase',	'Gamma-glutamyltransferase ',	'Total Bilirubin',	'Direct Bilirubin',	'Indirect Bilirubin',	'Alkaline phosphatase',	'Ionized calcium ',	'Strepto A',	'Magnesium',	'pCO2 (venous blood gas analysis)',	'Hb saturation (venous blood gas analysis)',	'Base excess (venous blood gas analysis)',	'pO2 (venous blood gas analysis)',	'Fio2 (venous blood gas analysis)',	'Total CO2 (venous blood gas analysis)',	'pH (venous blood gas analysis)',	'HCO3 (venous blood gas analysis)',	'Rods #',	'Segmented',	'Promyelocytes',	'Metamyelocytes',	'Myelocytes',	'Myeloblasts',	'Urine - Esterase',	'Urine - pH',	'Urine - Bile pigments',	'Urine - Ketone Bodies',	'Urine - Nitrite',	'Urine - Density',	'Urine - Protein',	'Urine - Sugar',	'Urine - Leukocytes',	'Urine - Red blood cells',	'Urine - Hyaline cylinders',	'Urine - Granular cylinders',	'Urine - Yeasts',	'Partial thromboplastin time (PTT) ',	'Relationship (Patient/Normal)',	'International normalized ratio (INR)',	'Lactic Dehydrogenase',	'Prothrombin time (PT), Activity',	'Vitamin B12',	'Creatine phosphokinase (CPK) ',	'Ferritin',	'Arterial Lactic Acid',	'Lipase dosage',	'D-Dimer',	'Albumin',	'Hb saturation (arterial blood gases)',	'pCO2 (arterial blood gas analysis)',	'Base excess (arterial blood gas analysis)',	'pH (arterial blood gas analysis)',	'Total CO2 (arterial blood gas analysis)',	'HCO3 (arterial blood gas analysis)',	'pO2 (arterial blood gas analysis)',	'Arteiral Fio2',	'Phosphor',	'ctO2 (arterial blood gas analysis)']] , dataset_blood['SARS-Cov-2 exam result'])

kviz_age.fit(dataset[['Patient age quantile']] , dataset_blood['SARS-Cov-2 exam result'])
kviz.predict(test[['Patient age quantile','Hematocrit',	'Hemoglobin',	'Platelets',	'Mean platelet volume ',	'Red blood Cells',	'Lymphocytes',	'Mean corpuscular hemoglobin concentration (MCHC)',	'Leukocytes',	'Basophils',	'Mean corpuscular hemoglobin (MCH)',	'Eosinophils',	'Mean corpuscular volume (MCV)',	'Monocytes',	'Red blood cell distribution width (RDW)',	'Serum Glucose',	'Neutrophils',	'Urea',	'Proteina C reativa mg/dL',	'Creatinine',	'Potassium',	'Sodium',	'Influenza B, rapid test',	'Influenza A, rapid test',	'Alanine transaminase',	'Aspartate transaminase',	'Gamma-glutamyltransferase ',	'Total Bilirubin',	'Direct Bilirubin',	'Indirect Bilirubin',	'Alkaline phosphatase',	'Ionized calcium ',	'Strepto A',	'Magnesium',	'pCO2 (venous blood gas analysis)',	'Hb saturation (venous blood gas analysis)',	'Base excess (venous blood gas analysis)',	'pO2 (venous blood gas analysis)',	'Fio2 (venous blood gas analysis)',	'Total CO2 (venous blood gas analysis)',	'pH (venous blood gas analysis)',	'HCO3 (venous blood gas analysis)',	'Rods #',	'Segmented',	'Promyelocytes',	'Metamyelocytes',	'Myelocytes',	'Myeloblasts',	'Urine - Esterase',	'Urine - pH',	'Urine - Bile pigments',	'Urine - Ketone Bodies',	'Urine - Nitrite',	'Urine - Density',	'Urine - Protein',	'Urine - Sugar',	'Urine - Leukocytes',	'Urine - Red blood cells',	'Urine - Hyaline cylinders',	'Urine - Granular cylinders',	'Urine - Yeasts',	'Partial thromboplastin time (PTT) ',	'Relationship (Patient/Normal)',	'International normalized ratio (INR)',	'Lactic Dehydrogenase',	'Prothrombin time (PT), Activity',	'Vitamin B12',	'Creatine phosphokinase (CPK) ',	'Ferritin',	'Arterial Lactic Acid',	'Lipase dosage',	'D-Dimer',	'Albumin',	'Hb saturation (arterial blood gases)',	'pCO2 (arterial blood gas analysis)',	'Base excess (arterial blood gas analysis)',	'pH (arterial blood gas analysis)',	'Total CO2 (arterial blood gas analysis)',	'HCO3 (arterial blood gas analysis)',	'pO2 (arterial blood gas analysis)',	'Arteiral Fio2',	'Phosphor',	'ctO2 (arterial blood gas analysis)']])
kviz_age.predict(test[['Patient age quantile']])
from sklearn import tree



# o classificador encontra padrões nos dados de treinamento

clf = tree.DecisionTreeClassifier() # instância do classificador

clf = clf.fit(dataset_blood[['Patient age quantile','Hematocrit',	'Hemoglobin',	'Platelets',	'Mean platelet volume ',	'Red blood Cells',	'Lymphocytes',	'Mean corpuscular hemoglobin concentration (MCHC)',	'Leukocytes',	'Basophils',	'Mean corpuscular hemoglobin (MCH)',	'Eosinophils',	'Mean corpuscular volume (MCV)',	'Monocytes',	'Red blood cell distribution width (RDW)',	'Serum Glucose',	'Neutrophils',	'Urea',	'Proteina C reativa mg/dL',	'Creatinine',	'Potassium',	'Sodium',	'Influenza B, rapid test',	'Influenza A, rapid test',	'Alanine transaminase',	'Aspartate transaminase',	'Gamma-glutamyltransferase ',	'Total Bilirubin',	'Direct Bilirubin',	'Indirect Bilirubin',	'Alkaline phosphatase',	'Ionized calcium ',	'Strepto A',	'Magnesium',	'pCO2 (venous blood gas analysis)',	'Hb saturation (venous blood gas analysis)',	'Base excess (venous blood gas analysis)',	'pO2 (venous blood gas analysis)',	'Fio2 (venous blood gas analysis)',	'Total CO2 (venous blood gas analysis)',	'pH (venous blood gas analysis)',	'HCO3 (venous blood gas analysis)',	'Rods #',	'Segmented',	'Promyelocytes',	'Metamyelocytes',	'Myelocytes',	'Myeloblasts',	'Urine - Esterase',	'Urine - pH',	'Urine - Bile pigments',	'Urine - Ketone Bodies',	'Urine - Nitrite',	'Urine - Density',	'Urine - Protein',	'Urine - Sugar',	'Urine - Leukocytes',	'Urine - Red blood cells',	'Urine - Hyaline cylinders',	'Urine - Granular cylinders',	'Urine - Yeasts',	'Partial thromboplastin time (PTT) ',	'Relationship (Patient/Normal)',	'International normalized ratio (INR)',	'Lactic Dehydrogenase',	'Prothrombin time (PT), Activity',	'Vitamin B12',	'Creatine phosphokinase (CPK) ',	'Ferritin',	'Arterial Lactic Acid',	'Lipase dosage',	'D-Dimer',	'Albumin',	'Hb saturation (arterial blood gases)',	'pCO2 (arterial blood gas analysis)',	'Base excess (arterial blood gas analysis)',	'pH (arterial blood gas analysis)',	'Total CO2 (arterial blood gas analysis)',	'HCO3 (arterial blood gas analysis)',	'pO2 (arterial blood gas analysis)',	'Arteiral Fio2',	'Phosphor',	'ctO2 (arterial blood gas analysis)']] , dataset_blood['SARS-Cov-2 exam result']) # fit encontra padrões nos dados

clf.feature_importances_

for feature,importancia in zip(dataset_blood.columns,clf.feature_importances_):

  print ("{}:{}".format(feature, importancia))
resultado = clf.predict(test[['Patient age quantile','Hematocrit',	'Hemoglobin',	'Platelets',	'Mean platelet volume ',	'Red blood Cells',	'Lymphocytes',	'Mean corpuscular hemoglobin concentration (MCHC)',	'Leukocytes',	'Basophils',	'Mean corpuscular hemoglobin (MCH)',	'Eosinophils',	'Mean corpuscular volume (MCV)',	'Monocytes',	'Red blood cell distribution width (RDW)',	'Serum Glucose',	'Neutrophils',	'Urea',	'Proteina C reativa mg/dL',	'Creatinine',	'Potassium',	'Sodium',	'Influenza B, rapid test',	'Influenza A, rapid test',	'Alanine transaminase',	'Aspartate transaminase',	'Gamma-glutamyltransferase ',	'Total Bilirubin',	'Direct Bilirubin',	'Indirect Bilirubin',	'Alkaline phosphatase',	'Ionized calcium ',	'Strepto A',	'Magnesium',	'pCO2 (venous blood gas analysis)',	'Hb saturation (venous blood gas analysis)',	'Base excess (venous blood gas analysis)',	'pO2 (venous blood gas analysis)',	'Fio2 (venous blood gas analysis)',	'Total CO2 (venous blood gas analysis)',	'pH (venous blood gas analysis)',	'HCO3 (venous blood gas analysis)',	'Rods #',	'Segmented',	'Promyelocytes',	'Metamyelocytes',	'Myelocytes',	'Myeloblasts',	'Urine - Esterase',	'Urine - pH',	'Urine - Bile pigments',	'Urine - Ketone Bodies',	'Urine - Nitrite',	'Urine - Density',	'Urine - Protein',	'Urine - Sugar',	'Urine - Leukocytes',	'Urine - Red blood cells',	'Urine - Hyaline cylinders',	'Urine - Granular cylinders',	'Urine - Yeasts',	'Partial thromboplastin time (PTT) ',	'Relationship (Patient/Normal)',	'International normalized ratio (INR)',	'Lactic Dehydrogenase',	'Prothrombin time (PT), Activity',	'Vitamin B12',	'Creatine phosphokinase (CPK) ',	'Ferritin',	'Arterial Lactic Acid',	'Lipase dosage',	'D-Dimer',	'Albumin',	'Hb saturation (arterial blood gases)',	'pCO2 (arterial blood gas analysis)',	'Base excess (arterial blood gas analysis)',	'pH (arterial blood gas analysis)',	'Total CO2 (arterial blood gas analysis)',	'HCO3 (arterial blood gas analysis)',	'pO2 (arterial blood gas analysis)',	'Arteiral Fio2',	'Phosphor',	'ctO2 (arterial blood gas analysis)']])

print(resultado)
!pip install ipywidgets

!pip3 install graphviz

!pip3 install pydot
import graphviz

import pydot

from sklearn.tree import DecisionTreeClassifier, export_graphviz
dot_data = export_graphviz(

    clf,

    out_file = None,

    feature_names = ['Patient age quantile','Hematocrit',	'Hemoglobin',	'Platelets',	'Mean platelet volume ',	'Red blood Cells',	'Lymphocytes',	'Mean corpuscular hemoglobin concentration (MCHC)',	'Leukocytes',	'Basophils',	'Mean corpuscular hemoglobin (MCH)',	'Eosinophils',	'Mean corpuscular volume (MCV)',	'Monocytes',	'Red blood cell distribution width (RDW)',	'Serum Glucose',	'Neutrophils',	'Urea',	'Proteina C reativa mg/dL',	'Creatinine',	'Potassium',	'Sodium',	'Influenza B, rapid test',	'Influenza A, rapid test',	'Alanine transaminase',	'Aspartate transaminase',	'Gamma-glutamyltransferase ',	'Total Bilirubin',	'Direct Bilirubin',	'Indirect Bilirubin',	'Alkaline phosphatase',	'Ionized calcium ',	'Strepto A',	'Magnesium',	'pCO2 (venous blood gas analysis)',	'Hb saturation (venous blood gas analysis)',	'Base excess (venous blood gas analysis)',	'pO2 (venous blood gas analysis)',	'Fio2 (venous blood gas analysis)',	'Total CO2 (venous blood gas analysis)',	'pH (venous blood gas analysis)',	'HCO3 (venous blood gas analysis)',	'Rods #',	'Segmented',	'Promyelocytes',	'Metamyelocytes',	'Myelocytes',	'Myeloblasts',	'Urine - Esterase',	'Urine - pH',	'Urine - Bile pigments',	'Urine - Ketone Bodies',	'Urine - Nitrite',	'Urine - Density',	'Urine - Protein',	'Urine - Sugar',	'Urine - Leukocytes',	'Urine - Red blood cells',	'Urine - Hyaline cylinders',	'Urine - Granular cylinders',	'Urine - Yeasts',	'Partial thromboplastin time (PTT) ',	'Relationship (Patient/Normal)',	'International normalized ratio (INR)',	'Lactic Dehydrogenase',	'Prothrombin time (PT), Activity',	'Vitamin B12',	'Creatine phosphokinase (CPK) ',	'Ferritin',	'Arterial Lactic Acid',	'Lipase dosage',	'D-Dimer',	'Albumin',	'Hb saturation (arterial blood gases)',	'pCO2 (arterial blood gas analysis)',	'Base excess (arterial blood gas analysis)',	'pH (arterial blood gas analysis)',	'Total CO2 (arterial blood gas analysis)',	'HCO3 (arterial blood gas analysis)',	'pO2 (arterial blood gas analysis)',	'Arteiral Fio2',	'Phosphor',	'ctO2 (arterial blood gas analysis)'],

    class_names = ['0','1'],

    filled = True, rounded = True,

    proportion = True,

    node_ids = True,

    rotate = False,

    label = 'all',

    special_characters = True

)



graph = graphviz.Source(dot_data)

graph