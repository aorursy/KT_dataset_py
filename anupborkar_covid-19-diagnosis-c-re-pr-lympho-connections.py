%reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
import os
%matplotlib inline
plt.rcParams.update({'figure.max_open_warning': 0}) #just to suppress warning for max plots of 20
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# Display output not only of last command but all commands in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Set pandas options to display results
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
# Load dataset
df = pd.read_csv("../input/uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
#Let's try to analyze the dataset based on what is availiable with us
df.info()
df.head()
df.describe()
# Segragate the sars_cov_2_exam_result into positive and negative
#df_positive = df[df['sars_cov_2_exam_result']=='positive']
#df_negative = df[df['sars_cov_2_exam_result']=='negative']
 
# Set Index as patient_id 
#df = df.set_index('patient_id')

missing_values = df.isna().sum()/len(df)

print("Total number of tuples:", len(df))
print("Percentage of  missing values:", round(missing_values.mean()*100,1),"%")

data_not_avlbl = []
for i in np.arange(0, len(df.columns), 10):
    data_not_avlbl.append(str(i)+"%");
plt.figure(figsize=[10,30]);

plt.yticks(np.arange(len(df.columns)), missing_values.index.values);
plt.xticks(np.arange(0, 1.1, .1), data_not_avlbl);

plt.ylim(0,len(df.columns));

plt.barh(np.arange(len(df.columns)), missing_values,color="darkcyan");
df_filtered = df[~np.isnan(df['hematocrit'])]
missing_values_filtered = df_filtered.isna().sum()/len(df_filtered)

print("Filtered Number of Tuples:", len(df_filtered))
print("Average percentage missing values:", round(missing_values_filtered.mean()*100,1),"%")


data_not_avlbl_filtered = []
for i in np.arange(0, 110, 10):
    data_not_avlbl_filtered.append(str(i)+"%");
plt.figure(figsize=[10,30]);

plt.yticks(np.arange(len(df_filtered.columns)), missing_values_filtered.index.values);
plt.xticks(np.arange(0, 1.1, .1), data_not_avlbl_filtered);

plt.ylim(0,len(df_filtered.columns));

plt.barh(np.arange(len(df_filtered.columns)), missing_values_filtered,color="darkcyan");
df_filtered = df_filtered[missing_values_filtered[missing_values_filtered<=.4].index.values]

missing_values_filtered = df_filtered.isna().sum()/len(df_filtered)

print("Total Number of Tuples:", len(df_filtered))
print("Average percentage missing values:", round(missing_values_filtered.mean()*100,1),"%")

data_not_avlbl = []
for i in np.arange(0, 110, 10):
    data_not_avlbl.append(str(i)+"%");
plt.figure(figsize=[10,20]);

plt.yticks(np.arange(len(df_filtered.columns)), missing_values_filtered.index.values);
plt.xticks(np.arange(0, 1.1, .1), data_not_avlbl);

plt.ylim(0,len(df_filtered.columns));

plt.barh(np.arange(len(df_filtered.columns)), missing_values_filtered,color="darkcyan");


df_pos_neg_treated = df_filtered
#filter out the categorical features from filtered dataset
cat_features = df_filtered.dtypes[df_filtered.dtypes == 'object'].index.values 
#fill NaN values for filtered categorical values
for feature in cat_features:
    df_pos_neg_treated[feature] = df_pos_neg_treated[feature].fillna(df_pos_neg_treated[feature].mode().values[0]) 
    
df_pos_neg_treated = df_pos_neg_treated.fillna(df_pos_neg_treated.median())

df_pos_neg_treated_dummies = pd.get_dummies(df_pos_neg_treated, drop_first=True, dtype='bool')

columns = list(df_pos_neg_treated_dummies.drop(labels=['sars_cov_2_exam_result_positive'],axis=1).columns.values)
columns.append('sars_cov_2_exam_result_positive')

df_pos_neg_treated_dummies = df_pos_neg_treated_dummies[columns]
ax = df_pos_neg_treated['sars_cov_2_exam_result'].value_counts().plot(kind='bar',figsize=(8,4),color="darkcyan");
ax.set_xticklabels(['negative', 'positive'], rotation=0, fontsize=15);

#df.columns.values
df['patient_addmited_to_regular_ward_1_yes_0_no'] = df['patient_addmited_to_regular_ward_1_yes_0_no'].map({
                                    't' : 1, 'f' : 0   })
df['patient_addmited_to_semi_intensive_unit_1_yes_0_no'] = df['patient_addmited_to_semi_intensive_unit_1_yes_0_no'].map({
                                    't' : 1, 'f' : 0   })
df['patient_addmited_to_intensive_care_unit_1_yes_0_no'] = df['patient_addmited_to_intensive_care_unit_1_yes_0_no'].map({
                                    't' : 1, 'f' : 0   })
unit_ward_types =  ['patient_addmited_to_regular_ward_1_yes_0_no',\
                'patient_addmited_to_semi_intensive_unit_1_yes_0_no',\
                 'patient_addmited_to_intensive_care_unit_1_yes_0_no']
def_hemogram_cols = list(df.columns[6:20])
parameters = ['lymphocytes', 'neutrophils', 'sodium', 'potassium', 'creatinine','proteina_c_reativa_mg_dl']

list_unit_ward_types = ['No-care', 'Regular', 'Semi-Intensive', 'Intensive']

cors = 'sars_cov_2_exam_result'

data_join = df[unit_ward_types + parameters].dropna().to_numpy(copy=True)
#df = pd.read_csv("diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
unit_ward_types_y_ = data_join[:,0:3] # ward_types columns
y = unit_ward_types_y_[:,0] #1st ward_type series 
y [unit_ward_types_y_[:,1]==1]=2 #2nd ward_type series semi-icu Assign value 2
y [unit_ward_types_y_[:,2]==1]=3 #3rd ward_type series icu Assign value 3
X = data_join[:,3:]  #Parameters columns

# Sodium vs. Potassium
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,2], X[idx,3], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Sodium');
plt.ylabel('Potassium');
plt.show();

# Lymphocytes vs. Neutrophils
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,0], X[idx,1], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Lymphocytes');
plt.ylabel('Neutrophils');
plt.show();

# Creatinine and CRP
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,4], X[idx,5], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Creatinine');
plt.ylabel('C Reactive Protein(CRP)');
plt.show();

plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,4], X[idx,5], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Creatinine');
plt.ylabel('C Reactive Protein(CRP)');
plt.show();
# Feature visualization only for SARS-Cov-2 positives
#df = pd.read_csv("diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
positive_df_ = df[ df['sars_cov_2_exam_result'] == 'positive']
data_join = positive_df_[unit_ward_types + parameters].dropna().to_numpy(copy=True)

unit_ward_types_y_ = data_join[:,0:3];
y = unit_ward_types_y_[:,0];
y [unit_ward_types_y_[:,1]==1]=2;
y [unit_ward_types_y_[:,2]==1]=3;
X = data_join[:,3:];

# Sodium vs. Potassium
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,2], X[idx,3], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Sodium');
plt.ylabel('Potassium');
plt.show();

# Lymphocytes vs. Neutrophils
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,0], X[idx,1], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Lymphocytes');
plt.ylabel('Neutrophils');
plt.show();

# Creatinine and CRP
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0]
    plt.scatter(X[idx,4], X[idx,5], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Creatinine');
plt.ylabel('C Reactive Protein (CRP)');
plt.show();
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0];
    plt.scatter(X[idx,0], X[idx,5], alpha=0.6, label=list_unit_ward_types[i]);
plt.legend();
plt.xlabel('Lymphocytes');
plt.ylabel('C Reactive Protein(CRP)');
plt.show();
plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0];
    plt.scatter(X[idx,0], X[idx,5], alpha=0.6, label=list_unit_ward_types[i]);
    
plt.plot([-2, 1.3, 1.3, -2, -2], [-0.4, -0.4, 3.5, 3.5, -0.4], 'r');# Coordinates forming a Rectangle
plt.plot([1.3, 1.3], [-0.4, 1.5], 'r', label='Risky Area');

plt.legend();
plt.xlabel('Lymphocytes');
plt.ylabel('C Reactive Protein(CRP)');
plt.show();
# Feature visualization- Hemogram data
#df = pd.read_csv("diagnosis-of-covid-19-and-its-clinical-spectrum.csv")

data_join = df[unit_ward_types + parameters].dropna().to_numpy(copy=True);

unit_ward_types_y_ = data_join[:,0:3];
y = unit_ward_types_y_[:,0];
y [unit_ward_types_y_[:,1]==1]=2;
y [unit_ward_types_y_[:,2]==1]=3;
X = data_join[:,3:];

scaler = StandardScaler();
scaler.fit(X);
X_ = scaler.transform(X);

# t-SNE visualization
tsne = TSNE(n_components=2);
Xtsne = tsne.fit_transform(X_);

plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0];
    plt.scatter(Xtsne[idx,0], Xtsne[idx,1], alpha=0.6, label=list_unit_ward_types[i]);

plt.title('t-SNE visualization');
plt.legend();
plt.show();
# Feature visualization - Hemogram parameters for SARS-Cov-2-positives
#df = pd.read_csv("diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
positive_df_ = df[ df['sars_cov_2_exam_result'] == 'positive'];
data_join = positive_df_[unit_ward_types + parameters].dropna().to_numpy(copy=True);

unit_ward_types_y_ = data_join[:,0:3];
y = unit_ward_types_y_[:,0];
y [unit_ward_types_y_[:,1]==1]=2;
y [unit_ward_types_y_[:,2]==1]=3;
X = data_join[:,3:];

scaler = StandardScaler();
scaler.fit(X);
X_ = scaler.transform(X);

# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30);#perplexity is related to the number of nearest neighbors
Xtsne = tsne.fit_transform(X_);

plt.figure();
for i in range(len(list_unit_ward_types)):
    idx = np.where(y == i)[0];
    plt.scatter(Xtsne[idx,0], Xtsne[idx,1], alpha=0.6, label=list_unit_ward_types[i]);

plt.title('t-SNE visualization');
plt.legend();
plt.show();