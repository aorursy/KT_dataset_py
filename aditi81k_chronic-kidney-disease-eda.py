!pip install plotly
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
train=pd.read_csv('../input/chronic-kidney-disease/kidney_disease_train.csv')

test=pd.read_csv('../input/chronic-kidney-disease/kidney_disease_test.csv')
col={"bp":"blood_pressure",

          "sg":"specific_gravity",

          "al":"albumin",

          "su":"sugar",

          "rbc":"red_blood_cells",

          "pc":"pus_cell",

          "pcc":"pus_cell_clumps",

          "ba":"bacteria",

          "bgr":"blood_glucose_random",

          "bu":"blood_urea",

          "sc":"serum_creatinine",

          "sod":"sodium",

          "pot":"potassium",

          "hemo":"hemoglobin",

          "pcv":"packed_cell_volume",

          "wc":"white_blood_cell_count",

          "rc":"red_blood_cell_count",

          "htn":"hypertension",

          "dm":"diabetes_mellitus",

          "cad":"coronary_artery_disease",

          "appet":"appetite",

          "pe":"pedal_edema",

          "ane":"anemia"}



train.rename(columns=col, inplace=True)

test.rename(columns=col, inplace=True)
print('We have total {} train sample and {} test sample'.format(train.shape[0],test.shape[0]))
train.info()
train.isnull().sum()
# Percentage of missing values

((train.isnull().sum()/train.shape[0])*100).sort_values(ascending=False)
#drop id column

train.drop(["id"],axis=1,inplace=True) 
train['red_blood_cell_count'] = pd.to_numeric(train['red_blood_cell_count'], errors='coerce')

train['white_blood_cell_count'] = pd.to_numeric(train['white_blood_cell_count'], errors='coerce')
train.describe(include='all').T
for i in train.columns:

    print('{} has unique values {}'.format(i,train[i].unique()),'\n')
#Replace incorrect values

train['diabetes_mellitus'] =train['diabetes_mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})

train['coronary_artery_disease'] = train['coronary_artery_disease'].replace(to_replace='\tno',value='no')
sns.countplot(x='classification',data=train)

plt.xlabel("classification")

plt.ylabel("Count")

plt.title("target Class")

plt.show()

print('Percent of chronic kidney disease sample: ',round(len(train[train['classification']=='ckd'])/len(train['classification'])*100,2),"%")

print('Percent of not a chronic kidney disease sample: ',round(len(train[train['classification']=='notckd'])/len(train['classification'])*100,2),"%")
corr_df = train.corr()

f,ax=plt.subplots(figsize=(15,15))

sns.heatmap(corr_df,annot=True,fmt=".2f",ax=ax,linewidths=0.5,linecolor="orange")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.title('Correlations between different predictors')

plt.show()
numerical_features=[feature for feature in train.columns if train[feature].dtypes=='float64']

print('total numerical column :',len(numerical_features))

print(numerical_features)
categorical_features=[feature for feature in train.columns if train[feature].dtypes=='O']

print('total categorical column :',len(categorical_features))

print(categorical_features)
train[categorical_features].describe(include='all').T
train[numerical_features].describe(include='all').T
((train[numerical_features].isnull().sum()/train.shape[0])*100).sort_values(ascending=False)
def violin(col): 

    fig = px.violin(train, y=col, x="classification", color="classification", box=True, points="all", hover_data=train.columns)

    return fig.show()

def kde_plot(feature):

    grid = sns.FacetGrid(train, hue="classification", aspect = 2)

    grid.map(sns.kdeplot, feature)

    grid.add_legend()
kde_plot('red_blood_cell_count')
train.groupby(['classification'])['red_blood_cell_count'].agg(['mean','median'])
kde_plot('packed_cell_volume')
kde_plot('hemoglobin')
train.groupby(['classification'])['hemoglobin'].agg(['mean','median'])
fig = px.scatter(train, x="red_blood_cell_count", y="hemoglobin", color="classification")

fig.show() 
fig = px.scatter(train, x="red_blood_cell_count", y="packed_cell_volume", color="classification")

fig.show()
fig = px.scatter(train, x="packed_cell_volume", y="red_blood_cell_count", color="classification")

fig.show()
fig = px.bar(train, x="red_blood_cells", y="red_blood_cell_count",color='classification', barmode='group',height=400)

fig.show()
train.groupby(['red_blood_cells','classification'])['red_blood_cell_count'].agg(['count','mean','median','min','max'])
violin('red_blood_cell_count')
violin('packed_cell_volume')
violin('hemoglobin')
def missing_value(feature): 

    a = train[(train[feature].isnull())]

    return a.groupby(['classification'])['classification'].agg(['count'])

    
print('missing values in RBC column:\n\n',missing_value('red_blood_cell_count'),'\n')

print('missing values in Packed cell volume column:\n\n',missing_value('packed_cell_volume'),'\n')

print('missing values in Hemoglobin column:\n\n',missing_value('hemoglobin'),'\n')
fig = px.bar(train, x="albumin", y="packed_cell_volume",color='classification', barmode='group',height=400)

fig.show()
train.groupby(['albumin','classification'])['albumin'].count()
fig = px.bar(train, x="albumin", y="hemoglobin",color='classification', barmode='group',height=400)

fig.show()
violin('albumin')
kde_plot('specific_gravity')
fig = px.bar(train, x="specific_gravity", y="packed_cell_volume",

             color='classification', barmode='group',

             height=400)

fig.show()
print("number of patient who's having packed cell volume<40 and specific gravity <1.02:\n\n",train[(train['packed_cell_volume']<40)&(train['specific_gravity']<1.02)].groupby(['classification'])['classification'].agg(['count']))

print("packed cell volume >=40 and specific gravity >=1.02:\n\n",train[(train['packed_cell_volume']>=40)&(train['specific_gravity']>=1.02)].groupby(['classification'])['classification'].agg(['count']))
fig = px.bar(train, x="specific_gravity", y="hemoglobin",

             color='classification', barmode='group',

             height=400)

fig.show()
print("number of patient who's having hemoglobin <12 and specific gravity <1.02:\n\n",train[(train['hemoglobin']<12)&(train['specific_gravity']<1.02)].groupby(['classification'])['classification'].agg(['count']))

print("hemoglobin >=12 and specific gravity >=1.02:\n\n",train[(train['packed_cell_volume']>=12)&(train['specific_gravity']>=1.02)].groupby(['classification'])['classification'].agg(['count']))
fig = px.bar(train, x="specific_gravity", y="red_blood_cell_count",

             color='classification', barmode='group',

             height=400)

fig.show()
print("number of patient who's having RBC <3.9 and specific gravity <1.02:\n\n",train[(train['red_blood_cell_count']<3.9)&(train['specific_gravity']<1.02)].groupby(['classification'])['classification'].agg(['count']))

print("RBC >=3.9 and specific gravity >=1.02:\n\n",train[(train['red_blood_cell_count']>=3.9)&(train['specific_gravity']>=1.02)].groupby(['classification'])['classification'].agg(['count']))
train[(train['packed_cell_volume']<40)&(train['specific_gravity']<1.02)&(train['hemoglobin']<12)&(train['red_blood_cell_count']<3.9)].groupby(['classification'])['classification'].agg(['count'])
violin('specific_gravity')
kde_plot('white_blood_cell_count')
train[(train['white_blood_cell_count']>=4300) &(train['white_blood_cell_count']<=11000)&(train['classification']=='ckd')]
violin('white_blood_cell_count')
kde_plot('potassium')
kde_plot('blood_urea')
fig = px.scatter(train, x="potassium", y="blood_urea", color="classification")

fig.show()
fig = px.scatter(train, x="potassium", y="serum_creatinine", color="classification")

fig.show()
violin('potassium')
violin('blood_urea')
fig = px.bar(train, x="pus_cell", y="blood_urea",color='classification', barmode='group',height=400)

fig.show()
fig = px.bar(train, x="pus_cell_clumps", y="blood_urea",color='classification', barmode='group',height=400)

fig.show()
kde_plot('sodium')
kde_plot('serum_creatinine')
fig = px.scatter(train, x="sodium", y="blood_pressure", color="classification")

fig.show()
fig = px.scatter(train, x="sodium", y="serum_creatinine", color="classification")

fig.show()
train[train['classification']=='notckd']['serum_creatinine'].agg(['min','max'])
kde_plot('blood_pressure')
train.groupby(['classification','hypertension'])['blood_pressure'].agg(['min','max','count','mean','median'])
fig = px.bar(train, x="hypertension", y="blood_pressure",color='classification', barmode='group',height=400)

fig.show()
violin('blood_pressure')
kde_plot('blood_glucose_random')
fig = px.bar(train, x="sugar", y="blood_glucose_random",

             color='classification', barmode='group',

             height=400)

fig.show()
fig = px.bar(train, x="diabetes_mellitus", y="blood_glucose_random",

             color='classification', barmode='group',

             height=400)

fig.show()
train.groupby(['classification'])['blood_glucose_random'].agg(['min','max','median','mean'])
violin('blood_glucose_random')
kde_plot('age')
train.groupby(['classification'])['age'].agg(['min','max','count','mean','median'])
violin('age')
col = ['bacteria','coronary_artery_disease', 'appetite', 'pedal_edema','anemia']

for i in col:

    sns.countplot(i, hue = 'classification', data = train)

    plt.show()
