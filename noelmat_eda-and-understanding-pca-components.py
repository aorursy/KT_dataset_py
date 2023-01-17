import pandas as pd

import numpy as np

from pathlib import Path

Path.ls = lambda x: list(x.iterdir())

import seaborn as sns

import matplotlib.pyplot as plt
path = Path("/kaggle/input/heart-failure-clinical-data/")

path.ls()
df = pd.read_csv(path/'heart_failure_clinical_records_dataset.csv')

df.head()
df.head().T
print('Information about the data columns along with their null counts')

df.info()
plt.figure(figsize=(8,5))

sns.countplot(df.age,palette='winter')

plt.title('Count Plot of Age',fontsize=14)

plt.xticks(rotation=90)

plt.show()
print('Checking the unique values in time:: ')

print(len(df.time.unique()))
plt.figure(figsize=(8,5))

sns.countplot(df.time,palette='winter')

plt.title('Count Plot of Time',fontsize=14)

plt.xticks(rotation=90)

plt.show()
def plot_hist(col, bins=40, title="",xlabel="",ax=None):

#     plt.figure(figsize=(12,8))

    sns.distplot(col, bins=bins,ax=ax)

    ax.set_title(f'Histogram of {title}',fontsize=20)

    ax.set_xlabel(xlabel)

    
fig, axes = plt.subplots(3,2,figsize=(20,20),constrained_layout=True)

plot_hist(df.creatinine_phosphokinase,

          title='Creatinine Phosphokinase',

          xlabel="Level of the CPK (mcg/L)",

          ax=axes[0,0])

plot_hist(df.platelets,

          bins=30,

          title='Platelets',

          xlabel='Platelets in the blood (kiloplatelets/mL)',

          ax=axes[0,1])

plot_hist(df.serum_creatinine,

          title='Serum Creatinine', 

          xlabel='Level of serum creatinine in the blood (mg/dL)',

          ax=axes[1,0])

plot_hist(df.serum_sodium,

          bins=30,

          title='Serum Sodium',

          xlabel='Level of serum sodium in the blood (mEq/L)',

          ax=axes[1,1])

plot_hist(df.ejection_fraction,

          title='Ejection Fraction', 

          xlabel='Percentage of blood leaving the heart at each contraction (percentage)',

          ax=axes[2,0])

plot_hist(df.time,

          bins=30,

          title='Time',

          xlabel='Follow-up period (days)',

          ax=axes[2,1])

plt.show()
def plot_categorical_var(x='DEATH_EVENT', col=None, title="",label="",ax=None):

    sns.countplot(data=df, x=col, hue=x,palette='winter',ax=ax)

    ax.set_title(title,fontsize=16)

    ax.set_xlabel(label)
fig, axes = plt.subplots(2,3,figsize=(20,10),constrained_layout=True)

plot_categorical_var(col='diabetes',

                     title='Death vs diabetes',

                     label='Diabetes',

                     ax=axes[0,0])

plot_categorical_var(col='high_blood_pressure',

                     title='Death vs high blood pressure',

                     label='High blood pressure',

                     ax=axes[0,1])

plot_categorical_var(col='sex',

                     title='Sex vs Death',

                     label='Sex',

                     ax=axes[0,2])

plot_categorical_var(col='smoking',

                     title='Smoking vs Death', 

                     label='Smoking Status',

                     ax=axes[1,0])

plot_categorical_var(col='anaemia',

                     title='Anaemia vs Death',

                     label='is anaemic?',

                     ax=axes[1,1])

plt.show()
totitle= lambda x: " ".join(x.split('_'))
def plot_multivar(df,x,y,hue_list=None):

    if hue_list is None:

        hue_list = ['DEATH_EVENT','sex']

    fig, axes = plt.subplots(1,len(hue_list), figsize=(6*len(hue_list),5),constrained_layout=True)

    fig.suptitle(f'{totitle(x)} vs {totitle(y)}'.title(),fontsize=18)

    if not isinstance(axes, np.ndarray):

        axes = np.array(axes)

    for i,(ax,hue) in enumerate(zip(axes.flatten(),hue_list)):

        sns.scatterplot(data=df, x=x,y=y,hue=hue,alpha=0.8,ax=ax,palette='rocket')

        ax.set_title(f'{totitle(hue)}'.title(),fontsize=18)

    plt.show()
plot_multivar(df,'serum_creatinine','creatinine_phosphokinase')
plot_multivar(df,'platelets','serum_creatinine')
plot_multivar(df,'platelets','creatinine_phosphokinase')
plot_multivar(df,'serum_creatinine','serum_sodium')
plot_multivar(df,'ejection_fraction','serum_creatinine')
plot_multivar(df,'time','serum_creatinine')
cont_cols = ['creatinine_phosphokinase', 'platelets','serum_creatinine', 'serum_sodium', 'ejection_fraction','age','time']

cat_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking','DEATH_EVENT']
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(df[['creatinine_phosphokinase', 'platelets','serum_creatinine', 'serum_sodium', 'ejection_fraction','age','time']])

scaled_df = df.copy()

scaled_df[cont_cols] = scaled_features

scaled_df.head()
from sklearn.decomposition import PCA



N_COMPONENTS = 6



pca = PCA(n_components = N_COMPONENTS)

pca.fit(scaled_df[cont_cols].values)

print(f"Explained variance: {pca.explained_variance_ratio_[:4].sum()}")



v = pd.DataFrame(pca.components_)
transformed = pca.transform(scaled_df[cont_cols])

transformed_df = pd.DataFrame(transformed)

transformed_df.columns = list(map(lambda x: f'pca_{x+1}', list(transformed_df.columns)))

transformed_df[cat_vars] = df[cat_vars]

transformed_df.head()
def display_component(v, features_list, component_num):

    

    row_idx = N_COMPONENTS - component_num

    

    v_1_row = v.iloc[:,row_idx]

    v_1 = np.squeeze(v_1_row.values)

    

    comps = pd.DataFrame(list(zip(v_1, features_list)),

                         columns=['weights', 'features'])

    

    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))

    sorted_weight_data = comps.sort_values('abs_weights',ascending=False).head()

    

    ax=plt.subplots(figsize=(10,6))

    ax=sns.barplot(data=sorted_weight_data,

                   x="weights",

                   y="features",

                   palette="Blues_d")

    ax.set_title("PCA Component Makeup, Component #" + str(component_num), fontsize=20)

    plt.show()
def show_component_details(num_component):

    print(f"Percent explained variance: {pca.explained_variance_ratio_[num_component-1]*100:.4f}","%")

    display_component(v,cont_cols,num_component)
show_component_details(1)
show_component_details(2)
show_component_details(3)
show_component_details(4)
plot_multivar(transformed_df, 'pca_1','pca_2',hue_list=['DEATH_EVENT'])
plot_multivar(transformed_df,'pca_1','pca_3',hue_list=['DEATH_EVENT'])
plot_multivar(transformed_df,'pca_1','pca_4',hue_list=['DEATH_EVENT'])