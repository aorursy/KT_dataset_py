import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize']=(20,10)

import seaborn as sns

sns.set_style("dark")

import warnings

warnings.filterwarnings('ignore')

from ipywidgets import interact
df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

df.head()
df['diagnosis'].value_counts()
categorical_cols = ['diagnosis','sex','anatom_site_general_challenge','benign_malignant','target']



fig,ax = plt.subplots(2,(len(categorical_cols)+1)//2,figsize=(30,15))



ratio = {}

for i,col in enumerate(categorical_cols+['age_approx']):

    ratio[col] = 100*df[col].value_counts(dropna=False)/df[col].value_counts(dropna=False).sum()

    if i==5:

        ax[1][2].hist(df['age_approx'])     

    else:

        ax[i%2][i//2].bar([str(x) for x in ratio[col].index],height=ratio[col])

    ax[i%2][i//2].set_title(col,fontdict={'fontsize':20})

    for tick in ax[i%2][i//2].get_xticklabels():

        tick.set_rotation(90)

        tick.set_fontsize('x-large')



fig.suptitle('Distribution of each categorical data',y=1.02,fontsize=25)

fig.tight_layout()
df_category = df.copy()

df_category['benign_malignant'] = (df_category['benign_malignant'] == 'malignant').apply(int)
print("Agreement between 'malignant' value and target:",100*((df['benign_malignant']=='malignant') == (df['target'])).sum()/len(df),'%')

print("Agreement between 'melanoma' value and target:",100*(((df['diagnosis'] =='melanoma') ) == (df['target'])).sum()/len(df),'%')
sex_c = df_category['sex'].value_counts()

sex_c = sex_c.sort_index()

fig,ax = plt.subplots() 

for idx in sex_c.index:

    sns.kdeplot(df.loc[df['sex']==idx,'age_approx'],shade=True,ax=ax)

ax.legend(sex_c.index)

ax.set_title('Age distribution for each sex',fontsize=20)

res = ax.set_xlabel('approx_age')
diag_c = df_category['diagnosis'].value_counts()

diag_c = diag_c.sort_index()

fig,ax = plt.subplots() 

for idx in diag_c.index:

    sns.kdeplot(df.loc[df['diagnosis']==idx,'age_approx'],shade=True,ax=ax)

ax.legend(diag_c.index)

ax.set_title('Relation between age & diagnosis',fontsize=20)

res = ax.set_xlabel('approx_age')
ax = sns.catplot(x="sex",

            y="target",

            kind="bar",

            hue='age_approx',

            data=df_category,

            height=9, 

            aspect=2.3);

_=ax.fig.suptitle('Influence of the age on likeliness that the diagnosis is melanoma',y=1.02, fontsize=20)
print("Number of men > 90 in the dataset:", len(df_category.loc[(df_category['age_approx']==90.0) & (df_category['sex']=='male'),'patient_id'].unique()))
anatom_c = df_category['anatom_site_general_challenge'].value_counts()

anatom_c = anatom_c.sort_index()

fig,ax = plt.subplots() 

for idx in anatom_c.index:

    sns.kdeplot(df.loc[df['anatom_site_general_challenge']==idx,'age_approx'],shade=True,ax=ax)

ax.legend(anatom_c.index)

ax.set_title('Relation between age & anatomic site',fontsize=20)

res = ax.set_xlabel('approx_age')
ax = sns.catplot(x="sex",

            y="target",

            hue="anatom_site_general_challenge",

            kind="bar",

            data=df_category,

            height=9, 

            aspect=2.3);

_=ax.fig.suptitle('Influence of the image site on the target, by sex', y=1.02,fontsize=20)
print("Number of female patient with an image of the oral/genital site:", len(df.loc[(df_category['sex']=="female") & (df_category['anatom_site_general_challenge']=="oral/genital"),'patient_id'].unique()))
print("Number of unique patients in the dataset:", len(df_category['patient_id'].unique()))
fig, ax = plt.subplots(figsize=(20,10))

df_category.groupby('patient_id').count()['image_name'].hist(ax=ax,bins=100)

ax.set_title('Number of pictures by patient', fontsize=20)

ax.grid(False)
ax= df_category.groupby('patient_id').aggregate({'age_approx':pd.Series.nunique}).hist()[0][0]

ax.set_title('Distribution of the number of different age_approx for each patient',fontsize=20)

ax.grid(False)
df_category_patient_with_melanoma = df_category.groupby('patient_id').apply(lambda x: 'melanoma' in set(x['diagnosis']))

set_category_patient_with_melanoma = {x[0] for x  in df_category_patient_with_melanoma.iteritems() if x[1]}

df_category_patient_with_melanoma = df_category.loc[df_category['patient_id'].isin(set_category_patient_with_melanoma)]



# We extract the id of the first image with melanoma for each patient with at least one melanoma 

first_melanoma_pictures = df_category_patient_with_melanoma.loc[df_category_patient_with_melanoma['diagnosis']=='melanoma'].drop_duplicates(['patient_id','diagnosis'],keep='first').index



# We select all the other images

df_other_images_patient_with_melanoma = df_category_patient_with_melanoma.loc[df_category_patient_with_melanoma['diagnosis'] != 'melanoma']

df_other_images_patient_with_melanoma =  df_category_patient_with_melanoma.drop(first_melanoma_pictures)

df_other_images_patient_with_melanoma['image_set'] = 'Other images of patients with melanoma'





df_all_images = df_category.copy()

df_all_images['image_set'] = 'Full dataset'



# We can concatenate all the images with only other images of patient with melanoma, in order to build comparative statics on the risk of occurence of melanoma

concatenated = pd.concat([df_all_images,

          df_other_images_patient_with_melanoma])



ax = sns.catplot(x="sex",

            y="target",

            hue="image_set",

            kind="bar",

            data=concatenated,

            height=8, 

            aspect=2.2);

_=ax.fig.suptitle('Influence on the diagnosis of the existence of a melanoma elsewhere',fontsize=20,y=1.02)
path_train_jpg = '../input/siim-isic-melanoma-classification/jpeg/train/'
diagnosis_list = ['unknown', 'nevus','melanoma','seborrheic keratosis','lentigo NOS',

 'lichenoid keratosis','solar lentigo','atypical melanocytic proliferation','cafe-au-lait macule']

def show_diagnosis(diagnosis='melanoma'):

    assert diagnosis in diagnosis_list

    fig, ax = plt.subplots(3,2,figsize=(10,10))

    samples = df.loc[df['diagnosis']==diagnosis].sample(6,replace=True)['image_name']

    ax = ax.ravel()

    for j, name in enumerate(samples): 

        ax[j].imshow(plt.imread(path_train_jpg+name+'.jpg'))

        ax[j].grid(False)

        # Hide axes ticks

        ax[j].set_xticks([])

        ax[j].set_yticks([])

    fig.suptitle("Diagnosis: "+diagnosis,fontsize=20, y=0.95)



    plt.show()

    

int = interact(show_diagnosis,diagnosis=diagnosis_list) 