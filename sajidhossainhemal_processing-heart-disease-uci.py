# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import the libraries
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib as mlb
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
heart_df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
heart_df.head()
heart_df.info()
import pandas_profiling as pp

pp.ProfileReport(heart_df)
heart_df.drop_duplicates()
heart_df.shape
heart_df.columns
heart_df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 
                 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate', 'exercise_induced_angina',
                 'st_depression', 'st_slope', 'major_vessel_num', 'thalassemia', 'condition']
heart_df.nunique()
heart_df['thalassemia'].value_counts()
heart_df['sex'] = heart_df['sex'].map({0:'female', 1:'male'})
heart_df['chest_pain_type'] = heart_df['chest_pain_type'].map({0:'typical_angina', 1:'non_anginal_pain', 2:'atypical_angina', 3:'asymptomatic'})
heart_df['fasting_blood_sugar'] = heart_df['fasting_blood_sugar'].map({0:'less_than_120mg/ml', 1:'greater_than_120mg/ml'})
heart_df['rest_ecg'] = heart_df['rest_ecg'].map({0:'normal', 1:'ST-T_wave_abnormality', 2:'left_ventricular_hypertrophy'})
heart_df['exercise_induced_angina'] = heart_df['exercise_induced_angina'].map({0:'no', 1:'yes'})
heart_df['st_slope'] = heart_df['st_slope'].map({0:'upsloping', 1:'flat', 2:'downsloping'})
heart_df['thalassemia'] = heart_df['thalassemia'].map({0:'normal', 1:'fixed_defect', 2:'normal', 3:'reversal_defect'})
heart_df['condition'] = heart_df['condition'].map({0:'no_diesease', 1:'has_diesease'})
heart_df.info()
heart_df.head()
heart_df.describe()
def cat_bar_plot(dataframe, rot = 0, alpha=0.70, color=['steelblue', 'crimson'],
                 title='Distribution', xlabel='Column Name', ylabel='Count of People'):
    dataframe.value_counts().plot(kind='bar', rot=rot, alpha=alpha, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
plt.figure(figsize=(30, 35))
plt.subplots_adjust(top=0.6, bottom=0.1, hspace=0.6, wspace=0.2)
sns.set(font_scale=1)

plt.subplot(331)
cat_bar_plot(heart_df['sex'], color=['steelblue', 'crimson'], title='Gender Distribution', xlabel='Gender')

plt.subplot(332)
cat_bar_plot(heart_df['chest_pain_type'], rot=45, color=['lime', 'orange', 'cyan', 'red'],
             title='Chest Pain Type Distribution', xlabel='Chest Pain Type')

plt.subplot(333)
cat_bar_plot(heart_df['fasting_blood_sugar'], color=['salmon', 'royalblue'],
             title='Fasting Blood Sugar Distribution', xlabel='Fasting Blood Sugar')

plt.subplot(334)
cat_bar_plot(heart_df['rest_ecg'], color=['violet', 'tomato', 'steelblue'],
             title='Rest ECG Distribution', xlabel='Rest ECG')

plt.subplot(335)
cat_bar_plot(heart_df['exercise_induced_angina'], color=['lime', 'cyan'],
             title='Exercise Induced Angina Distribution', xlabel='Exercise Induced Angina')

plt.subplot(336)
cat_bar_plot(heart_df['st_slope'], color=['crimson', 'violet', 'orange'],
             title='ST Slope Distribution', xlabel='ST Slope')

plt.subplot(337)
cat_bar_plot(heart_df['major_vessel_num'], color=['indianred', 'greenyellow', 'orange', 'violet', 'salmon'],
             title='Major Vessel Number Distribution', xlabel='Major Vessel Number')

plt.subplot(338)
cat_bar_plot(heart_df['thalassemia'], color=['steelblue', 'violet', 'red'],
             title='Thalassemia Distribution', xlabel='Thalassemia')

plt.subplot(339)
cat_bar_plot(heart_df['condition'], color=['cyan', 'lime'],
             title='Condition Distribution', xlabel='Condition')

plt.show()
def numerical_data_plot(dataframe, bins=15, hist_alpha=0.6, kde_alpha=0.8, hist_title='Distribution of Patients',
                        xlabel='Column Name', ylabel='Count of People', box_title='Column Name', 
                        kde_mul=1000, hist_color='crimson', kde_color='red', box_color='crimson'):
    plt.figure(figsize=(20, 6))
    sns.set(font_scale=1)
    
    plt.subplot(121)
    count, bin_edges = np.histogram(dataframe)
    dataframe.plot(kind='hist', bins=bins, alpha=hist_alpha, xticks=bin_edges, color=hist_color)
    
    # Let's add a KDE plot
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_x = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(dataframe)
    plt.plot(kde_x, kde.pdf(kde_x) * kde_mul, 'k--', alpha=kde_alpha, color=kde_color)
    
    plt.title(hist_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.subplot(122)
    red_circle = dict(markerfacecolor='r', marker='o')
    dataframe.plot(kind='box', color=box_color, flierprops=red_circle)
    plt.title(box_title)
    
# AGE
numerical_data_plot(heart_df['age'], hist_title='Age Distribution of Patients',
                    xlabel='Age', box_title='Age of Patients')

plt.show()
# Resting Blood Pressure
numerical_data_plot(heart_df['resting_blood_pressure'], bins=25, hist_title='Resting Blood Pressure Distribution of Patients',
                    xlabel='Resting Blood Pressure', box_title='Resting Blood Pressure of Patients', hist_color='violet',
                    kde_color='fuchsia', box_color='violet')

plt.show()
# Serum Cholesterol
numerical_data_plot(heart_df['serum_cholesterol'], bins=70, hist_title='Serum Cholesterol Distribution',
                    xlabel='Serum Cholesterol', box_title='Serum Cholesterol of Patients', hist_color='navy',
                    kde_color='blue', box_color='navy')

plt.show()
# Max Heart Rate
numerical_data_plot(heart_df['max_heart_rate'], bins=30, kde_alpha=0.9, hist_title='Max Heart Rate Distribution',
                    xlabel='Max Heart Rate', box_title='Max Heart Rate of Patients',
                    hist_color='limegreen', kde_color='green', box_color='limegreen')

plt.show()
# ST Depression
numerical_data_plot(heart_df['st_depression'], bins=15, hist_title='St Depression Distribution', xlabel='St Depression',
                    box_title='St Depression of Patients', kde_mul=100, hist_color='darksalmon',
                    kde_color='sienna', box_color='darksalmon')

plt.show()
def make_list(df):
    data = df.value_counts()
    dictionary = data.to_dict()
    return list(dictionary.values())
diesease_df = heart_df.loc[heart_df.condition == 'has_diesease']
no_diesease_df = heart_df.loc[heart_df.condition == 'no_diesease']


male_list = make_list(diesease_df['sex'])
female_list = make_list(no_diesease_df['sex'])

labels = ['Male',  'Female']
width = 0.80
x = [0.5, 2.5]

sns.set(font_scale=1.3)

fig, ax = plt.subplots()
ax.bar([0.1, 2.1], male_list, width, alpha=0.7, label='Has Diesease', color='midnightblue')
ax.bar([0.9, 2.9], female_list, width, alpha=0.7, label='No Diesease', color='crimson')

ax.set_ylabel('Count')
ax.set_xlabel('Gender')
ax.set_title('Gender vs Condition')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + 0.3, p.get_height() + 0.7), fontsize = 12)

plt.show()
heart_df.head()
