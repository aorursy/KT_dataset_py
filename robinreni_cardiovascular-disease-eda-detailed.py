import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt # for visualization

import seaborn as sns # for visualization

import sklearn # for scientific calculations

from sklearn import preprocessing

from matplotlib import rcParams

import warnings

warnings.filterwarnings("ignore")
# Data Loading

df_raw = pd.read_csv("/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv", sep=";")

df_raw.head()
# Basic information about dataset

df_raw.info()
# Checking any missing values

df_raw.isnull().sum()
# Checking for any duplicate values in dataset

df_raw.duplicated().sum()
# Convert the ages from number of days to categorical values

def calculate_age(days):

  days_year = 365.2425

  age = int(days // days_year)

  return age



# According to US National Library of Medicine National Institutes of Health the age groups are classified

def categorize_age(age):

  if 0 < age <= 2:

    return "Infants"

  elif 2 < age <= 5:

    return "Pre School Child"

  elif 5 < age <= 12:

    return "Child"

  elif 12 < age <= 19:

    return "Adolescent"

  elif 19 < age <= 24:

    return "Young Adult"

  elif 24 < age <= 44:

    return "Adult"

  elif 44 < age <= 65:

    return "Middle Aged"

  elif 65 < age:

    return "Aged"



def categorize_age_tees(age):

  if 0 < age <= 10:

    return "10s"

  elif 10 < age <= 30:

    return "20s"

  elif 20 < age <= 30:

    return "30s"

  elif 30 < age <= 40:

    return "40s"

  elif 40 < age <= 50:

    return "50s"

  elif 50 < age <= 60:

    return "60s"

  elif 60 < age <= 70:

    return "70+"
# age transformations

df_raw['age_new'] = df_raw['age'].apply(lambda x: calculate_age(x))

df_raw['age_cat'] = df_raw['age_new'].apply(lambda x: categorize_age(x))

df_raw['age_tees'] = df_raw['age_new'].apply(lambda x: categorize_age_tees(x))

df_raw.head()
# Visulization of age groups

rcParams['figure.figsize'] = 7, 7

sns.countplot(x ='age_cat', data = df_raw) 
# Visulization of age groups with cvdd info

rcParams['figure.figsize'] = 7, 7

sns.countplot(x ='age_cat', hue = 'cardio', data = df_raw) 
rcParams['figure.figsize'] = 11, 8

sns.countplot(x='age_new', hue='cardio', data = df_raw, palette="Set2");
rcParams['figure.figsize'] = 10, 7

sns.countplot(x='age_tees', hue='cardio', data = df_raw, palette="Set2", order = ['10s','20s','30s','40s','50s','60s','70+']);
# Height comparison 

df_raw.groupby('gender')['height'].mean()
# Alcohol consumption 

df_raw.groupby('gender')['alco'].sum()
# Gender Ratio

df_raw['gender'].value_counts()
# Calcualte the CVD distribution based on Gender

df_raw['cardio'].value_counts(normalize=True)
rcParams['figure.figsize'] = 7, 7

sns.countplot(x='gender', hue='cardio', data = df_raw, palette="Set1");
for col in ["height", "weight"]:

    sns.kdeplot(df_raw[col], shade=True)
# Height Distribution

df_melt = pd.melt(frame=df_raw, value_vars=['height'], id_vars=['gender'])

plt.figure(figsize=(7, 7))

ax = sns.violinplot(

    x='variable', 

    y='value', 

    hue='gender', 

    split=True, 

    data=df_melt, 

    scale='count',

    scale_hue=False,

    palette="Set2");
# Weight Distribution

df_melt = pd.melt(frame=df_raw, value_vars=['weight'], id_vars=['gender'])

plt.figure(figsize=(7, 7))

ax = sns.violinplot(

    x='variable', 

    y='value', 

    hue='gender', 

    split=True, 

    data=df_melt, 

    scale='count',

    scale_hue=False,

    palette="Set1");
# calculate the BMI score 

df_raw['BMI'] = df_raw['weight']/((df_raw['height']/100)**2)



# categorize normal & abnormal

def bmi_categorize(bmi_score):

  if 18.5 <= bmi_score <= 25:

    return "Normal"

  else:

    return "Abnormal"



df_raw["BMI_State"] = df_raw["BMI"].apply(lambda x: bmi_categorize(x))

df_raw["BMI_State"].value_counts().plot(kind='pie')
rcParams['figure.figsize'] = 7, 7

sns.countplot(x='BMI_State', hue='cardio', data = df_raw, palette="Set2");
# comparison plot with alcohol consumption with CVD patients

sns.catplot(x="BMI_State", y="BMI" , hue="alco", col="cardio", data=df_raw, kind="boxen", height=8, aspect=.6);
# Alcohol consumption 

df_raw["alco"].value_counts().plot(kind='pie')
# segregate data based on patients having Cvdd "1"

df_cardio_alco = df_raw.loc[df_raw["alco"] == 1]

with sns.axes_style('white'):

    g = sns.factorplot("cardio", data=df_cardio_alco, aspect=2,

                       kind="count", color='red')

    g.set_xticklabels(step=5)
df_raw["smoke"].value_counts().plot(kind='pie')
with sns.axes_style('white'):

    g = sns.factorplot("smoke", data=df_raw, aspect=4.0, kind='count',

                       hue='cardio', palette="Set2")

    g.set_ylabels('Number of Patients')
df_raw["active"].value_counts().plot(kind='pie')
with sns.axes_style('white'):

    g = sns.factorplot("active", data=df_raw, aspect=4.0, kind='count',

                       hue='cardio', palette="Set2")

    g.set_ylabels('Number of Patients')
# box plot got systolic blood pressure

sns.boxplot( y=df_raw["ap_hi"] )
# box plot got diastolic blood pressure

sns.boxplot( y=df_raw["ap_lo"] )
out_filter = ((df_raw["ap_hi"]>250) | (df_raw["ap_hi"]>250) | (df_raw["ap_lo"]>200) )

print("There is {} outlier".format(df_raw[out_filter]["cardio"].count()))
# removing outliers

df_outlier_raw = df_raw[~out_filter]
# checking the box plot got systolic blood pressure after outlier cleaning

sns.boxplot( y=df_outlier_raw["ap_hi"] )
# checking the box plot got diastolic blood pressure after outlier cleaning

sns.boxplot( y=df_outlier_raw["ap_lo"] )
df_raw.tail(5)
def categorize_blood_pressure(x):

  if x['ap_hi'] < 120 and x['ap_lo'] < 80:

    return "Normal"

  elif 120 <= x['ap_hi'] <= 129 and x['ap_lo'] < 80:

    return "Elevated"

  elif 130 <= x['ap_hi'] <= 139 or 80 <= x['ap_lo'] <= 89:

    return "High Blood Pressure(Stage 1)"

  elif  140 <= x['ap_hi'] <= 180 or 90 <= x['ap_lo'] <= 120:

    return "High Blood Pressure(Stage 2)"

  elif  (x['ap_hi'] > 180 and  x['ap_lo'] > 120) or (x['ap_hi'] > 180 or x['ap_lo'] > 120):

    return "Hypertensive Crisis"



# remove outliers

out_filter = ((df_raw["ap_hi"]>250) | (df_raw["ap_hi"]>250) | (df_raw["ap_lo"]>200) )

df_raw = df_raw[~out_filter]

# categorizing blood pressure

df_raw['blood_category'] = df_raw.apply(categorize_blood_pressure, axis=1)

 

df_raw.head()
# Visulization of blood pressure category

df_raw["blood_category"].value_counts().plot(kind='pie')
with sns.axes_style('white'):

    g = sns.factorplot("blood_category", data=df_raw, aspect=4.0, kind='count',

                       hue='cardio', palette="Set2", order=["Normal", "Elevated", "High Blood Pressure(Stage 1)", "High Blood Pressure(Stage 2)", "Hypertensive Crisis"])

    g.set_ylabels('Number of Patients')
with sns.axes_style('white'):

    g = sns.factorplot("cholesterol", data=df_raw, aspect=4.0, kind='count',

                       hue='cardio', palette="Set2")

    g.set_ylabels('Number of Patients')
with sns.axes_style('white'):

    g = sns.factorplot("gluc", data=df_raw, aspect=4.0, kind='count',

                       hue='cardio', palette="Set2")

    g.set_ylabels('Number of Patients')
# Filtering out the required features

new_df = df_raw[["gender","age_tees","BMI","BMI_State","cholesterol","gluc","active","blood_category","cardio"]].copy()

new_df.head()
# Checking any missing values

new_df.isnull().sum()
# Label encode the categorical columns BMI_State & blood category

le = preprocessing.LabelEncoder()



# BMI_State

le.fit(new_df['BMI_State'])

new_df['BMI_State'] = le.transform(new_df['BMI_State'])



# blood category

le.fit(new_df['blood_category'])

new_df['blood_category'] = le.transform(new_df['blood_category'])



# age tees

le.fit(new_df['age_tees'])

new_df['age_tees'] = le.transform(new_df['age_tees'])



new_df.head()
# plotting correlation map

corr = new_df.corr()

f, ax = plt.subplots(figsize = (15,15))

sns.heatmap(corr, annot=True, fmt=".3f", linewidths=0.5, ax=ax)