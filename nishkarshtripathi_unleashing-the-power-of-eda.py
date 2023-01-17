# Importing the libraries



import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly

import plotly.figure_factory as ff

import plotly.graph_objects as go

from plotly.offline import iplot
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
print("Number of rows in data :", data.shape[0])

print("Number of columns in data :", data.shape[1])
data.head()
data.info()
# Statistical properties of data

data.describe().round(3)
# Columns names



data.columns
data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',

       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
data.head()
new_data = data.copy() # for future use
# Let's see which columns has less or equal to 5 classes



for column in data.columns:

    if len(data[column].unique())<=5:

        print(f"{column} has {len(data[column].unique())} classes.")
# Let's map these class values to something meaning full information given in dataset description

# Note: As rest_ecg  and num_major_vessels doesn't have any description so I left that column



data.sex = data.sex.map({0:'female', 1:'male'})



data.chest_pain_type = data.chest_pain_type.map({1:'angina pectoris', 2:'atypical angina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})



data.fasting_blood_sugar = data.fasting_blood_sugar.map({0:'lower than 120mg/ml', 1:'greater than 120mg/ml'})



data.exercise_angina = data.exercise_angina.map({0:'no', 1:'yes'})



data.st_slope = data.st_slope.map({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})



data.thalassemia = data.thalassemia.map({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})



data.target = data.target.map({0:'No Heart Disease', 1:'Heart Disease'})
data.head()
# Drawing a correlation Plot



fig=plt.figure(figsize=(12,8))

sns.heatmap(new_data.corr(), annot= True, cmap='Blues')
print(f"Minimum Age : {min(data.age)} years")

print(f"Maximum Age : {max(data.age)} years")
sex_counts = data['sex'].value_counts().tolist()



print(f"Number of Male patients: {sex_counts[0]}")

print(f"Number of Female patients: {sex_counts[1]}")
# Count of male and female patients



sns.countplot('sex', hue="sex", data=data, palette="bwr")
# Let's look at the distribution of age



hist_data = [data['age']]

group_labels = ['age'] 



colors = ['#835AF1']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,

                         bin_size=10, show_rug=False)



# Add title

fig.update_layout(width=700, title_text='Age Distribution')

fig.show()
young_patients = data[(data['age']>=29)&(data['age']<40)]

middle_aged_patients = data[(data['age']>40)&(data['age']<55)]

old_aged_patients = data[(data['age']>55)]



print(f"Number of Young Patients : {len(young_patients)}")

print(f"Number of Middle Aged Patients: {len(middle_aged_patients)}")

print(f"Number of Old Aged Patients : {len(old_aged_patients)}")
# Plotting a pie chart for age ranges of patients



labels = ['Young Age','Middle Aged','Old Aged']

values = [

      len(young_patients), 

      len(middle_aged_patients),

      len(old_aged_patients)

]

colors = ['gold', 'mediumturquoise', 'darkorange']



trace = go.Pie(labels=labels, values=values,

               hoverinfo='value+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, line=dict(color='#000000', width=2)))



plotly.offline.iplot([trace], filename='styled_pie_chart')
# Age and Target based on sex



fig = px.bar(data, x=data['target'], y=data['age'], color='sex', height=500, width=800)

fig.update_layout(title_text='BarChart for Age Vs Target on Basis of Sex')

fig.show()
# Box plot for target and age based on sex



fig = px.box(data, x="target", y="age", points="all", color='sex')

fig.update_layout(title_text='BoxPlot of Age Vs Target')

fig.show()
# Plot of age and Maximum Heart Rate



df = px.data

fig = px.scatter(data, x="age", y="max_heart_rate", color="sex", hover_data=['age','max_heart_rate'])

fig.update_layout(title = "Scatter Plot of Age Vs Max Heart Rate")

fig.show()
# Plot of Age vs Resting Blood Pressure



df = px.data

fig = px.scatter(data, x="age", y="resting_blood_pressure", color="sex", hover_data=['age','resting_blood_pressure'])

fig.update_layout(title = "Scatter Plot of Age Vs Resting Blood Pressure")

fig.show()
# Plot of age vs Serum Cholesterol



df = px.data

fig = px.scatter(data, x="age", y="serum_cholesterol", color="sex", hover_data=['age','serum_cholesterol'])

fig.update_layout(title = "Scatter Plot of Age Vs Serum Cholesterol")

fig.show()
# Counts of Chest pain type among Heart Patients and Non-Heart Patients



sns.countplot(x="chest_pain_type", hue="target", data=data, palette="bwr")

plt.title("Chest Pain Types grouped by Targets")
# Counts of Chest pain type among male and female patients



sns.countplot(x="chest_pain_type", hue="sex", data=data, palette="bwr")
# Finding the maximum and minimum Resting Blood Pressure



print(f"Maximum Resting Blood Pressure : {data['resting_blood_pressure'].max()}")

print(f"Minimum Resting Blood Pressure : {data['resting_blood_pressure'].min()}")
# Let's look at the distribution of Resting Blood Pressure



hist_data = [data['resting_blood_pressure']]

group_labels = ['resting_blood_pressure'] 



colors = ['#008080']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,

                         bin_size=10, show_rug=False)



# Add title

fig.update_layout(width=700, title_text='Resting Blood Pressure Distribution')

fig.show()
# Plot of Resting Blood Pressure with Target



sns.barplot(x="target", y='resting_blood_pressure',data = data, palette="bwr")

plt.title('Resting Blood Pressure vs Target')

plt.show()
# Boxplot of Resting Blood Pressure with Target



fig = px.box(data, x="target", y="resting_blood_pressure", points="all", color='sex')

fig.update_layout(title_text='BoxPlot of Resting Blood Pressure Vs Target')

fig.show()
# Finding the maximum and minimum Serum Cholesterol



print(f"Maximum Serum Cholesterol : {data['serum_cholesterol'].max()}")

print(f"Minimum Serum Cholesterol : {data['serum_cholesterol'].min()}")
# Let's look at the distribution of Serum Cholesterol



hist_data = [data['serum_cholesterol']]

group_labels = ['serum_cholesterol'] 



colors = ['#DA70D6']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,

                         bin_size=10, show_rug=False)



# Add title

fig.update_layout(width=700, title_text='Serum Cholesterol Distribution')

fig.show()
# Plot of Serum Cholesterol and Target



sns.barplot(x="target", y='serum_cholesterol',data = data, palette="bwr")

plt.title('Serum Cholesterol vs Target')

plt.show()
# Box Plot on basis of Sex



fig = px.box(data, x="target", y="serum_cholesterol", points="all", color='sex')

fig.update_layout(title_text='BoxPlot of Serum Cholesterol Vs Target')

fig.show()
# Box plot on basis of st_slope



fig = px.box(data, x="target", y="serum_cholesterol", points="all", color='st_slope')

fig.update_layout(title_text='Serum Cholesterol Vs Target ')

fig.show()
# Counts of Heart Disease and No Heart Disease Patients with fasting blood sugar above 120 mg/dl

# and lower than 120 mg/dl



sns.set(rc={'figure.figsize':(8.7,5.27)})



sns.countplot(hue='fasting_blood_sugar',x ='target',data = data, palette="bwr")

plt.title('Fasting Blood Sugar > 120 mg/dl')

plt.show()
# Counts of Male and Female Patients with fasting blood sugar above 120 mg/dl

# and lower than 120 mg/dl



sns.countplot(hue='fasting_blood_sugar',x ='sex',data = data, palette="bwr")

plt.title('Fasting Blood Sugar > 120 mg/dl')

plt.show()
# Resting electrocardiographic results (values 0,1,2))



sns.countplot(x='rest_ecg', hue ='target', data = data, palette="bwr")

plt.title('Resting electrocardiographic Results')

plt.show()
# Resting electrocardiographic results (values 0,1,2))



sns.countplot(x='rest_ecg', hue ='sex', data = data, palette="bwr")

plt.title('Resting electrocardiographic Results')

plt.show()
print(f"Maximum Max Heart Rate : {data['max_heart_rate'].max()}")

print(f"Minimum Max Heart Rate: {data['max_heart_rate'].min()}")
# Let's look at the distribution of Maximum Heart Rate



hist_data = [data['max_heart_rate']]

group_labels = ['max_heart_rate'] 



colors = ['#808000']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,

                         bin_size=10, show_rug=False)



# Add title

fig.update_layout(width=700, title_text='Maximum Blood Pressure Distribution')

fig.show()
# Maximum heart rate and target 



sns.barplot(x="target", y='max_heart_rate',data = data, palette="bwr")

plt.title('Maximum Heart Rate vs Target')

plt.show()
# Box plot on basis of sex



fig = px.box(data, x="target", y="max_heart_rate", points="all", color='sex')

fig.update_layout(title_text='Maximum Heart Rate Vs Target')

fig.show()
# Box plot on basis of st_slope



fig = px.box(data, x="target", y="max_heart_rate", points="all", color='st_slope')

fig.update_layout(title_text='Maximum Heart Rate Vs Target ')

fig.show()
# Plot on basis of Target



sns.countplot(x='exercise_angina', hue ='target', data = data, palette="bwr")

plt.title('Exercise Induced Angina')

plt.show()
# Count of Male and Female patients with and withour Exercise Induced Angina



sns.countplot(x = 'exercise_angina', hue ='sex', data = data, palette="bwr")

plt.title('Exercise Induced Angina')

plt.show()
print(f"Maximum Depression : {data['st_depression'].max()}")

print(f"Minimum Depression : {data['st_depression'].min()}")
# Let's look at the distribution of ST Depression



hist_data = [data['st_depression']]

group_labels = ['st_depression'] 



colors = ['#808000']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,

                         bin_size=0.2, show_rug=False)



# Add title

fig.update_layout(width=700, title_text='ST Depression Distribution')

fig.show()
# Box plot on basis of sex



fig = px.box(data, x="target", y="st_depression", points="all", color='sex')

fig.update_layout(title_text='ST Depression Vs Target')

fig.show()
# The slope of the peak exercise ST segment



sns.countplot(hue='st_slope',x ='target',data = data, palette="winter_r")

plt.title('Slope of the peak exercise ST segment')

plt.show()
# the slope of the peak exercise ST segment



sns.countplot(hue='st_slope',x ='sex',data = data, palette="winter_r")

plt.title('Slope of the peak exercise ST segment')

plt.show()
# Number of major vessels (0-4)



sns.countplot(hue='num_major_vessels',x ='target',data = data, palette="rainbow_r")

plt.title('Number of major vessels (0-4) colored by flourosopy')

plt.show()
# Thalassemia types based on target



sns.countplot(hue='thalassemia',x ='target',data = data, palette="gist_ncar")

plt.title('Thalassmia')

plt.show()
# Thalassemia types based on Chest Pain Type



plt.figure(figsize=(10,5))

sns.countplot(x="chest_pain_type", hue="thalassemia", data=data, palette="YlOrRd_r")
# Thalassemia types based on st_slope



plt.figure(figsize=(10,5))

sns.countplot(x="st_slope", hue="thalassemia", data=data, palette="YlOrRd_r")
# Thalassemia types based on sex



sns.countplot(hue='thalassemia',x ='sex',data = data, palette="gist_ncar")

plt.title('Thalassmia')

plt.show()