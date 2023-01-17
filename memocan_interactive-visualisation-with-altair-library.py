# Imports
import altair as alt
import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import seaborn as sns
#alt.renderers.enable('notebook')
heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart.head()
# Rename the columns names
heart.columns = ['Age','Sex','Chest_Pain_Type','Blood_Pressure', 'Cholesterol', 'Blood_Sugar','ECG','Max_HeartRate','Exercise_InducedAgina','Oldpeak','Slope_of_Peak','Number_Vessels','Thal','Target'] 
# Data analysis
# Change the values of the categorical data to improve the readability on visualization. 
heart['Chest_Pain_Type'][heart['Chest_Pain_Type'] == 0] = 'typical angina'
heart['Chest_Pain_Type'][heart['Chest_Pain_Type'] == 1] = 'atypical angina'
heart['Chest_Pain_Type'][heart['Chest_Pain_Type'] == 2] = 'non-anginal pain'
heart['Chest_Pain_Type'][heart['Chest_Pain_Type'] == 3] = 'asymptomatic'
heart['Target'][heart['Target'] == 1] = 'Not Patient'
heart['Target'][heart['Target'] == 0] = 'Patient'
heart['Sex'][heart['Sex'] == 1] = 'Male'
heart['Sex'][heart['Sex'] == 0] = 'Women'
# Actually I created these new columns for my second design but I have learned that this design will not be implemented
# I use these columns for understanding intervals which has a large number of patients in my exploratory data analysis. 
heart['age_range'] = pd.cut(x=heart['Age'], bins=[25, 29, 34, 39, 44, 49, 54, 59, 64, 80], labels=['25-29', '30-34', '35-39','40-44','45-49','50-54','55-59','60-64', '65+'])
heart['blood_pre_range'] = pd.cut(x=heart['Blood_Pressure'], bins=[90, 99, 109, 119, 129, 139, 149, 159, 169, 200 ], labels=['90-99', '100-109', '110-119', '120-129', '130-139', '140-149', '150-159', '160-169', '170+'])
# My task is about Age and Blood Pressure, I change their data types float to integer for convenience. 
heart['Age'] = heart['Age'].astype('int')
heart['Blood_Pressure'] = heart['Blood_Pressure'].astype('int')
# First 5 rows of our data set
heart.head()
#Description of my data set
heart.describe()
# Check the data type of features
heart.dtypes
#Check the missing value
sns.heatmap(heart.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
# Fortunately we have not missing value
# Exploratory Data Analysis because of understanding basics of my task
sns.set_style('whitegrid')
sns.countplot(x = 'Target', data = heart, palette= ['r', 'cyan'])
# my data set has more non-patient than patient
sns.countplot(x = 'Target', hue = 'Sex', data = heart, palette = 'RdBu_r')
# I can expect more male patient than women patient. I should be more careful when displaying age and blood pressure ranges of male patients 
sns.countplot(x = 'Target', data = heart, hue = 'Chest_Pain_Type')
plt.legend(loc = 'upper right')

pd.crosstab(heart.age_range,heart.Target).plot(kind="bar",figsize=(20,6),colormap = 'seismic')
plt.title('Heart Disease Frequency for Ages')
# I should pay attention to 50+ ages
pd.crosstab(heart.blood_pre_range,heart.Target).plot(kind="bar",figsize=(20,6),colormap = 'seismic')
plt.title('Heart Disease Frequency for Blood Pressure')
# Logistic Regression
# We should create dummy variables because logistic regression can't understand categorical variable
sex = pd.get_dummies(heart['Sex'],drop_first = True)
chest_pain_type = pd.get_dummies(heart['Chest_Pain_Type'],drop_first = True)
heart = pd.concat([heart,sex,chest_pain_type],axis =1)
heart.head()
#Everything is okey. I don't drop first appereance of dummies because of visualization part and readability
y = heart['Target']
X = heart[['Age','Women','atypical angina','non-anginal pain','typical angina']]
#I take selected features as a parameter based on my task
#Import
from sklearn.model_selection import train_test_split
# I divided the X and y into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4, random_state = 101)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import classification_report
#Compare the prediction and actual values
print(classification_report(y_test,y_pred))
# precision value is not bad
chest = ['asymptomatic', 'non-anginal pain', 'atypical angina',
       'typical angina'] 
sex = ['Male','Women']
brush = alt.selection(type='interval')
select = alt.selection_single(name = 'Select',
                              fields=['Sex','Chest_Pain_Type'],init={'Sex': sex[0],'Chest_Pain_Type' : chest[0]},
                              bind= {'Sex' : alt.binding_radio(options = sex), 'Chest_Pain_Type' : alt.binding_select(options = chest)})
scale = alt.Scale(domain=['Patient','Not Patient'],
                  range=['#d62728', '#17becf'])
color = alt.Color('Target:N', scale=scale)
points = alt.Chart(heart).mark_point(filled = True,size = 75).add_selection(select).encode(
    x= alt.X('Age:Q',scale = alt.Scale(domain = [20,80])),
    y=alt.Y('Blood_Pressure:Q',scale = alt.Scale(domain = [80,200])),
    color=alt.condition(brush,color,alt.value('lightgray')),
    opacity=alt.condition(select, alt.value(0.9), alt.value(0.01)),
    tooltip = [alt.Tooltip('Age:Q'),alt.Tooltip('Blood_Pressure:Q')]
).add_selection(
    brush
).properties(
    width=650,
    height=400
)

bars = alt.Chart(heart).mark_bar().encode(
    y='Target:N',
    x='count():Q',
    color=alt.Color('Target:N', scale = scale)
).transform_filter(brush).transform_filter(select)
text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=3  
).encode(text = 'count(Target):Q')

alt.vconcat(
    points,
    bars + text,
    data=heart,
    title = 'Age versus Blood Pressure'
).configure_title(anchor = 'middle',fontWeight = 'bold',fontSize = 20).configure_legend(labelFontSize = 15,titleFontSize = 15
).configure_axis(titleFontSize = 15)
