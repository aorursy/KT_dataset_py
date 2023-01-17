import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#import data

stroke = pd.read_csv('../input/healthcare-dataset-stroke-data/train_2v.csv')

stroke.head()
stroke[('stroke')].value_counts()

##dataset is very unbalanced and will affect the ML accuracy
shuffled_data = stroke.sample(frac=1,random_state=4)

stroke_df = stroke.loc[stroke['stroke'] == 1]

non_stroke_df = stroke.loc[stroke['stroke'] == 0].sample(n= 3500,random_state= 101)

# non-stroke sufferers were reduced to 3500 to balance the data set.
normalized_stroke = pd.concat([stroke_df, non_stroke_df])
sns.countplot('stroke', data= normalized_stroke, palette= "colorblind")

plt.title('Stroke Analysis')

plt.show()

#Dataset split by stroke.
sns.countplot(x='stroke', hue = 'gender', data = normalized_stroke, palette = "Set1")

plt.title('Gender Split')

plt.show()

#Split by gender
sns.violinplot(x = 'stroke', y = 'age', hue = "gender", data=normalized_stroke, palette= "Set1")

#Split by gender and age
plt.figure(figsize=(8,7))

sns.boxplot(x = 'stroke', y = 'bmi', hue = 'gender', data= normalized_stroke, palette= "winter")

plt.title('Subject BMIs')

plt.show()

#The mean BMI across genders was checked.
sns.heatmap(normalized_stroke.isnull(), yticklabels=False, cbar=False, cmap='viridis')
def input_bmi(cols):

    bmi = cols[0]

    stroke = cols [1]

   

    

    if pd.isnull(bmi):

        return 28.6

    else:

        return bmi

# Ifelse used to fill out missing BMI using the mean BMI 28.6
normalized_stroke['bmi'] = stroke[['bmi', 'stroke']].apply(input_bmi, axis=1)
sns.heatmap(normalized_stroke.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# Heatmap showing totall filled BMI numbers.
sns.countplot(x='stroke', hue = 'Residence_type', data =normalized_stroke, palette = 'GnBu')

plt.title('Residence Type')

plt.show()

# Count plot to check the occurence of stroke across rural and urban areas
sns.countplot(x='ever_married', hue = 'stroke', data = normalized_stroke)

plt.title('Marital Status')

plt.show()

# Count plot to check the occurence of stroke by marital status
sns.countplot(x='hypertension', hue = 'stroke', data = normalized_stroke)

plt.title('Hypertension Check')

plt.show()

# Count plot to check subjects with/without hypertension and stroke
sns.countplot(x='heart_disease', hue = 'stroke', data = normalized_stroke)

plt.title('Heart Condition')

plt.show()

# Count plot to check subjects with/without heart disease and stroke
sns.countplot(x='work_type', hue = 'stroke', data = normalized_stroke)

plt.title('Occupation')

plt.show()

# Count plot to check the occurence of stroke by occupation
sns.barplot(x='stroke', y = 'avg_glucose_level', data = normalized_stroke)

plt.title('Blood Glucose Level')

plt.show()

# Count plot to check subjects with/without blood sugar and stroke
residence = pd.get_dummies(normalized_stroke['Residence_type'])

residence.head()
residence = pd.get_dummies(normalized_stroke["Residence_type"], drop_first= True)
normalized_stroke.drop(["Residence_type"], axis = 1, inplace = True)
normalized_stroke = pd.concat([normalized_stroke, residence], axis = 1)
normalized_stroke.head()

#Checking feature engineering for Residence_type
normalized_stroke.rename(columns={'Urban':'Residence_type'}, 

                 inplace=True)
sex = pd.get_dummies(normalized_stroke['gender'])

sex = pd.get_dummies(normalized_stroke["gender"], drop_first= True)

normalized_stroke.drop(["gender"], axis = 1, inplace = True)

normalized_stroke = pd.concat([normalized_stroke, sex], axis = 1)
marital_status = pd.get_dummies(normalized_stroke['ever_married'])
marital_status = pd.get_dummies(normalized_stroke["ever_married"], drop_first= True)
normalized_stroke.drop(["ever_married", "smoking_status"], axis = 1, inplace = True)
normalized_stroke = pd.concat([normalized_stroke, marital_status], axis = 1)
normalized_stroke.rename(columns={'Yes':'marital_status'}, 

                 inplace=True)
occupation = pd.get_dummies(normalized_stroke['work_type'])
normalized_stroke.drop(["work_type"], axis = 1, inplace = True)
normalized_stroke = pd.concat([normalized_stroke, occupation], axis = 1)
normalized_stroke.drop(["avg_glucose_level"], axis = 1, inplace = True)
normalized_stroke.head()

# Dataset fully engineered to accurately represent the underlying structure of the data and 

# to create the best model.
normalized_stroke.drop(["id"], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split
X = normalized_stroke.drop('stroke', axis = 1)

y = normalized_stroke['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
logmodel.score(X_test, y_test)

#ML Accuracy: 84%. Prediction could improve with a more balanced dataset.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), annot= True, cmap = 'viridis', fmt="2")

plt.title('Confusion Matrix')

plt.show()