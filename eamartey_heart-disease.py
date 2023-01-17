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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.rename(columns={"thalach": "max_heart_rate", "oldpeak": "ST_depression", 'cp':'chestpain', 'trestbps': 'restbps'}, inplace = True)
df.head()
df.thal.value_counts()
df.corr()
plt.figure(figsize = (10, 8))
sns.heatmap(df.corr())
sns.catplot('target', 'age', kind = 'box', data = df) 

sns.catplot('target', 'chestpain', kind = 'violin', data = df)  

sns.catplot('target', 'chol', kind = 'box', data = df)
sns.catplot('target', 'chol', kind = 'box', data = df, hue = 'sex') 
sns.catplot('target', 'restecg', kind = 'violin', data = df) 
sns.catplot('target', 'max_heart_rate', kind = 'box', data = df ) 
sns.catplot('target', 'slope', kind = 'violin', data = df) 
sns.catplot('target', 'ca', kind = 'violin', data = df)  
sns.catplot('target', 'exang', kind = 'violin', data = df) 
df.head()
sns.catplot('target', 'thal', kind = 'violin', data = df) 
sns.catplot('target', 'ST_depression', kind = 'box', data = df) 
##sns.catplot('target', 'restecg', kind = 'violin', data = df, hue = 'sex') #For both men and women those with a higher resting ecg have heart disease present
#sns.catplot('target', 'fbs', kind = 'violin', data = df) ### Fasting blood glucose did not determine the presence or absence of heart disease 
## sns.catplot('target', 'restbps', kind = 'box', data = df) #Resting blood pressure doesn't play a role in the presence or absence of a heart disease
fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(16, 5))   
sns.set_style("darkgrid")
sns.violinplot('target', 'fbs', data = df, ax=ax1)
sns.violinplot('target', 'restbps', data = df, palette='rocket',ax=ax2)
sns.violinplot('target', 'sex', data = df,palette='deep', ax=ax3) 
ax1.set_title('Plot of target with Fasting blood sugar', color='black')
ax2.set_title('Plot of target with Resting blood pressure', color='black')
ax3.set_title('Plot of target with Sex', color='black')
plt.tight_layout()
plt.show()

df.head()
#sns.catplot('target', 'sex', kind = 'violin', data = df)## Sex doesn't play a role in the presence or absence of heart disease
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2, figsize=(12,12))
sns.set_style("darkgrid")
sns.boxplot('sex', 'age', data = df,palette='dark', ax=ax1)
sns.boxplot('chestpain', 'age', data = df,palette='viridis', ax=ax2)
sns.scatterplot('restbps', 'age', data = df, color = 'green',ax=ax3)
sns.scatterplot('chol', 'age', data = df,palette='vlag', ax=ax4)
sns.boxplot('fbs', 'age', data = df,palette='dark', ax=ax5)
sns.boxplot('restecg', 'age', data = df,palette='Spectral', ax=ax6)
ax1.set_title('Plot of Age with Sex', color='black')
ax2.set_title('Plot of Age with Chest pain', color='black')
ax3.set_title('Plot of Age with Resting Blood Pressure', color='black')
ax4.set_title('Plot of Age with Serum Cholesterol', color='black')
ax5.set_title('Plot of Age with Fasting Blood Glucose', color='black')
ax6.set_title('Plot of Age with Resting ECG', color='black')
 
plt.tight_layout()
plt.show()
fig, ((ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(3,2, figsize=(12,12))
sns.set_style("darkgrid")
sns.scatterplot('max_heart_rate', 'age', data = df, color = 'green',ax=ax7)
sns.scatterplot('ST_depression', 'age', data = df,color = 'red', ax=ax8)
sns.boxplot('exang', 'age', data = df,palette='dark', ax=ax9)

sns.boxplot('slope', 'age', data = df,palette='viridis', ax=ax10)
sns.boxplot('ca', 'age', data = df,palette='YlOrBr', ax=ax11)
sns.boxplot('thal', 'age', data = df,palette='rocket', ax=ax12)
ax7.set_title('Plot of Age with Max Heart Rate', color='black')
ax8.set_title('Plot of Age with Chest pain', color='black')
ax9.set_title('Plot of Age with Resting Blood Pressure', color='black')
ax10.set_title('Plot of Age with Serum Cholesterol', color='black')
ax11.set_title('Plot of Age with Fasting Blood Glucose', color='black')
ax12.set_title('Plot of Age with Resting ECG', color='black')
plt.tight_layout()
plt.show()    
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2, figsize=(15,15))
sns.set_style("darkgrid")
sns.violinplot('sex', 'chestpain', data = df,palette='dark', ax=ax1)
sns.boxplot('sex', 'restbps', data = df,palette='rocket', ax=ax2)
sns.violinplot('sex', 'chol', data = df, color = 'green',ax=ax3)
sns.violinplot('sex', 'fbs', data = df,color = 'green', ax=ax4)
sns.violinplot('sex', 'restecg', data = df,palette='dark', ax=ax5)
sns.boxplot('sex', 'ST_depression', data = df, palette='rocket', ax=ax6)
ax1.set_title('Plot of Sex with Chest pain', color='black')
ax2.set_title('Plot of Sex with Resting Blood Pressure', color='black')
ax3.set_title('Plot of Sex with ST Depression', color='black')
ax6.set_title('Plot of Sex with Serum Cholesterol', color='black')
ax4.set_title('Plot of Sex with Fasting Blood Glucose', color='black')
ax5.set_title('Plot of Sex with Resting ECG', color='black')
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize=(15, 10))
sns.set_style("darkgrid")
sns.violinplot('sex', 'exang', data = df,palette='dark', ax=ax1)
sns.violinplot('sex', 'slope', data = df, color = 'green',ax=ax3)
sns.violinplot('sex', 'ca', data = df,color = 'green', ax=ax2)
sns.violinplot('sex', 'thal', data = df,palette='dark', ax=ax4)
ax1.set_title('Plot of Sex with Excercise Induced Angina', color='black')
ax2.set_title('Plot of Sex with Slope', color='black')
ax3.set_title('Plot of Sex with Colored vessels on Flouroscopy', color='black')
ax4.set_title('Plot of Sex with Thalium Stress Test', color='black')
 
 
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2, figsize=(15,16))
sns.set_style("darkgrid")
sns.violinplot('chestpain', 'chol', data = df,palette='dark', ax=ax1)
sns.boxplot('chestpain', 'restbps', data = df, ax=ax2)
sns.violinplot('chestpain', 'fbs', data = df, color = 'green',ax=ax3)
sns.violinplot('chestpain', 'exang', data = df,palette='viridis', ax=ax4)
sns.violinplot('chestpain', 'restecg', data = df,palette='dark', ax=ax5)
sns.boxplot('chestpain', 'max_heart_rate', data = df,palette='YlOrBr', ax=ax6)
ax1.set_title('Plot of Chestpain with Sex', color='black')
ax4.set_title('Plot of Chestpain with Exercise Induced Angina', color='black')
ax5.set_title('Plot of Chestpain with Resting Blood Pressure', color='black')
ax1.set_title('Plot of Chestpain with Serum Cholesterol', color='black')
ax3.set_title('Plot of Chestpain with Fasting Blood Glucose', color='black')
ax6.set_title('Plot of Chestpain with Max Heart Rate', color='black')
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize=(15,15))
sns.set_style("darkgrid")
sns.violinplot('chestpain', 'ST_depression', data = df,palette='dark', ax=ax1)
sns.violinplot('chestpain', 'slope', data = df,palette='rocket', ax=ax2)
sns.violinplot('chestpain', 'ca', data = df, palette='YlOrBr',ax=ax3)
sns.violinplot('chestpain', 'thal', data = df,palette='viridis', ax=ax4)
ax1.set_title('Plot of Chestpain with ST_depression', color='black')
ax2.set_title('Plot of Chestpain with Slope', color='black')
ax3.set_title('Plot of Chestpain with Colored Vessels on Flouroscopy ', color='black')
ax4.set_title('Plot of Chestpain with Thalium Stress Test', color='black')
df.head()
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2, figsize=(15,15))
sns.set_style("darkgrid")
sns.scatterplot('restbps', 'chol', data = df,palette='dark', ax=ax1)
sns.scatterplot('ST_depression','restbps', data = df, color = 'green', ax=ax2)
sns.boxplot('fbs','restbps', data = df, color = 'green',ax=ax3)
sns.boxplot('exang','restbps', data = df,color = 'orange', ax=ax4)
sns.boxplot('restecg','restbps', data = df,palette='dark', ax=ax5)
sns.scatterplot('restbps', 'max_heart_rate', data = df,color = 'red', ax=ax6)
ax1.set_title('Plot of Resting Blood pressure with Serum Cholesterol', color='black')
ax2.set_title('Plot of Resting Blood pressure with  ST Depression', color='black')
ax3.set_title('Plot of Resting Blood pressure with Fasting Blood Glucose', color='black')
ax4.set_title('Plot of Resting Blood pressure with Exercise Induced Angina', color='black')
ax5.set_title('Plot of Resting Blood pressure with REsting ECG', color='black')
ax6.set_title('Plot of Resting Blood pressure with Max Heart Rate', color='black')
fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(18,6))
sns.set_style("darkgrid")
sns.violinplot('slope','restbps', data = df,palette='dark', ax=ax1)
sns.violinplot('ca','restbps', data = df,palette='rocket', ax=ax2)
sns.violinplot('thal','restbps', data = df, color = 'green',ax=ax3)
ax1.set_title('Plot of Resting Blood pressure with Slope', color='black')
ax2.set_title('Plot of Resting Blood pressure with Colored vessels on Flouroscopy', color='black')
ax3.set_title('Plot of Resting Blood pressure with Thallium', color='black')
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2, figsize=(15,15))
sns.set_style("darkgrid")
sns.violinplot('slope','chol', data = df,palette='dark', ax=ax6)
sns.scatterplot('ST_depression','chol', data = df, color = 'green', ax=ax2)
sns.boxplot('fbs','chol', data = df, color = 'green',ax=ax3)
sns.boxplot('exang','chol', data = df,color = 'orange', ax=ax4)
sns.boxplot('restecg','chol', data = df,palette='dark', ax=ax5)
sns.scatterplot('max_heart_rate','chol', data = df,color = 'red', ax=ax1)
ax6.set_title('Plot of Serum Cholestrol with Slope', color='black')
ax2.set_title('Plot of Serum Cholestrol with  ST Depression', color='black')
ax3.set_title('Plot of Serum Cholestrol with Fasting Blood Glucose', color='black')
ax4.set_title('Plot of Serum Cholestrol with Exercise Induced Angina', color='black')
ax5.set_title('Plot of Serum Cholestrol with Resting ECG', color='black')
ax1.set_title('Plot of Serum Cholestrol with Max Heart Rate', color='black')
fig, (ax2,ax3) = plt.subplots(1,2, figsize=(18,6))
sns.set_style("darkgrid")

sns.violinplot('ca','chol', data = df,palette='rocket', ax=ax2)
sns.violinplot('thal','chol', data = df, color = 'green',ax=ax3)
ax2.set_title('Plot of Serum Cholestrol with Colored vessels on Flouroscopy', color='black')
ax3.set_title('Plot of Serum Cholestrol  with Thallium', color='black')
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15,20))
sns.set_style("darkgrid")
sns.violinplot('fbs','slope', data = df,palette='dark', ax=ax1)
sns.violinplot('fbs','ST_depression', data = df, color = 'green', ax=ax2)
sns.violinplot('fbs','ca', data = df, color = 'green',ax=ax3)
sns.violinplot('fbs','exang', data = df,color = 'orange', ax=ax4)
sns.violinplot('fbs','restecg', data = df,palette='dark', ax=ax5)
sns.violinplot('fbs','max_heart_rate', data = df,color = 'red', ax=ax6)
sns.violinplot('fbs','thal', data = df,palette='dark', ax=ax7)
ax1.set_title('Plot of Fasting Blood Glucose with Slope', color='black')
ax2.set_title('Plot of Fasting Blood Glucose with  ST Depression', color='black')
ax3.set_title('Plot of Fasting Blood Glucose with Colored vessels on flouroscopy', color='black')
ax4.set_title('Plot of Fasting Blood Glucose with Exercise Induced Angina', color='black')
ax5.set_title('Plot of Fasting Blood Glucose with Resting ECG', color='black')
ax6.set_title('Plot of Fasting Blood Glucose with Max Heart Rate', color='black')
ax7.set_title('Plot of Fasting Blood Glucose with Thallium stress test', color='black')
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(15,15))
sns.set_style("darkgrid")
sns.violinplot('restecg','slope', data = df,palette='dark', ax=ax1)
sns.violinplot('restecg','ST_depression', data = df, color = 'green', ax=ax2)
sns.violinplot('restecg','ca', data = df, color = 'green',ax=ax3)
sns.violinplot('restecg','exang', data = df,color = 'orange', ax=ax4)
sns.violinplot('restecg','thal', data = df,palette='dark', ax=ax5)
sns.violinplot('restecg','max_heart_rate', data = df,color = 'red', ax=ax6)
 
ax6.set_title('Plot of Resting ECG with Slope', color='black')
ax2.set_title('Plot of Resting ECG with  ST Depression', color='black')
ax3.set_title('Plot of Resting ECG with Colored vessels on flouroscopy', color='black')
ax4.set_title('Plot of Resting ECG with Exercise Induced Angina', color='black')
ax5.set_title('Plot of Resting ECG with Thallium stress test', color='black')
ax1.set_title('Plot of Resting ECG with Max Heart Rate', color='black')
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(15,15))
sns.set_style("darkgrid")
sns.violinplot('slope','max_heart_rate', data = df,palette='dark', ax=ax1)
sns.scatterplot('ST_depression','max_heart_rate', data = df, color = 'green', ax=ax2)
sns.violinplot('ca','max_heart_rate', data = df, color = 'green',ax=ax3)
sns.violinplot('exang','max_heart_rate', data = df,color = 'orange', ax=ax4)
sns.violinplot('thal','max_heart_rate', data = df,palette='dark', ax=ax5) 
 
ax1.set_title('Plot of max_heart_rate with Slope', color='black')
ax2.set_title('Plot of max_heart_rate with  ST Depression', color='black')
ax3.set_title('Plot of max_heart_rate with Colored vessels on flouroscopy', color='black')
ax4.set_title('Plot of max_heart_rate with Exercise Induced Angina', color='black')
ax5.set_title('Plot of max_heart_rate with Thallium stress test', color='black')

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
sns.set_style("darkgrid")
sns.violinplot('exang','slope', data = df,palette='dark', ax=ax1)
sns.violinplot('exang','ST_depression', data = df, color = 'green', ax=ax2)
sns.violinplot('exang','ca', data = df, color = 'green',ax=ax3)
sns.violinplot('exang','thal', data = df,palette='dark', ax=ax4) 
 
ax1.set_title('Plot of max_heart_rate with Slope', color='black')
ax2.set_title('Plot of max_heart_rate with  ST Depression', color='black')
ax3.set_title('Plot of max_heart_rate with Colored vessels on flouroscopy', color='black')
 
ax4.set_title('Plot of max_heart_rate with Thallium stress test', color='black')

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
sns.set_style("darkgrid")
sns.violinplot('slope','ST_depression', data = df,palette='dark', ax=ax1)
sns.violinplot('ca','ST_depression', data = df, color = 'green',ax=ax2)
sns.violinplot('thal','ST_depression', data = df,palette='dark', ax=ax3) 
 
ax1.set_title('Plot of ST_depression with Slope', color='black')
 
ax2.set_title('Plot of ST_depression Colored vessels on flouroscopy', color='black')
 
ax3.set_title('Plot of ST_depression with Thallium stress test', color='black')
df.head(1)
fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(15,10))
sns.set_style("darkgrid")
 
sns.violinplot('ca','slope', data = df, color = 'green',ax=ax1)
sns.violinplot('thal','slope', data = df,palette='dark', ax=ax2) 
 
 
ax2.set_title('Plot of ST_depression Colored vessels on flouroscopy', color='black')
 
ax3.set_title('Plot of ST_depression with Thallium stress test', color='black')
fig, ((ax1)) = plt.subplots(1, 1, figsize=(15,10))
sns.set_style("darkgrid")
 
sns.violinplot('ca','thal', data = df, color = 'green',ax=ax1)
 
 
ax2.set_title('Plot of ST_depression Colored vessels on flouroscopy', color='black')
 
ax3.set_title('Plot of ST_depression with Thallium stress test', color='black')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
df.head()
#df.drop(['restbps', 'fbs', 'sex'], axis = 1, inplace = True)
df.head()
x = df.iloc[:, :-1]
y = df.iloc[:, -1:]
y.head()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 1, test_size = 0.2)
lr = LogisticRegression()
dt = DecisionTreeClassifier(random_state = 1)
gr = GradientBoostingClassifier(learning_rate = 0.9)
rf = RandomForestClassifier(random_state = 1)
ss = StandardScaler()
svc = SVC(C = 0.1, kernel = 'linear')
svc.fit(xtrain, ytrain)
svc.score(xtrain, ytrain)
svc.score(xtest, ytest)
svr = SVC(C = 0.2, kernel = 'sigmoid')
svr.fit(xtrain, ytrain)
svr.score(xtrain, ytrain)
svr.score(xtest, ytest)
pipe = Pipeline([('ss', ss), ('rf', rf)])
pipe.fit(xtrain, ytrain)
pipe.score(xtrain, ytrain)
pipe.score(xtest, ytest)
lr.fit(xtrain, ytrain)
lr.score(xtrain, ytrain)
lr.score(xtest, ytest)
gr.fit(xtrain, ytrain)
gr.score(xtrain, ytrain)
gr.score(xtest, ytest)
dt.fit(xtrain, ytrain)
dt.score(xtrain, ytrain)
dt.score(xtest, ytest)
rf.fit(xtrain, ytrain)
rf.score(xtrain, ytrain)
rf.score(xtest, ytest)
estimators = [('lr', lr),('svc', svc)]
sr = StackingClassifier(estimators=estimators)
sr.fit(xtrain, ytrain)
sr.score(xtrain, ytrain)
sr.score(xtest, ytest)
df.head()
fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(16, 5))   
sns.set_style("darkgrid")
sns.violinplot('target', 'ST_depression', data = df,palette='viridis', ax=ax1)
sns.violinplot('target', 'slope', data = df, palette='rocket',ax=ax2)
sns.violinplot('target', 'ca', data = df,palette='dark', ax=ax3) 
ax1.set_title('Plot of target with ST depression', color='black')
ax2.set_title('Plot of target with Slope of ST', color='black')
ax3.set_title('Plot of target with Colored vessels on flourosopy', color='black')
ax4.set_title('"Age" in test dataset, by "Sex"', color='red')
plt.tight_layout()
plt.show()

### Slope is the slope of the peak exercise ST segment (has 3 values)
### During exercise slope becomes depressed in a healthy heart, with maximum depression 
###occuring at peak exercise. The normal ST segment during exercise therefore slopes sharply upwards.
### So those with higher slopes have heart disease present
### Those with lower ST depression have the presence of heart disease
## ca is the number of major vessels (0-3) colored by flourosopy
### Those with lower amounts of major blood vessels colored by flourosopy have heart disease, probably indicating that those blood vessels are clogged.One would think that that should result in angina during exercise, but the major blood vessels are not specified in the data description, so further investigation is needed
### as to why no angina is induced by people with heart disease in this data, that's a question that needs further investigating
fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(16, 5))   
sns.set_style("darkgrid")
sns.violinplot('target', 'restecg', data = df,palette='YlOrBr', ax=ax1)
sns.violinplot('target', 'max_heart_rate', data = df, palette='cubehelix',ax=ax2)
sns.violinplot('target', 'exang', data = df,palette='dark', ax=ax3) 
ax1.set_title('Plot of target with Resting ECG', color='black')
ax2.set_title('Plot of target with Maximum heart rate', color='black')
ax3.set_title('Plot of target with Excercise induced angina', color='black')
ax4.set_title('"Age" in test dataset, by "Sex"', color='red')
plt.tight_layout()
plt.show()

### Those with a higher resting ecg have heart disease present
### Those who achieved higher maximum heart rate have heart disease
### Exang is exercise induced angina (pain in the heart caused by excercise) (1 = yes; 0 = no)
### Those who don't have exercise induced angina have heart disease. This could be a signal that the heart itself might be diseased
### Further research on this would be needed to get to the bottom of this
fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(16, 5))   
sns.set_style("darkgrid")
sns.violinplot('target', 'age', data = df, ax=ax1)
sns.violinplot('target', 'chestpain', data = df, palette='rocket',ax=ax2)
sns.violinplot('target', 'chol', data = df,palette='dark', ax=ax3)
ax1.set_title('Plot of target with age', color='black')
ax2.set_title('Plot of target with chestpain', color='black')
ax3.set_title('Plot of target with serum cholesterol', color='black')
ax4.set_title('"Age" in test dataset, by "Sex"', color='red')
plt.tight_layout()
plt.show()

### Heart disease is occuring in those who are younger than those who are older. To understand why this is happening is another phase of the research, which isn't available in this dataset.
### As far as those with chest pain, heart disease is present in those with chest pain
### Serum cholesterol is lower in people who had the heart disease
