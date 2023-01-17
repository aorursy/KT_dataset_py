import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import shap

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split
df=pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

df.describe()
def age(row):

    if row<=40:

        return 1

    if row>40 and row<=60:

        return 2

    if row>60:

        return 3    
def co(row):

    if row<200:

        return 1

    if row>=200and row<239:

        return 2

    if row>=239:

        return 3    
df['agetype']=df['age'].apply(age)

df['cholevel']=df['chol'].apply(co)
df.head()
df.loc[df.target== 1].age.min()
df.loc[(df.target== 1)].agetype.value_counts()
df.loc[(df.target== 1)].cp.value_counts()
df.loc[(df.cholevel == 1) & (df.target== 1)].sum()
df.loc[(df.target== 1)].sex.value_counts()
#scatterplot

# Set the width and height of the figure

plt.figure(figsize=(14,7))

sns.scatterplot(x=df['age'], y=df['chol'])

#common regression line

sns.regplot(x=df['age'], y=df['chol'])
#scatterplot

# Set the width and height of the figure

plt.figure(figsize=(14,7))

#color coded plot 

sns.scatterplot(x=df['trestbps'], y=df['chol'], hue=df['target'])
#Barplot

# Set the width and height of the figure

plt.figure(figsize=(10,6))

# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df['agetype'], y=df['target'])
#scatterplot

# Set the width and height of the figure

plt.figure(figsize=(14,7))

sns.scatterplot(x=df['trestbps'], y=df['thalach'])

#common regression line

sns.regplot(x=df['trestbps'], y=df['thalach'])
# Histogram

plt.figure(figsize=(10,6))

sns.distplot(a=df['thalach'], kde=False)
# KDE plot (density plot)(continueous)

plt.figure(figsize=(10,6))

sns.kdeplot(data=df['oldpeak'], shade=True)
# 2D KDE plot

plt.figure(figsize=(10,6))

sns.jointplot(x=df['trestbps'], y=df['oldpeak'], kind="kde")
df['cholevel'].value_counts()
d=[50,94,159]

lab=['normal','moderate','high']
# Creating plot 

fig = plt.figure(figsize =(16, 9))

plt.title('Cholestrol level of people ')

plt.pie(d,labels=lab )   

# show plot 

plt.show() 
y=df['target']

X=df.drop(['target','chol','age'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 9, criterion = 'entropy', random_state = 0)

clf.fit(X_train,y_train)
# Predicting the Test set results

y_pred = clf.predict(X_test)

print('accuracy of the model: ',accuracy_score(y_test,y_pred)*100)
# Predicting the Test set results

y_pred = clf.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
#SHAP VALUES

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1], X_test, plot_type="bar")