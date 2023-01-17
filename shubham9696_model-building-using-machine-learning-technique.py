import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.getcwd()    #to get current working directory
dataset = pd.read_csv('../input/dectecting-phishing-website-using-machine-learning/dataset.csv')    # read dataset

dataset.head(5)    # To display 1st 5 obervations
dataset.info()    # Information of all dataset
dataset.isnull().sum()    # For count missing values
dataset = dataset.drop_duplicates()    # For remove duplicates
dataset = dataset.drop('index',axis = 1)    # For drop index column
dataset.head()    # For display top 5 observation
dataset.head(10).T
dataset.columns.to_list()
from sklearn.preprocessing import LabelEncoder

# Create an object of the label encoder class
labelencoder = LabelEncoder()

# Apply labelencoder object on columns
dataset = dataset.apply(labelencoder.fit_transform)
dataset.head()
sns.set(style="darkgrid")    # To set background
sns.countplot('Result', data = dataset)    # Countplot
plt.title('Class Distribution Plot')    # Title
from matplotlib.pyplot import show
total = float(len(dataset)) # one person per row 
sns.set(style="darkgrid")

ax = sns.countplot(x="Result",hue = 'SSLfinal_State',data=dataset)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
plt.title('Multiple Bar Plot of SSLfinal_State v/s Result')
show()
from matplotlib.pyplot import show
total = float(len(dataset)) # one person per row 
sns.set(style="darkgrid")

ax = sns.countplot(x="Result",hue = 'Links_in_tags',data=dataset)    # Countplot

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,          # Loop for calculate percentages for each bar
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
plt.title('Multiple Bar Plot of Links_in_tags v/s Result')    # Title of plot
show()
from matplotlib.pyplot import show
total = float(len(dataset)) # one person per row 
sns.set(style="darkgrid")

ax = sns.countplot(x="Result",hue = 'web_traffic',data=dataset)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title('Multiple Bar Plot of web_traffic v/s Result')    # Title of plot
show()
from matplotlib.pyplot import show
total = float(len(dataset)) # one person per row 
sns.set(style="darkgrid")

ax = sns.countplot(x="Result",hue = 'URL_of_Anchor',data=dataset)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
plt.title('Multiple Bar Plot of URL_of_Anchor v/s Result')    # Title of plot
show()
from matplotlib.pyplot import show
total = float(len(dataset)) # one person per row 
sns.set(style="darkgrid")

ax = sns.countplot(x = "Result",hue = 'having_Sub_Domain',data=dataset)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
    
plt.title('Multiple Bar Plot of having_Sub_Domain v/s Result')
show()
plt.figure()
plt.figure(figsize = (20,40))    # For Figure size

for i in range(1,31):
    sns.set(style = 'darkgrid')
    plt.subplot(11,3,i)    # Create Subplot 
    sns.countplot(dataset.columns[i],data = dataset)   # Create Countplot
    plt.tight_layout()    # For tight graph layout
    
    
plt.title('Distributions of Features')
plt.show()
data_count = dataset.apply(pd.value_counts)    # For count values

data_count = data_count.T.iloc[:-2, : ]    # For draw last 2 rows

data_count.fillna(0, inplace = True)    # For fill missing value

data_count.style.background_gradient(cmap = 'Blues')    # For background style

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split    # import train_test_split module

x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:,:-1], dataset.iloc[:,-1], test_size = .20, random_state = 7)    # Split dataset into train, test

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.tree import DecisionTreeClassifier    # import DecisionTreeClassifier

model = DecisionTreeClassifier()    # Call model

model.fit(x_train, y_train)    # Fit model

y_pred = model.predict(x_test)    # Prediction

acc = accuracy_score(y_test, y_pred)    # Accuracy Score
print(acc)
model = LogisticRegression()    # Call model

model.fit(x_train, y_train)    # Fit model

y_pred = model.predict(x_test)    # Prediction

acc = accuracy_score(y_test, y_pred)    # Accuracy Score
print(acc)
model = KNeighborsClassifier()    # Call model

model.fit(x_train, y_train)    # Fit model

y_pred = model.predict(x_test)    # Prediction

acc = accuracy_score(y_test, y_pred)    # Accuracy Score
print(acc)
model = GaussianNB()    # Call model

model.fit(x_train, y_train)    # Fit model

y_pred = model.predict(x_test)    # Prediction

acc = accuracy_score(y_test, y_pred)    # Accuracy Score
print(acc)
model = SVC()    # Call model

model.fit(x_train, y_train)    # Fit model

y_pred = model.predict(x_test)    # Prediction

acc = accuracy_score(y_test, y_pred)
print(acc)
model = MLPClassifier()    # Call model

model.fit(x_train, y_train)    # Fit model

y_pred = model.predict(x_test)    # Prediction

acc = accuracy_score(y_test, y_pred)
print(acc)
model = RandomForestClassifier()    # Call model

model.fit(x_train, y_train)    # Fit model

y_pred = model.predict(x_test)    # Prediction

acc = accuracy_score(y_test, y_pred)    # Accuracy Score
print(acc)
pd.DataFrame({'Accuracy' : [0.9647, 0.9376, 0.9570, 0.6296, 0.9611, 0.9656, 0.9756]}, index = ['Decision_Tree', 'Logistic regression',
                                                                                    'K Nearest Neighbour', 'Navie Bayes', 'Support Vector Machine', 'ANN', 'Random Forest']).plot(kind = 'bar')
cr = classification_report(y_test, y_pred, target_names = ['phishing', 'No phishing'])    # Classification Report
print(cr)
sns.heatmap(confusion_matrix(y_test, y_pred),cmap = 'Blues', annot = True, fmt = '.2f',)
roc = plot_roc_curve(model, x_test, y_test)    # Plot Roc Curve
plt.title("ROC Curve")
plot_precision_recall_curve(model, x_test, y_test)    # Plot precision recall curve
plt.title("Precision Recall Curve")
feature_importance = model.feature_importances_    # Important Features

indices=np.argsort(feature_importance)[::-1]    # Reature importance in descending order
names=[dataset.columns[:-1][i] for i in indices ]     # Rearrange names


plt.figure(figsize=(10,4))    # Set Figure Size
plt.title("Features_Importnace")    # Add title
plt.bar(range(30), feature_importance[indices])    # Create Barplot
plt.xticks(range(30),names,rotation=90)    # Xticks for each Bar
plt.show()
import joblib
joblib.dump(model, 'Random_Forest.pkl')    # save model