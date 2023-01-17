# Elemental Libraries 

import pandas as pd 

import numpy as np



# Basic Visualization

import seaborn as sns 

import matplotlib.pyplot as plt 

%matplotlib inline 

sns.set_style(style="whitegrid")

import cufflinks as cf

cf.go_offline()



#Other Visualiation

import joypy as jp

from matplotlib import cm

from IPython.display import display

from PIL import Image



import warnings

warnings.filterwarnings('ignore')
from PIL import Image

display(Image.open('../input/imagenes/osemn.jpeg'))

df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')



print("----Techinical Information----")

print('Data Set Shape = {}'.format(df.shape))

print('Data Set Memory Usage = {:.2f} MB'.format(df.memory_usage().sum()/1024**2))

print('\n')

df.describe()
print('df_columns are ={}'.format(df.columns.to_list()))

print('-------------')

print('Species in data are = {}'.format(df['species'].value_counts()))

print('-------------')

print('Target Column is = {}'.format(df.columns[-1]))
# 2. Finding Missing Values 

df.isnull().sum()
# 3. Tranforming Data

from sklearn.preprocessing import OrdinalEncoder



Ordi_Var = df.iloc[:,4:]    # From Text to Numbers using Ordinal Encoder

OE = OrdinalEncoder().fit_transform(Ordi_Var)

df['target'] = OE

df.head()
sns.pairplot(data=df, hue='species', palette='Set2')

plt.show()
label = 'species'



f,axes = plt.subplots(2,2, figsize = (10,10) , dpi=100)

sns.violinplot(x = label , y = 'sepal_length', data = df , ax= axes[0,0])

sns.violinplot(x = label , y = 'sepal_width', data = df , ax= axes[0,1])

sns.violinplot(x = label , y = 'petal_length', data = df , ax= axes[1,0])

sns.violinplot(x = label  , y = 'petal_width', data = df , ax= axes[1,1])

plt.show()
!pip install joypy

import joypy



joypy.joyplot(data=df.iloc[:,0:], by = 'species'  ,overlap=7, figsize=(10,10), legend = True)

plt.show()
display(Image.open('../input/imagenes/dont understan.jfif'))
display(Image.open('../input/imagenes/i_image.jpg'))
# Analizying Target Variable 

df['species'].value_counts()



# We want to predict or classify Iris (flowers) , therefore we should have a balanced data set

# 3 flowers to predict , with n-rows for each one
# Dropping columns

df.drop('species',axis = 1, inplace=True)



def corr_heat (frame):   #<---Heat Map

    correlation = frame.corr()

    f,ax = plt.subplots(figsize=(15,10))

    mask = np.triu(correlation)

    sns.heatmap(correlation, annot=True, mask=mask,ax=ax,cmap='viridis')

    bottom,top = ax.get_ylim()

    ax.set_ylim(bottom+ 0.5, top - 0.5)

    

    

corr_heat(df)
df.iloc[:,:4].iplot(kind= 'box' , boxpoints = 'outliers')

plt.show()
# Extracting Outliers 



#Finding by index

Outliers = df[(df['sepal_width']>4) | (df['sepal_width']<2.2)].index



#Visualization

df[(df['sepal_width']>4) | (df['sepal_width']<2.2)]
# Dropping 

df.drop(Outliers, axis = 0 , inplace=True)



df.iloc[:,:4].iplot(kind= 'box' , boxpoints = 'outliers')

plt.show()
# Scenario 1 

SC_X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]



# Scenario 2 

SC_1X = df[['sepal_length', 'petal_length', 'petal_width']]



#Variable to predict 

y = df.iloc[:,-1]
from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
#---> Splitting Data

X_train, X_test, y_train, y_test = train_test_split(SC_X, y, test_size=0.2, random_state=104)





# 2. Second Bring the model 

LGR = LogisticRegression(random_state=0) #Logistic Regression

DT = DecisionTreeClassifier() #Decision Tree

RDF = RandomForestClassifier () #Randome Forest 





# --------------> 3.Fit models

#-->Scenario 1 

LGR = LGR.fit(X_train,y_train)

DT = DT.fit(X_train,y_train)

RDF = RDF.fit(X_train,y_train)



#4. Prediction

y_pred_LGR = LGR.predict(X_test)

y_pred_DT = DT.predict(X_test)

y_pred_RDF = RDF.predict(X_test)
#---> Splitting Data

X1_train, X1_test, y_train, y_test = train_test_split(SC_1X, y, test_size=0.2, random_state=104)





# 2. Second Bring the model 

LGR1 = LogisticRegression(random_state=0) #Logistic Regression

DT1 = DecisionTreeClassifier() #Decision Tree

RDF1 = RandomForestClassifier () #Randome Forest 





# --------------> 3.Fit models

#-->Scenario 1 

LGR1 = LGR1.fit(X1_train,y_train)

DT1 = DT1.fit(X1_train,y_train)

RDF1 = RDF1.fit(X1_train,y_train)



#4. Prediction

y_pred_LGR1 = LGR1.predict(X1_test)

y_pred_DT1 = DT1.predict(X1_test)

y_pred_RDF1 = RDF1.predict(X1_test)


#Confusion Matriz

print("-------------------------Scenario 1-----------------------")

print("\n")

print("The acurracy score of Logistical regression is {}".format(accuracy_score(y_test,y_pred_LGR)),"\n",confusion_matrix(y_test,y_pred_LGR))

print("\n")

print("The acurracy score of Decision Tree is {}".format(accuracy_score(y_test,y_pred_DT)),"\n",confusion_matrix(y_test,y_pred_DT))

print("\n")

print("The acurracy score of Randome Forest is {}".format(accuracy_score(y_test,y_pred_RDF)),"\n",confusion_matrix(y_test,y_pred_RDF))

print("\n")


#Confusion Matriz

print("-------------------------Scenario 2-----------------------")

print("\n")

print("The acurracy score of Logistical regression-2nd Scenario is {}".format(accuracy_score(y_test,y_pred_LGR1)),"\n",confusion_matrix(y_test,y_pred_LGR1))

print("\n")

print("The acurracy score of Decision Tree-2nd Scenario is {}".format(accuracy_score(y_test,y_pred_DT1)),"\n",confusion_matrix(y_test,y_pred_DT1))

print("\n")

print("The acurracy score of Randome Forest-2nd Scenario is {}".format(accuracy_score(y_test,y_pred_RDF1)),"\n",confusion_matrix(y_test,y_pred_RDF1))

print("\n")