import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

!pip install plotly

import plotly.offline as py 

import plotly.graph_objs as go

import plotly.express as px

from collections import Counter  

from subprocess import call

from IPython.display import Image

############################################################################################

%matplotlib inline 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
credit=pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")

print("The dataset is {} credit record".format(len(credit)))
credit.head(2)
credit=credit.iloc[:, 1:]
credit.info()
credit.describe()
credit['Sex'].value_counts()
SA = credit.loc[:,['Sex','Age']]

fig = px.box(SA, x="Sex", y="Age", points="all",color="Sex")

fig.update_layout(

    title={

          'text':"Sex Vs Age Cross tabulation",

        'y':.95,

        'x':.5,

        'xanchor': 'center',

        'yanchor': 'top'

    },

    xaxis_title="Sex",

    yaxis_title="Age",

   

)

fig.show()
SC =credit.loc[:,['Sex','Credit amount']]

fig = px.box(SC, x="Sex", y="Credit amount", points="all", color="Sex")

fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default

fig.update_layout(

    title={

          'text':"Sex Vs Credit Amount Cross tabulation",

        'y':.95,

        'x':.5,

        'xanchor': 'center',

        'yanchor': 'top'

    },

    xaxis_title="Sex",

    yaxis_title="Age",

   

)

fig.show()
Purpose = credit['Purpose']

fig = px.histogram(credit, x="Purpose", color="Purpose")

fig.update_layout(

    title={

          'text':"Purpose breakdown",

        'y':.95,

        'x':.5,

        'xanchor': 'center',

        'yanchor': 'top'

    }

   

)

fig.show()
SC =credit.loc[:,['Purpose','Credit amount']]

fig = px.box(SC, x="Purpose", y="Credit amount", color="Purpose")

fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default

fig.update_layout(

    title={

          'text':"Purpose Vs Credit Amount Cross tabulation",

        'y':.95,

        'x':.5,

        'xanchor': 'center',

        'yanchor': 'top'

    },

    xaxis_title="Purpose",

    yaxis_title="Credit amount",

   

)

fig.show()
credit['Risk'] = credit['Risk'].map({'bad':1, 'good':0})
credit['Saving accounts'] = credit['Saving accounts'].fillna('Others')

credit['Checking account'] = credit['Checking account'].fillna('Others')
credit_clean=credit.copy()
cat_features = ['Sex','Housing', 'Saving accounts', 'Checking account','Purpose']

num_features=['Age', 'Job', 'Credit amount', 'Duration','Risk']

for variable in cat_features:

    dummies = pd.get_dummies(credit_clean[cat_features])

    df1= pd.concat([credit_clean[num_features], dummies],axis=1)



Risk= df1['Risk']          

df2=df1.drop(['Risk'],axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(df2,Risk,test_size=0.20,random_state = 30)
random_forest = RandomForestClassifier( random_state = 100)
#Standardization

sc=StandardScaler()

X_train_std=sc.fit_transform(X_train)

X_test_std=sc.transform(X_test)



n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]# Number of trees in random forest

max_features = ['auto', 'sqrt']# Number of features to consider at every split

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]# Minimum number of samples required to split a node

min_samples_leaf = [1, 2, 4]# Minimum number of samples required at each leaf node

bootstrap = [True, False]# Method of selecting samples for training each tree



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



random_forest = RandomForestClassifier(random_state = 100)

rf_random = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=4, scoring='recall', random_state=42, n_jobs = -1)

rf_random.fit(X_train_std, Y_train)

rf_random.best_params_
Y_test_pred = rf_random.predict(X_test_std)
confusion_matrix= confusion_matrix(Y_test, Y_test_pred)

confusion_matrix
y_true = ["bad", "good"]

y_pred = ["bad", "good"]

df_cm = pd.DataFrame(confusion_matrix, columns=np.unique(y_true), index = np.unique(y_true))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

df_cm.dtypes



plt.figure(figsize = (8,5))

plt.title('Confusion Matrix')

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
total=sum(sum(confusion_matrix))



sensitivity_recall = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

print('Sensitivity_recall : ',sensitivity_recall )



Specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

print('Specificity: ', Specificity)



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])

print('Precision: ', precision)



accuracy =(confusion_matrix[0,0]+confusion_matrix[1,1])/(confusion_matrix[0,0]+confusion_matrix[0,1]+

                                                         confusion_matrix[1,0]+confusion_matrix[1,1])

print('Accuracy: ', accuracy)
fpr, tpr, thresholds = roc_curve(Y_test, Y_test_pred)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12



plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)



print("\n")

print ("Area Under Curve: %.2f" %auc(fpr, tpr))

print("\n")