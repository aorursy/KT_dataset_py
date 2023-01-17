
#1.0 Clear memory
%reset -f

# 1.1 Call data manipulation libraries
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

# 1.2 Feature creation Classes
from sklearn.preprocessing import PolynomialFeatures            # Interaction features
from sklearn.preprocessing import KBinsDiscretizer  


# 1.3 Data transformation classes
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Construct a transformer from an arbitrary callable.
from sklearn.preprocessing import FunctionTransformer

# 1.4 Fill missing values
from sklearn.impute import SimpleImputer


# 1.5  Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1.6 RandomForest modeling
from sklearn.ensemble import RandomForestClassifier 

# 1.7 Misc
import os, gc

#Graphing
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly.express as px
from matplotlib.colors import LogNorm
import seaborn as sns

# to display all outputs of one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#hide warning
import warnings
warnings.filterwarnings('ignore')
os.chdir('/kaggle/input')
os.listdir()
dfci=pd.read_csv('caravan-insurance-challenge/caravan-insurance-challenge.csv')
dfci.head()
print("No of Observatios:",dfci.shape[0])
print("No of Features:",dfci.shape[1])
dfci.columns
dfci.head()
dfci.columns[dfci.isnull().any()]
#no column has null value so need to fix null values

dfci.dtypes.value_counts()
#All columns are int 64 except one ,

str_features = dfci.select_dtypes(include='object').columns
str_features
#Lets check the unique values of ORIGIN 
dfci['ORIGIN'].value_counts()
#check summary
dfci.describe()
#Fetch Train Data
train_data= dfci[dfci['ORIGIN']=='train']
#drop ORIGIN col from train_data
train_data.drop(['ORIGIN'],axis=1,inplace=True)

test_data=dfci[dfci['ORIGIN']=='test']
#drop ORIGIN col from test_data
test_data.drop(['ORIGIN'],axis=1,inplace=True)
train_data['CARAVAN'].value_counts().plot(kind='bar', title='CARAVAN Classification Train Data', grid=True)
test_data['CARAVAN'].value_counts().plot(kind='bar', title='CARAVAN Classification Test Data', grid=True)
y = train_data.pop('CARAVAN')
train_data.head()
#check standard deviation.if std() is zero drop that columns
s= []
s = [col for col in train_data.columns if train_data[col].std() == 0]
s

dg=(train_data.nunique() < 5)
cat_columns = dg[dg == True].index.tolist()
num_columns = dg[dg == False].index.tolist()
print("No of cat cols",len(cat_columns))
print("No of num cols",len(num_columns))
import math
plt.figure(figsize=(15,18))
noofrows= math.ceil(len(num_columns)/3)
noofrows
#set false.Other wise error if  bandwidth =0 
sns.distributions._has_statsmodels=False

for i in range(len(num_columns)):
    plt.subplot(noofrows,3,i+1)
    out=sns.distplot(train_data[num_columns[i]])
    
plt.tight_layout()


ct=ColumnTransformer([
    ('rs',RobustScaler(),num_columns),
    ('ohe',OneHotEncoder(),cat_columns),
    ],
    remainder="passthrough"
    )
ct.fit_transform(train_data)
X=train_data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30)
rf= RandomForestClassifier(oob_score = True,bootstrap=True)
pipe =Pipeline(
    [
     ('ct',ct),
     ('rf',rf)
    ]
    )
rf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print("Accuracy is:",accuracy)
print("out-of-bag score computed by sklearn is an estimate of the classification accuracy we might expect to observe on new data")
print("Out-of-bag score estimation::",rf.oob_score_)


from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, predicted))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicted vs Actual')
cm
from sklearn.tree import export_graphviz
import graphviz

feature_list=train_data.columns
tree = rf.estimators_[5]
# Export  to a dot_data
dot_data = export_graphviz(tree, out_file=None,
                     feature_names=train_data.columns,
                     filled=True, rounded=True,
                     special_characters=True)
# Set graph and plot
graph = graphviz.Source(dot_data)
graph


importances = list(rf.feature_importances_)

dffeature_importance=pd.DataFrame({'Feature_Name':feature_list, 'Imporatance':importances})



# Get which feature has max importance
dffeature_importance[dffeature_importance['Imporatance'] == dffeature_importance['Imporatance'].max()]
# Get which feature has max importance
dffeature_importance[dffeature_importance['Imporatance'] == dffeature_importance['Imporatance'].min()]

plt.figure(figsize=(20,18))

# list of x locations for plotting
x_values = list(range(len(importances)))
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
# Make a bar chart
out=plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

