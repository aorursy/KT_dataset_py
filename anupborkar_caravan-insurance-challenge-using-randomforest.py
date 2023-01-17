%reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn import preprocessing

#Data transformation classes
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import confusion_matrix

# Miscellaneous
import os, gc
import warnings

%matplotlib inline
plt.rcParams.update({'figure.max_open_warning': 0}) #just to suppress warning for max plots of 20
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

# Use white grid plot background from seaborn
sns.set(font_scale=0.5, style="ticks")
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6})
# Display output not only of last command but all commands in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Set pandas options to display results
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
#Browse the Dataset Path
os.chdir('/kaggle/input/caravan-insurance-challenge/')
os.listdir()
# Load dataset
cc = pd.read_csv('caravan-insurance-challenge.csv')
#Let's try to analyze the dataset based on what is availiable with us
cc.info()
cc.head()
cc.describe()
def missing_columns_data(data):
    miss      = data.isnull().sum()
    miss_pct  = 100 * data.isnull().sum()/len(data)
    
    miss_pct      = pd.concat([miss,miss_pct], axis=1)
    missing_cols = miss_pct.rename(columns = {0:'Missings', 1: 'Missing pct'})
    missing_cols = missing_cols[missing_cols.iloc[:,1]!=0].sort_values('Missing pct', ascending = False).round(1)
    
    return missing_cols  

missing = missing_columns_data(cc)
missing
null_counts = cc.isnull().sum()/len(cc);
plt.figure(figsize=(15,4));
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical');
plt.ylabel('Fraction of Rows with missing data');
plt.bar(np.arange(len(null_counts)),null_counts);
cc_no_origin_train = cc[cc['ORIGIN']=='train'].drop(['ORIGIN'], axis=1)
cc_no_origin_train.head()
cc_no_origin_test = cc[cc['ORIGIN']=='test'].drop(['ORIGIN'], axis=1)
cc_no_origin_test.head()
fig = plt.figure(figsize=(10,10));

# Plot Telling the total count of different values in CARAVAN
plt.subplot(3,1,1);
cc_no_origin_train['CARAVAN'].value_counts().plot(kind='bar', title='Classifying CARAVAN', color='#10bbd4', grid=True);

# Plot Telling the total count of different values in customer subtype
plt.subplot(3,1,2);
cc_no_origin_train['MOSTYPE'].value_counts().plot(kind='bar', align='center', title='Classifying Customer Subtypes', color='#10bbd4', grid=True);
categorysubtype_caravan = pd.crosstab(cc_no_origin_train['MOSTYPE'], cc_no_origin_train['CARAVAN']);
categorysubtype_caravan_percentage = categorysubtype_caravan.div(categorysubtype_caravan.sum(1).astype(float), axis=0);
categorysubtype_caravan_percentage.plot(figsize= (15,10), kind='barh', stacked=True, color=['#10bbd4', 'Crimson'], title='Category Subtype vs Caravan', grid=True);
plt.xlabel('Category Subtype');
plt.ylabel('CARAVAN or not');
cc_no_origin_train['MGEMLEEF'].hist(figsize=(5,3), fc='#10bbd4', grid=True);
plt.xlabel('age');
plt.ylabel('count');
age_caravan = pd.crosstab(cc_no_origin_train['MGEMLEEF'], cc_no_origin_train['CARAVAN']);
age_caravan_percentage = age_caravan.div(age_caravan.sum(1).astype(float),axis=0);
age_caravan_percentage.plot(figsize=(5,3), kind='barh', stacked=True, color=['#10bbd4', 'Crimson'], title='Dependency of Caravan on age groups', grid=True);
plt.xlabel('Age Groups');
plt.ylabel('Caravan');
cc_no_origin_train['MOSHOOFD'].value_counts().plot(kind='barh', color='#10bbd4', grid=True);
plt.xlabel('Customer Types');
plt.ylabel('count');
cust_type_caravan = pd.crosstab(cc_no_origin_train['MOSHOOFD'], cc_no_origin_train['CARAVAN']);
cust_type_caravan_percentage = cust_type_caravan.div(cust_type_caravan.sum(1).astype(float), axis=0);
cust_type_caravan_percentage.plot(kind='barh', stacked=True, color = ['#10bbd4', 'Crimson']);
plt.xlabel('Customer types');
plt.ylabel('Caravan');
#Remove unnecessary columns
#check standard deviation.if std() is zero drop those columns
#Removal list initialize
rem = []

#Add constant columns as they don't help in prediction process
for c in cc_no_origin_train.columns:
    if cc_no_origin_train[c].std() == 0: #standard deviation is zero
        rem.append(c)

#drop the columns        
cc_no_origin_train.drop(rem,axis=1,inplace=True)

print(rem)

#Following columns are dropped

y = cc_no_origin_train.pop('CARAVAN')
cc_no_origin_train.head()
features=(cc_no_origin_train.nunique() < 5)

cat_attributes = features[features == True].index.tolist()
num_attributes = features[features == False].index.tolist()

print("No. of Categorical Columns",len(cat_attributes))
print("No. of Numerical Columns",len(num_attributes))
plt.figure(figsize=(30,40));

for i in range(len(num_attributes)):
    plt.subplot(25,4,i+1);
    sns.boxplot(cc_no_origin_train[num_attributes[i]],  showmeans=True, palette=[sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["medium green"]]);
    
plt.tight_layout();
ct=ColumnTransformer([
    ('num_attributes',RobustScaler(),num_attributes),
    ('cat_attributes',OneHotEncoder(),cat_attributes),
    ],
    remainder="passthrough"
    )
ct.fit_transform(cc_no_origin_train)
X=cc_no_origin_train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25)
clf= RandomForestClassifier(oob_score = True,bootstrap=True,n_estimators=100)
pipe =Pipeline(
    [
     ('ct',ct),
     ('rf',clf)
    ]
    );
#Fit/train the object on training data
forest_fit=clf.fit(X_train, y_train);
forest_fit;
#Make prediction
predict = clf.predict(X_test)
accuracy = accuracy_score(y_test, predict)
print('Accuracy is :'+"{:.2f}".format(accuracy*100),"%")

#mean absolute error
errors = abs(predict - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Display confusion matrix
def display_conf_matrix(y_test, y_pred):
    # Confusion matrix: row -> actual, column -> predicted
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))
display_conf_matrix(y_test, predict)

confusion_matrix = pd.DataFrame(confusion_matrix(y_test, predict))
confusion_matrix
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.xlabel('Predicted');
plt.ylabel('Actual');
plt.title('Predicted vs Actual');
feature_list=X.columns
importances = list(clf.feature_importances_)
feature_importance=pd.DataFrame({'Feature_Name':feature_list, 'Importance':importances})
feature_importance.sort_values(by ='Importance',ascending=False)
plt.figure(figsize=(8,6))
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', stacked=True, color = ['#10bbd4', 'Crimson']);
plt.xlabel('Importance Level');
plt.ylabel('Feature');