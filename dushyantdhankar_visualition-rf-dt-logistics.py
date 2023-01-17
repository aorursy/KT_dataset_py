import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import re

from scipy import stats
from functools import reduce

# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Some matplotlib options
%matplotlib inline
matplotlib.style.use("ggplot")

# General pandas options
pd.set_option('display.max_colwidth', -1)  # Show the entire column 
pd.options.display.max_columns = 100 
pd.options.display.max_rows = 10000 

# Seaborn options
sns.set_style("whitegrid")


# model to implement
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE


# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.shape)
df.head()
df.info()
df.isnull().sum()
df.Attrition = df.Attrition.astype("category")
df.Attrition = df.Attrition.cat.reorder_categories(['No','Yes'])
df.Attrition = df.Attrition.cat.codes
df.Attrition.dtype
df.BusinessTravel.value_counts() # I am considering them in order.
df.BusinessTravel = df.BusinessTravel.astype("category")
df.BusinessTravel = df.BusinessTravel.cat.reorder_categories(['Non-Travel','Travel_Rarely','Travel_Frequently'])
df.BusinessTravel = df.BusinessTravel.cat.codes
df.Department.value_counts() # This is nominal data here label encoding and just assigning nos. won't work so I create dummy variables.
df.EducationField.value_counts()  # This is nominal data here label encoding and just assigning nos. won't work so I create dummy variables.
df.Gender.value_counts()  # This is nominal data here label encoding and just assigning nos. won't work so I create dummy variables.
df.JobRole.value_counts() # This is nominal data here label encoding and just assigning nos. won't work so I create dummy variables.
df.MaritalStatus.value_counts() # This is nominal data here label encoding and just assigning nos. won't work so I create dummy variables.
df.Over18.value_counts() # constant so delete
df.OverTime.value_counts()
df.OverTime = df.OverTime.astype("category")
df.OverTime = df.OverTime.cat.reorder_categories(['No','Yes'])
df.OverTime = df.OverTime.cat.codes
# Plot for all variables distribution + Count
# Graph distribution
df.hist (bins=50, figsize=(20,15), color = 'deepskyblue')
plt.show()
#seprating numerical columns from dataframe
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','int8']

newdf = df.select_dtypes(include=numerics)
newdf.columns # numerical variable
# Create a figure space matrix consisting of 3 columns and 2 rows
fig, ax = plt.subplots(figsize=(20,15), ncols=3, nrows=5)
# The amount of space above titles
"""y_title_margin = .2
ax[0][0].set_title("Age",y = y_title_margin)
ax[0][1].set_title("BusinessTravel",y = y_title_margin)
ax[0][2].set_title("DailyRate",y = y_title_margin)
ax[1][0].set_title("DistanceFromHome",y = y_title_margin)
ax[1][1].set_title("EnvironmentSatisfaction",y = y_title_margin)
ax[1][2].set_title("JobSatisfaction",y = y_title_margin)
ax[2][0].set_title("MonthlyRate",y = y_title_margin)
ax[2][1].set_title("OverTime",y = y_title_margin)
ax[2][2].set_title("PerformanceRating",y = y_title_margin)
ax[3][0].set_title("RelationshipSatisfaction",y = y_title_margin)
ax[3][1].set_title("TotalWorkingYears",y = y_title_margin)
ax[3][2].set_title("WorkLifeBalance",y = y_title_margin)
ax[4][0].set_title("YearsAtCompany",y = y_title_margin)
ax[4][1].set_title("YearsSinceLastPromotion",y = y_title_margin)
ax[4][2].set_title("YearsWithCurrManage",y = y_title_margin)"""

sns.distplot(df.Age,kde=False,color="b", ax=ax[0][0])
sns.distplot(df.BusinessTravel,kde=False,color="b", ax=ax[0][1])
sns.distplot(df.DailyRate,kde=False,color="b", ax=ax[0][2])
sns.distplot(df.DistanceFromHome,kde=False,color="b", ax=ax[1][0])
sns.distplot(df.EnvironmentSatisfaction,kde=False,color="b", ax=ax[1][1])
sns.distplot(df.JobSatisfaction,kde=False,color="b", ax=ax[1][2])
sns.distplot(df.MonthlyRate,kde=False,color="b", ax=ax[2][0])
sns.distplot(df.OverTime,kde=False,color="b", ax=ax[2][1])
sns.distplot(df.PerformanceRating,kde=False,color="b", ax=ax[2][2])
sns.distplot(df.RelationshipSatisfaction,kde=False,color="b", ax=ax[3][0])
sns.distplot(df.TotalWorkingYears,kde=False,color="b", ax=ax[3][1])
sns.distplot(df.WorkLifeBalance,kde=False,color="b", ax=ax[3][2])
sns.distplot(df.YearsAtCompany,kde=False,color="b", ax=ax[4][0])
sns.distplot(df.YearsSinceLastPromotion,kde=False,color="b", ax=ax[4][1])
sns.distplot(df.YearsWithCurrManager,kde=False,color="b", ax=ax[4][2])


#separting categorical columns 
cat = ['object']

newdf1 = df.select_dtypes(include=cat)
newdf1.columns
# Create a figure space matrix consisting of 3 columns and 2 rows
fig, ax = plt.subplots(figsize=(20,15), ncols=3, nrows=2)
sns.countplot(x="Department",data=df,palette="Greens_d",ax= ax[0][0])
sns.countplot(x="EducationField",data=df,palette="Greens_d",ax= ax[0][1])
sns.countplot(x="Gender",data=df,palette="Greens_d",ax= ax[0][2])
sns.countplot(x="JobRole",data=df,palette="Greens_d",ax= ax[1][0])
sns.countplot(x="MaritalStatus",data=df,palette="Greens_d",ax= ax[1][1])
sns.countplot(x="Over18",data=df,palette="Greens_d",ax= ax[1][2]) # drop Over18
df.columns
# Create a figure space matrix consisting of 3 columns and 2 rows ## box plot for categorical vs numerical
sns.boxplot(x="BusinessTravel",y="Age",hue="Attrition",data=df) 
# we can conclude that most employers who are in range of 27-38 leave company.this can be due to career switch or want salary hike.
sns.countplot(x="Department",data=df,hue='Attrition')  # no specific relation
sns.boxplot(x="Attrition",y='DistanceFromHome',hue="Attrition",data=df) # no specific relation
sns.countplot(x="Education",data=df,hue='Attrition') # no specific relation
sns.countplot(x="EducationField",data=df,hue='Attrition') # no specific relation
sns.countplot(x="EnvironmentSatisfaction",data=df,hue='Attrition') # we can see that % of attrition for environment satisfaction = 1,2 will be more that of 3 and 4.
sns.countplot(x="Gender",data=df,hue='Attrition') #no specific relation
sns.countplot(x="JobSatisfaction",data=df,hue='Attrition') # we can see that % of attrition for job satisfaction = 1,2 will be more that of 3 and 4.
sns.boxplot(x="Attrition",y="YearsSinceLastPromotion",hue="Attrition",data=df)
True_Class = print(sum(df['Attrition']==1))
Total_length = print(len(df['Attrition']))
print((237/1470)*100) # percentage of class 1, checking class imbalance # it is not highly imbalanced but still we will use SMOTE for 1 model and do it without SMOTE for other model
#Checking class imbalance
sns.countplot(x ='Attrition',data = df)
pandas_profiling.ProfileReport(df)
df = df.drop(['EmployeeCount','MonthlyIncome','Over18','StandardHours'],axis =1)
df = pd.get_dummies(df)
df.head()
# Choose the dependent variable column (churn) and set it as target
target = df.Attrition
# Drop column churn and set everything else as features
features = df.drop("Attrition",axis=1)
# Import the train_test_split method
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

# Split data into train and test sets as well as for validation and testing
# Use that function to create the splits both for target and for features
# Set the test sample to be 25% of your observations
target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=1)
# Import the classification algorithm
from sklearn.tree import DecisionTreeClassifier

# Initialize it and call model by specifying the random_state parameter
model = DecisionTreeClassifier(random_state=42,class_weight='balanced')

# Apply a decision tree model to fit features to the target
model.fit(features_train,target_train)
# Do k-fold cross-validation
cv_results = cross_val_score(model, # Pipeline
                             features_train, # Feature matrix
                             target_train, # Target vector
                             cv=kf, # Cross-validation technique
                             scoring="accuracy", # Loss function
                             n_jobs=-1) # Use all CPU scores
# Calculate mean # cross validated score
CV_mean = cv_results.mean()
print(CV_mean*(100))
# overfited model
# Check the accuracy score of the prediction for the training set
print(model.score(features_train,target_train)*100)

# Check the accuracy score of the prediction for the test set
print(model.score(features_test,target_test)*100)
# generate max depth range
depth = [i for i in range (5,21,1)]
samples = [i for i in range(50,450,1)]
Parameters = dict(max_depth = depth, min_samples_leaf = samples)
from sklearn.model_selection import GridSearchCV
Param_search = GridSearchCV(model,Parameters)
Param_search.fit(features_train,target_train)
print(Param_search.best_params_)
model1 = DecisionTreeClassifier(random_state=42,class_weight='balanced',max_depth = 5, min_samples_leaf = 368)
model1.fit(features_train,target_train)
# Do k-fold cross-validation
cv_results1 = cross_val_score(model1, # Pipeline
                             features_train, # Feature matrix
                             target_train, # Target vector
                             cv=kf, # Cross-validation technique
                             scoring="accuracy", # Loss function
                             n_jobs=-1) # Use all CPU scores
# Calculate mean # cross validated score
CV_mean1 = cv_results1.mean()
print(CV_mean1*(100))
# No overfiting
# Check the accuracy score of the prediction for the training set
print(model1.score(features_train,target_train)*100)

# Check the accuracy score of the prediction for the test set
print(model1.score(features_test,target_test)*100)
from sklearn.metrics import confusion_matrix,classification_report
print (confusion_matrix(target_test, model1.predict(features_test)))
print (classification_report(target_test, model1.predict(features_test)))
important_features = model.feature_importances_
feature_list = list(features)
relative_importances = pd.DataFrame(index = feature_list, data = important_features, columns = ['Important'])
relative_importances.sort_values(by='Important', ascending = False)
selected_features = relative_importances[relative_importances.Important> 0.02]
selected_list = selected_features.index
feature_train_selected = features_train[selected_list]
feature_test_selected = features_test[selected_list]
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

seed = 0   # We set our random seed to zero for reproducibility

# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 800,
    'warm_start': True, 
    'max_features': 0.3,
    'max_depth': 9,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

# Instantiate rf
rf = RandomForestClassifier(**rf_params)
            
# Fit rf to the training set    
rf.fit(features_train, target_train)
# Do k-fold cross-validation
cv_results2 = cross_val_score(rf, # Pipeline
                             features_train, # Feature matrix
                             target_train, # Target vector
                             cv=kf, # Cross-validation technique
                             scoring="accuracy", # Loss function
                             n_jobs=-1) # Use all CPU scores
# Calculate mean # cross validated score
CV_mean2 = cv_results2.mean()
print(CV_mean2*(100))
# Slight overfiting
# Check the accuracy score of the prediction for the training set
print(rf.score(features_train,target_train)*100)

# Check the accuracy score of the prediction for the test set
print(rf.score(features_test,target_test)*100)
from sklearn.metrics import confusion_matrix,classification_report
print (confusion_matrix(target_test, rf.predict(features_test)))
print (classification_report(target_test, rf.predict(features_test)))
import matplotlib.pyplot as plt
#Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= features_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(features_train,target_train)
# Do k-fold cross-validation
cv_results2 = cross_val_score(lr, # Pipeline
                             features_train, # Feature matrix
                             target_train, # Target vector
                             cv=kf, # Cross-validation technique
                             scoring="accuracy", # Loss function
                             n_jobs=-1) # Use all CPU scores
# Calculate mean # cross validated score
CV_mean2 = cv_results2.mean()
print(CV_mean2*(100))
# No overfiting
# Check the accuracy score of the prediction for the training set
print(lr.score(features_train,target_train)*100)

# Check the accuracy score of the prediction for the test set
print(lr.score(features_test,target_test)*100)
from sklearn.metrics import confusion_matrix,classification_report
print (confusion_matrix(target_test, lr.predict(features_test)))
print (classification_report(target_test, lr.predict(features_test)))
print(lr.coef_)
print(lr.intercept_)
probability = lr.predict_proba(features_test)
from sklearn.metrics import roc_auc_score
roc_auc_score(target_test,probability[:,1])
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(target_test, lr.predict(features_test))
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();