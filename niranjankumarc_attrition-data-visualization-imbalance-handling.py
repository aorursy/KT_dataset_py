import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#using the seaborn style for graphs

plt.style.use("seaborn")
## Read the dataset

employee_data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
employee_data.head()
##looking for any missing values



employee_data.isnull().sum()
employee_data.info()
## basic descriptive statistics

employee_data.describe()
#Mapping the attrition 1 - yes and 0 - no in the new column



employee_data["left"] = np.where(employee_data["Attrition"] == "Yes",1,0)
employee_data.head()
#supressing all the warnings

import warnings

warnings.filterwarnings('ignore')
def NumericalVariables_targetPlots(df,segment_by,target_var = "Attrition"):

    """A function for plotting the distribution of numerical variables and its effect on attrition"""

    

    fig, ax = plt.subplots(ncols= 2, figsize = (14,6))    



    #boxplot for comparison

    sns.boxplot(x = target_var, y = segment_by, data=df, ax=ax[0])

    ax[0].set_title("Comparision of " + segment_by + " vs " + target_var)

    

    #distribution plot

    ax[1].set_title("Distribution of "+segment_by)

    ax[1].set_ylabel("Frequency")

    sns.distplot(a = df[segment_by], ax=ax[1], kde=False)

    

    plt.show()
def CategoricalVariables_targetPlots(df, segment_by,invert_axis = False, target_var = "left"):

    

    """A function for Plotting the effect of variables(categorical data) on attrition """

    

    fig, ax = plt.subplots(ncols= 2, figsize = (14,6))

    

    #countplot for distribution along with target variable

    #invert axis variable helps to inter change the axis so that names of categories doesn't overlap

    if invert_axis == False:

        sns.countplot(x = segment_by, data=df,hue="Attrition",ax=ax[0])

    else:

        sns.countplot(y = segment_by, data=df,hue="Attrition",ax=ax[0])

        

    ax[0].set_title("Comparision of " + segment_by + " vs " + "Attrition")

    

    #plot the effect of variable on attrition

    if invert_axis == False:

        sns.barplot(x = segment_by, y = target_var ,data=df,ci=None)

    else:

        sns.barplot(y = segment_by, x = target_var ,data=df,ci=None)

        

    ax[1].set_title("Attrition rate by {}".format(segment_by))

    ax[1].set_ylabel("Average(Attrition)")

    plt.tight_layout()



    plt.show()
# we are checking the distribution of employee age and its related to attrition or not



NumericalVariables_targetPlots(employee_data,segment_by="Age")
#Analyzing the daily wage rate vs employee left the company or not



NumericalVariables_targetPlots(employee_data,"DailyRate")
NumericalVariables_targetPlots(employee_data,"MonthlyIncome")
NumericalVariables_targetPlots(employee_data,"HourlyRate")
NumericalVariables_targetPlots(employee_data,"PercentSalaryHike")
NumericalVariables_targetPlots(employee_data,"TotalWorkingYears")
sns.lmplot(x = "TotalWorkingYears", y = "PercentSalaryHike", data=employee_data,fit_reg=False,hue="Attrition",size=6,

           aspect=1.5)



plt.show()
NumericalVariables_targetPlots(employee_data,"DistanceFromHome")
#cross tabulation between attrition and JobInvolvement

pd.crosstab(employee_data.JobInvolvement,employee_data.Attrition)
#calculating the percentage of people having different job involvement rate

round(employee_data.JobInvolvement.value_counts()/employee_data.shape[0] * 100,2)
CategoricalVariables_targetPlots(employee_data,"JobInvolvement")
CategoricalVariables_targetPlots(employee_data,"JobSatisfaction")
#checking the number of categories under performance rating

employee_data.PerformanceRating.value_counts()
#calculate the percentage of performance rating per category in the whole dataset

round(employee_data.PerformanceRating.value_counts()/employee_data.shape[0] * 100,2)
CategoricalVariables_targetPlots(employee_data,"PerformanceRating")
#percentage of each relationship satisfaction category across the data

round(employee_data.RelationshipSatisfaction.value_counts()/employee_data.shape[0],2)
CategoricalVariables_targetPlots(employee_data,"RelationshipSatisfaction")
#percentage of worklife balance rating across the company data

round(employee_data.WorkLifeBalance.value_counts()/employee_data.shape[0],2)
CategoricalVariables_targetPlots(employee_data,"WorkLifeBalance")
CategoricalVariables_targetPlots(employee_data,"OverTime")
CategoricalVariables_targetPlots(employee_data,segment_by="BusinessTravel")
employee_data.Department.value_counts()
CategoricalVariables_targetPlots(employee_data,segment_by="Department")
employee_data.EducationField.value_counts()
CategoricalVariables_targetPlots(employee_data,"EducationField",invert_axis=True)
plt.figure(figsize=(10,8))

sns.barplot(y = "EducationField", x = "left", hue="Education", data=employee_data,ci=None)

plt.show()
CategoricalVariables_targetPlots(employee_data,"EnvironmentSatisfaction")
sns.boxplot(employee_data['Gender'], employee_data['MonthlyIncome'])

plt.title('MonthlyIncome vs Gender Box Plot', fontsize=20)      

plt.xlabel('MonthlyIncome', fontsize=16)

plt.ylabel('Gender', fontsize=16)

plt.show()
CategoricalVariables_targetPlots(employee_data,"Gender")
fig,ax = plt.subplots(2,3, figsize=(20,20))               # 'ax' has references to all the four axes

plt.suptitle("Comparision of various factors vs Gender", fontsize=20)

sns.barplot(employee_data['Gender'],employee_data['DistanceFromHome'],hue = employee_data['Attrition'], ax = ax[0,0],ci=None); 

sns.barplot(employee_data['Gender'],employee_data['YearsAtCompany'],hue = employee_data['Attrition'], ax = ax[0,1],ci=None); 

sns.barplot(employee_data['Gender'],employee_data['TotalWorkingYears'],hue = employee_data['Attrition'], ax = ax[0,2],ci=None); 

sns.barplot(employee_data['Gender'],employee_data['YearsInCurrentRole'],hue = employee_data['Attrition'], ax = ax[1,0],ci=None); 

sns.barplot(employee_data['Gender'],employee_data['YearsSinceLastPromotion'],hue = employee_data['Attrition'], ax = ax[1,1],ci=None); 

sns.barplot(employee_data['Gender'],employee_data['NumCompaniesWorked'],hue = employee_data['Attrition'], ax = ax[1,2],ci=None); 

plt.show()
CategoricalVariables_targetPlots(employee_data,"JobRole",invert_axis=True)
CategoricalVariables_targetPlots(employee_data,"MaritalStatus")
from sklearn.model_selection import train_test_split



#for fitting classification tree

from sklearn.tree import DecisionTreeClassifier



#to create a confusion matrix

from sklearn.metrics import confusion_matrix



#import whole class of metrics

from sklearn import metrics
employee_data.Attrition.value_counts().plot(kind = "bar")

plt.xlabel("Attrition")

plt.ylabel("Count")

plt.show()
employee_data["Attrition"].value_counts()
#copying the main employee data to another dataframe

employee_data_new = employee_data.copy()
#dropping the not significant variables

employee_data_new.drop(["EmployeeCount","EmployeeNumber","Gender","HourlyRate","Over18","StandardHours","left"], axis=1,inplace=True)
#data types of variables

dict(employee_data_new.dtypes)
#segregating the variables based on datatypes



numeric_variable_names  = [key for key in dict(employee_data_new.dtypes) if dict(employee_data_new.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]



categorical_variable_names = [key for key in dict(employee_data_new.dtypes) if dict(employee_data_new.dtypes)[key] in ["object"]]
categorical_variable_names
#store the numerical variables data in seperate dataset



employee_data_num = employee_data_new[numeric_variable_names]
#store the categorical variables data in seperate dataset



employee_data_cat = employee_data_new[categorical_variable_names]

#dropping the attrition 

employee_data_cat.drop(["Attrition"],axis=1,inplace=True)
#converting into dummy variables



employee_data_cat = pd.get_dummies(employee_data_cat)
#Merging the both numerical and categorical data



employee_data_final = pd.concat([employee_data_num, employee_data_cat,employee_data_new[["Attrition"]]],axis=1)
employee_data_final.head()
#final features

features =  list(employee_data_final.columns.difference(["Attrition"]))
features
#seperating the target and predictors



X = employee_data_final[features]

y = employee_data_final[["Attrition"]]
X.shape
# Function for creating model pipelines

from sklearn.pipeline import make_pipeline



#function for crossvalidate score

from sklearn.model_selection import cross_validate



#to find the best 

from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y,random_state = 100)
#Checks

#Proportion in training data

y_train.Attrition.value_counts()/len(y_train)
#Checks

#Proportion in training data

pd.DataFrame(y_train.Attrition.value_counts()/len(y_train)).plot(kind = "bar")

plt.show()
#Proportion of test data

y_test.Attrition.value_counts()/len(y_test)
#make a pipeline for decision tree model 



pipelines = {

    "clf": make_pipeline(DecisionTreeClassifier(max_depth=3,random_state=100))

}
scores = cross_validate(pipelines['clf'], X_train, y_train,return_train_score=True)
scores['test_score'].mean()
decisiontree_hyperparameters = {

    "decisiontreeclassifier__max_depth": np.arange(3,12),

    "decisiontreeclassifier__max_features": np.arange(3,10),

    "decisiontreeclassifier__min_samples_split": [2,3,4,5,6,7,8,9,10,11,12,13,14,15],

    "decisiontreeclassifier__min_samples_leaf" : np.arange(1,3)

}
pipelines['clf']
#Create a cross validation object from decision tree classifier and it's hyperparameters



clf_model = GridSearchCV(pipelines['clf'], decisiontree_hyperparameters, cv=5, n_jobs=-1)
#fit the model with train data

clf_model.fit(X_train, y_train)
#Display the best parameters for Decision Tree Model

clf_model.best_params_
#Display the best score for the fitted model

clf_model.best_score_
#In Pipeline we can use the string names to get the decisiontreeclassifer



clf_model.best_estimator_.named_steps['decisiontreeclassifier']
#saving into a variable to get graph



clf_best_model = clf_model.best_estimator_.named_steps['decisiontreeclassifier']
#Making a dataframe of actual and predicted data from test set



tree_test_pred = pd.concat([y_test.reset_index(drop = True),pd.DataFrame(clf_model.predict(X_test))],axis=1)

tree_test_pred.columns = ["actual","predicted"]



#setting the index to original index

tree_test_pred.index = y_test.index
tree_test_pred.head()
#keeping only positive condition (yes for attrition)



pred_probability = pd.DataFrame(p[1] for p in clf_model.predict_proba(X_test))

pred_probability.columns = ["predicted_prob"]

pred_probability.index = y_test.index
#merging the predicted data and its probability value



tree_test_pred = pd.concat([tree_test_pred,pred_probability],axis=1)
tree_test_pred.head()
#converting the labels Yes --> 1 and No --> 0 for further operations below



tree_test_pred["actual_left"] = np.where(tree_test_pred["actual"] == "Yes",1,0)

tree_test_pred["predicted_left"] = np.where(tree_test_pred["predicted"] == "Yes",1,0)
tree_test_pred.head()
#confusion matrix

metrics.confusion_matrix(tree_test_pred.actual,tree_test_pred.predicted,labels=["Yes","No"])
#confusion matrix visualization using seaborn heatmap



sns.heatmap(metrics.confusion_matrix(tree_test_pred.actual,tree_test_pred.predicted,

                                    labels=["Yes","No"]),cmap="Greens",annot=True,fmt=".2f",

           xticklabels = ["Left", "Not Left"] , yticklabels = ["Left", "Not Left"])



plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
#Area Under ROC Curve



auc_score_test = metrics.roc_auc_score(tree_test_pred.actual_left,tree_test_pred.predicted_left)

print("AUROC Score:",round(auc_score_test,4))
##Plotting the ROC Curve



fpr, tpr, thresholds = metrics.roc_curve(tree_test_pred.actual_left, tree_test_pred.predicted_prob,drop_intermediate=False)





plt.figure(figsize=(8, 6))

plt.plot( fpr, tpr, label='ROC curve (area = %0.4f)' % auc_score_test)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic cuve')

plt.legend(loc="lower right")

plt.show()

#calculating the recall score



print("Recall Score:",round(metrics.recall_score(tree_test_pred.actual_left,tree_test_pred.predicted_left) * 100,3))
#calculating the precision score



print("Precision Score:",round(metrics.precision_score(tree_test_pred.actual_left,tree_test_pred.predicted_left) * 100,3))
print(metrics.classification_report(tree_test_pred.actual_left,tree_test_pred.predicted_left))
# conda install pydot graphviz

#! pip install pydotplus
from sklearn.tree import export_graphviz
!pip install pydotplus
import pydotplus as pdot
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus as pdot
#write the dot data

dot_data = StringIO()
#export the decision tree along with the feature names into a dot file format



export_graphviz(clf_best_model,out_file=dot_data,filled=True,

                rounded=True,special_characters=True,feature_names = X_train.columns.values,class_names = ["No","Yes"])
#make a graph from dot file 

graph = pdot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
#export the tree diagram

graph.write_png("employee_attirtion.png")