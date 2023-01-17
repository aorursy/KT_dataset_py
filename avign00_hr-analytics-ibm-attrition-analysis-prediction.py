import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



# To avoid Warning message inbetween ...

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/Attrition1.csv")
#Quick Analysis on Dataset : DataTypes, Rows and Columns ,Null values, Unique values ...

def quick_analysis(df):

    print("Data Types:")

    print(df.dtypes)

    print("\nRows and Columns:")

    print(df.shape)

    print("\nColumn names:")

    print(df.columns)

    print("\nNull Values")

    print(df.apply(lambda x: sum(x.isnull()) / len(df)))

    print("\nUnique values")

    print(df.nunique())



quick_analysis(df)
#Dropping the unwanted columns: Those having only one unique value.

df=df.drop(["EmployeeCount","Over18","StandardHours","EmployeeNumber"],axis=1)
#Visual Exploratory Data Analysis (EDA) And Your First Model

#EDA on Feature Variables

print(list(set(df.dtypes.tolist())))

df_object = df.select_dtypes(include=["object"]).copy()

df_int = df.select_dtypes(include=['int64']).copy()



categorical = df_object.columns

numerical = df_int.columns



print("Datashape of Object Dataframe:",df_object.shape)

print("Datashape of Interger Dataframe:",df_int.shape)
# Univariate Analysis

# EDA with Categorical Variables



fig,ax = plt.subplots(3,2, figsize=(20,20))

for variable,subplot in zip(categorical,ax.flatten()):

    sns.countplot(df[variable],ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(20)
# EDA with Numerical Variables

df[numerical].hist(bins=50,figsize=(16,20),layout=(8,3))
# Bivariate analysis - Categorical (Target variable) vs Numerical ( Feature Variables)

fig , ax =plt.subplots(4,6,figsize=(30,30))

for var,subplot in zip(numerical,ax.flatten()):

    sns.boxplot(x="Attrition",y=var,data=df, ax=subplot)
#fig , ax =plt.subplots(3,8,figsize=(30,30))

for var,subplot in zip(numerical,ax.flatten()):

    facet = sns.FacetGrid(df,hue="Attrition",aspect=4)

    facet.map(sns.kdeplot,var,shade= True)

    facet.set(xlim=(0,df[var].max()))

    facet.add_legend()

    plt.show()

    #sns.boxplot(x="Attrition",y=var,data=df, ax=subplot) 
g = sns.FacetGrid(df, col="Attrition",row="Gender",aspect=1,height=4,hue="Department") 

g.map(sns.distplot, "YearsAtCompany")

g.set(xlim=(0,df["YearsAtCompany"].max()))

#plt.xlim(0,6)

g.add_legend()
g = sns.FacetGrid(df, col="Attrition",row="Department",hue="Gender",aspect=1,height=5) 

g.map(sns.distplot, "MonthlyIncome")

g.add_legend()

g.set(xlim=(0,df["MonthlyIncome"].max()))
g = sns.FacetGrid(df, col="Attrition",row="JobRole",hue="Gender",aspect=1,height=5) 

g.map(sns.distplot, "MonthlyIncome")

g.add_legend()

plt.ylim(0,0.0010)

g.set(xlim=(0,df["MonthlyIncome"].max()))
figure = plt.figure(figsize=(20,8))

plt.hist([df[df['Attrition'] == 1]['MonthlyIncome'], df[df['Attrition'] == 0]['MonthlyIncome']], 

         stacked=True,

         bins = 80, label = ['Attrition','Not_Attrition'])

plt.xlabel('MonthlyIncome')

plt.ylabel('Number of Employees')

plt.legend();
df.groupby(["JobRole","Gender"]).Attrition.value_counts()
labels = df['JobRole'].astype('category').cat.categories.tolist()

counts = df['JobRole'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

print(sizes)

fig1, ax1 = plt.subplots(figsize=(6,6))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot

ax1.axis('equal')

plt.show()
# Data Manuplation in the Dataset

# Data types changes for :Features Variables



# Variables in df needs to be changed to the object type from int64.

df["Education"] = df["Education"].astype(object)

df["EnvironmentSatisfaction"] = df["EnvironmentSatisfaction"].astype(object)

df["JobInvolvement"] = df["JobInvolvement"].astype(object)

df["JobLevel"] = df["JobLevel"].astype(object)

df["JobSatisfaction"] = df["JobSatisfaction"].astype(object)

df["PerformanceRating"] = df["PerformanceRating"].astype(object)

df["RelationshipSatisfaction"] = df["RelationshipSatisfaction"].astype(object)

df["StockOptionLevel"] = df["StockOptionLevel"].astype(object)

df["TrainingTimesLastYear"] = df["TrainingTimesLastYear"].astype(object)

df["WorkLifeBalance"] = df["WorkLifeBalance"].astype(object)
df["TotalWorkingYears"] =np.sqrt(df["TotalWorkingYears"])

df["YearsAtCompany"] =np.sqrt(df["YearsAtCompany"])



# Taking log tranformation,so the data distribution look normal Distribution ...

df["MonthlyIncome"] = np.log(df["MonthlyIncome"])
df.loc[ df['YearsInCurrentRole'] <= 2,'YearsInCurrentRole'] = 0

df.loc[ (df['YearsInCurrentRole'] >=3) & (df['YearsInCurrentRole'] <= 6) ,'YearsInCurrentRole'] = 1

df.loc[ (df['YearsInCurrentRole'] >=7) & (df['YearsInCurrentRole'] <= 10), 'YearsInCurrentRole'] = 2

df.loc[ (df['YearsInCurrentRole'] >=11) & (df['YearsInCurrentRole'] <= 18), 'YearsInCurrentRole'] = 3
df.loc[ df['YearsSinceLastPromotion'] <= 1,'YearsSinceLastPromotion'] = 0

df.loc[ (df['YearsSinceLastPromotion'] >=2) & (df['YearsSinceLastPromotion'] <= 4) ,'YearsSinceLastPromotion'] = 1

df.loc[ (df['YearsSinceLastPromotion'] >=5) & (df['YearsSinceLastPromotion'] <= 7), 'YearsSinceLastPromotion'] = 2

df.loc[ (df['YearsSinceLastPromotion'] >=8) & (df['YearsSinceLastPromotion'] <= 15), 'YearsSinceLastPromotion'] = 3
df.loc[ df['YearsWithCurrManager'] < 1,'YearsWithCurrManager'] = 0

df.loc[ (df['YearsWithCurrManager'] >=2) & (df['YearsWithCurrManager'] <= 3) ,'YearsWithCurrManager'] = 1

df.loc[ (df['YearsWithCurrManager'] >=4) & (df['YearsWithCurrManager'] <= 6), 'YearsWithCurrManager'] = 2

df.loc[ (df['YearsWithCurrManager'] >=7) & (df['YearsWithCurrManager'] <= 9), 'YearsWithCurrManager'] = 3

df.loc[ (df['YearsWithCurrManager'] >=10) & (df['YearsWithCurrManager'] <= 17), 'YearsWithCurrManager'] = 4
df.loc[ df['DistanceFromHome'] <= 2,'DistanceFromHome'] = 0

df.loc[ (df['DistanceFromHome'] >=3) & (df['DistanceFromHome'] <= 5) ,'DistanceFromHome'] = 1

df.loc[ (df['DistanceFromHome'] >=6) & (df['DistanceFromHome'] <= 8), 'DistanceFromHome'] = 2

df.loc[ (df['DistanceFromHome'] >=9) & (df['DistanceFromHome'] <= 12), 'DistanceFromHome'] = 3

df.loc[ (df['DistanceFromHome'] >=13) & (df['DistanceFromHome'] <= 20), 'DistanceFromHome'] = 4

df.loc[ (df['DistanceFromHome'] >=21) & (df['DistanceFromHome'] <= 29), 'DistanceFromHome'] = 5
df.loc[ df['NumCompaniesWorked'] <= 1,'NumCompaniesWorked'] = 0

df.loc[ (df['NumCompaniesWorked'] >=2) & (df['NumCompaniesWorked'] <= 4) ,'NumCompaniesWorked'] = 1

df.loc[ (df['NumCompaniesWorked'] >=5) & (df['NumCompaniesWorked'] <= 6), 'NumCompaniesWorked'] = 2

df.loc[ (df['NumCompaniesWorked'] >=7) & (df['NumCompaniesWorked'] <= 9), 'NumCompaniesWorked'] = 3
df.loc[ df['PercentSalaryHike'] <= 12,'PercentSalaryHike'] = 0

df.loc[ (df['PercentSalaryHike'] >=13) & (df['PercentSalaryHike'] <= 14) ,'PercentSalaryHike'] = 1

df.loc[ (df['PercentSalaryHike'] >=15) & (df['PercentSalaryHike'] <= 18), 'PercentSalaryHike'] = 2

df.loc[ (df['PercentSalaryHike'] >=19) & (df['PercentSalaryHike'] <= 21), 'PercentSalaryHike'] = 3

df.loc[ (df['PercentSalaryHike'] >=22) & (df['PercentSalaryHike'] <= 25), 'PercentSalaryHike'] = 4
df["YearsInCurrentRole"] = df["YearsInCurrentRole"].astype(object)

df["YearsSinceLastPromotion"] = df["YearsSinceLastPromotion"].astype(object)

df["YearsWithCurrManager"] = df["YearsWithCurrManager"].astype(object)

df["DistanceFromHome"] = df["DistanceFromHome"].astype(object)

df["NumCompaniesWorked"] = df["NumCompaniesWorked"].astype(object)

df["PercentSalaryHike"] = df["PercentSalaryHike"].astype(object)
#EDA on Feature Variables

print(list(set(df.dtypes.tolist())))

df_object = df.select_dtypes(include=["object"]).copy()

df_int = df.select_dtypes(include=['int64','float64']).copy()



categorical = df_object.columns

numerical = df_int.columns



print("Datashape of Object Dataframe:",df_object.shape)

print("Datashape of Interger Dataframe:",df_int.shape)
# Features encoding and scaling

# Preprocessing packages 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler



# Model selection for Train and Test split the dataset

from sklearn.model_selection import train_test_split
def feature_imp_Dataset(df):

    #Target Columns

    target_col = ["Attrition"]



    #Categorical Columns

    cat_cols = df.nunique()[df.nunique() < 10].keys().tolist()

    cat_cols = [x for x in cat_cols if x not in target_col]



    #numerical columns

    num_cols = [x for x in df.columns if x not in cat_cols + target_col]



    #Binary columns with 2 values

    bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()



    #Columns more than 2 values

    multi_cols = [i for i in cat_cols if i not in bin_cols] 

    

    df_feature_imp = df.copy()

    

    #Label encoding Binary columns

    le = LabelEncoder()

    for i in cat_cols:

        df_feature_imp[i] = le.fit_transform(df_feature_imp[i])



    #Dulpicating columns for Multiple value columns

    #df = pd.get_dummies(data=df,columns= multi_cols,drop_first=True)

    df_feature_imp= pd.get_dummies(data=df_feature_imp,columns= multi_cols)

    

    return df_feature_imp
# Feature selection



# Inorder to avoid the Dummy trap, we are removing the less Importance Dummy Varaible columns...

# For this we are using the Random Forest to select the Importance Feature...

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')



# Loading the dataset:

df_feature_imp = feature_imp_Dataset(df)



# Def X and Y for Unscaled Dataset

target_col = ["Attrition"]

y = pd.DataFrame(df_feature_imp,columns=target_col)

#y = df_unscaled["Attrition"]

X = df_feature_imp.drop('Attrition',1)



# Fit the Model with the X and y ...

clf = clf.fit(X, y)

features = pd.DataFrame()

features['feature'] = X.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))
# As per the Importance Features Techinque With the help of Random Forest Classifier we could see the below Dummy Variables

# has less Importance compared to other Dummy variables, so we are removing those variables.



to_drop_dummy_variable_trap= ['BusinessTravel_0','Department_0','DistanceFromHome_2','Education_4','EducationField_0','EnvironmentSatisfaction_1',

 'JobInvolvement_3','JobLevel_3','JobRole_4','JobSatisfaction_1','MaritalStatus_0','NumCompaniesWorked_1','PercentSalaryHike_3',

 'RelationshipSatisfaction_1','StockOptionLevel_2','TrainingTimesLastYear_5','WorkLifeBalance_3','YearsInCurrentRole_3','YearsInCurrentRole_3',

 'YearsWithCurrManager_4']

#df = df.drop(columns=to_drop_dummy_variable_trap)
# Correlation Matrix - Orginal Dataset ...



#correlation for Orginal Dataset

correlation = df.corr()



#tick labels

#matrix_cols = correlation.columns.tolist()

#convert to array

#corr_array  = np.array(correlation)



# Viewing the Correlation with respect to Attrition ...

corr_list = correlation['Attrition'].sort_values(axis=0,ascending=False)#.iloc[1:]

#corr_list
from sklearn.utils.class_weight import compute_class_weight

def _compute_class_weight_dictionary(y):

    # helper for returning a dictionary instead of an array

    classes = np.unique(y)

    class_weight = compute_class_weight("balanced", classes, y)

    class_weight_dict = dict(zip(classes, class_weight))

    return class_weight_dict   
y=df["Attrition"]

print("Class Weight for the Attrition Attribute:")

_compute_class_weight_dictionary(y)
def unscaled_data(df):

    #global to_drop_dummy_variable_trap

    #Target Columns

    target_col = ["Attrition"]



    #Categorical Columns

    cat_cols = df.nunique()[df.nunique() < 10].keys().tolist()

    cat_cols = [x for x in cat_cols if x not in target_col]



    #numerical columns

    num_cols = [x for x in df.columns if x not in cat_cols + target_col]



    #Binary columns with 2 values

    bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()



    #Columns more than 2 values

    multi_cols = [i for i in cat_cols if i not in bin_cols]

    

    df_unscaled = df.copy()



    #Label encoding Binary columns

    le = LabelEncoder()

    for i in cat_cols:

        df_unscaled[i] = le.fit_transform(df_unscaled[i])



    #Dulpicating columns for Multiple value columns

    #df = pd.get_dummies(data=df,columns= multi_cols,drop_first=True)

    df_unscaled= pd.get_dummies(data=df_unscaled,columns= multi_cols)



    #Dropping original values merging scaled values for numerical columns

    #f_unscaled = df.copy()

    

    ###############################################################################

    # Remove collinear features for Unscaled Dataset ...

    # Threshold to remove correlated Variables

    threshold = 0.7

    #0.8 - Initailly i have taken as



    # Absolute value of corelation matrix

    corr_matrix = df_unscaled.corr().abs()

    corr_matrix.head()



    # Upper triangle of correlations

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    upper.head()



    # Select columns with correlations above threshold

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print('There are %d columns to remove:' %(len(to_drop)))

    print("Threshold more than %s \n" %threshold ,to_drop)

    df_unscaled = df_unscaled.drop(columns=to_drop)

    

    #Columns thats is to avoid dummy variable trap: to_drop_dummy_variable_trap

    

    to_drop_dummy_variable_trap_un = [i for i in to_drop_dummy_variable_trap if i not in to_drop]

    df_unscaled = df_unscaled.drop(columns=to_drop_dummy_variable_trap_un)

    print("\nRemoving variables to avoid Dummy variable trap:\n")

    print(to_drop_dummy_variable_trap_un)

    print("\n")

    

    ###############################################################################

    #y=df_unscaledl["Attrition"]

    #print("Class Weight for the Attrition Attribute:\n")

    #_compute_class_weight_dictionary(y)

    #print("\n")

    

    ###############################################################################

    # Prepare dataset

    # Define (X, y)



    # Def X and Y for Unscaled Dataset

    y = pd.DataFrame(df_unscaled,columns=target_col)

    #y = df_unscaled["Attrition"]

    X = df_unscaled.drop('Attrition',1)

    

    # execute this step if you need the Orginal "Train test split :X_train, X_test, y_train, y_test" 

    random_state =0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = random_state)

    # Defining Cols variables to store the Column names of X Unscaled dataframe.

    cols = X_train.columns

    

    return X_train,X_test,y_train,y_test,cols,X,y
def scaled_data(df):

    #Target Columns

    target_col = ["Attrition"]



    #Categorical Columns

    cat_cols = df.nunique()[df.nunique() < 10].keys().tolist()

    cat_cols = [x for x in cat_cols if x not in target_col]



    #numerical columns

    num_cols = [x for x in df.columns if x not in cat_cols + target_col]



    #Binary columns with 2 values

    bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()



    #Columns more than 2 values

    multi_cols = [i for i in cat_cols if i not in bin_cols]

    

    df_scaled = df.copy()

    

    #Label encoding Binary columns

    le = LabelEncoder()

    for i in cat_cols:

        df_scaled[i] = le.fit_transform(df_scaled[i])



    #Dulpicating columns for Multiple value columns

    #df = pd.get_dummies(data=df,columns= multi_cols,drop_first=True)

    df_scaled = pd.get_dummies(data=df_scaled,columns= multi_cols)



    #Scaling the Numerical columns

    std = StandardScaler()

    scaled = std.fit_transform(df_scaled[num_cols])

    scaled = pd.DataFrame(scaled,columns=num_cols)



    #Dropping original values merging scaled values for numerical columns

    df_scaled = df_scaled.drop(columns= num_cols,axis=1)

    df_scaled = df_scaled.merge(scaled,left_index=True,right_index=True,how="left")

    

    ###############################################################################

    # Threshold to remove correlated Variables

    threshold = 0.7

    #0.8 - Initailly i have taken as



    # Absolute value of corelation matrix

    corr_matrix = df_scaled.corr().abs()

    corr_matrix.head()



    # Upper triangle of correlations

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    upper.head()



    # Select columns with correlations above threshold

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print('There are %d columns to remove:' %(len(to_drop)))

    print("Threshold more than %s \n" %threshold ,to_drop)

    df_scaled = df_scaled.drop(columns=to_drop)

    print(to_drop)

    

    #Columns thats is to avoid dummy variable trap: to_drop_dummy_variable_trap

    

    to_drop_dummy_variable_trap_un = [i for i in to_drop_dummy_variable_trap if i not in to_drop]

    df_scaled = df_scaled.drop(columns=to_drop_dummy_variable_trap_un)

    print("\nRemoving variables to avoid Dummy variable trap:\n")

    print(to_drop_dummy_variable_trap_un)

    print("\n")

    

    ###############################################################################

    # Def X and Y for Scaled Dataset

    y_scale = pd.DataFrame(df_scaled,columns=target_col)

    #y = df_unscaled["Attrition"]

    X_scale = df_scaled.drop('Attrition',1)

    

    ###############################################################################   

    # execute this step if you need the Scaled "Train test split :X_train, X_test, y_train, y_test" 

    random_state = 0

    X_train, X_test, y_train, y_test = train_test_split(X_scale,y_scale, test_size = 0.30, random_state = random_state)

    # Defining Cols variables to store the Column names of X scaled dataframe.

    cols = X_train.columns

    

    return X_train, X_test, y_train, y_test,cols,X_scale,y_scale
def cross_validate_(model,X,y,num_validations=5):

    accuracy_train = cross_val_score(model,X,y,scoring="accuracy",cv=num_validations)

    precision_train = cross_val_score(model,X,y,scoring="precision",cv=num_validations)

    recall_train = cross_val_score(model,X,y,scoring="recall",cv=num_validations)

    f1_train = cross_val_score(model,X,y,scoring="f1_weighted",cv=num_validations)                                  

    

    print("Cross Validation of : {}".format(model.__class__.__name__))

    print('*********************')

    print(" Model :",model)

    #print("Transforming {}".format(transformer.__class__.__name__))

    print("Accuracy: " , round(100*accuracy_train.mean(), 2))

    print("Precision: ",  round(100*precision_train.mean(), 2))

    print("Recall: ",  round(100*recall_train.mean(), 2))

    print("F1 Score: ",  round(100*f1_train.mean(), 2))

    print('**************************************************************************\n')

# Modelling

# Baseline Model



# Support functions

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate

from scipy.stats import uniform



# Fit models

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from xgboost import XGBClassifier

import statsmodels.api as sm





# Scoring functions

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.metrics import roc_auc_score,roc_curve,scorer

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import mean_squared_error

# Basics function for Prediction

# Function which can predict the Accuracy and Area under the curve(AUC) of test data from Train dataset in a single shot ...



def Attrition_prediction(algorthim,train_x,test_x,train_y,test_y,cols):

    

    #model

    algorthim.fit(train_x,train_y)

    predictions = algorthim.predict(test_x)

    #probabilities = algorthim.predict_proba(test_x)

     

    #roc_auc_score

    model_roc_auc = roc_auc_score(test_y,predictions)

    

    #RMSE values of Train Model ...

    train_y_pred = algorthim.predict(train_x)

    confusion_matrix(train_y,train_y_pred)



    final_mse = mean_squared_error(train_y,train_y_pred) 

    train_final_rmse = np.sqrt(final_mse)

    

    #RMSE values of Test Model ...

    final_mse = mean_squared_error(test_y,predictions) 

    test_final_rmse = np.sqrt(final_mse)

    

    #Confusion Matrix for Train Model ...

    confuse_train = confusion_matrix(train_y, train_y_pred)

    

    #Confusion Matrix for Train Model ...

    confuse_test = confusion_matrix(test_y, predictions)

    

    print("Algorthims parameters used :\n\n",algorthim)

    print("\n Classification Report :\n", classification_report(test_y,predictions))

    print("Accuracy Score of Train :", accuracy_score(train_y,train_y_pred),"\n")

    print("Accuracy Score of Test :", accuracy_score(test_y,predictions),"\n")

    print("Area under the curve :",model_roc_auc,"\n")

    print("RMSE of the Train Model :",train_final_rmse,"\n")

    print("Confusion Matrix of the Train Model :\n",confuse_train)

    print("RMSE of the Test Model :",test_final_rmse,"\n")

    print("Confusion Matrix of the Test Model :\n",confuse_test)

logreg = LogisticRegression()

logreg_cv = LogisticRegressionCV()

random_f = RandomForestClassifier()

knn = KNeighborsClassifier()

svc = SVC(kernel='linear')

xgb = XGBClassifier()



classifiers = [logreg, logreg_cv, random_f, knn, svc, xgb]
X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

print('**************************************************************************\n')

#for model in classifiers:

    #cross_validate_(model,X,y,num_validations=7)
X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

print('**************************************************************************\n')

for model in classifiers:

    cross_validate_(model,X,y,num_validations=7)
logreg = LogisticRegression()

logreg_cv = LogisticRegressionCV()

random_f = RandomForestClassifier()

knn = KNeighborsClassifier()

svm = SVC()

xgb = XGBClassifier()

from sklearn.ensemble import BaggingClassifier

classifiers = [logreg, logreg_cv, random_f, knn, svm, xgb]
X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

print('**************************************************************************\n')

for model in classifiers:

    print("Bagging Techinque on :{}".format(model.__class__.__name__))

    print("**********************")

    print("Model used:", model)

    bag_model = BaggingClassifier(base_estimator=model,n_estimators=100,bootstrap=True)

    bag_model = bag_model.fit(X_train,y_train)

    ytest_pred = bag_model.predict(X_test)

    print("Bagging Accuarcy :", bag_model.score(X_test,y_test))

    print("Confusin Matrix :\n ", confusion_matrix(y_test,ytest_pred))

    print("***************************************************************************\n")

    
X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

print('**************************************************************************\n')

for model in classifiers:

    print("Bagging Techinque on :{}".format(model.__class__.__name__))

    print("**********************")

    print("Model used:", model)

    bag_model = BaggingClassifier(base_estimator=model,n_estimators=100,bootstrap=True)

    bag_model = bag_model.fit(X_train,y_train)

    ytest_pred = bag_model.predict(X_test)

    print("Bagging Accuarcy :", bag_model.score(X_test,y_test))

    print("Confusin Matrix :\n ", confusion_matrix(y_test,ytest_pred))

    print("***************************************************************************\n")

    
from tpot import TPOTClassifier

from tpot import TPOTRegressor
tpot = TPOTClassifier(generations=5,verbosity=2)

X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

tpot.fit(X_train.values,y_train.values)
tpot.score(X_test.values,y_test.values)
#tpot.export('tpot_Attrition_modeling_pipeline.py')
tpot = TPOTClassifier(generations=5,verbosity=2)

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

tpot.fit(X_train.values,y_train.values)
tpot.score(X_test.values,y_test.values)
#tpot.export('tpot_Attrition_modeling_scaled_pipeline.py')
#Logistic Regression...



# Using Unscaled Data set for Logistic Regression...



classifier = LogisticRegression()

X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
#Logistic Regression...



# Using Scaled Data set for Logistic Regression...



classifier = LogisticRegression()

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Grid Search on Logistic Regression ...



# Fit the parameters for logistic regression ...



param_grid = {'C':np.logspace(-3,3,8),'penalty':["l1","l2"],'max_iter':[100],'intercept_scaling':[1]}

log_param_grid = GridSearchCV(LogisticRegression(),param_grid=param_grid,cv=10,refit=True,verbose=1)



X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

#Applying Grid Search on Orginal Dataset ...

log_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=log_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)
# Grid Search on Logistic Regression ...



# Fit the parameters for logistic regression ...



param_grid = {'C':np.logspace(-3,3,8),'penalty':["l1","l2"],'max_iter':[100],'intercept_scaling':[0.97,0.98,1]}

log_param_grid = GridSearchCV(LogisticRegression(),param_grid=param_grid,cv=7,refit=True,verbose=1)



X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

#Applying Grid Search on Orginal Dataset ...

log_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=log_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)
# KNN ...

# Using Unscaled Data set for KNN...

classifier = KNeighborsClassifier()

X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Using Scaled Data set for KNN...

classifier = KNeighborsClassifier()

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Grid Search on KNN Scaled Dataset ...



# Fit the parameters for KNN ...

param_grid = {'n_neighbors':[3,5,7,9],'weights':['uniform','distance'],'metric':['euclidean','manhattan','minkowski'],

             'leaf_size':[40,45,50,60]}

knn_param_grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=7,refit=True,n_jobs=-1,verbose=1)

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)



#Applying Grid Search on Orginal Dataset ...

knn_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=knn_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)
# Grid Search on KNN unScaled Dataset ...



# Fit the parameters for KNN ...

param_grid = {'n_neighbors':[3,5,7,9,11],'weights':['uniform','distance'],'metric':['euclidean','manhattan','minkowski'],

             'leaf_size':[60,90,100,150,200,300]}

knn_param_grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=7,refit=True,n_jobs=-1,verbose=1)

X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)



#Applying Grid Search on Orginal Dataset ...

knn_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=knn_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)

# Random Forest 



#Create a Gaussian Classifier ...

# Using scaled Data set for Random Forest ...

classifier = RandomForestClassifier()

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)

#Create a Gaussian Classifier ...

# Using Unscaled Data set for Random Forest ...

classifier = RandomForestClassifier()

X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Grid Search on Random Forest Scaled Dataset ...



# Fit the parameters for Random Forest ...

param_grid = {'max_depth':[3,4,5,6],'max_features':['sqrt', 'auto', 'log2'],'n_estimators':[50,100],

             'min_samples_split':[2,3,5,6,7],'bootstrap':[True,False],'min_samples_leaf':[1,3,10]}

class_weight = dict({0: 0.5961070559610706, 1: 3.1012658227848102})

cross_validation = StratifiedKFold(n_splits=10)

RF_param_grid = GridSearchCV(RandomForestClassifier(class_weight=class_weight),param_grid,cv=cross_validation,refit=True,n_jobs=-1,verbose=1)



X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)



#Applying Grid Search on Orginal Dataset ...

RF_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=RF_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)

# Grid Search on Random Forest UnScaled Dataset ...



# Fit the parameters for Random Forest ...

param_grid = {'max_depth':[3,4,5,6],'max_features':['sqrt', 'auto', 'log2'],'n_estimators':[50,100],

             'min_samples_split':[2,3,5,6,7],'bootstrap':[True,False],'min_samples_leaf':[1,3,10]}

cross_validation = StratifiedKFold(n_splits=10)

class_weight = dict({0: 0.5961070559610706, 1: 3.1012658227848102})

RF_param_grid = GridSearchCV(RandomForestClassifier(class_weight=class_weight),param_grid,cv=cross_validation,refit=True,n_jobs=-1,verbose=1)



X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)



#Applying Grid Search on Orginal Dataset ...

RF_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=RF_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)

# Support Vector Machine



# Using Unscaled Data set.

classifier = SVC(kernel='linear')

X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Using Scaled Data set ...

classifier = SVC(kernel='linear')

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Grid Search on SVM Scaled Dataset ...



# Fit the parameters for SVM ...

param_grid = {'C':[0.45,0.5,0.51,0.53,0.55,1,1.5,5],'kernel': ['linear']}

cross_validation = StratifiedKFold(n_splits=10)

SVC_param_grid = GridSearchCV(SVC(),param_grid,cv=cross_validation,refit=True,n_jobs=-1,verbose=1)

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)



#Applying Grid Search on Orginal Dataset ...

SVC_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=SVC_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)
# Grid Search on SVM-RBF Scaled Dataset ...



# Fit the parameters for SVM ...

param_grid = {'C':[0.5,1,1.5,5],'gamma':[1,0.1,0.01,0.001],'probability':[True,False],'kernel': ['rbf']}

cross_validation = StratifiedKFold(n_splits=5)

SVC_param_grid = GridSearchCV(SVC(),param_grid,cv=cross_validation,refit=True,n_jobs=-1,verbose=1)

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)



#Applying Grid Search on Orginal Dataset ...

SVC_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=SVC_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)
# Grid Search on SVM Scaled Dataset ...



# Fit the parameters for SVM ...

param_grid = {'C':[0.5,1,1.5,5],'gamma':[1,0.1,0.01,0.001],'probability':[True,False],'kernel': ['poly'],

             'degree':[2,3]}

cross_validation = StratifiedKFold(n_splits=5)

SVC_param_grid = GridSearchCV(SVC(),param_grid,cv=cross_validation,refit=True,n_jobs=-1,verbose=1)

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)



#Applying Grid Search on Orginal Dataset ...

SVC_param_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=SVC_param_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)
# Extreme Gradient boosting classifier using Scaled Data ...



classifier = XGBClassifier()

X_train,X_test,y_train,y_test,cols,X,y=scaled_data(df)

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Extreme Gradient boosting classifier using UnScaled Data ...



classifier = XGBClassifier()

X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# A parameter grid for XGBoost

params = {

        'n_estimators' : [100, 200, 500, 750],

        'learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.25],

        'min_child_weight': [1, 5, 7, 10],

        'gamma': [0.1, 0.5, 1, 1.5, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5, 10, 12]

        }



folds = 5

param_comb = 800

from sklearn.model_selection import RandomizedSearchCV

# Extreme Gradient boosting classifier using UnScaled Data ...

classifier = XGBClassifier()

xgb_grid = RandomizedSearchCV(classifier, param_distributions=params,n_iter=param_comb, cv=5,n_jobs=-1, refit=True, verbose=1)



X_train,X_test,y_train,y_test,cols,X,y=unscaled_data(df)

#Applying Grid Search on Orginal Dataset ...

xgb_grid.fit(X_train,y_train)

# Find the best estimator from the model ...

final_model=xgb_grid.best_estimator_

# Predicting the Accuracy and AUC value from the function we defined above ...

Attrition_prediction(final_model,X_train,X_test,y_train,y_test,cols)
# Applying PCA function on training 

# and testing set of X component 



from sklearn.decomposition import PCA

X_train,X_test,y_train,y_test,cols,X,y= scaled_data(df)



pca = PCA().fit(X)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Attrition Level')

explained_variance = pca.explained_variance_ratio_ 

len(explained_variance)
# execute this step if you need the "PCA" Scaled "Train test split :X_train, X_test, y_train, y_test" 

pca = PCA(0.95)

X_unscale_pca = pca.fit_transform(X)

X_train,X_test,y_train,y_test,cols,X,y= scaled_data(df)

random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = random_state)

# Defining Cols variables to store the Column names of X scaled dataframe.



#Logistic Regression...



# Using PCA scaled Data set for Logistic Regression...



classifier = LogisticRegression()

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)

print("************************************************************")

classifier = SVC(kernel='linear')

Attrition_prediction(classifier,X_train,X_test,y_train,y_test,cols)
# Load the Imbalance Librarires for the further processing ...

from imblearn.over_sampling import SMOTE,RandomOverSampler

from imblearn.under_sampling import ClusterCentroids,NearMiss,RandomUnderSampler

from imblearn.combine import SMOTEENN,SMOTETomek

#from imblearn.ensemble import BalanceCascade



from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, auc,roc_auc_score,roc_curve, precision_recall_curve
# Helper functions



def benchmark(sampling_type,X,y):

    #clf = LogisticRegression(penalty='l2')

    clf = model

    param_grid = {'C':[0.01,0.1,1,10]}

    """LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='warn',

          n_jobs=None, penalty='l2', random_state=None, solver='warn',

          tol=0.0001, verbose=0, warm_start=False)"""

    grid_search_lr = GridSearchCV(estimator=clf,param_grid=param_grid,scoring='accuracy',cv=10,verbose=1,refit=True,n_jobs=-1)

    grid_search_lr = grid_search_lr.fit(X.values,y.values.ravel())

    

    return sampling_type,grid_search_lr.best_score_,grid_search_lr.best_params_['C']



def transform(transformer,X,y):

    print("Transforming {}".format(transformer.__class__.__name__))

    X_resampled,y_resampled =transformer.fit_sample(X.values,y.values.ravel())

    return transformer.__class__.__name__,pd.DataFrame(X_resampled),pd.DataFrame(y_resampled)

    

        


X_train,X_test,y_train,y_test,cols,X,y= unscaled_data(df)



datasets = []

datasets.append(("base",X_train,y_train))

datasets.append(transform(SMOTE(n_jobs=-1),X_train,y_train))

datasets.append(transform(RandomOverSampler(),X_train,y_train))

datasets.append(transform(NearMiss(n_jobs=-1),X_train,y_train))

datasets.append(transform(RandomUnderSampler(),X_train,y_train))

datasets.append(transform(SMOTEENN(),X_train,y_train))

datasets.append(transform(SMOTETomek(),X_train,y_train))

benchmark_scores =[]

for sample_type,X,y in datasets:

    print('__________________________________________________________________________')

    print('{}'.format(sample_type))

    benchmark_scores.append(benchmark(sample_type,X,y))

    print('__________________________________________________________________________')
benchmark_scores
# Train/evaluate models for each of tranformed datasets



scores=[]

# Train model based on benchmark params ...



for sample_type,score,parm in benchmark_scores:

    print("Training on {}".format(sample_type))

    clf = LogisticRegression(penalty='l1',C=parm)

    for s_type,X,y in datasets:

        if s_type == sample_type:

            clf.fit(X.values,y.values.ravel())

            pred_test = clf.predict(X_test.values)

            pred_test_probs = clf.predict_proba(X_test.values)

            probs = clf.decision_function(X_test.values)

            fpr, tpr , thresholds = roc_curve(y_test.values.ravel(),pred_test)

            p,r,t = precision_recall_curve(y_test.values.ravel(),probs)

            scores.append((sample_type,

                          f1_score(y_test.values.ravel(),pred_test), 

                          precision_score(y_test.values.ravel(),pred_test),

                           recall_score(y_test.values.ravel(),pred_test),

                           accuracy_score(y_test.values.ravel(),pred_test),

                           auc(fpr,tpr),

                           auc(p,r,reorder=True),

                           confusion_matrix(y_test.values.ravel(),pred_test)))
#Tabulate results

sampling_results_unscaled = pd.DataFrame(scores,columns=['Sampling Type','F1 Score','Precision','Recall','Accuracy','AUC_Score',

                                               'AUC_PR','Confusion Matrix'])

sampling_results_unscaled


X_train,X_test,y_train,y_test,cols,X,y= scaled_data(df)



datasets = []

datasets.append(("base",X_train,y_train))

datasets.append(transform(SMOTE(n_jobs=-1),X_train,y_train))

datasets.append(transform(RandomOverSampler(),X_train,y_train))

datasets.append(transform(NearMiss(n_jobs=-1),X_train,y_train))

datasets.append(transform(RandomUnderSampler(),X_train,y_train))

datasets.append(transform(SMOTEENN(),X_train,y_train))

datasets.append(transform(SMOTETomek(),X_train,y_train))
benchmark_scores =[]

for sample_type,X,y in datasets:

    print('__________________________________________________________________________')

    print('{}'.format(sample_type))

    benchmark_scores.append(benchmark(sample_type,X,y))

    print('__________________________________________________________________________')
# Train/evaluate models for each of tranformed datasets



scores=[]

# Train model based on benchmark params ...



for sample_type,score,parm in benchmark_scores:

    print("Training on {}".format(sample_type))

    clf = LogisticRegression(penalty='l1',C=parm)

    for s_type,X,y in datasets:

        if s_type == sample_type:

            clf.fit(X.values,y.values.ravel())

            pred_test = clf.predict(X_test.values)

            pred_test_probs = clf.predict_proba(X_test.values)

            probs = clf.decision_function(X_test.values)

            fpr, tpr , thresholds = roc_curve(y_test.values.ravel(),pred_test)

            p,r,t = precision_recall_curve(y_test.values.ravel(),probs)

            scores.append((sample_type,

                          f1_score(y_test.values.ravel(),pred_test), 

                          precision_score(y_test.values.ravel(),pred_test),

                           recall_score(y_test.values.ravel(),pred_test),

                           accuracy_score(y_test.values.ravel(),pred_test),

                           auc(fpr,tpr),

                           auc(p,r,reorder=True),

                           confusion_matrix(y_test.values.ravel(),pred_test)))
#Tabulate results

sampling_results_scaled = pd.DataFrame(scores,columns=['Sampling Type','F1 Score','Precision','Recall','Accuracy','AUC_Score',

                                               'AUC_PR','Confusion Matrix'])

sampling_results_scaled