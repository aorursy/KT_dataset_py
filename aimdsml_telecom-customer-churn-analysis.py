#import numpy , pandas , scipy , matplotlib , seaborn , sklearn
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from scipy import stats, integrate
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score

plt.rc("font", size=14)
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
# load DataSet
churn_df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn_df.info()
churn_df.head()
list(churn_df.columns.values)
# check missing data
churn_df.isnull().sum()
# check unique value
churn_df.nunique()
id_col     = ['customerID']
target_col = ['Churn']
category_cols = [col for col in churn_df.columns if churn_df[col].nunique() <= 4 and col != 'Churn']
numeric_cols = [col for col in churn_df.columns if col not in category_cols and col != 'Churn' and col != 'customerID']
print(category_cols)
print(numeric_cols)

#visualize categorical data
fig , axes = plt.subplots(nrows = 6 ,ncols = 3,figsize = (15,30))
for i, item in enumerate(category_cols):
    if i < 3:
        ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[0,i],rot = 0)
    elif i >=3 and i < 6:
        ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[1,i-3],rot = 0)
    elif i >= 6 and i < 9:
        ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[2,i-6],rot = 0)
    elif i < 12:
        ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[3,i-9],rot = 0)
    elif i < 15:
        ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[4,i-12],rot = 0)
    elif i < 18:
        ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[5,i-15],rot = 0)
    ax.set_title(item)
sns.countplot(x= 'Churn'   , data=churn_df);
#visualize categorical data as a relation to churn 
fig , axes = plt.subplots(nrows = 6 ,ncols = 3,figsize = (20,50))
for i, col in enumerate(category_cols):
    if i < 3:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[0,i],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[0,i])
    elif i >=3 and i < 6:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[1,i-3],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[1,i-3])
    elif i >= 6 and i < 9:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[2,i-6],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[2,i-6])
    elif i < 12:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[3,i-9],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[3,i-9])
    elif i < 15:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[4,i-12],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[4,i-12])
    elif i < 18:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[5,i-15],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[5,i-15])
    ax.set_title(col)
   
# map  'No internet service' to No 
internet_dependent_service = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in internet_dependent_service : 
    churn_df[i]  = churn_df[i].replace({'No internet service' : 'No'})
    
churn_df['MultipleLines'] = churn_df['MultipleLines'].replace({'No phone service' : 'No'})
#visualize categorical data as a relation to churn 
fig , axes = plt.subplots(nrows = 6 ,ncols = 3,figsize = (20,50))
for i, col in enumerate(category_cols):
    if i < 3:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[0,i],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[0,i])
    elif i >=3 and i < 6:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[1,i-3],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[1,i-3])
    elif i >= 6 and i < 9:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[2,i-6],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[2,i-6])
    elif i < 12:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[3,i-9],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[3,i-9])
    elif i < 15:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[4,i-12],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[4,i-12])
    elif i < 18:
        #ax = churn_df[item].value_counts().plot(kind = 'bar',ax=axes[5,i-15],rot = 0)
        ax = sns.countplot(x=col ,  hue = churn_df['Churn']  , data=churn_df , ax=axes[5,i-15])
    ax.set_title(col)
   
#quantative description for categorical data as a relation to churn
pd.crosstab(churn_df['Churn'] , churn_df[col] , margins = True)
for col in category_cols: 
    a =  pd.crosstab(churn_df['Churn'] , churn_df[col] , margins = True)
    a = a.apply(lambda r: r/r.sum() * 100 , axis=1)
    print(a)
    print('-----------------------------------------------------------------------------')

            
#impute total charge missing value
churn_df['TotalCharges'] = churn_df["TotalCharges"].replace(" ",np.nan)
churn_df["TotalCharges"] = churn_df["TotalCharges"].fillna( churn_df["MonthlyCharges"] * churn_df["tenure"] )
churn_df["TotalCharges"] = churn_df["TotalCharges"].astype(float) 
churn_df["AccumlatedCharges"] = churn_df["MonthlyCharges"] * churn_df["tenure"];

# visualize numeric variables
fig , axes = plt.subplots(nrows = 2 ,ncols = 3,figsize = (15,12))
for i, col in enumerate(numeric_cols):
    ax = sns.distplot(churn_df[col] , ax=axes[0,i])
    ax.set_title(col)

sns.distplot(churn_df['AccumlatedCharges'] , ax=axes[1,0])
ax.set_title('AccumlatedCharges');
#total charge = accumlated charge we can visualize the relation via coorelation map
corr_matrix = churn_df[['MonthlyCharges','TotalCharges'  , 'tenure']].corr()
plt.figure(figsize=(15, 15))
corrmap = sns.heatmap( corr_matrix , square=True , annot=True)
ax = sns.kdeplot(churn_df['MonthlyCharges'][(churn_df["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(churn_df['MonthlyCharges'][(churn_df["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')
ax = sns.kdeplot(churn_df['TotalCharges'][(churn_df["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(churn_df['TotalCharges'][(churn_df["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total Charges')
ax.set_title('Distribution of monthly charges by churn')
ax = sns.kdeplot(churn_df['tenure'][(churn_df["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(churn_df['tenure'][(churn_df["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('tenure')
ax.set_title('Distribution of tenure  by churn')
plt.figure(figsize=(8,8))
sns.distplot(churn_df.loc[churn_df['Churn']=='No', 'MonthlyCharges'], label='Churn: No')
sns.distplot(churn_df.loc[churn_df['Churn']=='Yes', 'MonthlyCharges'], label='Churn: Yes')
# PCA 
# Separating out the numeric features
x_numeric = churn_df.loc[:,numeric_cols].values
# Standardizing the features
x_normalized = StandardScaler().fit_transform(x_numeric)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x_normalized)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, churn_df[target_col]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Yes', 'No']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Churn'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#Binary columns with 2 values
binary_cols   = churn_df.nunique()[churn_df.nunique() == 2].keys().tolist()

#Columns more than 2 values
multi_category_cols = [i for i in category_cols if i not in binary_cols]



#Label encoding Binary columns
le = LabelEncoder()
for i in binary_cols :
    churn_df[i] = le.fit_transform(churn_df[i])
    
#Duplicating columns for multi value columns
churn_df = pd.get_dummies(data = churn_df,columns = multi_category_cols )

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(churn_df[numeric_cols])
scaled = pd.DataFrame(scaled,columns=numeric_cols)

list(churn_df)
#splitting train and test data 
train,test = train_test_split(churn_df,test_size = .25 ,random_state = 111)
 
#based on previous analysis via visualiztion and mathematical analysis we can deduce the relevant features and test it 
features = ['tenure' , 'MonthlyCharges' , 'SeniorCitizen' , 'Partner' , 'Dependents' ,
             'OnlineSecurity',
             'OnlineBackup',
             'DeviceProtection',
             'TechSupport',
             'StreamingTV',
             'StreamingMovies',
             'PaperlessBilling',
             'InternetService_DSL',
             'InternetService_Fiber optic',
             'InternetService_No',
             'Contract_Month-to-month',
             'Contract_One year',
             'Contract_Two year',
             'PaymentMethod_Bank transfer (automatic)',
             'PaymentMethod_Credit card (automatic)',
             'PaymentMethod_Electronic check',
             'PaymentMethod_Mailed check']
target = ['Churn']

train_X = train[features]
train_Y = train['Churn']
test_X  = test[features]
test_Y  = test['Churn']
def churn_prediction(ml_model,train_x,train_y,test_x,test_y) :
                            
    #model
    ml_model.fit(train_x,train_y)
    predictions   = ml_model.predict(test_x)
    scores = cross_val_score(ml_model , train_x , train_y , cv=5)
    
    print (ml_model)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {:.2f}".format(scores.mean()))
    print ("Classification report :",classification_report(test_y,predictions))
    print ("Accuracy Score : ",accuracy_score(test_y,predictions))
   
   
    
logistic_regression_model = LogisticRegression()
churn_prediction(logistic_regression_model,train_X,train_Y,test_X,test_Y)
decision_tree_model = DecisionTreeClassifier()
churn_prediction(decision_tree_model,train_X,train_Y,test_X,test_Y)
rf_model = RandomForestClassifier(criterion='entropy')
churn_prediction(rf_model,train_X,train_Y,test_X,test_Y)
linear_svm_model = SVC(kernel='linear',C=0.1,gamma=0.1,degree=3)
churn_prediction(linear_svm_model,train_X,train_Y,test_X,test_Y)
gaussian_svm_model = SVC(kernel='rbf',C=1,gamma=0.1,degree=3)
churn_prediction(gaussian_svm_model,train_X,train_Y,test_X,test_Y)
