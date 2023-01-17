import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np   
import pandas as pd    
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,train_test_split,cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy.stats import randint
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Models
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")
%matplotlib inline
# Loading the data
# df = pd.read_csv('train.csv')
df = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
test1 = test.copy()
# Checking top 5 rows of data
df.head()
# Checking Shape
df.shape
# Checking Information
df.info()
# Converting Region_Code & Policy_Sales_Channel to integer
df.Region_Code = df.Region_Code.astype('int64')
df.Policy_Sales_Channel = df.Policy_Sales_Channel.astype('int64')
test.Region_Code = test.Region_Code.astype('int64')
test.Policy_Sales_Channel = test.Policy_Sales_Channel.astype('int64')
# Checking null values
df.isnull().sum()
# Checking description of numerical columns
df.describe(include = 'number')
# Checking value counts for numerical columns
for col in df.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
        print(col,":",df[col].nunique())
# Checking description of object columns
df.describe(include = 'object')
# Checking for duplicates
dups = df.duplicated()
dups.sum()
# Dropping the duplicates
df.drop_duplicates(inplace=True)
df.shape
# Target variable count
graph = sns.countplot(df.Response)
for p in graph.patches:
    graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),ha='center', va='bottom',color= 'black')
df.Response.value_counts().plot(kind='pie', autopct='%1.0f%%');
# Counts of Categorical variables
fig=plt.figure(figsize=(20,12))
fig.subplots_adjust(hspace = .3, wspace=.2)
x = ['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']
for i in range(0,len(x)):
    ax=fig.add_subplot(2,3,i+1).set_title(x[i])
    graph = sns.countplot(df[x[i]])
    
    for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
# Analyzing continuous variables
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(14, 14)
a = sns.distplot(df['Age'] , ax=axes[0][0])
a.set_title("Age",fontsize=15)
a = sns.boxplot(df['Age'] , orient = "v" , ax=axes[0][1])
a.set_title("Age",fontsize=15)

a = sns.distplot(df['Annual_Premium'] , ax=axes[1][0])
a.set_title("Annual_Premium",fontsize=15)
a = sns.boxplot(df['Annual_Premium'] , orient = "v" , ax=axes[1][1])
a.set_title("Annual_Premium",fontsize=15)
plt.show()

plt.show()
# Top 10 regions with highest number of insurers
labels= df['Region_Code'].value_counts()[:10].keys()
values= df['Region_Code'].value_counts()[:10]

plt.figure(figsize = (15, 5))
graph = sns.barplot(x = labels, y = values)

for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
# Top 10 policy channels covering highest number of insurers
labels= df['Policy_Sales_Channel'].value_counts()[:10].keys()
values= df['Policy_Sales_Channel'].value_counts()[:10]

plt.figure(figsize = (15, 5))
graph = sns.barplot(x = labels, y = values)

for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
sns.distplot(df.Vintage)
# Pearson Correlation
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,mask=np.triu(df.corr(),+1))
fig=plt.figure(figsize=(20,12))
fig.subplots_adjust(hspace = .3, wspace=.2)
x = ['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']
for i in range(0,len(x)):
    ax=fig.add_subplot(2,3,i+1).set_title(x[i])
    graph = sns.countplot(df[x[i]],hue = df['Response'])
    
    for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
pd.crosstab(df['Vehicle_Damage'],df['Previously_Insured'])
pd.crosstab(df['Previously_Insured'],df['Response'])
pd.crosstab(df['Vehicle_Age'],df['Previously_Insured'])
fig=plt.figure(figsize=(20,12))
fig.subplots_adjust(hspace = .3, wspace=.2)
x = ['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']
for i in range(0,len(x)):
    ax=fig.add_subplot(2,3,i+1).set_title(x[i])
    sns.pointplot(df[x[i]],df['Response'])
fig, axes = plt.subplots(nrows=3,ncols=3,  figsize=(20,20))
fig.subplots_adjust(hspace = .3, wspace=.2)
x = ['Age','Annual_Premium','Vintage']
for i in range(0,len(x)):
    sns.barplot(df['Response'],df[x[i]],ax=axes[i][0])
    sns.violinplot(df['Response'],df[x[i]],ax=axes[i][1])
    sns.boxplot(df['Response'],df[x[i]],ax=axes[i][2])
fig=plt.figure(figsize=(20,12))
fig.subplots_adjust(hspace = .3, wspace=.2)
x = ['Gender','Vehicle_Age','Vehicle_Damage','Driving_License','Previously_Insured']
for i in range(0,len(x)):
    ax=fig.add_subplot(2,3,i+1).set_title(x[i])
    sns.barplot(df[x[i]],df['Annual_Premium'],hue = df['Response'])
sns.barplot('Vehicle_Age','Age',data=df)
sns.barplot('Vehicle_Damage','Age',data=df)
sns.pointplot('Vehicle_Damage','Driving_License',data=df)
df_feature_importance = df.copy()
df_feature_importance.replace({'< 1 Year': 0,'1-2 Year': 1,'> 2 Years': 2},inplace=True)
df_feature_importance = pd.get_dummies(df_feature_importance, columns=['Gender','Vehicle_Damage'],drop_first=True)
def feature_importance(model):
    x=pd.DataFrame(model.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
    plt.figure(figsize=(12,7))
    sns.barplot(x[0],x.index,palette='rainbow')
    plt.ylabel('Feature Name')
    plt.xlabel('Feature Importance in %')
    plt.title('Feature Importance Plot')
    plt.show()
# Copy all the predictor variables into X dataframe
X = df_feature_importance.drop('Response', axis=1)

# Copy target into the y dataframe
y = df_feature_importance['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)
import lightgbm as lgb
lgb_classfr = lgb.LGBMClassifier(objective='binary', 
         boosting_type = 'gbdt', 
         n_estimators = 10000)
lgb_classfr.fit(X_train, y_train, early_stopping_rounds=200, verbose = 200, eval_set = [(X_test, y_test)])
feature_importance(lgb_classfr)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
feature_importance(rf)
df_outlier = df.copy()
Q1 = df_outlier['Annual_Premium'].quantile(0.25)
Q3 = df_outlier['Annual_Premium'].quantile(0.75)
IQR = Q3 - Q1
lower_range= Q1-(3 * IQR)
upper_range= Q3+(3 * IQR)
print('Number of Outliers:')
((df_outlier['Annual_Premium'] < (lower_range)) | (df_outlier['Annual_Premium'] > (upper_range))).sum()
((((df_outlier['Annual_Premium'] < (lower_range)) | (df_outlier['Annual_Premium'] > (upper_range))).sum())/df_outlier.shape[0])*100
print('Upper Range: {}\nLower Range:{}'.format(upper_range,lower_range))
df_outlier['Annual_Premium']=np.where(df_outlier['Annual_Premium']>upper_range,upper_range,df_outlier['Annual_Premium'])
df_outlier['Annual_Premium']=np.where(df_outlier['Annual_Premium']<lower_range,lower_range,df_outlier['Annual_Premium'])
sns.boxplot(df_outlier['Annual_Premium'])
sns.distplot(df_outlier['Annual_Premium'])
Q1 = test['Annual_Premium'].quantile(0.25)
Q3 = test['Annual_Premium'].quantile(0.75)
IQR = Q3 - Q1
lower_range= Q1-(3 * IQR)
upper_range= Q3+(3 * IQR)
print('Number of Outliers:')
((test['Annual_Premium'] < (lower_range)) | (test['Annual_Premium'] > (upper_range))).sum()
test['Annual_Premium']=np.where(test['Annual_Premium']>upper_range,upper_range,test['Annual_Premium'])
test['Annual_Premium']=np.where(test['Annual_Premium']<lower_range,lower_range,test['Annual_Premium'])
# Age and annual premium highly skewed. Not normally distributed. Both right skewed with non negative values
# function to plot a histogram and a Q-Q plot
# side by side, for a certain variable
def diagnostic_plots(X,var):
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    plt.title(var)
    X.hist()

    plt.subplot(1, 2, 2)
    stats.probplot(X, dist="norm", plot=plt)

    plt.show()
def box_plots(X,Y,var1,var2):
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    plt.title(var1)
    sns.boxplot(X)

    plt.subplot(1, 2, 2)
    plt.title(var2)
    sns.boxplot(Y)

    plt.show()
for i in ['Age','Annual_Premium']:
    diagnostic_plots(df_outlier[i],i)
    print('Skewness for',i,":",df_outlier[i].skew())
for i in ['Age','Annual_Premium']:
    diagnostic_plots(np.log(df_outlier[i]+1),i)
    print('Skewness for',i,":",np.log(df_outlier[i]+1).skew())
box_plots(np.log(df_outlier['Age']+1),np.log(df_outlier['Annual_Premium']+1),'Age','Annual_Premium')
for i in ['Age','Annual_Premium']:
    diagnostic_plots(df_outlier[i]**(1/2),i)
    print('Skewness for',i,":",(df_outlier[i]**(1/2)).skew())
box_plots(df_outlier['Age']**(1/2),df_outlier['Annual_Premium']**(1/2),'Age','Annual_Premium')
for i in ['Age','Annual_Premium']:
    diagnostic_plots(df_outlier[i]**(1/3),i)
    print('Skewness for',i,":",(df_outlier[i]**(1/3)).skew())
box_plots(df_outlier['Age']**(1/3),df_outlier['Annual_Premium']**(1/3),'Age','Annual_Premium')
for i in ['Age','Annual_Premium']:
    diagnostic_plots(1/(df_outlier[i]+1),i)
    print('Skewness for',i,":",(1/(df_outlier[i]+1)).skew())
box_plots(1/(df_outlier['Age']+1),1/(df_outlier['Annual_Premium']+1),'Age','Annual_Premium')
df_outlier['Age_boxcox'], param = stats.boxcox(df_outlier.Age) 
df_outlier['Annual_Premium_boxcox'], param = stats.boxcox(df_outlier.Annual_Premium) 
test['Age_boxcox'], param = stats.boxcox(test.Age) 
test['Annual_Premium_boxcox'], param = stats.boxcox(test.Annual_Premium) 
for i in ['Age_boxcox','Annual_Premium_boxcox']:
    diagnostic_plots(df_outlier[i],i)
    print('Skewness for',i,":",df_outlier[i].skew())
box_plots(df_outlier['Age_boxcox'],df_outlier['Annual_Premium_boxcox'],'Age','Annual_Premium')
df_outlier.drop(['id','Age','Annual_Premium','Driving_License'],axis=1,inplace=True)
test.drop(['id','Age','Annual_Premium','Driving_License'],axis=1,inplace=True)
df_outlier.head()
df1 = df_outlier.copy()
df1.replace({'< 1 Year': 0,'1-2 Year': 1,'> 2 Years': 2},inplace=True)
test.replace({'< 1 Year': 0,'1-2 Year': 1,'> 2 Years': 2},inplace=True)
# dummy variable encoding
df1 = pd.get_dummies(df1, columns=['Gender','Vehicle_Damage'],drop_first=True)
test = pd.get_dummies(test, columns=['Gender','Vehicle_Damage'],drop_first=True)
df1.head() 
# Pearson Correlation
plt.figure(figsize=(12,8))
sns.heatmap(df1.corr(),annot=True,mask=np.triu(df1.corr(),+1))
# Checking level of multicollinearity with VIF of all variables
X = df1.drop('Response', axis=1)
vif = [variance_inflation_factor(X.values, ix) for ix in range(X.shape[1])] 
i=0
for column in X.columns:
  print (column ,"--->",  vif[i])
  i = i+1
df1.drop('Age_boxcox',axis=1,inplace=True)
test.drop('Age_boxcox',axis=1,inplace=True)
X = df1.drop('Response', axis=1)
vif = [variance_inflation_factor(X.values, ix) for ix in range(X.shape[1])] 
i=0
for column in X.columns:
  print (column ,"--->",  vif[i])
  i = i+1
print("Unique values in Policy_Sales_Channel: {}\nUnique values in Region_Code: {}".format(df1.Policy_Sales_Channel.nunique(),df1.Region_Code.nunique()))
## Mean encoding for Policy_Sales_Channel and Region_Code
df_mean_encode = df1.copy()
encod_type_Region_Code = df_mean_encode.groupby('Region_Code')['Response'].mean()
encod_type_Policy_Sales_Channel = df_mean_encode.groupby('Policy_Sales_Channel')['Response'].mean()
df_mean_encode.loc[:, 'Region_Code'] = df_mean_encode['Region_Code'].map(encod_type_Region_Code)
df_mean_encode.loc[:, 'Policy_Sales_Channel'] = df_mean_encode['Policy_Sales_Channel'].map(encod_type_Policy_Sales_Channel)
test.loc[:, 'Region_Code'] = test['Region_Code'].map(encod_type_Region_Code)
test.loc[:, 'Policy_Sales_Channel'] = test['Policy_Sales_Channel'].map(encod_type_Policy_Sales_Channel)
df_mean_encode.head() 
## Frequency encoding for Policy_Sales_Channel and Region_Code
df_frequency_encode = df1.copy()
df_frequency_map_region = df_frequency_encode.Region_Code.value_counts().to_dict()
df_frequency_map_Policy_Sales_Channel = df_frequency_encode.Policy_Sales_Channel.value_counts().to_dict()
df_frequency_encode['Region_Code'] = df_frequency_encode['Region_Code'].map(df_frequency_map_region)
df_frequency_encode['Policy_Sales_Channel'] = df_frequency_encode['Policy_Sales_Channel'].map(df_frequency_map_Policy_Sales_Channel)
df_frequency_encode.head()
## KDD encoding for Policy_Sales_Channel and Region_Code
df_KDD_encode = df1.copy()
# function to create the dummy variables for the most frequent labels
# we can vary the number of most frequent labels that we encode
def one_hot_encoding_top_x(df, variable, x):
    
    top_x_labels = [y for y in df[variable].value_counts().sort_values(ascending=False).head(x).index]
    
    for label in top_x_labels:
        df[variable+'_'+str(label)] = np.where(df[variable]==label, 1, 0)
one_hot_encoding_top_x(df_KDD_encode, 'Region_Code', 10)
one_hot_encoding_top_x(df_KDD_encode, 'Policy_Sales_Channel', 10)
df_KDD_encode.drop(['Region_Code','Policy_Sales_Channel'],axis=1,inplace=True)
df_KDD_encode.head()
mm = MinMaxScaler()
def split_data(dataframe):
    # Copy all the predictor variables into X dataframe
    X = dataframe.drop('Response', axis=1)
    
    # Copy target into the y dataframe
    y = dataframe['Response']
    
    # Split X and y into training and test set in 70:30 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)
    
    # Transform data using MinMaxScaler
    X_trains = mm.fit_transform(X_train)
    X_tests = mm.transform(X_test)
    return X_trains,X_tests,y_train,y_test
X_trains,X_tests,y_train,y_test = split_data(df_mean_encode);
test = mm.transform(test)
results_mean_encode = pd.DataFrame({'accuracy_train':[],'accuracy_test':[],'f1_score_train':[],'f1_score_test': [],'recall_train': [], 'recall_test': [],'precision_train': [],'precision_test': [],'AUC_train': [],'AUC_test': []})
models = [GaussianNB(), DecisionTreeClassifier(random_state=1), RandomForestClassifier(random_state=1), LogisticRegression(random_state = 1), LinearDiscriminantAnalysis(), xgb.XGBClassifier(random_state=1), LGBMClassifier(random_state=1)]
for model in models:
    model.fit(X_trains, y_train)
    results_mean_encode = results_mean_encode.append({'accuracy_train':metrics.accuracy_score(y_train, model.predict(X_trains)),
                                                      'accuracy_test': metrics.accuracy_score(y_test, model.predict(X_tests)),
                                                      'f1_score_train': metrics.f1_score(y_train, model.predict(X_trains)),
                                                      'f1_score_test': metrics.f1_score(y_test, model.predict(X_tests)),
                                                      'recall_train': metrics.recall_score(y_train, model.predict(X_trains)), 
                                                      'recall_test': metrics.recall_score(y_test, model.predict(X_tests)),
                                                      'precision_train': metrics.precision_score(y_train, model.predict(X_trains)),
                                                      'precision_test': metrics.precision_score(y_test, model.predict(X_tests)),
                                                      'AUC_train': roc_auc_score(y_train,model.predict_proba(X_trains)[:,1]),
                                                      'AUC_test': roc_auc_score(y_test,model.predict_proba(X_tests)[:,1])
                                                     }, ignore_index=True)
    
results_mean_encode['Models'] = ['Naive Bayes','Decision Tree','Random Forest','Logistic Regression','LDA','XGBoost','LightGBM']
results_mean_encode
X_trains,X_tests,y_train,y_test = split_data(df_frequency_encode);
results_frequency_encode = pd.DataFrame({'accuracy_train':[],'accuracy_test':[],'f1_score_train':[],'f1_score_test': [],'recall_train': [], 'recall_test': [],'precision_train': [],'precision_test': [],'AUC_train': [],'AUC_test': []})
for model in models:
    model.fit(X_trains, y_train)
    results_frequency_encode = results_frequency_encode.append({'accuracy_train':metrics.accuracy_score(y_train, model.predict(X_trains)),
                                                                'accuracy_test': metrics.accuracy_score(y_test, model.predict(X_tests)),
                                                                'f1_score_train': metrics.f1_score(y_train, model.predict(X_trains)),
                                                                'f1_score_test': metrics.f1_score(y_test, model.predict(X_tests)),
                                                                'recall_train': metrics.recall_score(y_train, model.predict(X_trains)), 
                                                                'recall_test': metrics.recall_score(y_test, model.predict(X_tests)),
                                                                'precision_train': metrics.precision_score(y_train, model.predict(X_trains)),
                                                                'precision_test': metrics.precision_score(y_test, model.predict(X_tests)),
                                                                'AUC_train': roc_auc_score(y_train,model.predict_proba(X_trains)[:,1]),
                                                                'AUC_test': roc_auc_score(y_test,model.predict_proba(X_tests)[:,1])
                                                               }, ignore_index=True)
    
results_frequency_encode['Models'] = ['Naive Bayes','Decision Tree','Random Forest','Logistic Regression','LDA','XGBoost','LightGBM']
results_frequency_encode
X_trains,X_tests,y_train,y_test = split_data(df_KDD_encode);
results_KDD_encode = pd.DataFrame({'accuracy_train':[],'accuracy_test':[],'f1_score_train':[],'f1_score_test': [],'recall_train': [], 'recall_test': [],'precision_train': [],'precision_test': [],'AUC_train': [],'AUC_test': []})
for model in models:
    model.fit(X_trains, y_train)
    results_KDD_encode = results_KDD_encode.append({'accuracy_train':metrics.accuracy_score(y_train, model.predict(X_trains)),
                                                    'accuracy_test': metrics.accuracy_score(y_test, model.predict(X_tests)),
                                                    'f1_score_train': metrics.f1_score(y_train, model.predict(X_trains)),
                                                    'f1_score_test': metrics.f1_score(y_test, model.predict(X_tests)),
                                                    'recall_train': metrics.recall_score(y_train, model.predict(X_trains)), 
                                                    'recall_test': metrics.recall_score(y_test, model.predict(X_tests)),
                                                    'precision_train': metrics.precision_score(y_train, model.predict(X_trains)),
                                                    'precision_test': metrics.precision_score(y_test, model.predict(X_tests)),
                                                    'AUC_train': roc_auc_score(y_train,model.predict_proba(X_trains)[:,1]),
                                                    'AUC_test': roc_auc_score(y_test,model.predict_proba(X_tests)[:,1])
                                                    }, ignore_index=True)
    
results_KDD_encode['Models'] = ['Naive Bayes','Decision Tree','Random Forest','Logistic Regression','LDA','XGBoost','LightGBM']
results_KDD_encode
X_trains,X_tests,y_train,y_test = split_data(df_mean_encode);
test = mm.transform(test)
XGB = xgb.XGBClassifier(random_state=1)
XGB.fit(X_trains,y_train)
result =  XGB.predict(test)
submit = pd.DataFrame({'id': test1.id, 'Response': result})
submit.to_csv('Submission.csv', index=False)
print("Your submission was successfully saved!")