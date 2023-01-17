import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.svm import SVC
%matplotlib inline
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target
m=X[:, 0]
n=X[:, 1]
svm = SVC(C=0.5, kernel='linear')
svm.fit(X, y)
def plot_decision_boundary(X,y,clf=None):
    fig=plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y,cmap='viridis',s=30, zorder=3)
    ax.axis('tight')
    ax.axis('on')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='viridis',
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
plot_decision_boundary(X,y,clf=svm)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('SVM on Iris')
# Import general libraries such as pandas, numpy, matplotlib.pyplot and seaborn, that will be used in the program.
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import ast
# Read the data set from local into a dataframe called df.
df=pd.read_csv('../input/data.csv')
# Check the dataframe
df.head(5)
# As shown above in the dataset, because we have many variables consist of dictionaries or lists.
# We need to convert the columns and split into a normal column version.
#import json and to convert the columns
import json
from pandas.io.json import json_normalize
# Read all data of list of dict in the column customDimensions and resign it to a new customDimensions column
df['customDimensions']=df['customDimensions'].apply(ast.literal_eval)
# convert the list of dict into dict and resign
df['customDimensions']=df['customDimensions'].str[0]
# format null-value data
df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
# Using json to normalize the dict colomn into a dataframe
df_customDimensions = json_normalize(df['customDimensions'])
# define the name of columns
df_customDimensions.columns = [f"customDimension_{subcolumn}" for subcolumn in df_customDimensions.columns]
df_customDimensions.shape
# Read all data of list of dict in the column customDimensions and resign it to a new hits column
df['hits']=df['hits'].apply(ast.literal_eval)
# convert the list of dict into dict and resign
df['hits']=df['hits'].str[0]
# format null-value data
df['hits']=df['hits'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
# Using json to normalize the dict colomn into a dataframe
df_hits = json_normalize(df['hits'])
# define the name of columns
df_hits.columns = [f"hits_{subcolumn}" for subcolumn in df_hits.columns]
df_hits.shape
# Using json to normalize the dict colomn into a dataframe
#check the shape of the dataframe
df_device=json_normalize(df['device'].apply(eval))
df_device.shape
# Using json to normalize the dict colomn into a dataframe
#check the shape of the dataframe
df_trafficSource=json_normalize(df['trafficSource'].apply(eval))
df_trafficSource.shape
# Using json to normalize the dict colomn into a dataframe
#check the shape of the dataframe
df_geoNetwork=json_normalize(df['geoNetwork'].apply(eval))
df_geoNetwork.shape
# define a new dataframe which is a concat of all five json dataframe then delete original redundent columns 
df=pd.concat([df,df_device,df_customDimensions,df_geoNetwork,df_hits,df_trafficSource],axis=1,sort=True)
df.drop(columns=['device','customDimensions','geoNetwork','trafficSource','hits'],inplace=True)
df.head()
#Create a list of columns those are going to be deleted except target variable.
missing_drop=['hits_index','hits_value','hits_promotionActionInfo.promoIsClick','hits_page.searchCategory',
             'hits_page.searchKeyword','hits_eventInfo.eventLabel','hits_eventInfo.eventCategory',
             'hits_eventInfo.eventAction','hits_contentGroup.contentGroupUniqueViews1','totals_totalTransactionRevenue',
             'totals_transactions','adContent','hits_contentGroup.contentGroupUniqueViews3','adwordsClickInfo.adNetworkType',
             'adwordsClickInfo.isVideoAd','adwordsClickInfo.page','adwordsClickInfo.slot','adwordsClickInfo.gclId']
#Then define a new dataframe called df_missing_drop and drop all columns defined at the last step.
df_missing_drop=df.drop(columns=missing_drop)
df_missing_drop.head()
# Then I'm going to fill all missing values with 0 and create a new dataframe called df_missing_replace
df_missing_replace=df_missing_drop.fillna(0)
df_missing_replace.head()
# As shown in the dataframe, we can find the data consists of both numeric and catagorical variables. So I'm going to 
# preprocess the data within two steps interm of numeric and catagorical variables.
# Check the numeric data in the dataset.
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_numerics_row = df_missing_replace.select_dtypes(include=numerics)
df_numerics_row.head()
# Then I'm going to delete some incorrelate columns within the dataframe and then define a new dataframe to store the data
numerics_columns_drop=['date','visitId','visitStartTime']
df_numerics_drop=df_missing_replace.drop(columns=numerics_columns_drop)
df_numerics_drop.head()
# As we can see in the output, there are seven columns containing list data. So I'm going to delete these columns firstly.
list_drop=['hits_customDimensions','hits_customMetrics','hits_customVariables','hits_experiment',
          'hits_product','hits_promotion','hits_publisher_infos']
df_list_drop=df_numerics_drop.drop(columns=list_drop)
df_list_drop.head()
# Then I'm going to drop more incorrelate catagorical variables in the dataframe.
# look through the feature and define lists of columns which are needed to be deleted for orginal variables.
socialEngagementType=['socialEngagementType']
customDimensions=['customDimension_index']
fullVistorId=['fullVisitorId']
device=['browserSize','browserVersion','flashVersion','language','mobileDeviceBranding','mobileDeviceInfo',
        'mobileDeviceMarketingName','mobileDeviceModel','mobileInputSelector','operatingSystemVersion',
        'screenColors','screenResolution']
geoNetwork=['cityId','metro','region','networkDomain','networkLocation','longitude','latitude']
df_socialEngagementType_drop=df_list_drop.drop(columns=socialEngagementType)
df_customDimensions_drop=df_socialEngagementType_drop.drop(columns=customDimensions)
df_fullVistorID_drop=df_customDimensions_drop.drop(columns=fullVistorId)
df_device_drop=df_fullVistorID_drop.drop(columns=device)
df_geoNetwork_drop=df_device_drop.drop(columns=geoNetwork)
df_geoNetwork_drop.head()
# Then I'm going to drop some irrelated features
Drop=['hits_appInfo.exitScreenName','hits_appInfo.landingScreenName','hits_appInfo.screenName',
     'hits_page.hostname','hits_page.pagePath','hits_page.pagePathLevel1','hits_page.pagePathLevel2',
     'hits_page.pagePathLevel3','hits_page.pagePathLevel4','hits_page.pageTitle','campaign','keyword',
     'referralPath','hits_referer','browser']
df_geoNetwork_drop=df_geoNetwork_drop.drop(columns=Drop)
df_geoNetwork_drop.head()
# After data cleansing, the next step I'm going to do is to scale the data with a proper scaling model.
# I'm going to use Dicvectorizer as my scaling model because in this dataset it has both numeric and catagorical 
# variables
# Then I'm going to assign this dataframe from the scaling process (with transaction_revenue>0) to the final 
# dataframe as my main dataframe to continue on work.
# Split features and target into different dataframes and also give data scaling
from sklearn.feature_extraction import DictVectorizer
vec1 = DictVectorizer(sparse=False, dtype=int)
vec2 = DictVectorizer(sparse=False, dtype=int)
vec3 = DictVectorizer(sparse=False, dtype=int)
scaled_df=df_geoNetwork_drop.drop(columns='totals_transactionRevenue')
scaled_target=df_geoNetwork_drop[['totals_transactionRevenue']]
scaled_data_all=vec1.fit_transform(df_geoNetwork_drop.to_dict('records'))
scaled_data=vec2.fit_transform(scaled_df.to_dict('records'))
scaled_data_target=vec3.fit_transform(scaled_target.to_dict('records'))  
df_final=pd.DataFrame(scaled_data_all,columns=vec1.get_feature_names())
df_scaled=pd.DataFrame(scaled_data,columns=vec2.get_feature_names())
df_final.head()
# Add a column called cluster in df_final and assign 0 if revenue is 0 otherwise assign 1 if revenue >0
# define a method and implement to the add-in column process
def converter(revenue):
    if revenue==0:
        return 0
    else:
        return 1
# Create a new column called cluster which contains 1 for positive revenues and 0 for non revenues
df_final['Cluster'] = df_final['totals_transactionRevenue'].apply(converter)
df_final.head()
# Then find the not set variables 
List=[]
for name in df_final.columns:
    if 'not' in name:
        List.append(name)
print(List)
# Then delete some not set variables
Not_drop=['adwordsClickInfo.criteriaParameters=not available in demo dataset', 'city=(not set)', 
'city=not available in demo dataset', 'continent=(not set)', 'country=(not set)', 
'hits_contentGroup.contentGroup1=(not set)', 'hits_contentGroup.contentGroup2=(not set)', 
'hits_contentGroup.contentGroup3=(not set)', 'hits_contentGroup.contentGroup4=(not set)', 
'hits_contentGroup.contentGroup5=(not set)', 'hits_dataSource=(not set)', 'hits_social.socialNetwork=(not set)'
, 'operatingSystem=(not set)', 'subContinent=(not set)']
df_final=df_final.drop(columns=Not_drop)
df_final.head()
# Then I'm going to run a logistic regression model which is a classification model
from sklearn.model_selection import train_test_split
X=df_final.drop(columns=['totals_transactionRevenue','Cluster'])
y=df_final['Cluster']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
# I'm going to use classsification report to evaluate the prediction model.
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
# Then i'm foing to do another classification which is called KNN  
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
# I'm going to use classsification report to evaluate the prediction model.
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
# Then I'm going to run a regression model based on the positive revenue record
from sklearn.linear_model import LinearRegression
df_revenue=df_final[df_final['totals_transactionRevenue']>0]
X=df_revenue.drop(columns=['totals_transactionRevenue','Cluster'])
y=df_revenue['totals_transactionRevenue']
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=101)
# Using PCA to reduce dimensionality 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
scaled_data=scaler.transform(X)
pca = PCA().fit(scaled_data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# Create a pca model which components is 132 because 132 components explain most variance
pca1=PCA(n_components=132)
pca1.fit(scaled_data)
x_pca1=pca1.transform(scaled_data)
# Check out th ecomponents
pca1.components_
# split the data into train and test datasets
X_scaled=x_pca1
y_scaled=df_revenue['totals_transactionRevenue']
Xtrain, Xtest, ytrain,ytest=train_test_split(X_scaled,y_scaled,test_size=0.2,random_state=101)
# Using gridSearchCV to find the best parameters for linear regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
param_grid = {'fit_intercept':[True,False],
              'normalize':[True,False], 
              'copy_X':[True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=7)
grid.fit(X_scaled,y_scaled)
grid.best_params_
# Create the linear regression model
linear=LinearRegression(copy_X=True,fit_intercept=False,normalize=True)
linear.fit(Xtrain,ytrain)
# See the result of the linear regression model
linear_pred=linear.predict(Xtest)
plt.scatter(ytest,linear_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Revenue')
df_null=df_final[df_final['totals_transactionRevenue']==0]
df_null.shape
