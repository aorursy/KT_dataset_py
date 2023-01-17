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
#Checkout the dataframe columns
df.columns
# I'm going to list all of JSON columns
# CustomDimensions device geoNetwork hits trafficSource
#I'm going to check out each JSON column
df['customDimensions'][0]
df['device'][0]
df['trafficSource'][0]
df['geoNetwork'][0]
df['hits'][0]
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
#check the shape of the dataframe
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
#check the shape of the dataframe
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
# define a methond a find all missing values in the dataframe
def find_missing(df):
    count_missing=df.isnull().sum().values
    total=df.shape[0]
    ratio_missing=count_missing/total
    return pd.DataFrame({'missing':count_missing,'missing_ratio':ratio_missing},index=df.columns)
# define a dataframe called df_missing and use the function defined in the last step with the df dataframe
df_missing=find_missing(df)
# Check the dataframe df_missing
df_missing[df_missing['missing_ratio']>0].sort_values('missing_ratio',ascending=True).head(30)
#Then I'm going to visualize the missing data
missing_values = df.isnull().sum(axis=0).reset_index()
missing_values.columns = ['column_name', 'missing_count']
missing_values = missing_values.loc[missing_values['missing_count']>0]
missing_values = missing_values.sort_values(by='missing_count')
ind = np.arange(missing_values.shape[0])
width = 0.1
fig, ax = plt.subplots(figsize=(12,3))
rects = ax.barh(ind, missing_values.missing_count.values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(missing_values.column_name.values, rotation='horizontal')
ax.set_xlabel("Missing Observations Count")
ax.set_title("Missing Categorical Observations in Train Dataset")
plt.show()
#Then I'm going to delete columns containing more than 90 percent.
df_missing[df_missing['missing_ratio']>0].sort_values('missing_ratio',ascending=False).head(20)
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
# Then I'm going to deal with the catagorical variables
# look at the whole dataset to find which columns contain list cells.
for col in df_numerics_drop.columns:
    try:
        print(col, ':', df_numerics_drop[col].nunique(dropna=False))
    except TypeError:
        a=df_numerics_drop[col].astype('str')
        #print(a)
        print( col, ':', a.nunique(dropna=False), ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LIST')
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
# Then I'm going to look through other features to figure out if there is any other features to delete
for col in df_geoNetwork_drop.columns:
    print(col," : ",df_geoNetwork_drop[col].unique())
    print("=================================")
# Then I'm going to create a list of columns  which are going to be deleted in the next step
# Those columns are going to be which contains large variance and those just have "not available" cells.
other_drop=['operatingSystem','hits_appInfo.exitScreenName',
           'hits_appInfo.landingScreenName','hits_page.pagePath','hits_page.pagePathLevel1','hits_page.pagePathLevel2',
           'hits_page.pagePathLevel3','hits_page.pagePathLevel4','hits_page.pageTitle','hits_referer',
            'hits_social.socialInteractionNetworkAction','hits_transaction.currencyCode','campaign','keyword',
            'referralPath','source','hits_appInfo.screenDepth','hits_contentGroup.contentGroup4','hits_contentGroup.contentGroup5',
           'adwordsClickInfo.criteriaParameters','hits_item.currencyCode','hits_contentGroup.contentGroup2',
            'hits_contentGroup.contentGroup4','hits_contentGroup.contentGroup5','hits_contentGroup.previousContentGroup1',
           'hits_contentGroup.previousContentGroup2','hits_contentGroup.previousContentGroup3',
           'hits_contentGroup.previousContentGroup4','hits_contentGroup.previousContentGroup5','hits_hitNumber',
           'hits_hour','hits_eCommerceAction.action_type','hits_dataSource','hits_minute','hits_page.hostname','hits_isEntrance',
           'hits_isExit','hits_isInteraction','hits_social.socialNetwork','hits_time','continent','hits_promotionActionInfo.promoIsView',
           'isTrueDirect','medium','hits_social.hasSocialSourceReferral','hits_appInfo.screenName','hits_contentGroup.contentGroupUniqueViews2',
            'hits_eCommerceAction.step','hits_contentGroup.contentGroup3','hits_contentGroup.contentGroup1']
df_all_drop=df_geoNetwork_drop.drop(columns=other_drop)
df_all_drop_total=df_all_drop[df_all_drop['totals_transactionRevenue']>0]
df_all_drop_total.head()
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
scaled_df=df_all_drop_total.drop(columns='totals_transactionRevenue')
scaled_target=df_all_drop_total[['totals_transactionRevenue']]
scaled_data_all=vec1.fit_transform(df_all_drop_total.to_dict('records'))
scaled_data1=vec2.fit_transform(scaled_df.to_dict('records'))
scaled_data_target=vec3.fit_transform(scaled_target.to_dict('records'))                                
df_final=pd.DataFrame(scaled_data_all,columns=vec1.get_feature_names())
df_scaled=pd.DataFrame(scaled_data1,columns=vec2.get_feature_names())
df_target=pd.DataFrame(scaled_target,columns=vec3.get_feature_names())
df_final.head()
# Have a look on correlations between varied features
fx=plt.figure(figsize=(12,6))
sns.heatmap(df_final.corr(),cmap='coolwarm')
# Import train_test_split 
# Then assign all features except the target feature totals_transactionRevenue as X
# Assign totals_transactionRevenue as y
# Split the dataframe into X_train,X_test,y_train,y_test and give a test size as 0.25
from sklearn.model_selection import train_test_split
X=df_final.drop(columns='totals_transactionRevenue')
y=df_final['totals_transactionRevenue']
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=101)
# First, I'm tring to find the best parameters for the linear regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
param_grid = {'fit_intercept':[True,False],
              'normalize':[True,False], 
              'copy_X':[True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=7)
grid.fit(X,y)
grid.best_params_
# Then I'm going to run a supervised model which is linear regression
model=LinearRegression(copy_X=True,fit_intercept=False,normalize=True)
model.fit(X_train,y_train)
# I'm going to use regression evaluation metrics to evaluate the model which are Mean Absolute Error (MAE),
# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
from sklearn import metrics
# Check out the coefficient of the model
model.coef_
# Check out the intercept of the model
model.intercept_
# Predict the totals_transactionRevenue and plot the prediction
predictions=model.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Revenue')
# Print three metrics
# We can see both of the three metrics are large so the regression model is not good
print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
#Check the r square which shows the model is overfit
from sklearn.metrics import r2_score
r2_score(y_test, predictions)
# print out the residual histogram to see the difference between predcitons and actual dataset.
print(predictions)
sns.distplot(tuple(y_test-predictions),bins=50)
# Define a dataframe to see top 10 features having large coeffecient.
coeffecients=pd.DataFrame(model.coef_,X.columns)
coeffecients.columns=['Coeffecient']
coeffecients.sort_values('Coeffecient',ascending=False).head(10)
# First, I'm tring to find the best parameters for the KNN
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
gridlog = GridSearchCV(LogisticRegression(), param_grid,cv=None)
gridlog.fit(X,y)
gridlog.best_params_
# Then I'm going to create a model called logmodel with the best parameters.
logmodel = LogisticRegression(C=0.001)
logmodel.fit(X_train,y_train)
# make predictions with the logistic regression model
predictions = logmodel.predict(X_test)
predictions
# I'm going to use classsification report to evaluate the prediction model.
# As we can see in the report, because the target feature is not binary, and also because the amount of predicting variables
# is huge. So the logistic model does not make any sence about the prediction.
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
sns.distplot(tuple(y_test-predictions),bins=50)
#Check the r square which shows the model is overfit
from sklearn.metrics import r2_score
r2_score(y_test, predictions)
# Then I'm going to use an unsupervised learning model called pca to reduce dimensionality
# I'm going to try to see how many components should be assigned
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(scaled_data1)
scaled_data=scaler.transform(scaled_data1)
pca = PCA().fit(scaled_data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# As we can see above, around 8 components can expalain most variance,but at this step, I'm going to assign components 
# as 20.
pca=PCA(n_components=20)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=df_final['totals_transactionRevenue'],cmap='plasma')
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
# Check out the components
#In this numpy matrix array, each row represents a principal component, and each column relates back to the 
#original features
pca.components_
# Check out the components dataframe
df_comp=pd.DataFrame(pca.components_,columns=df_scaled.columns)
df_comp.head()
# Using heatmap to represent the correlation between varied features and then principle component itself.
plt.figure(figsize=(15,6))
sns.heatmap(df_comp,cmap='plasma')
# Then I'm going to use K-Means clustering unserpervied learning model.
# Create a K-means model with 4 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_final.drop(columns='totals_transactionRevenue',axis=1))
# Check out the cluster center
kmeans.cluster_centers_
# Have a look on scatter plot to see the difference from dataset before and after clustering.
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(df_final.iloc[:,-4],df_final.iloc[:,-3],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(df_final.iloc[:,-4],df_final.iloc[:,-3],c=df_final['totals_transactionRevenue'],cmap='rainbow')
# Define a method to convert the clustering model.
def converter(revenue):
    if 1000000<revenue<10000000:
        return 0
    elif 10000000<=revenue<1000000000:
        return 1
    elif 100000000<=revenue<1000000000:
        return 2
    else:
        return 3
# Add a new column cluster
df_final_cluster=df_final
df_final_cluster['Cluster'] = df_final_cluster['totals_transactionRevenue'].apply(converter)
# Check the dataframe
df_final_cluster.head()
# Using confusion matrix and classification report to evaluate the model.
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df_final['Cluster'],kmeans.labels_))
print(classification_report(df_final['Cluster'],kmeans.labels_))
# In this step, I'm going to run another pca model which contains 10 principle components
from sklearn.decomposition import PCA  
model = PCA(n_components=10)           
model.fit(scaled_data)                      
X_2D = model.transform(scaled_data) 
df_final_pca=df_final
# Assign two columns PCA1 PCA2 into the dataframe and visualize the PCA
df_final_pca['PCA1'] = X_2D[:, 0]
df_final_pca['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='totals_transactionRevenue', data=df_final,legend=False,fit_reg=False)
# I'm going to create a clustering model which contains 4 clusters
from sklearn.mixture import GaussianMixture                     
model = GaussianMixture(n_components=4, covariance_type='full')  
model.fit(scaled_data1)                                               
y_gmm = model.predict(scaled_data1)                                    
# Assign a new column cluster in the dataframe and visualize the clustering
df_cluster=df_final
df_cluster['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=df_cluster, hue='totals_transactionRevenue', 
           col='cluster', fit_reg=False, palette = 'tab10',legend=False);
# Now Im going to use a new pca to repredict with linear regression and logistic regression 
# Reassign the df_final dataframe
df_final=df_final.drop(columns=['Cluster','cluster','PCA1','PCA2'])
df_final.head()
# I'm going to use the pca created in the previous step to predict the data but make make n_components to be 40.
from sklearn.decomposition import PCA
pca1=PCA(n_components=40)
pca1.fit(scaled_data)
x_pca1=pca1.transform(scaled_data)
plt.figure(figsize=(8,6))
plt.scatter(x_pca1[:,0],x_pca1[:,1],c=df_final['totals_transactionRevenue'],cmap='plasma')
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
# Check out the components
pca1.components_
#Assign a new dataframe and store all components then print it out.
df_comp1=pd.DataFrame(pca1.components_,columns=df_scaled.columns)
df_comp1.head()
# Using heatmap to represent the correlation between varied features and then principle component itself.
plt.figure(figsize=(15,6))
sns.heatmap(df_comp1,cmap='viridis')
# Rerun the linear regression with the same parameters as preivious one and use the result of pca instead of original data
from sklearn.model_selection import train_test_split
X_scaled=x_pca1
y_scaled=scaled_data_target
Xtrain, Xtest, ytrain,ytest=train_test_split(X_scaled,y_scaled,test_size=0.2,random_state=101)
from sklearn.linear_model import LinearRegression
model1=LinearRegression(copy_X=True,fit_intercept=False,normalize=True)
model1.fit(Xtrain,ytrain)
# Check the coefficient
model1.coef_
model1.intercept_
# Plot the linear regression model
predictions1=model1.predict(Xtest)
plt.scatter(ytest,predictions1)
plt.xlabel('Y Test')
plt.ylabel('Predicted Revenue')
# Evaluate the result
print('MAE:',metrics.mean_absolute_error(ytest,predictions1))
print('MSE:',metrics.mean_squared_error(ytest,predictions1))
print("RMSE:",np.sqrt(metrics.mean_squared_error(ytest,predictions1)))
#Check the r square which shows the model is overfit
from sklearn.metrics import r2_score
r2_score(ytest, predictions1)
sns.distplot(tuple(ytest-predictions1),bins=10)
# Rerun the logistic regression with the same parameters as privious one and use the result from pca
from sklearn.model_selection import GridSearchCV
logmodel1 = LogisticRegression(C=0.001)
logmodel1.fit(X_scaled,y_scaled)
# Check the predictions
predictions_log = logmodel1.predict(Xtest)
predictions_log
# Evaluate the result
# We can see the difference from this and previous one although the model is not good enough as well.
from sklearn.metrics import classification_report
print(classification_report(ytest,predictions_log))

