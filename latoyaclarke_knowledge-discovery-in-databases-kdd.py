# Libraries needed 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import warnings
import re
warnings.filterwarnings('ignore')
%matplotlib inline
#random_state = 1
plt.rcParams['figure.figsize']=8,8
import os
print(os.listdir("../input"))
# load wine dataset into notebook
data_path = "../input/winemag-data-130k-v2.csv"
wine_data = pd.read_csv(data_path)
print("Wine Data Dimension:",wine_data.shape)
wine_data.head(3) #disply first 5 elements in dataset
wine_data.tail(3) #displays last 3 elements in dataset
# check to see if there are spaces in column names
wine_data.columns
wine_data.describe()
# Useful information about the data
wine_data.info()
# Disttribution of points for initial dataset
sns.distplot(wine_data.points) 
plt.xlabel("Points",size=15)
plt.title("Fig.1: Points Distribution", size=20)
# Wine Tasters and the Amount of Wines Evaluated
wine_data['taster_name'].value_counts().plot(kind='bar')
plt.xticks(fontsize=11)
plt.xlabel("Taster Names",size=15)
plt.ylabel('No. of wines tasted', size=15)
plt.title("Fig. 2: Wine Tasters and the Amount of Wines Evaluated", size=20)
# Selecting sample of the data that will be used (i.e. the wines evaluated by Roger Voss)
wine_data = wine_data[(wine_data['taster_name']=='Roger Voss') | (wine_data['taster_twitter_handle']=='@vossroger')]
print("Wine Data Sample Dimension:",wine_data.shape,"(i.e. Wines evaluated by Roger Voss)") #dataset dimension
wine_data.head()
# Measures of the numerical data (i.e. wines tasted by Roger Voss)
wine_data.describe()
wine_data.info()
# Distribution of points for wines Roger Voss evaluated
sns.distplot(wine_data.points) 
plt.xlabel("Points",size=15)
plt.title("Fig.3: Points Distribution for Wines Evaluated by Roger Voss ", size=20)
sns.countplot(x='country',data=wine_data, orient="h")
plt.ylabel('Country Count',size=12)
plt.xlabel("Country",size=12)
plt.xticks(rotation=45)
plt.suptitle("Fig.4: Countries per bottle of Wine Evaluted by Roger Voss ", size=20)
# Rename column 'serial' to 'wine_id'
wine_data.rename(columns={'serial':'wine_Id'}, inplace=True)
wine_data.head(1)
#check for duplicates in dataset and remove if any
print(wine_data.duplicated(subset=None, keep='first').sum(),"duplicate record(s)")
# Perform feature extraction to impute the year of each wine
wine_data['year'] = wine_data['title'].str.extract('(\d\d\d\d)', expand=True)
# Check to see if there are any null years
wine_data['year'].isnull().value_counts()
# wines that does not have a year in the title
#Wines without a year are classified as Non-Vintage wines
wine_data.title[wine_data['year'].isnull()].head()
# convert year to int so as to make searches for preprocessing easier
wine_data.year = pd.to_numeric(wine_data.year, errors='coerce').fillna(0).astype(np.int64)
# check fo erroneous years (NB: its year 2018, any year above this is invalid)
print((wine_data['year']>2018).sum(),"invalid year(s)")
# Applying feature engineering to create type of wine (Vintage/n\Non-Vintage)
wine_data['type']= None
wine_data.type[wine_data['year']!=0] = 'Vintage'
wine_data.type[wine_data['year']==0] = 'Non-Vintage'
# Create loation by feature extraction from title
no_location = wine_data['title'].str.split('(', expand=True, n=1)
#wine_data['location'] = no_location.str.extract('(', expand=True)
#wine_data
no_location=no_location[1].str.split(')', expand=True, n=1)
wine_data['location']=no_location[0]

#wine_data[wine_data['location'].isnull()==True]
# impute location from region_2,region_1,province 
wine_data['location'].fillna(wine_data.region_2, inplace = True) 
wine_data['location'].fillna(wine_data.region_1, inplace = True)
wine_data['location'].fillna(wine_data.province, inplace = True)
# look for missing locations and country
print(wine_data['location'].isnull().sum(),"missing location(s) and",wine_data['country'].isnull().sum(),"missing countries") #check for null locations

#impute missing missing location and country from title research
wine_data.location.fillna('Bordeaux',inplace=True)
wine_data.country.fillna('France',inplace=True)

print("are attributed to 'Bordeaux' region in 'France' based on research of wine titles")
# look for missing prices
print(wine_data['price'].isnull().sum(),"missing price(s)") #check for null prices

#impute missing prices with the median price
wine_data.price.fillna(wine_data['price'].median(),inplace=True)
print("imputed from median price")
# Drop columns that are not needed
wine_data_2 = wine_data.drop(['designation','region_1','region_2','taster_twitter_handle','description','province','taster_name'],axis=1)
wine_data_2.head()
wine_data_2.info() # confirm that there are no missing values
# Transformation
# Label encoder transforms nominal features into numerical labels which algorithms can make sense of
def create_label_encoder_dict(df):
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder_dict = {}
    
    for column in df.columns:
        if not np.issubdtype(df[column].dtype, np.number) and column != 'year':
            label_encoder_dict[column]= LabelEncoder().fit(df[column])
    return label_encoder_dict

label_encoders = create_label_encoder_dict(wine_data_2)
#print("Encoded Values for each Label")
#print("="*32)
#for column in label_encoders:
 #   print("="*32)
 #   print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
  #  print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
    
### Apply each encoder to the data set to obtain transformed values
wd3 = wine_data_2.copy() # create copy of initial data set
for column in wd3.columns:
    if column in label_encoders:
        wd3[column] = label_encoders[column].transform(wd3[column])

print("Transformed data set")
print("="*32)
wd3.head()

# Function to do K-Fold Cross Validation
def cross_validate(x,y,kf_split):
    from sklearn.model_selection import KFold
    
    #K-Fold Cross Validation
    kf =KFold(n_splits=kf_split,shuffle=True,random_state=1)
    
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    return [X_train,y_train,X_test,y_test]
# Algorithms without Hyper Parameter Tuning
def pred_techniques(x,y,kf_split): 
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neural_network import MLPRegressor

    train_test = cross_validate(x,y,kf_split) #perform kfold cross validation

    # Decision Tree Regressor
    reg = DecisionTreeRegressor(random_state=1) 
    reg.fit(train_test[0], train_test[1])
    
    
    # Multi-Layer Perceptron Regressor
    clf = MLPRegressor(solver='adam', alpha=1e-5, activation='relu',learning_rate_init =0.01,shuffle=True,
                    hidden_layer_sizes=(7, 4),random_state=1)
    clf.fit(train_test[0],train_test[1])
    

    return [reg,clf,train_test[2],train_test[3],train_test[0],train_test[1]]
# separate data into dependent (Y) and independent(X) variables
feature_cols =  ['variety','winery','location', 'year']
x_data = wd3[feature_cols]
y_data = wd3['points']

l = pred_techniques(x_data,y_data,100)
print("Fig.5: Feature Significance") 
pd.DataFrame([ "%.2f%%" % perc for perc in (l[0].feature_importances_ * 100) ], index = x_data.columns, columns = ['Feature Significance in Decision Tree'])    
# Accuracy Test Scores of both techniques 
r2_tree = l[0].score(l[2],l[3])
r2_nn = l[1].score(l[2],l[3])

print("Decision Tree Regressor")
print("="*32)
print("R Square:",r2_tree )


print("\nMulti-Layer Perceptron Regressor")
print("="*32)
print("R Square:",r2_nn)

# Actual points and Predicted points by both models plus 4 predictor variables on left
results= l[2].copy()
results['Actual Points']=l[3]
pred_tree=l[0].predict(l[2])
pred_mlp=l[1].predict(l[2])
results['Dec_Tree_Reg']=pred_tree
results['MLP_Regressor']=pred_mlp
print("Fig.6: Actual Points and Predicted Points yielded from both Models")
results.head()

# Calculate Variance in both models
mse_treg = mean_squared_error(l[3],pred_tree)
mse_nn = mean_squared_error(l[3],pred_mlp)

# Calculate Standard Deviation in both models
rmse_treg = math.sqrt(mean_squared_error(l[3],pred_tree))
rmse_nn = math.sqrt(mean_squared_error(l[3],pred_mlp))

# Calcualte Mean Absolute Error in both models
mae_treg = mean_absolute_error(l[3],pred_tree)
mae_nn = mean_absolute_error(l[3],pred_mlp)

# Print evaluation metrics of both models
print("Decision Tree Regressor")
print("="*32)
print("MSE:",mse_treg)
print("RMSE:",rmse_treg)
print("MAE:",mae_treg)

print("\nMulti-Layer Perceptron Regressor")
print("="*32)
print("MSE:",mse_nn)
print("RMSE:",rmse_nn)
print("MAE:",mae_nn)
print("Decision Tree Number of Perfect Predictions:")
results[results['Dec_Tree_Reg']==results['Actual Points']].Dec_Tree_Reg.count()
print("Neural Network Number of Perfect Predictions:")
results[results['MLP_Regressor']==results['Actual Points']].MLP_Regressor.count()
sns.distplot( results["Actual Points"] , color="skyblue", label="Actual Points")
sns.distplot( results["Dec_Tree_Reg"] , color="orange", label="Decision Tree Predicted Points")
plt.legend()
plt.xlabel("Points",size=15)
plt.title("Fig.7: Actual Points vs Decision Tree Predicted Points", size=20)
sns.distplot( results["Actual Points"] , color="skyblue", label="Actual Points")
sns.distplot( results["MLP_Regressor"] , color="red", label="NN Predicted Points")
plt.legend()
plt.xlabel("Points",size=15)
plt.title("Fig.8: Actual Points vs Neural Network Predicted Points", size=20)
print("Fig.9: Summary of Evaluation Metrics")
pd.DataFrame(dict(R_Square= [r2_tree,r2_nn],
                  MSE=[mse_treg,mse_nn], RMSE=[rmse_treg,rmse_nn],MAE=[mae_treg,mae_nn]),
                index=['Dec Tree Reg','MLP Reg'])

results.describe()
sns.regplot(x="price", y="points", data=wine_data, fit_reg = False)
plt.xlabel("Price",size=12)
plt.ylabel("Points",size=12)
plt.title("Fig.10: Correlation between Price and Points",size=20)
# Coverage of Vintage vs. Non-Vintage
wine_data['type'].value_counts().plot(kind="pie",autopct='%1.0f%%')
labels = 'Vintage', 'Non-Vintage'
plt.legend(labels)
plt.suptitle("Fig.11: Vintage vs. Non-Vintage Wine", size=20)
plt.ylabel('')
# Non-Vintage Wine Points Distribution
#wine_data[wine_data['year']== 0]
no_year= wine_data[wine_data['year']==0]
sns.boxplot(x=no_year.points)
plt.xlabel("Points",size=15)
plt.title("Fig.12: Non-Vintage Wine Points Distribution  ", size=20)
plt.figure(figsize=(10,7))
sns.heatmap(wd3.corr(),cmap=plt.cm.Reds,)
plt.xticks(size=12,rotation=45)
plt.yticks(size=12)
plt.title('Fig.13: Correlation between Transformed Data Columns ',size=20)
