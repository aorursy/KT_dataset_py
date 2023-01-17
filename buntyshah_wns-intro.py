import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.decomposition import PCA,KernelPCA
train= pd.read_csv('../input/WNS_Train.csv')
test = pd.read_csv('../input/WNS_test.csv')
sample = pd.read_csv('../input/wns_sample.csv')
#train['depart&Edu'] = train['department'] + train['education'] + train['region']
#test['depart&Edu'] = test['department'] + test['education'] + test['region']
#train['sex&reg'] = train['region'] + train['gender']
#test['sex&reg'] = test['region'] + test['gender']

train['previous_year_rating'] = train['previous_year_rating'].astype(str)
test['previous_year_rating'] = test['previous_year_rating'].astype(str)
#Lets Bin Avg training score

#train['avg_training_score'] = train['avg_training_score'].apply(lambda x: 'bin1' if x<51 else ('bin2' if (x>=51 and x<63) else ('bin3' if (x>=63 and x<75) else ('bin4' if (x>=75 and x<87) else 'bin5')) ))
#test['avg_training_score'] = test['avg_training_score'].apply(lambda x: 'bin1' if x<51 else ('bin2' if (x>=51 and x<63) else ('bin3' if (x>=63 and x<75) else ('bin4' if (x>=75 and x<87) else 'bin5')) ) )                                                          
#test['avg_training_score_bin'] = test['avg_training_score'].apply(lambda x: 'bin1' if x<59 else ('bin2' if (x>=59 and x<79) else 'bin3') )
train['age'] = train['age'].apply(lambda x: 'bin1' if (x >=20 and x<30) else ('bin2' if ( x>=30 and x<40) else ('bin3' if (x>=40 and x<50) else 'bin4')) )
test['age'] = test['age'].apply(lambda x: 'bin1' if (x >=20 and x<30) else ('bin2' if ( x>=30 and x<40) else ('bin3' if (x>=40 and x<50) else 'bin4')) )
#train= train.drop('region',axis=1)
#test =test.drop('region',axis=1)
train.head()
test.head()
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values_train = missing_values_table(train)
missing_values_test = missing_values_table(test)
print("Train Data")
print(missing_values_train)
print("Test Data")
print(missing_values_test)
train['education'].unique()
train['education'] = train['education'].apply(lambda x : 'other' if pd.isnull(x) else x)
test['education'] = test['education'].apply(lambda x : 'other' if pd.isnull(x) else x)
train['depart&Sex'] = train['education'] + train['gender']
test['depart&Sex'] = test['education'] + test['gender']
#Check Department unique value in train and test
print("department in train data is",train.department.nunique())
print("department in test data is ",test.department.nunique())
#Check region unique value in train and test
#print("region in train data is",train.region.nunique())
#print("region in test data is ",test.region.nunique())
#Check education unique value in train and test
print("education in train data is",train.education.nunique())
print("education in test data is ",test.education.nunique())
#Check recruitment_channel unique value in train and test
print("recruitment_channel in train data is",train.recruitment_channel.nunique())
print("recruitment_channel in test data is ",test.recruitment_channel.nunique())
#lets check data imbalance
sns.set(style="darkgrid")
sns.countplot(x='is_promoted',data=train)
sns.catplot(x="KPIs_met >80%", hue="previous_year_rating", col="is_promoted",
                data=train, kind="count",
                height=4, aspect=.7);
#lets do Lable enconding coding to make more features 

le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in train:
    if train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train[col].unique())) <= 2:
            # Train on the training data
            le.fit(train[col])
            # Transform both training and testing data
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
train.head()
train = pd.get_dummies(train)
test = pd.get_dummies(test)

print('Training Features shape: ', train.shape)
print('Testing Features shape: ', test.shape)
#Creating custom columns 

#We have seen previous year rating is not avilable , lets create a new column 

#train['is_emp_new'] = train['previous_year_rating'].apply(lambda x : 1 if pd.isnull(x) else 0)
#test['is_emp_new'] = test['previous_year_rating'].apply(lambda x : 1 if pd.isnull(x) else 0)
#train= train.set_index('employee_id')
#test= test.set_index('employee_id')
sns.set(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(train.corr(), cmap=cmap, vmax=.3, center=0,annot=False,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr = train.corr()
corr = pd.DataFrame(corr)
corr.to_csv('corr.csv',index=False)
submission = pd.read_csv('../input/wns_sample.csv')
submission.head()
train.head()

#train= train.reindex(
 #   np.random.permutation(train.index))
train = train.drop('employee_id',axis=1)
test = test.drop('employee_id',axis=1)
y = train['is_promoted']
X = train.drop('is_promoted',axis=1)

X = X.astype(float)
y=y.astype(float)
from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
def run_lgb(X_train, X_test, y_train, y_test, test_df):
    params = {
        "objective" : "binary",
       "n_estimators":10000,
       "reg_alpha" : 0.1,
       "reg_lambda":0.1,
       "n_jobs":-1,
       "colsample_bytree":.8,
       "min_child_weight":8,
       "subsample":0.8715623,
       "min_data_in_leaf":100,
       "nthread":4,
       "metric" : "f1",
       "num_leaves" : 600,
       "learning_rate" : 0.01,
       "verbosity" : -1,
       "seed": 120,
       "max_bin":60,
       'max_depth':15,
       'min_gain_to_split':.0222415,
       'scale_pos_weight':2
    }
    
    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgval = lgb.Dataset(X_test, label=y_test)
    evals_result = {}
    model = lgb.train(params, lgtrain, 10000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=100, 
                      evals_result=evals_result,feval=lgb_f1_score)
    
    pred_test_y = model.predict(test_df, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
pred_test, model, evals_result = run_lgb(X_train, X_test, y_train, y_test, test)
print("LightGBM Training Completed...")
submission['is_promoted'] = pred_test
submission.to_csv('Sub_with_prob.csv',index=False)
submission['is_promoted'] = submission['is_promoted'].apply(lambda x : 1 if (x>=0.44) else 0)
submission['is_promoted'].sum()
submission.to_csv('second_sub.csv',index=False)
len(pred_test)
