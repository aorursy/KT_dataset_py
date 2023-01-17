import pandas as pd
import numpy as np
import matplotlib as pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
%matplotlib inline
# to see all the comands result in a single kernal 
%load_ext autoreload
%autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# to increase no. of rows and column visibility in outputs
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
train = pd.read_csv(r'../input/janatahack-crosssell-prediction/train.csv')
test = pd.read_csv(r'../input/janatahack-crosssell-prediction/test.csv')
sample_submmission = pd.read_csv(r'../input/janatahack-crosssell-prediction/sample_submission.csv')
train.shape
test.shape
sample_submmission.shape
test.info()
%matplotlib inline
import matplotlib.pyplot as plt
train.hist(bins=50, figsize=(20,15))
plt.show()
train.head()
train.isna().sum().sum()
test.isna().sum().sum()
numeric_data = train.select_dtypes(include=np.number) # select_dtypes selects data with numeric features
numeric_col = numeric_data.columns 

print("Numeric Features:")
print(numeric_data.head())
print("===="*20)
categorical_data = train.select_dtypes(exclude=np.number) # we will exclude data with numeric features
categorical_col = categorical_data.columns                                                                              # we will store the categorical features in a variable

print("Categorical Features:")
print(categorical_data.head())
print("===="*20)
train['Response'].value_counts()/len(train)
train['Gender'].value_counts()
train['Vehicle_Age'].value_counts()
train['Vehicle_Damage'].value_counts()
le = LabelEncoder()
train['Vehicle_Age'] = le.fit_transform(train['Vehicle_Age'])
train['Gender'] = le.fit_transform(train['Gender'])
train['Vehicle_Damage'] = le.fit_transform(train['Vehicle_Damage'])
test['Gender'] = le.fit_transform(test['Gender'])
test['Vehicle_Age'] = le.fit_transform(test['Vehicle_Age'])
test['Vehicle_Damage'] = le.fit_transform(test['Vehicle_Damage'])
train.head()
train.columns
col_1=['Gender', 'Age','Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
# categorical column 
cat_col=['Gender','Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
train['Region_Code']=train['Region_Code'].astype(int)
test['Region_Code']=test['Region_Code'].astype(int)
train['Policy_Sales_Channel']=train['Policy_Sales_Channel'].astype(int)
test['Policy_Sales_Channel']=test['Policy_Sales_Channel'].astype(int)

X = train[col_1]
y = train['Response']
X_test = test[col_1]
y_valid_pred = 0
y_test_pred = 0
K = 4
kf = KFold(n_splits = K, random_state=150307, shuffle = True)
OPTIMIZE_ROUNDS = False
# Run CV
model = CatBoostClassifier()

for i, (train_i, test_i) in enumerate(kf.split(train)):
    
    # Create data for this fold
    y_train, y_eval = y.iloc[train_i], y.iloc[test_i]
    X_train, X_eval = X.iloc[train_i,:], X.iloc[test_i,:]
    print( "\nFold ", i)
    
    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        fit_model = model.fit( X_train, y_train, 
                               eval_set=[X_eval, y_eval],
                               use_best_model=True
                             )
        print( "  N trees = ", model.tree_count_ )
    else:
        fit_model = model.fit( X_train, y_train,cat_features=cat_col,eval_set=(X_eval, y_eval),early_stopping_rounds=40,verbose=200 )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_eval)[:,1]
    print( "  ROCAUC = ", roc_auc_score(y_eval, pred) )
    y_valid_pred.iloc[test_i] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(test[col_1])[:, 1]
    
y_test_pred /= K  # Average test set predictions

print( "\nROCAUC for full training set:" )
roc_auc_score(y, y_valid_pred)
feat_importances = pd.Series(fit_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(15).plot(kind='barh')
#feat_importances.nsmallest(20).plot(kind='barh')
plt.show()
# Create submission file
submmission = pd.DataFrame()
submmission['id'] = test['id'].values
submmission['Response'] = y_test_pred
submmission.to_csv('cat_submitfinal.csv', float_format='%.6f', index=False)
submmission.head()
# 85.86 on Public leaderboard