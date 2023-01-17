import pandas as pd

import numpy as np



SEED = 42 # for reproducability



import warnings

warnings.filterwarnings('ignore') # ignoring any warnings
TRAIN_PATH = '../input/home-data-for-ml-course/train.csv'



train = pd.read_csv(TRAIN_PATH)
train.shape
train.head()
cat_columns = list(train.select_dtypes(include='object').columns)

print(len(cat_columns), 'Categorical columns')

print(train.shape[1] - len(train.select_dtypes(include='object').columns), 'Numerical columns')
train['SalePrice'].describe()
import matplotlib.pyplot as plt



# Histogram

plt.figure(figsize=(6,6))

plt.hist(train['SalePrice'], bins=30, color='salmon')

plt.title('Sale Price Histogram', fontdict={'fontsize': 18}, pad = 20)

plt.xlabel('Price (US Dollars)')

plt.show()
# Pie Chart



plt.figure(figsize=(5,5))

labels = ['< $200,000', '> $200,000']

sizes = [len(train[train['SalePrice'] < 200000]), 

         len(train[train['SalePrice'] > 200000])]



plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=[0,0.08], startangle=90)

plt.title('Price Distribution', fontdict={'fontsize': 16})

plt.axis('equal')

plt.show()
print('There are', len(train[train['SalePrice'] == 200000]), 'Houses that are exactly $200,000')
corr_matrix = train.corr()

corr_matrix['SalePrice'].sort_values(ascending=False).head(15)
train = train[train['SalePrice'] != 200000]
target = train['SalePrice'].apply(lambda x: 1 if x > 200000 else 0)
features = train.drop(columns=['SalePrice','Id'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,

                                                 target,

                                                 test_size=0.2,

                                                 random_state=SEED)
x_train[x_train.columns[x_train.isna().any()].tolist()].info()
# Drop columns that don't have much data



columns_to_drop = ['MiscFeature', 'PoolQC', 'Fence', 'FireplaceQu', 'Alley', 'LotFrontage']



x_train = x_train.drop(columns=columns_to_drop)

x_test = x_test.drop(columns=columns_to_drop) # make sure to do same thing to test set as well



for i in columns_to_drop: # remove any dropped column from the category columns list, since I'm going to use it later

  if i in cat_columns:

    cat_columns.remove(i)
columns_to_impute = ['MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','Electrical','BsmtFinType1',

                     'BsmtFinType2','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond'] 
# Impute Garage Year Built nulls with mean



x_train['GarageYrBlt'] = x_train['GarageYrBlt'].fillna(np.mean(x_train['GarageYrBlt']))

x_test['GarageYrBlt'] = x_test['GarageYrBlt'].fillna(np.mean(x_train['GarageYrBlt']))



# Impute 'MasVnrArea' nulls with 0.0



x_train['MasVnrArea'] = x_train['MasVnrArea'].fillna(0.0)

x_test['MasVnrArea'] = x_test['MasVnrArea'].fillna(0.0)
# Imputing



from sklearn.impute import SimpleImputer



imp = SimpleImputer(strategy='most_frequent').fit(x_train)



x_train_imp = pd.DataFrame(imp.transform(x_train), index = x_train.index, columns = x_train.columns)

x_test_imp = pd.DataFrame(imp.transform(x_test), index = x_test.index, columns = x_test.columns)
import category_encoders as ce
# Encoding categorical values



enc = ce.target_encoder.TargetEncoder(cols=cat_columns).fit(x_train_imp,y_train)



x_train_enc = enc.transform(x_train_imp)

x_test_enc = enc.transform(x_test_imp)
# Scaling the data



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_enc), index=x_train_enc.index, columns=x_train_enc.columns)

x_test_scaled = pd.DataFrame(scaler.transform(x_test_enc), index=x_test_enc.index, columns=x_test_enc.columns)
from sklearn.metrics import roc_auc_score



preds = [] # holds test predictions

auc_list = [] # holds auc scores



# Function to train model, make test data predictions, and print AUC score

def fit_and_score(model):

  model.fit(x_train_scaled,y_train)

  temp_preds = np.array(model.predict_proba(x_test_scaled))[:,1]

  preds.append(temp_preds)

  print(type(model), 'AUC score: ', roc_auc_score(y_test,temp_preds))

  auc_list.append(round(roc_auc_score(y_test,temp_preds),5))
from xgboost import XGBClassifier



model_1 = XGBClassifier(random_state=SEED)



fit_and_score(model_1)
from sklearn.linear_model import LogisticRegression



model_2 = LogisticRegression(random_state=SEED)



fit_and_score(model_2)
from sklearn.svm import SVC



model_3 = SVC(random_state = SEED, probability=True)



fit_and_score(model_3)
avg_preds = np.mean(preds,axis=0)

auc_list.append(round(roc_auc_score(y_test,avg_preds),5))

print('Combined AUC score: ', roc_auc_score(y_test,avg_preds))
model_list = ['XGBoost','Logistic Regression','Support Vector Machine','Average Ensemble']





models_ranked_df = pd.DataFrame(data={'Model': model_list,'AUC Score': auc_list})

models_ranked_df.sort_values(by='AUC Score', ascending = False)
from sklearn import metrics

import seaborn as sn
### Plot confusion matrix heat map



data = metrics.confusion_matrix(y_test, [int(round(i)) for i in avg_preds]) # Need to round each probability to get class

df_cm = pd.DataFrame(data, columns = [0,1], index = [0,1])

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (7,5))

sn.set(font_scale=1.3)

sn.heatmap(df_cm, cmap = "Blues", annot = True, annot_kws = {"size": 16})
print(metrics.classification_report(y_test,[int(round(i)) for i in avg_preds]))
# Plot ROC AUC graph



fpr, tpr, threshold = metrics.roc_curve(y_test, avg_preds)

roc_auc = metrics.auc(fpr, tpr)



plt.figure(figsize=(7,7))

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
features_ranked_df = pd.DataFrame(data={'feature': x_train_scaled.columns, 

                                        'importance': model_1.feature_importances_}

                                  ).sort_values(by='importance', ascending = False)
features_ranked_df[:10]