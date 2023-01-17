import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
import time, sys
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv')
test = pd.read_csv('../input/Kaggle_Test_Dataset_v2.csv')

#Preview Data
print("Train Data :",train.shape)
print("Test Data :",test.shape)
train.sample(10)
train.describe(include = 'all')
print('-'*50,'\n','Train columns with null values:\n','-'*50, '\n',train.isnull().sum().sort_values(),'\n','-'*50,'\n')

print('Test/Validation columns with null values:\n','-'*50,'\n', test.isnull().sum().sort_values(),'\n','-'*50,'\n')

def numericalCol(x):            
     return x.select_dtypes(include=[np.number]).columns.values

def categoricaCol(x):            
     return x.select_dtypes(include=[np.chararray]).columns.values

# Delete the columns
def ColDelete(x,drop_column):
    x.drop(drop_column, axis=1, inplace = True)

def deleteNAValues(x):
    x.dropna(axis=0, how='any',inplace = True)
    
def categoricalToNumerical(dataset,categorical_columns_names,replace_value_map) :
    for categorical_column_name in categorical_columns_names:
        dataset[categorical_column_name] = dataset[categorical_column_name].map(replace_value_map).astype(int)
    return 
     
#   Categorical and Numerical Columns 
#   Number of unique values per column. Maybe some columns
#   are actually categorical in nature
numerical_columns_name = numericalCol(train)
categorical_columns_names = [ 'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk','stop_auto_buy', 'rev_stop', 'went_on_backorder']

train[categorical_columns_names].sample(10)

replace_value_map = {'Yes': 1, 'No': 0}
data_cleaner = [train,test]

for dataset in data_cleaner:   
    ### Replace NA with Mean value in "lean_time" column with help of Imputer module in SKlearn  
    dataset['lead_time'] = Imputer(strategy="mean").fit_transform(dataset['lead_time'].values.reshape(-1, 1))
    ### Drop All NA values
    deleteNAValues( dataset )
    ### Drop SKU columns
    ColDelete(dataset,"sku")
    ### Categorical Columns to Numerical
    categoricalToNumerical(dataset,categorical_columns_names,replace_value_map)

        

preProcessData = pd.concat(data_cleaner, axis = 'index')
preProcessData.shape
# Plot

plt.figure(figsize=(14,8))
plt.rc('font', size=14)          # controls default text sizes
preProcessData['went_on_backorder'].value_counts().plot(kind = 'pie',autopct='%1.2f%%')
plt.axis('equal')
plt.title("Only 0.72% records are belong to Yes")
plt.show()
print(" Target Result \n",preProcessData['went_on_backorder'].value_counts(),"\n\n")
plt.figure(figsize=(12,8))

sns.heatmap(preProcessData.corr())
plt.show()

print("Note :-A correlation of 0 means that no relationship exists between the two variables, whereas a correlation of 1 indicates a perfect positive relationship.")
# Rename went_on_backorder columns to target column
train.rename(columns={'went_on_backorder': 'target'}, inplace=True)
test.rename(columns={'went_on_backorder': 'target'}, inplace=True)
train.head()
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


h2o.init(nthreads = -1)

train_hf = h2o.H2OFrame(train)
test_hf = h2o.H2OFrame(test)

# For binary classification, response should be a factor
train_hf['target'] = train_hf['target'].asfactor()
test_hf['target'] = test_hf['target'].asfactor()
## Split Dataset - Target is true (all backorder ) as 'test' and non backorder as 'train'  

x_test_ac  = train_hf[train_hf['target'] == '1']
x_train_ac = train_hf[train_hf['target'] == '0']
x_test_ac.shape
x_train_ac.shape
X= list(range(0,22))
### Auto Encoder Model
autoencoder_model = H2OAutoEncoderEstimator(  activation="Tanh",
                                          hidden=[50,20,5,20,50],
                                          ignore_const_cols = False,
                                           stopping_metric='MSE', 
                                            stopping_tolerance=0.00001,
                                              epochs=200)



autoencoder_model.train(x =X, training_frame = x_train_ac)


print("MSE = ",autoencoder_model.mse())


train_rec_error = autoencoder_model.anomaly(x_train_ac)
test_rec_error = autoencoder_model.anomaly(x_test_ac)

train_rec_error = train_rec_error.as_data_frame()
train_rec_error['id'] = train_rec_error.index.values + 1
train_rec_error['target'] = 0

count_train_records = len(train_rec_error) 
test_rec_error = test_rec_error.as_data_frame()
test_rec_error['id'] = test_rec_error.index.values + count_train_records + 1
test_rec_error['target'] = 1

rec_error = pd.concat([train_rec_error,test_rec_error], axis = 'index')


# Plot the Scatter graph

sns.lmplot('id', 'Reconstruction.MSE', hue='target', data=rec_error, fit_reg=False);
plt.axvline(x=count_train_records,linewidth=1)
plt.show()

rec_error[rec_error['target'] == 1].plot(kind='scatter', x='id', y='Reconstruction.MSE',c='red',marker='x', label='Back-Order')
rec_error[rec_error['target'] == 0].plot(kind='scatter', x='id', y='Reconstruction.MSE',c='blue',marker='o', label='Normal Order')
plt.legend(loc='upper right')
plt.show()

rec_error_hf = autoencoder_model.anomaly(test_hf)
rec_error_hf['actual'] = test_hf['target']
rec_error_hf = rec_error_hf.as_data_frame()
rec_error_hf['id'] = rec_error_hf.index.values
rec_error_hf['predict'] = 0

#predict = [1 if e > threshold else 0 for e in rec_error_fh.reconstruction_error.values]

rec_error_hf['predict'][rec_error_hf['Reconstruction.MSE']  >0.079] = 1


df_confusion = pd.crosstab(rec_error_hf['actual'], rec_error_hf['predict'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)
print("---"*25)
print("\n % \n")
print(pd.crosstab(rec_error_hf['actual'], rec_error_hf['predict'], rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum()))



from sklearn.metrics import confusion_matrix

LABELS = LABELS = ['Normal Order', 'Back-Order']
conf_matrix = confusion_matrix(rec_error_hf['actual'], rec_error_hf['predict'])
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
#### Shutdown H2o server
#h2o.shutdown()    
#h2o.cluster().shutdown()