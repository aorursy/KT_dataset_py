#Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn import base
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
!pip install category_encoders
!pip install rfpimp
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,confusion_matrix
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
#Read The Dataset

url = "https://drive.google.com/uc?id=1TiEhIjpjB6KUxvqqgVeps9SkKTff1hvY"

data = pd.read_csv(url)
data.head()
data.shape
data.info()
#Inserting KEY

#In this data, we define a column with a unique value of 65188 as a key.

data.insert(0,"KEY",data.CLNHGVS)
data['KEY'].nunique()==len(data)
data.describe()
%matplotlib inline
# Histogram of the target categories
def histogram(df,feature):
    ncount = len(df)
    ax = sns.countplot(x = feature, data=df ,palette="hls")
    sns.set(font_scale=1)
    ax.set_xlabel('Target Segments')
    plt.xticks(rotation=90)
    ax.set_ylabel('Number of Observations')
    fig = plt.gcf()
    fig.set_size_inches(12,5)
    # Make twin axis
    ax2=ax.twinx()
    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()
    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequency [%]')
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.2f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))
    # Fix the frequency range to 0-100
    ax2.set_ylim(0,100)
    ax.set_ylim(0,ncount)
    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    ax2.grid(None)
    plt.title('Histogram of Binary Target Categories', fontsize=20, y=1.08)
    plt.show()
    plt.savefig('target_histogram.png')
    del ncount, x, y
histogram(data,"CLASS")
def MissingUniqueStatistics(df):
  
  total_entry_list = []
  total_missing_value_list = []
  missing_value_ratio_list = []
  data_type_list = []
  unique_values_list = []
  number_of_unique_values_list = []
  variable_name_list = []
  
  for col in df.columns:

    variable_name_list.append(col)
    missing_value_ratio = round((df[col].isna().sum()/len(df[col])),4)
    total_entry_list.append(df[col].shape[0] - df[col].isna().sum())
    total_missing_value_list.append(df[col].isna().sum())
    missing_value_ratio_list.append(missing_value_ratio)
    data_type_list.append(df[col].dtype)
    unique_values_list.append(list(df[col].unique()))
    number_of_unique_values_list.append(len(df[col].unique()))

  data_info_df = pd.DataFrame({'Variable':variable_name_list,'#_Total_Entry':total_entry_list,\
                           '#_Missing_Value':total_missing_value_list,'%_Missing_Value':missing_value_ratio_list,\
                           'Data_Type':data_type_list,'Unique_Values':unique_values_list,\
                           '#_Uniques_Values':number_of_unique_values_list})
  
  return data_info_df.sort_values(by="#_Missing_Value",ascending=False)
data_info = MissingUniqueStatistics(data)
data_info = data_info.set_index("Variable")
data_info
drop_list = list(data_info[data_info["%_Missing_Value"] >= 0.99].index)

data.drop(drop_list,axis = 1,inplace = True) # --> MAIN DF CHANGED
data["Protein_position"].unique() , data["CDS_position"].unique()  , data["cDNA_position"].unique()
def value_correction(df,columns):
  
  for col in columns:

    value_correction = pd.DataFrame(df[col][df[col].notnull()].str.split("-").tolist(),columns=["X","Y"])
    value_correction["X"][value_correction["X"]=="?"] = value_correction["Y"]
    key = df[[col,"KEY"]][df[col].notnull()]["KEY"]

    counter = 0

    for i in key.index:

      df[col][i] = value_correction["X"][counter]
      counter += 1

    df[col] = df[col].astype(float)
  return df
data = value_correction(data,["CDS_position","cDNA_position","Protein_position"]) # --> MAIN DF CHANGED
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(data.corr(), annot=True)
plt.show()
data.drop(["CDS_position","cDNA_position"],axis = 1, inplace = True) # --> MAIN DF CHANGED
data[["EXON","INTRON"]][data["INTRON"].notnull() & data["EXON"].notnull()].head(10)
def convert_to_float(df,columns):
  
  for col in columns:
       
    convert_to_float = pd.DataFrame(df[col][df[col].notnull()].str.split("/").tolist(),columns=["Numerator","Denominator"])
    convert_to_float = convert_to_float.astype("float")
    convert_to_float["Result"] = convert_to_float["Numerator"] / convert_to_float["Denominator"]
    key =df[[col,"KEY"]][df[col].notnull()]["KEY"]

    counter = 0
    for i in key.index:

      df[col][i] = convert_to_float["Result"][counter]
      counter += 1
    df[col] = df[col].astype(float)

  return df
data = convert_to_float(data,["INTRON","EXON"]) # --> MAIN DF CHANGED
MissingUniqueStatistics(data)
msno.bar(data,color='#79ccb3',sort='descending')
plt.show()
msno.matrix(data,color=(0.45,0.45,0.64),figsize=(27, 10), width_ratios=(10, 0))
plt.show()
msno.dendrogram(data);
msno.heatmap(data)
plt.show()
data["EXON"][data["EXON"].isnull()]
data["INTRON"][data["INTRON"].notnull()]
data["EXON"][data["EXON"].isnull()] = data["INTRON"][data["INTRON"].notnull()] # --> MAIN DF CHANGED
data.drop(["INTRON"], axis = 1, inplace = True) # --> MAIN DF CHANGED
data_info = MissingUniqueStatistics(data)
data_info = data_info.set_index("Variable")
data_info
#Creating new column for add of variable type
data_info["Variable_Type"] = ["Ordinal","Ordinal","Nominal","Nominal","Nominal","Nominal","Nominal","Continuous","Nominal",
                              "Continuous","Continuous","Continuous","Continuous","Continuous","Nominal","Nominal","Nominal",
                              "Nominal","Nominal","Nominal","Nominal","Ordinal","Nominal","Nominal","Nominal","Nominal","Nominal","Nominal",
                              "Continuous","Continuous","Continuous","Nominal","Nominal","Cardinal","Ordinal"]
data_info
# 1- Row Uniqueness (Drop Duplicates) 
len(data_info.index) == data_info.shape[0]
# 2- Column Uniqueness (Drop Singletons)
numerical_columns = list(data_info.loc[(data_info.loc[:,"Variable_Type"]=="Cardinal") |
                                       (data_info.loc[:,"Variable_Type"]=="Continuous")].index)
len(numerical_columns), numerical_columns
categorical_columns = list(data_info.loc[(data_info.loc[:,"Variable_Type"]=="Nominal") |
                                       (data_info.loc[:,"Variable_Type"]=="Ordinal")].index)
len(categorical_columns), categorical_columns
def ZeroVarianceFinder(df, numerical_columns):
  
  import pandas as pd
  import numpy as np

  zerovariance_numerical_features=[]
  for col in numerical_columns:
      try:
          if pd.DataFrame(df[col]).describe().loc['std'][0] == 0.00 or \
          np.isnan(pd.DataFrame(df[col]).describe().loc['std'][0]):
              zerovariance_numerical_features.append(col)
      except:
          print("Error:",col)
  return zerovariance_numerical_features
zerovariance_numerical_features = ZeroVarianceFinder(data,numerical_columns)
zerovariance_numerical_features
singleton_categorical_features=[]
for col in categorical_columns:
    if len(data[col].unique()) <= 1:
        singleton_categorical_features.append(col)
len(singleton_categorical_features), singleton_categorical_features
y = data.loc[:,"CLASS"]
x1 = data.iloc[:,1:15]
x2 = data.iloc[:,16:]
x = pd.concat([x1,x2],axis = 1)
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.33,random_state=42)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
histogram(X_train,Y_train)
histogram(X_test,Y_test)
df_train = data.copy()
numerical_columns_remains = numerical_columns
        
sparse_columns = []
for col in numerical_columns_remains:
    if (df_train[col].quantile(0.01)==df_train[col].quantile(0.25)==df_train[col].mode()[0]):
        sparse_columns.append(col)

sparse_columns_2 = []
for col in numerical_columns_remains:
    if (df_train[col].quantile(0.01)==df_train[col].quantile(0.25)):
        sparse_columns_2.append(col)

len(numerical_columns_remains), len(sparse_columns), len(sparse_columns_2)
from pylab import rcParams

def box_plot(x,y,data):

  rcParams['figure.figsize'] = 20, 10
  fig, axs = plt.subplots(2,5)
  plt.tight_layout()
  fig.subplots_adjust(top=0.7)
  sns.set(style="ticks", palette="rainbow")

  j = 0
  k = 0
  for i in range(len(y)):
    sns.boxplot(x=x, y=y[i], data=data,ax=axs[j,k])
    if(k==4):
      k = 0
      j += 1
    else:
      k += 1

  plt.tight_layout()
  plt.show()

box_plot(Y_train,numerical_columns,X_train)
"""
Algorithm 'HER(Hard-Edges Method)' applies induction to the elements of a value line which are:

    - lower than the 1th quantile to that quantile and
    - upper than the 99th quantile to that quantile.
    
Main aim is to diminish negative effects of outlier values on analytical operations being performed.
"""

def HardEdgeReduction(df,numerical_columns,sparse_columns,upper_quantile=0.99,lower_quantile=0.01):
    
    import pandas as pd

    import psutil, os, gc, time
    print("HardEdgeReduction process has began:\n")
    proc = psutil.Process(os.getpid())
    gc.collect()
    mem_0 = proc.memory_info().rss
    start_time = time.time()
    
    # Do outlier cleaning in only one loop
    epsilon = 0.0001 # for zero divisions

    # Define boundaries that we will use for Reduction operation

    df_outlier_cleaned = df.copy()


    print("Detected outliers will be replaced with edged quantiles/percentiles: 1% and 99%!\n")
    print("Total number of rows is: %s\n"%df_outlier_cleaned.shape[0])

    outlier_boundries_dict={}

    for col in numerical_columns:

        if col in sparse_columns:

            # First ignore the 'sparse' data points:
            nonsparse_data = pd.DataFrame(df_outlier_cleaned[df_outlier_cleaned[col] !=\
                                                             df_outlier_cleaned[col].mode()[0]][col]) 
            
            #we used only mode to catch sparse points, since we know/proved it is enough to do that.

            # Find Outlier Thresholds:
            # Note: All columns are right-skewed
            # For lower threshold (left-hand-side)
            if nonsparse_data[col].quantile(lower_quantile) < df_outlier_cleaned[col].mode()[0]: #Unexpected case
                lower_bound_sparse = nonsparse_data[col].quantile(lower_quantile)
            else:
                lower_bound_sparse = df_outlier_cleaned[col].mode()[0]
            
            # For upper threshold (right-hand-side)
            if nonsparse_data[col].quantile(upper_quantile) < df_outlier_cleaned[col].mode()[0]: #Unexpected case
                upper_bound_sparse = df_outlier_cleaned[col].mode()[0]
            else:
                upper_bound_sparse = nonsparse_data[col].quantile(upper_quantile)

            outlier_boundries_dict[col]=(lower_bound_sparse,upper_bound_sparse)

            # Inform user about the cardinality of Outlier existence:
            number_of_outliers = len(df_outlier_cleaned[(df_outlier_cleaned[col] < lower_bound_sparse) |\
                                                        (df_outlier_cleaned[col] > upper_bound_sparse)][col])
            print("Sparse: Outlier number in {} is equal to: ".format(col),round(number_of_outliers/(nonsparse_data.shape[0] -
                                                                                       nonsparse_data.isnull().sum()),2))

            # Replace Outliers with Edges --> 1% and 99%:
            if number_of_outliers > 0:

                # Replace 'left-hand-side' outliers with its 1% quantile value
                df_outlier_cleaned.loc[df_outlier_cleaned[col] < lower_bound_sparse,col] = lower_bound_sparse - epsilon # --> MAIN DF CHANGED

                # Replace 'right-hand-side' outliers with its 99% quantile value
                df_outlier_cleaned.loc[df_outlier_cleaned[col] > upper_bound_sparse,col] = upper_bound_sparse + epsilon # --> MAIN DF CHANGED

        else:
            # Find Edges:
            number_of_outliers = len(df_outlier_cleaned[(df_outlier_cleaned[col] < \
                                                         df_outlier_cleaned[col].quantile(lower_quantile))|\
                                                        (df_outlier_cleaned[col] > \
                                                         df_outlier_cleaned[col].quantile(upper_quantile))]\
                                     [col])
            print("Other: Outlier number in {} is equal to: ".format(col),round(number_of_outliers/(df[col].shape[0] -
                                                                                       df[col].isnull().sum()),2)) 

            # Replace 'Standard' outliers:
            if number_of_outliers > 0:
                # Replace all outliers with its %99 quartile
                lower_bound_sparse = df_outlier_cleaned[col].quantile(lower_quantile)
                df_outlier_cleaned.loc[df_outlier_cleaned[col] < \
                                       lower_bound_sparse,col] \
                = lower_bound_sparse  - epsilon

                upper_bound_sparse = df_outlier_cleaned[col].quantile(upper_quantile)
                df_outlier_cleaned.loc[df_outlier_cleaned[col] > \
                                       upper_bound_sparse,col] \
                = upper_bound_sparse  + epsilon

            outlier_boundries_dict[col]=(lower_bound_sparse,upper_bound_sparse)


    print('HardEdgeReduction process has been completed!')
    print("--- in %s minutes ---" % ((time.time() - start_time)/60))

    return df_outlier_cleaned, outlier_boundries_dict


X_train, outlier_boundries_dict = HardEdgeReduction(X_train,numerical_columns,sparse_columns)
outlier_boundries_dict
# Do outlier cleaning in only one loop
epsilon = 0.0001 # for zero divisions

# Define boundaries that we will use for Reduction operation
upper_quantile = 0.99
lower_quantile = 0.01

df_test_outlier_cleaned = X_test.copy()

print("Detected outliers will be replaced with edged quantiles/percentiles: 1% and 99%!\n")
print("Total number of rows is: %s\n"%df_test_outlier_cleaned.shape[0])

for col in numerical_columns_remains:

      lower_bound = outlier_boundries_dict[col][0]
      upper_bound = outlier_boundries_dict[col][1]
        
      # Inform user about the cardinality of Outlier existence:
      number_of_outliers = len(df_test_outlier_cleaned[(df_test_outlier_cleaned[col] < lower_bound) |\
                                                        (df_test_outlier_cleaned[col] > upper_bound)][col])
      print("Outlier number in {} is equal to: ".format(col), round(number_of_outliers/
            (df_test_outlier_cleaned[col].shape[0] - df_test_outlier_cleaned[col].isnull().sum()),2))

      # Replace Outliers with Edges --> 1% and 99%:
      if number_of_outliers > 0:

          # Replace 'left-hand-side' outliers with its 1% quantile value
          df_test_outlier_cleaned.loc[df_test_outlier_cleaned[col] < lower_bound,col] = lower_bound  - epsilon # --> MAIN DF CHANGED
          
          # Replace 'right-hand-side' outliers with its 99% quantile value
          df_test_outlier_cleaned.loc[df_test_outlier_cleaned[col] > upper_bound,col] = upper_bound  + epsilon # --> MAIN DF CHANGED
        

box_plot(Y_train,numerical_columns,X_train)
X_test = df_test_outlier_cleaned
Zero_MR_variables_list = list(data_info[data_info['%_Missing_Value']==0].index)
Low_MR_variables_list = list(data_info[(data_info['%_Missing_Value']>0)&
                                       (data_info['%_Missing_Value']<=0.05)].index)
Moderate_MR_variables_list = list(data_info[(data_info['%_Missing_Value']>0.05)&\
                                                      (data_info['%_Missing_Value']<=0.25)].index)
High_MR_variables_list = list(data_info[(data_info['%_Missing_Value']>0.25)&\
                                                  (data_info['%_Missing_Value']<=0.50)].index)
Extreme_MR_variables_list = list(data_info[(data_info['%_Missing_Value']>0.50)&
                                           (data_info['%_Missing_Value']<=0.95)].index)
Drop_MR_variables_list = list(data_info[data_info['%_Missing_Value']>0.95].index)

len(Zero_MR_variables_list),len(Low_MR_variables_list),len(Moderate_MR_variables_list),len(High_MR_variables_list),\
len(Extreme_MR_variables_list),\
len(Zero_MR_variables_list)+len(Low_MR_variables_list)+len(Moderate_MR_variables_list)+len(High_MR_variables_list)+\
len(Extreme_MR_variables_list) == len(data_info)
Low_MR_variables_list
def SimpleImputer(df,data_info,variable_list):
  for col in variable_list:
    
    if(col in numerical_columns):
      
      print("Total null values: {}".format(df[[str(col)]].isnull().sum()))

      average = float(df[col].mean())
      std = float(df[col].std())
      count_nan = int(df[col].isnull().sum())
      rand = np.random.normal(loc=average,scale=std,size =count_nan)
      slice_col = pd.Series(df[col].copy())
      slice_col[pd.isnull(slice_col)] = rand
      df[col] = slice_col

      print("Numerical variable {} have been imputed.".format(col))

    else:

      print("Total null values: {}".format(df[[str(col)]].isnull().sum()))
      df.loc[df.loc[:,col].isnull(),col] = np.random.choice(sorted(list(df.loc[:,col].dropna().unique())),
                                                            size=int(df.loc[df.loc[:,col].isnull(),col].shape[0]),
                                                            p=[pd.Series(df.groupby(col).size()/df.loc[:,col].dropna().shape[0]).iloc[i] for i in 
                                                               np.arange(0,len(df.loc[:,col].dropna().unique()))])
      
      print("Categorical variable {} have been imputed.".format(col))
SimpleImputer(X_train, data_info, Low_MR_variables_list)
SimpleImputer(X_test,data_info,Low_MR_variables_list)
MissingUniqueStatistics(X_train.loc[:,Low_MR_variables_list])
MissingUniqueStatistics(X_test.loc[:,Low_MR_variables_list])
class KFoldTargetEncoderTrain(base.BaseEstimator,
                               base.TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self,X):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)
        
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold,
                   shuffle = False, random_state=2020)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = \
            X_val[self.colnames].map(X_tr.groupby(self.colnames)
                                     [self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace = True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'\
                  .format(col_mean_name,self.targetName,
                          np.corrcoef(X[self.targetName].values,
                                      encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X
def StringConverterTrain(df,target_name,variable_list):
    for col in variable_list:
      targetc = KFoldTargetEncoderTrain(col,target_name,n_fold=4)
      new_train = targetc.fit_transform(df)
    return new_train
nominal_variable = list(data_info[data_info["Variable_Type"]=="Nominal"].index)
nominal_lst = [item for item in Moderate_MR_variables_list+High_MR_variables_list+Extreme_MR_variables_list if item in nominal_variable]
nominal_lst
df_trial = pd.concat([X_train,Y_train],axis=1).copy()
df_output_train = StringConverterTrain(df=df_trial,target_name="CLASS",variable_list=nominal_lst)
for item in nominal_lst:
  print(df_output_train.loc[:,[item+"_Kfold_Target_Enc"]].isnull().sum())
for item in nominal_lst:
  X_train[item] = df_output_train[item+"_Kfold_Target_Enc"]
nominal_encoding_variable_lst =["Consequence","REF","ALT","CLNDISDB","CLNDN","Allele","Feature","SYMBOL"]

df_encoding = pd.concat([X_train,Y_train],axis=1).copy()
df_encoding_train = StringConverterTrain(df=df_encoding,target_name="CLASS",variable_list=nominal_encoding_variable_lst)
for item in nominal_encoding_variable_lst:
  print(df_encoding_train.loc[:,[item+"_Kfold_Target_Enc"]].isnull().sum())
for item in nominal_encoding_variable_lst:
  X_train[item] = df_encoding_train[item+"_Kfold_Target_Enc"]
MissingUniqueStatistics(X_train)
df_output_test = X_test.copy()
mean_of_target = df_output_train['CLASS'].copy().mean()
target_mean_list = nominal_lst                                                 
for col in target_mean_list:
    df_output_test[col] = df_output_test[col].map(df_output_train.groupby(col)[col+'_Kfold_Target_Enc'].mean())
    df_output_test[col].fillna(mean_of_target, inplace = True)
for item in nominal_lst:
  print(df_output_test.loc[:,[item]].isnull().sum())
X_test[nominal_lst] = df_output_test[nominal_lst]
X_test_encoder = X_test.copy()
mean_of_target = df_encoding_train['CLASS'].copy().mean()
target_mean_list = nominal_encoding_variable_lst                                                 
for col in target_mean_list:
    X_test_encoder[col+'_Kfold_Target_Enc'] = X_test_encoder[col].map(df_encoding_train.groupby(col)[col+'_Kfold_Target_Enc'].mean())
    X_test_encoder[col+'_Kfold_Target_Enc'].fillna(mean_of_target, inplace = True)
for item in nominal_encoding_variable_lst:
  X_test[item] = X_test_encoder[item+"_Kfold_Target_Enc"]
  
MissingUniqueStatistics(X_test[nominal_encoding_variable_lst])
X_train.drop("CLNHGVS", axis = 1, inplace = True)
X_test.drop("CLNHGVS", axis = 1, inplace = True)
def MBI(df,columns,train_or_test,lst_numerical):

  data_binary_encoded=df.copy()
  le=LabelEncoder()

  for col in columns:
    
    if(train_or_test == "test"):

      le.fit(X_train[col].copy().astype(str))
      data_binary_encoded[col]=le.transform(df[col].copy().astype(str))

    else:

      data_binary_encoded[col] = le.fit_transform(df[col].copy().astype(str))

  data_scaled=data_binary_encoded.copy()

  for col in numerical_columns:

    scaler = StandardScaler()

    if(train_or_test == "test"):

      scaler.fit(np.array(X_train.loc[:,col]).reshape(-1,1))
      data_scaled.loc[:,col] = scaler.transform(np.array(data_scaled.loc[:,col]).reshape(-1,1))

    else:
      data_scaled.loc[:,col] = scaler.fit_transform(np.array(data_scaled.loc[:,col]).reshape(-1,1))

  for col in lst_numerical:

    target_dropped_fullcases = data_scaled.drop(col,axis=1).loc[:,list(set(Zero_MR_variables_list+Low_MR_variables_list)-
                                                                                  set(["CLASS","KEY","CLNHGVS"]))].copy()
    
    target = data_scaled.loc[:,col]
    null_mask = target.isna()
    print(col)

    if(col in numerical_columns):
      
      mlp = MLPRegressor(hidden_layer_sizes=(100,10,),
                        activation='tanh',
                        solver='adam',
                        learning_rate='adaptive',
                        max_iter=1000,
                        learning_rate_init=0.01,
                        alpha=0.01,
                        early_stopping = False)
    else:
      mlp = MLPClassifier(hidden_layer_sizes=(100,10,),
                        activation='tanh',
                        solver='adam',
                        learning_rate='adaptive',
                        max_iter=1000,
                        learning_rate_init=0.01,
                        alpha=0.01,
                        early_stopping = False)
    
    mlp.fit(target_dropped_fullcases[~null_mask],target[~null_mask])
    data_scaled.loc[null_mask,col] = mlp.predict(target_dropped_fullcases[null_mask])

  print(data_scaled.loc[:,lst_numerical].isnull().sum());
  return data_scaled
lst_numerical = [item for item in Moderate_MR_variables_list if item in numerical_columns]
lst_numerical.append("SIFT")
lst_numerical.append("PolyPhen")
lst_numerical
encoding_col_list =["CHROM","CLNVC","Feature_type","BIOTYPE","IMPACT"]

X_train_scaled = MBI(X_train,encoding_col_list,"train",lst_numerical)
X_train_scaled
X_test_scaled = MBI(X_test,encoding_col_list,"test",lst_numerical)
def Label_Encoder(df,columns,train_or_test):
  le = LabelEncoder()
  for col in columns:
    if(train_or_test == "test"):

      le.fit(X_train_scaled[col].copy().astype(str))
      df[col] = le.transform(df[col].copy().astype(str))

    else:
      df[col] = le.fit_transform(df[col].copy().astype(str))

  return df
X_test_scaled = Label_Encoder(X_test_scaled,["SIFT","PolyPhen"],"test")
X_train_scaled = Label_Encoder(X_train_scaled,["SIFT","PolyPhen"],"train")
rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train_scaled, Y_train)

features = X_train_scaled.columns
importances = rnd_clf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(20,10))
feat_importances = pd.Series(importances, index=features)
feat_importances.nlargest(len(indices)).plot(kind='bar',color = '#79CCB3');
# Creating an empty Dataframe with Scores
df_accur_roc_score = pd.DataFrame(columns=['Roc_Auc_Score'])
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)
clf.fit(X_train_scaled, Y_train)

y_preds = clf.predict_proba(X_test_scaled)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score.loc['Logistic_regression'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='Logistic R. AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=150,random_state=0,learning_rate=0.25,eta=0.4,booster="gbtree",base_score=0.8,colsample_bylevel=0.9009229642844634,gamma=0.49967765132613584,
                        max_depth=6,min_child_weight=7,reg_lambda=0.27611902459972926,subsample=0.9300916052594785)

xgb_model.fit(X_train_scaled, Y_train)

y_preds = xgb_model.predict_proba(X_test_scaled)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score.loc['XGBoost_Classifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='XGBoost Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_scaled,Y_train)
y_preds = knn.predict_proba(X_test_scaled)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score.loc['KNeighborsClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='KNeighbors Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from sklearn.tree import DecisionTreeClassifier

reg_dtr = DecisionTreeClassifier(random_state=0)
reg_dtr.fit(X_train_scaled,Y_train)

y_preds = reg_dtr.predict_proba(X_test_scaled)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score.loc['DecisionTreeClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='DecisionTree Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
        max_depth=6,
        n_estimators=100,random_state=0,learning_rate=0.1,eta=0.4,base_score=0.8,colsample_bylevel=0.9009229642844634,gamma=0.49967765132613584,
                        min_child_weight=9,reg_lambda=0.27611902459972926,subsample=0.9300916052594785,min_samples_split=2,min_samples_leaf=0.1)

lgbm.fit(X_train_scaled, Y_train)

y_preds = lgbm.predict_proba(X_test_scaled)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score.loc['LGBMClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='LGBM Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                   max_depth=7, random_state=0).fit(X_train_scaled, Y_train)

y_preds = gradient_boosting_clf.predict_proba(X_test_scaled)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score.loc['GradientBoostingClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='GradientBoosting Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


hist_gradient_boosting_clf = HistGradientBoostingClassifier(learning_rate=0.25,
                                                   max_depth=4, random_state=0).fit(X_train_scaled, Y_train)

y_preds = hist_gradient_boosting_clf.predict_proba(X_test_scaled)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score.loc['HistGradientBoostingClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.plot(fpr, tpr, label='HistGradientBoosting Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')
plt.title('ROC Curve')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()

df_accur_roc_score

df_accur_roc_score.sort_values(by=['Roc_Auc_Score'],ascending=False).plot(kind='bar', y='Roc_Auc_Score',figsize=(20,8),color='#79ccb3', rot=0,title="Model outputs by roc score before feature importance");
lst_importance_drop = []

for item in range(0,feat_importances.shape[0]):
  
  if(feat_importances[item] < 0.004):
    lst_importance_drop.append(features[item])

X_train_importance = X_train_scaled.drop(lst_importance_drop,axis=1)
X_test_importance = X_test_scaled.drop(lst_importance_drop,axis=1)

lst_importance_drop
# Creating an empty Dataframe with Scores
df_accur_roc_score_importance = pd.DataFrame(columns=['Roc_Auc_Score'])
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0)
clf = logreg.fit(X_train_importance, Y_train)

y_preds = clf.predict_proba(X_test_importance)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score_importance.loc['Logistic_regression'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='Logistic Regression AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=150,random_state=0,learning_rate=0.1,eta=0.4,booster="gbtree",base_score=0.8,colsample_bylevel=0.9009229642844634,gamma=0.49967765132613584,
                        max_depth=6,min_child_weight=7,reg_lambda=0.27611902459972926,subsample=0.9300916052594785)

xgb_model.fit(X_train_importance, Y_train)
y_preds = xgb_model.predict_proba(X_test_importance)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score_importance.loc['XGBoost_Classifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='XGBoost Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_importance,Y_train)
y_preds = knn.predict_proba(X_test_importance)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score_importance.loc['KNeighborsClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='KNeighbors Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from sklearn.tree import DecisionTreeClassifier

reg_dtr = DecisionTreeClassifier(random_state=0)
reg_dtr.fit(X_train_importance,Y_train)

y_preds = reg_dtr.predict_proba(X_test_importance)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score_importance.loc['DecisionTreeClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='DecisionTree Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
        max_depth=6,
        n_estimators=100,random_state=0,learning_rate=0.25,eta=0.4,base_score=0.8,colsample_bylevel=0.9009229642844634,gamma=0.49967765132613584,
                        min_child_weight=7,reg_lambda=0.27611902459972926,subsample=0.9300916052594785,min_sample_split=2)

lgbm.fit(X_train_importance, Y_train)

y_preds = lgbm.predict_proba(X_test_importance)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score_importance.loc['LGBMClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='LGBM Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                   max_depth=5, random_state=0).fit(X_train_importance, Y_train)


y_preds = gradient_boosting_clf.predict_proba(X_test_importance)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score_importance.loc['GradientBoostingClassifier'] = [auc_score]

plt.subplots(figsize=(8, 6))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label='GradientBoosting Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
from sklearn.ensemble import HistGradientBoostingClassifier

hist_gradient_boosting_clf = HistGradientBoostingClassifier(learning_rate=0.1,
                                                   max_depth=7, random_state=0).fit(X_train_importance, Y_train)

y_preds = hist_gradient_boosting_clf.predict_proba(X_test_importance)
preds = y_preds[:,1]

fpr, tpr, _ = metrics.roc_curve(Y_test, preds)

auc_score = metrics.auc(fpr, tpr)
df_accur_roc_score_importance.loc['HistGradientBoostingClassifier'] = [auc_score]
plt.subplots(figsize=(8, 6))
plt.plot(fpr, tpr, label='HistGradientBoosting Classifier AUC = {:.2f}'.format(auc_score))
plt.plot([0,1],[0,1],'r--')
plt.title('ROC Curve')


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
df_accur_roc_score_importance.sort_values(by=['Roc_Auc_Score'],ascending=False).plot(kind='bar', y='Roc_Auc_Score',figsize=(20,8),color='#79ccb3', rot=0,title="Model outputs by roc score before feature importance");
f,ax = plt.subplots(figsize = (9,10))
sns.barplot(x=df_accur_roc_score_importance.Roc_Auc_Score,y=df_accur_roc_score_importance.index,color='red',alpha = 0.5,label='After Feature Importance' )
sns.barplot(x=df_accur_roc_score.Roc_Auc_Score,y=df_accur_roc_score.index,color='blue',alpha = 0.7,label='Before Feature Importance')

ax.legend(frameon = True)
ax.set(xlabel='Scores', ylabel='Models',title = "Auc Score ")
plt.show()
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Before Feature Importance', y=df_accur_roc_score.Roc_Auc_Score, x=df_accur_roc_score.index,text=round(df_accur_roc_score.Roc_Auc_Score,3),textposition='auto'),
    go.Bar(name='After Feature Importance', y=df_accur_roc_score_importance.Roc_Auc_Score, x=df_accur_roc_score_importance.index,text=round(df_accur_roc_score_importance.Roc_Auc_Score,3),textposition='auto',)
    
])
fig.update_layout(barmode='group')
fig.show()