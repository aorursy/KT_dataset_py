import pandas as pd
import os
from IPython import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#import seaborn as sns
%matplotlib inline
def read_metadata(file):
    return pd.read_csv(file, index_col=0, header=None)
    
def read_data(file):
    return pd.read_csv(file, index_col=[0, 1, 2], header=[0,1], na_values=['(X)', '-', '**'])

def read_prepped(file):
    return pd.read_csv(file, header=[0,1], na_values=['-'])

def ingnore_DS_Store(directory):
    return filter(lambda f: f != '_DS_Store', os.listdir(directory))

def collect_info_for_dep (dept_dir):
    """
    This function collects the '.csv' files into pandas dataframes.
    The return value is a hash where the keys refer to the original file names.
    """
    base_dir = "../input/cpe-data/{}".format(dept_dir)
    data_directories = list(filter(lambda f: f.endswith("_data"), os.listdir(base_dir)))
    info = {'dept' : dept_dir}
    assert len(data_directories) == 1, "found {} data directories".format(len(data_directories))
    for dd in data_directories:
        directory = "{}/{}".format(base_dir, dd)
        dd_directories = ingnore_DS_Store(directory)
        #print(dd_directories)
        for ddd in dd_directories:
            ddd_directory = "{}/{}".format(directory, ddd)
            files = list(ingnore_DS_Store(ddd_directory))
            #print(files)
            assert len(files) == 2, "found {} files in {}".format(len(files), directory)
            full_file_names = ["{}/{}".format(ddd_directory, file) for file in files]
            dataframes = [read_metadata(file) if file.endswith('_metadata.csv') else read_data(file) for file in full_file_names]
            info[ddd] = dict(zip(files, dataframes))
    prepped_files = list(filter(lambda f: f.endswith("_prepped.csv"), os.listdir(base_dir)))
    for pf in prepped_files:
        info[pf] = read_prepped("{}/{}".format(base_dir, pf))
    return info
department_names = [
#    'Dept_11-00091',
#    'Dept_23-00089',
    'Dept_35-00103',
    'Dept_37-00027',
    'Dept_37-00049',
#    'Dept_49-00009',
]

departments = {dep: collect_info_for_dep(dep) for dep in department_names}
def investigate_dept(dept):
    print(dept['dept'])
    print('=' * 20)
    print(dept.keys())
for dep in departments.keys():
    investigate_dept(departments[dep])
    print()
prepped_dfs = [departments['Dept_35-00103']['35-00103_UOF-OIS-P_prepped.csv'], departments['Dept_37-00027']['37-00027_UOF-P_2014-2016_prepped.csv'], departments['Dept_37-00049']['37-00049_UOF-P_2016_prepped.csv']]
print(prepped_dfs[0].shape)
print(prepped_dfs[1].shape)
print(prepped_dfs[2].shape)
from functools import reduce

columns = [list(zip(*pre.columns))[0] for pre in prepped_dfs]
common_columns = reduce(lambda x, y: list(set(x).intersection(y)), columns)
common_columns
# rearrange a little

common_columns = [
 'INCIDENT_DATE',
 'LOCATION_LONGITUDE',
 'LOCATION_LATITUDE',
 'LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION',
 'SUBJECT_GENDER',
 'SUBJECT_RACE',
 'SUBJECT_INJURY_TYPE',
]
prepped_dfs[1].head()
class TransformAndSelectForDept_35_00103(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        #print("TransformAndSelectForDept_35_00103")
        return self
    
    def transform(self, X, y = None):
        columns = common_columns
        ret_df = X[columns].copy()
        ret_df.columns = columns
        ret_df['SUBJECT_GENDER'] = ret_df['SUBJECT_GENDER'].map({'Male': 'M', 'Female': 'F'})
        return ret_df
class TransformAndSelectForDept_37_00027(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        #print("TransformAndSelectForDept_37_00027")
        return self
    
    def transform(self, X, y = None):
        columns1 = [
         'INCIDENT_DATE',
        ]
        ret_df = pd.DataFrame()
        ret_df[('Y_COORDINATE', 'Y-Coordinate')] = - X[('Y_COORDINATE', 'Y-Coordinate')] / 100000.0        
        ret_df[('Y_COORDINATE', 'X-Coordinate')] = X[('Y_COORDINATE', 'X-Coordinate')] / 100000.0
        columns2 = [
         'LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION',
         'SUBJECT_GENDER',
         'SUBJECT_RACE',
         'SUBJECT_INJURY_TYPE',
        ]
        ret_df = pd.concat([X[columns1], ret_df, X[columns2]], axis=1)
        ret_df.columns = common_columns
        return ret_df
class TransformAndSelectForDept_37_00049(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        #print("TransformAndSelectForDept_37_00049")
        return self
    
    def transform(self, X, y = None):
        columns = common_columns
        ret_df = X[columns].copy()
        ret_df.columns = columns
        ret_df['SUBJECT_GENDER'] = ret_df['SUBJECT_GENDER'].map({'Male': 'M', 'Female': 'F'})
        return ret_df
transformations = [
    TransformAndSelectForDept_35_00103(),
    TransformAndSelectForDept_37_00027(),
    TransformAndSelectForDept_37_00049()
]

dfs = [trans.fit_transform(df1) for df1, trans in zip(prepped_dfs, transformations)]
for d in dfs:
    print(d.shape)
df = pd.concat(dfs)
df.shape
df.info()
df['INCIDENT_DATE'] = pd.to_datetime(df['INCIDENT_DATE'])
dti = pd.DatetimeIndex(df['INCIDENT_DATE'])
df['year'] = dti.year
df['month'] = dti.month
df['dayofweek'] = dti.dayofweek
df['dayofyear'] = dti.dayofyear
df = df.dropna()

lb_gender = LabelEncoder()
lb_race = LabelEncoder()
lb_injury_type = LabelEncoder()

df["subject_gender_code"] = lb_gender.fit_transform(df["SUBJECT_GENDER"])
df['subject_race_code'] = lb_race.fit_transform(df["SUBJECT_RACE"]) 
df['subject_injury_type_code'] = lb_injury_type.fit_transform(df['SUBJECT_INJURY_TYPE'])
df.info()
columns_for_prediction = [
    'LOCATION_LONGITUDE',
    'LOCATION_LATITUDE',
    'year',              
    'month',             
    'dayofweek',         
    'dayofyear',         
    'subject_gender_code',
    'subject_race_code',  
    'subject_injury_type_code',    
]
len(columns_for_prediction)
# display large dataframes in an html iframe
def ldf_display(df, lines=500):
    txt = ("<iframe " +
           "srcdoc='" + df.head(lines).to_html() + "' " +
           "width=1000 height=500>" +
           "</iframe>")

    return display.HTML(txt)
ldf_display(df, lines=20)
for col in df.columns:
    print(col)
    print('=' * 20)
    print(df[col].value_counts())
prepped_dfs[0]['SUBJECT_GENDER']['INDIVIDUAL_GENDER'].value_counts()
prepped_dfs[1]['SUBJECT_GENDER']['Subject Sex'].value_counts()
prepped_dfs[2]['SUBJECT_GENDER']['CitSex'].value_counts()
#from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix

scatter_matrix(df[columns_for_prediction], figsize=(20, 20), alpha=0.2, diagonal='kde')
plt.show()
