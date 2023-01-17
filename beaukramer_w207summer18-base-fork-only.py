{"cells":[{"metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","trusted":true},"cell_type":"code","source":"# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load in \n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the \"../input/\" directory.\n# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n\nimport os\nprint(os.listdir(\"../input\"))\n\n# Any results you write to the current directory are saved as output.","execution_count":null,"outputs":[]},{"metadata":{"_cell_guid":"79c7e3d0-c299-4dcb-8224-4455121ee9b0","collapsed":true,"_uuid":"d629ff2d2480ee46fbb7e2d37f6b5fab8052498a","trusted":false},"cell_type":"code","source":"","execution_count":null,"outputs":[]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.6.4","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"}},"nbformat":4,"nbformat_minor":1}
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import cross_validation # Use for train-test split

# Packages for Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.dummy import DummyClassifier # For baselines
# import warnings
# warnings.filterwarnings('ignore')
dataset = pd.read_csv("../input/train.csv") 
#Drop the first column 'Id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]
#Look at a summary of the dataset
display(dataset.describe())
# No missing data!!!!

# soil_type7 annd soil_type15 are constant so can be removed
dataset.drop(['Soil_Type7','Soil_Type15'], axis=1, inplace=True)

dataset.groupby('Cover_Type').size()
# All classes have equal representation
r,c = dataset.shape
# Extract just the values from the dataset
array = dataset.values
X = array[:,0:(c-1)] # Take all but the last column as the inputs
Y = array[:,(c-1)] # Take the last column as the output (Cover_Type)
seed = 0
val_size=0.1
X_train, X_dev, y_train, y_dev = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
continuouscatbreakpoint = 10 # the first ten columns are continuous variables, the remainder are categorical
# STANDARD SCALER
X_temp = StandardScaler().fit_transform(X_train[:,0:continuouscatbreakpoint])
X_dev_temp = StandardScaler().fit_transform(X_dev[:,0:continuouscatbreakpoint])
# MINMAX SCALER
# X_temp = MinMaxScaler().fit_transform(X_train[:,0:continuouscatbreakpoint])
# X_dev_temp = MinMaxScaler().fit_transform(X_dev[:,0:continuouscatbreakpoint])
# Normalizer
# X_temp = Normalizer().fit_transform(X_train[:,0:continuouscatbreakpoint])
# X_dev_temp = Normalizer().fit_transform(X_dev[:,0:continuouscatbreakpoint])


X_train = np.concatenate((X_temp,X_train[:,continuouscatbreakpoint:]),axis=1)
X_dev = np.concatenate((X_dev_temp,X_dev[:,continuouscatbreakpoint:]),axis=1)
# EXAMINE THAT ONLY CONTINUOUS VARIABLES WERE CHANGED AND THAT COLUMNS ARE STILL CORRECT
df = pd.DataFrame(X_train)
df.columns = dataset.columns[:-1]
display(df.describe())


# We will use these pandas dataframes of the training and dev data for some baselines
dataset_train = pd.DataFrame(data=(X_train))
dataset_train['Cover_Type'] = pd.Series(y_train)
dataset_train.columns = dataset.columns

dataset_dev = pd.DataFrame(data=(X_dev))
dataset_dev['Cover_Type'] = pd.Series(y_dev)
dataset_dev.columns = dataset.columns
for variable in range(X_train.shape[1]):
    rho = np.corrcoef(X_train[:,variable], y_train)[0,1]
    if abs(rho) > 0.7:
        print(dataset_train.columns[variable], np.corrcoef(X_train[:,variable], y_train)[0,1], "***")
    elif abs(rho) > 0.5:
        print(dataset_train.columns[variable], np.corrcoef(X_train[:,variable], y_train)[0,1], "**")
    elif abs(rho) > 0.2:
        print(dataset_train.columns[variable], np.corrcoef(X_train[:,variable], y_train)[0,1], "*")
    else:
        print(dataset_train.columns[variable], np.corrcoef(X_train[:,variable], y_train)[0,1])

# Look at correlations of continuous variables
sns.heatmap(df.iloc[:,:continuouscatbreakpoint].corr(), center=0, cmap="vlag",annot=True)

# Run the various dummy classification strategies
dummy_mf = DummyClassifier(strategy='most_frequent',random_state=0)
dummy_mf.fit(X_train, y_train)
print("Most Frequent ",dummy_mf.score(X_dev, y_dev))

dummy_st = DummyClassifier(strategy='stratified', random_state=0)
dummy_st.fit(X_train, y_train)
print("Stratified", dummy_st.score(X_dev, y_dev))

dummy_pr = DummyClassifier(strategy='prior', random_state=0)
dummy_pr.fit(X_train, y_train)
print("Prior", dummy_pr.score(X_dev, y_dev))

dummy_un = DummyClassifier(strategy='uniform', random_state=0)
dummy_un.fit(X_train, y_train)
print("Uniform", dummy_un.score(X_dev, y_dev))
# Undo one hot encoding
def get_soil(row):
    for c in dataset_train.columns[14:]:
        if row[c]==1:
            return c
dataset_train['Soil_Type'] = dataset_train.apply(get_soil, axis=1)
dataset_dev['Soil_Type'] = dataset_dev.apply(get_soil, axis=1)
dataset_train.sort_values("Soil_Type") # add column for soil_type
soil_counts = dataset_train.groupby('Soil_Type').Cover_Type.apply(lambda x: x.mode()) # Find the most common cover_type by soil_type
# Clean up index
soil_counts = pd.DataFrame(soil_counts)
soil_counts.reset_index(inplace=True)
soil_counts.drop("level_1",axis=1, inplace=True)
# Add column labels
soil_counts.columns = ["Soil_Type","Most_Frequent_By_Soil_Type"]

# Assign the predicted cover_type to each datapoint
dataset_train = dataset_train.merge(soil_counts[["Soil_Type","Most_Frequent_By_Soil_Type"]],on=["Soil_Type"])
dataset_dev = dataset_dev.merge(soil_counts[["Soil_Type","Most_Frequent_By_Soil_Type"]],on=["Soil_Type"])

# this is moving covertype so it is on the end of the dataframe
df1 = dataset_train.pop('Cover_Type') # remove column b and store it in df1
dataset_train['Cover_Type'] = df1
df1 = dataset_dev.pop('Cover_Type') # remove column b and store it in df1
dataset_dev['Cover_Type'] = df1

# Calculate the accuracy of this strategy
correct = 0
for row in range(dataset_dev.shape[0]):
    if dataset_dev.iloc[row]["Cover_Type"] == dataset_dev.iloc[row]["Most_Frequent_By_Soil_Type"]:
        correct += 1
    else:
        pass
correct / dataset_dev.shape[0]
parameters = np.arange(1,20)
for n in parameters:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_dev)
    print(n,accuracy_score(Y_dev, pred))


