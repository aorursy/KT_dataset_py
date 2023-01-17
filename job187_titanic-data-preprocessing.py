import kaggle
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder #For creating dummy variables from one discrete variable (CategoricalEncoder is newer and quicker though)
from sklearn.preprocessing import StandardScaler

kaggle.api.authenticate()

comptetition = 'titanic'
downloadpath = 'D:\Datasets\Titanic Machine Learning from Disaster\kaggle datasets'
trainname = 'train.csv' 
testname = 'test.csv'
def GetDatasets(competition, dlpath):
    file_info_list = kaggle.api.competitions_data_list_files(competition)
    for file_info in file_info_list:
        print('file info: ' + str(file_info))
        file_name = file_info['name']
        kaggle.api.competition_download_file(competition, file_name, path=dlpath)
def load_data(train, test):
    train = pd.read_csv(downloadpath + '\\' + train)
    test = pd.read_csv(downloadpath + '\\' + test)
    return train, test
# Get the titanic datasets
# Sometimes needs to be run twice since some files are missed. There is also a "competition_download_files" method but then we get a compressed file
GetDatasets(comptetition, downloadpath)

# Load the data into dataframes
train, score_data = load_data(trainname, testname)

score_data.to_csv("D:\\Datasets\\Titanic Machine Learning from Disaster\\kaggle datasets\\test_kaggle.csv", index=None)
# Inspect the dataframe
print("\n", train.head(5))
# Quick look for missing values and data types
print("\n", train.info())
# Inspect categorical variable content
print("\n",train["Sex"].value_counts())
print("\n",train["Embarked"].value_counts())
#all_trainDescription of the numerical variables 
print("\n", train.describe())
#Check histograms of the variables
plot = train.hist(bins=50, figsize=(20,15))
plt.pyplot.show()
train_set, test_set = train_test_split(train, test_size=0.2, random_state=42)
corr_matrix = train_set.corr() 
corr_matrix["Survived"].sort_values(ascending=False)
titanic_labels = train_set["Survived"]
titanic = train_set.drop("Survived", axis=1)
titanic_num = titanic.drop(["Name" ,"Sex", "Ticket", "Cabin", "Embarked", "PassengerId"], axis = 1)
titanic_cat = titanic[["Sex", "Embarked"]] #Only load Sex and Embarked for now since the others are either very incomplete and or cryptic
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# "Train" the imputer on the titanic numerical dataset
imputer.fit(titanic_num)

# Create a numpy array with the titanic_num values and imputed values when missing
X = imputer.transform(titanic_num)

# Convert back to Dataframe
titanic_tr = pd.DataFrame(X, columns=titanic_num.columns)
# Fill in missing embarked based on the average fare. A more complex, conditional rule would have been had to be constructed if the test dataset also had missing values. However, it does not. 
titanic_cat.loc[titanic_cat["Embarked"].isna(), "Embarked"] = 'C'
titanic_cat_encoded_emb, titanic_categories_emb = pd.factorize(titanic_cat["Embarked"])
titanic_cat_encoded_sex, titanic_categories_sex = pd.factorize(titanic_cat["Sex"])

# To deal with the issue of the discrete variable being interpreted as an ordinal we emplay one-hot encoding (create dummys). (Scikits CategoricalEncoder does all of this in one step.)
hot_encoder = OneHotEncoder()

# fit_transform expects a 2D array. OneHotEncoder produces a scipy.sparse matrix which is very space efficient. If we want to transform back to a numpy array then use .toarray()
titanic_cat_encoded_emb_1hot = hot_encoder.fit_transform(titanic_cat_encoded_emb.reshape(-1,1))
scaler = StandardScaler().fit(titanic_tr)
titanic_scaled = pd.DataFrame(data=scaler.transform(titanic_tr), columns=titanic_tr.columns)
# Rejoin the categorical dataframes
tmp_df1 = pd.DataFrame(titanic_cat_encoded_sex,columns=["Sex"])
tmp_df2 = pd.DataFrame(titanic_cat_encoded_emb_1hot.toarray())
titanic_cat_encoded_final = tmp_df1.join(tmp_df2)

#Join the numerical and categorial datasets to the final dataset and output to csv
titanic_prepared_final = titanic_scaled.join(titanic_cat_encoded_final)
titanic_prepared_final.to_csv("D:\\Datasets\\Titanic Machine Learning from Disaster\\kaggle datasets\\titanic_train_features.csv" , index=False)
titanic_labels.to_csv("D:\\Datasets\\Titanic Machine Learning from Disaster\\kaggle datasets\\titanic_train_labels.csv" ,  header="Survived",index=False)
titanic_prepared_final
