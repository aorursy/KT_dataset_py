import pandas as pd
import numpy as np
import seaborn as sns # plotting library
x_train = pd.read_csv("/kaggle/input/titanic/train.csv")

x_train.head()
# 1. check missing value of dataset

def count_missing_values(data) : 
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_values.sort_values(inplace=True) 
    
    missing_values = pd.DataFrame({"ColumnName": missing_values.index, "MissingCount": missing_values.values})

    missing_values["MissingPercentage(%)"] = missing_values['MissingCount'].apply(lambda x: round(x / data.shape[0] * 100, 2))
    return missing_values

count_missing_values(x_train)
# 2. split label and delete unecessary column

y_train = x_train["Survived"].values
x_train.drop("Survived", axis=1, inplace=True)
x_train.drop("PassengerId", axis=1, inplace=True)

x_train.head()
# 3.1 fill missing value - Embarked

sns.countplot(x="Embarked", data=x_train)

# because Embarked "S" is the Largest, so fill "S" into the place of NaN 
x_train["Embarked"] = x_train["Embarked"].fillna("S")

display(count_missing_values(x_train))
display(x_train.head())
# 3.2 fill missing value - Age

sns.distplot(x_train["Age"])

# the name feature maybe is useful for filling missing value of age feature
name_of_titles = {}
for name in x_train["Name"].values.tolist():
    title = name.split(", ")[1].split(".")[0]
    if title not in list(name_of_titles.keys()):
        name_of_titles[title] = 1
    else:
        name_of_titles[title] += 1

for title, count in name_of_titles.items():
    age_mean = int(x_train[x_train["Name"].str.contains("{}\.".format(title))]["Age"].mean())
    x_train.loc[x_train["Name"].str.contains("{}\.".format(title)) & x_train["Age"].isna(), "Age"] = age_mean

x_train.drop("Name", axis=1, inplace=True)
    
display(count_missing_values(x_train))
display(x_train.head())

# the missing values of "cabin" is too many, so we skip and don't use it.

x_train.drop("Cabin", axis=1, inplace=True)
# 4. next, I will use feature engineering for data
# 4.1 feature "Sex"

def one_hot_encoding(data, column_name):
    unique_data = np.unique(data[column_name].values)
    
    encoding_dict = {}
    for index in range(len(unique_data)):
        encoding_dict[unique_data[index]] = index

    return data[column_name].map(encoding_dict)

sns.countplot(x_train["Sex"], hue=y_train)

x_train["Sex"] = one_hot_encoding(x_train, "Sex")

x_train.head()
# 4.2 feature "Embarked"

x_train["Embarked"] = one_hot_encoding(x_train, "Embarked")

x_train.head()
# 4.3 feature "Fare"
sns.distplot(x_train["Fare"])

split_index = 15
for number in range(int(x_train["Fare"].max() / split_index) + 1):
    x_train.loc[(x_train["Fare"] > number * split_index) & (x_train["Fare"] < (number + 1) * split_index), "Fare"] = number + 1

x_train.head()
# 4.4 familty number = sister or brothers + parent or children

x_train["FamilyNumber"] = x_train["SibSp"] + x_train["Parch"]

x_train.drop("SibSp", axis=1, inplace=True)
x_train.drop("Parch", axis=1, inplace=True)

x_train.head()
# 4.5 drop the feature "Ticket" that we don't use
x_train.drop("Ticket", axis=1, inplace=True)

x_train.head()
# 5. build the model
from sklearn.neighbors import KNeighborsClassifier

best_model = 0
best_accuracy = 0
best_k = 0
for k in range(1, 51, 1):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train.values, y_train)
    
    accuracy = model.score(x_train.values, y_train)
    
    if (accuracy > best_accuracy):
        best_accuracy = accuracy
        best_k = k
        best_model = model

print(best_k)
print(best_accuracy)
# 6. we must use feature engineering for x_test data before predict time

# we can predict test data after get best c
# best_model.predict()

x_test = pd.read_csv("/kaggle/input/titanic/test.csv")

x_test.head()
# 7. check missing value and drop unecessary column
display(count_missing_values(x_test))

x_test.drop("PassengerId", axis=1, inplace=True)
# 8.1 fill missing value - Fare

# we know the people that no fare information now
display(x_test.iloc[np.where(np.isnan(x_test["Fare"].values))[0]])

# load original x_train data
x_train = pd.read_csv("/kaggle/input/titanic/train.csv")

# get title of name is Mr. and "Embarked" is S
display(x_test.loc[x_test["Name"].str.contains("Mr\.") & ( x_test["Age"] > 50)])

# the mean value is about 20.93
print(x_test.loc[(x_test["Embarked"] == "S") & x_test["Name"].str.contains("Mr\.")]["Fare"].mean())
print(x_train.loc[(x_train["Embarked"] == "S") & x_train["Name"].str.contains("Mr\.")]["Fare"].mean())

# fill nan
x_test.loc[x_test["Fare"].isna(), "Fare"] = 20.93

display(count_missing_values(x_test))

# 在查詢資料的時候也發現
# Ticket 有包含 PC 的，Fare 通常都比較高
# Ticket 2 開頭的， Fare 都比較低
# 8.2 fill missing value - Age
# recall 3.2

# the age distplot is similar to the age distplot of training data
sns.distplot(x_test["Age"])

# the name feature maybe is useful for filling missing value of age feature
name_of_titles = {}
for name in x_test["Name"].values.tolist():
    title = name.split(", ")[1].split(".")[0]
    if title not in list(name_of_titles.keys()):
        name_of_titles[title] = 1
    else:
        name_of_titles[title] += 1

for title, count in name_of_titles.items():
    
    # because the test data have no age of "Ms.", we replace the training data into it
    if title == "Ms":
        age_mean = x_train.loc[x_train["Name"].str.contains("Ms\.")]["Age"].mean()
    else: 
        age_mean = int(x_test[x_test["Name"].str.contains("{}\.".format(title))]["Age"].mean())
    
    x_test.loc[x_test["Name"].str.contains("{}\.".format(title)) & x_test["Age"].isna(), "Age"] = age_mean

x_test.drop("Name", axis=1, inplace=True)

display(count_missing_values(x_test))
display(x_test.head())

# the missing values of "cabin" is too many, so we skip and don't use it.
x_test.drop("Cabin", axis=1, inplace=True)

# check data
display(x_test.head())
# 8.3 familty number = sister or brothers + parent or children
# recall 4.4 

x_test["FamilyNumber"] = x_test["SibSp"] + x_test["Parch"]

x_test.drop("SibSp", axis=1, inplace=True)
x_test.drop("Parch", axis=1, inplace=True)

x_test.head()
# 8.4 one-hot encoding

x_test["Sex"] = one_hot_encoding(x_test, "Sex")
x_test["Embarked"] = one_hot_encoding(x_test, "Embarked")

x_test.head()
# 8.5 drop the feature "Ticket" that we don't use
x_test.drop("Ticket", axis=1, inplace=True)

x_test.head()
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(data, title = "Download CSV file", filename = "data.csv"):  
    csv = data.to_csv()
    base_64_file = base64.b64encode(csv.encode())
    payload = base_64_file.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

submission_sample = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

answer = np.expand_dims(model.predict(x_test), axis=1)

submission_sample["Survived"] = answer
create_download_link(submission_sample)