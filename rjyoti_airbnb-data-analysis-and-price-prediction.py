import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/airbnb-price-prediction/train.csv")
df.head()
df.shape
print("The dataset has \nrows :- {}\ncolumns :- {}".format(df.shape[0],df.shape[1]))
dataset_columns = list(df.columns)
dataset_columns
df.info()
df.dtypes

df.isnull().sum()
for column in df.columns:
    if df[column].isnull().sum() != 0:
        print("========================================")
        print("{} :- {},  dtypes : {}".format(column,df[column].isnull().sum(),df[column].dtypes))
# the last_review column in the dataset is a datetime field and it contains 15827 missing values
df["last_review"] = pd.to_datetime(df["last_review"])
df["first_review"] = pd.to_datetime(df["first_review"])
df["host_since"] = pd.to_datetime(df["host_since"])
df.last_review
df.first_review
df.host_since
df.last_review.fillna(method="ffill",inplace=True)
df.first_review.fillna(method="ffill",inplace=True)
df.host_since.fillna(method="ffill",inplace=True)
df.last_review
df.first_review
print("Missing values in datetime type columns ")
print("last_review :- ",df.last_review.isnull().sum())
print("first review :- ",df.first_review.isnull().sum())
print("host_since :- ",df.host_since.isnull().sum())

sns.set(font_scale=2.5)
plt.figure(figsize=(30,20))
sns.heatmap(df.corr(), annot=True)
df.bathrooms.mean()
df.bathrooms.unique()
# the bathrooms column contains values in the multiples of 0.5 
# Hence replacing the missing values in this column by the roundoff value of mean value of this column
# Since the mean of this column 1.2352 is not a multiple of 0.5
df["bathrooms"] = df['bathrooms'].fillna(round(df["bathrooms"].mean()))
# All the missing values of bathroom column are handled
df.bathrooms.isnull().sum()
print("Mean:- ",df.review_scores_rating.mean())
print("Min :- ",df.review_scores_rating.min())
print("Max :- ",df.review_scores_rating.max())
df[["number_of_reviews","review_scores_rating"]][df.number_of_reviews == 0]
# the above table shows that the missing values in review_scores_rating is NaN which has 0 number_of_reviews
# Hence replacing this missing values with 0
# Replacing the missing values in the review_scores_rating with 0
df["review_scores_rating"] = df["review_scores_rating"].fillna(0)
# All the missing values of review_scores_rating column are handled
df.review_scores_rating.isnull().sum()
df.bedrooms.mean()
df.bedrooms.unique()
df.bedrooms.describe()
# Bedrooms column contain only integer values and mean is 1.26
# Replacing missing values with 1
df["bedrooms"] = df["bedrooms"].fillna(1.0)
# All the missing values of bedrooms column are handled
df.bedrooms.isnull().sum()
df.beds.mean()
df.beds.unique()
df.beds.describe()
# Beds column also contain only integer values and mean is 1.71
# Replacing missing values with 2
df["beds"] = df["beds"].fillna(2.0)
# All the missing values of beds column are handled
df.beds.isnull().sum()

df.host_has_profile_pic
df.host_identity_verified
# replacing t and f in these columns with 1 and 0 respectively and changing datatype to bool
df.replace(to_replace = "t", value = 1,inplace=True) 
df.replace(to_replace = "f", value = 0,inplace=True) 
print(df.host_has_profile_pic.dtypes)
print(df.host_identity_verified.dtypes)
df["host_has_profile_pic"] = df["host_has_profile_pic"].astype("bool")
df["host_identity_verified"] = df["host_identity_verified"].astype("bool")
# changed the datatype to bool of these two columns

# Dropping host_response_rate and zipcode and thumbnail_url column ---> bcoz not relevant on determining the log_price
df.drop(["host_response_rate"],axis=1,inplace=True)
df.drop(["zipcode"],axis=1,inplace=True)
df.drop(["thumbnail_url"],axis=1,inplace=True)

df.dtypes

sns.set(font_scale=2)
sns.catplot(x = "city", kind="count", data=df,height=10);
sns.set(font_scale=1.5)
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = list(df.city.unique())
students =list(df.city.value_counts())
ax.pie(students, labels = langs,autopct='%1.2f%%')
plt.show()
data = df.neighbourhood.value_counts()[:15]
plt.figure(figsize=(22,22))
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("Most popular Neighbourhood")
plt.ylabel("Neighbourhood Area")
plt.xlabel("Number of guest who host in this area")

plt.barh(x,y)
# Function to plot catplot graphs
def plot_catplot(h,v,he):
    sns.set(font_scale=1.5)
    sns.catplot(x=h,kind=v,data=df,height=he)
# Function to plot catplot graphs
def plot_piechart(h):
    sns.set(font_scale=1.5)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    langs = list(df[h].unique())
    students =list(df[h].value_counts())
    ax.pie(students, labels = langs,autopct='%1.2f%%')
    plt.show()
plot_catplot("room_type","count",8)

plot_piechart("room_type")

plot_catplot("cancellation_policy","count",12)

plot_piechart("cancellation_policy")

plot_catplot("cleaning_fee","count",6)
plot_piechart("cleaning_fee")
def plot_violinplot(h,v):
    plt.figure(figsize=(15,8))
    sns.set(font_scale=1.5)
    sns.violinplot(data=df, x=h, y=v, palette='GnBu_d')
    plt.title('Density and distribution of prices ', fontsize=15)
    plt.xlabel(h)
    plt.ylabel(v)
plot_violinplot("city","log_price")

plot_violinplot("room_type","log_price")

plot_violinplot("cancellation_policy","log_price")
plot_violinplot("bed_type","log_price")
plot_catplot("bed_type","count",8)
sns.scatterplot(df.longitude,df.latitude)


df.amenities
# We can replce the amenties column by its count of amentities provided in the set
amenities_count = []
for i in df["amenities"]:
    amenities_count.append(len(i))
len(amenities_count)
# Replacing the count values in the amenities feature
df["amenities"] = amenities_count
df.amenities
df.amenities.describe()


categorical_col = []
for column in df.columns:
    if df[column].nunique() <= 50 and df[column].dtypes != "float64" and df[column].dtypes != "int64" and df[column].dtypes != "bool":
        categorical_col.append(column)
categorical_col
for i in categorical_col:
    print(i," --> ",df[i].dtypes," --> ",df[i].nunique())
print("The categorical features in the dataset are : \n",categorical_col)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_col:
    df[col] = le.fit_transform(df[col])
for col in categorical_col:
    print("-----------------------")
    print(col)
    print("Unique value count :- ",df[col].nunique())
    print(df[col].describe())

df.dtypes

df["cleaning_fee"] = df["cleaning_fee"].astype("int")
df["host_has_profile_pic"] = df["host_has_profile_pic"].astype("int")
df["host_identity_verified"] = df["host_identity_verified"].astype("int")
df.hist(edgecolor="black", linewidth=1.2, figsize=(30, 30));


# plt.figure(figsize=(30,30))
# sns.pairplot(df,height=2,diag_kind="hist")



df.corr()
sns.set(font_scale=2.25)
plt.figure(figsize=(50,50))
sns.heatmap(df.corr(), annot=True)
plt.savefig("heatmap_of_correlation_matrix.png")


plt.figure(figsize=(30, 30))
sns.set(font_scale=1)
i = 1
for column in df.columns:
    if df[column].dtype == "float64" or df[column].dtype == "int64":
        plt.subplot(5, 5, i)
        df.corr()[column].sort_values().plot(kind="barh")
        i += 1
plt.figure(figsize=(10,10))
df.corr()["log_price"].sort_values().plot(kind="barh")
plt.figure(figsize=(10,10))
df.corr()["accommodates"].sort_values().plot(kind="barh")
df.columns
from sklearn.model_selection import train_test_split

X = df.drop(["id","name","log_price","description","first_review","host_since","last_review","neighbourhood"],axis = 1)
y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
X.head(10)
y.head(10)

from sklearn import metrics

def print_evaluate(true,predicted):
    mae = metrics.mean_absolute_error(true,predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true,predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('\nMAE:', mae)
    print('\nMSE:', mse)
    print('\nRMSE:', rmse)
    print('\nR2 Square', r2_square)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
print("Error/ Accuracy Analysis :- ")
print_evaluate(y_test,lin_reg.predict(X_test))
def final_result(y_test,y_pred):
    dic = {"actual_y" : y_test.values, "predicted_y" : y_pred}
    result = pd.DataFrame(dic)
    return result
final_result(y_test,y_pred).head(10)

X = df[["bathrooms","room_type","bedrooms"]]
y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
print("Error/Accuracy Analysis :- ")
print_evaluate(y_test,y_pred)
# The error is increased in this case 
final_result(y_test,y_pred).head(10)
df.columns
X = df[["bathrooms","room_type","bedrooms","accommodates","property_type"]]
y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
print("Error/Accuracy Analysis :- ")
print_evaluate(y_test,y_pred)
final_result(y_test,y_pred).head(10)

X = df[["room_type"]]
y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
print("Error/Accuracy Analysis :- ")
print_evaluate(y_test,y_pred)
X = df[["room_type","accommodates"]]
y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
print("Error/Accuracy Analysis :- ")
print_evaluate(y_test,y_pred)
# From the above two cells we can see that the error is improved when 2 input features 
# are considered for predicting the output rather than single feature

# However that doesn't mean increasing number of features will improve the accuracy of the model
X = df[["property_type","amenities","latitude","longitude","cancellation_policy","number_of_reviews","bedrooms"]]
y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
print("Error/Accuracy Analysis :- ")
print_evaluate(y_test,y_pred)
# The above cell clearly shows that increasing the number of input features doesn't ensure the better 
# performance of the model


# Rather the features which are higly correlated with dependent variable performs better in predicting the output
