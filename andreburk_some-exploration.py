# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# load datasets
train = pd.read_csv("../input/gender_age_train.csv", dtype = {"device_id": np.str, "age": np.int8})
events = pd.read_csv("../input/events.csv", dtype = {"device_id": np.str},\
                    infer_datetime_format = True, parse_dates = ["timestamp"])
brands = pd.read_csv("../input/phone_brand_device_model.csv", dtype = {"device_id": np.str},\
                    encoding = "UTF-8")
app_events = pd.read_csv("../input/app_events.csv")
app_labels = pd.read_csv("../input/app_labels.csv")
label_categories = pd.read_csv("../input/label_categories.csv")
# merging datasets and dropping unnecessary items
print(str(train.shape) + " shape of train")
### events
train = train.merge(events, how = "left", on = "device_id")
del events
print(str(train.shape) + " shape after event merge")

### brands
train = train.merge(brands, how = "left", on = "device_id")
del brands
train.drop("device_id", axis = 1, inplace = True)
print(str(train.shape) + " shape after brands merge")

### app_events
train = train.merge(app_events, how = "left", on = "event_id")
del app_events
train.drop("event_id", axis = 1, inplace = True)
print(str(train.shape) + " shape after app_events merge")

### app_labels and label_categories
train = train.merge(app_labels, how = "left", on = "app_id")
train = train.merge(label_categories, how = "left", on = "label_id")
del app_labels, label_categories
train.drop(["app_id", "label_id"], axis = 1, inplace = True)
print(str(train.shape) + " shape after labels merge")
print("------------------------------------------------")
print(train.head())
print("------------------------------------------------")
print(train.info())
# checking for NAs
print(train.isnull().sum())
# filling NAs
### for categories
train["is_installed"].fillna("NotAvailable", inplace = True)
train["is_active"].fillna("NotAvailable", inplace = True)
train["category"].fillna("NotAvailable", inplace = True)
### for numbers
train["timestamp"].fillna(-1000, inplace = True)
train["longitude"].fillna(-1000, inplace = True)
train["latitude"].fillna(-1000, inplace = True)

# check NA again
print(train.isnull().sum())
# showing target variable
sns.countplot(y = "group", data = train, \
              order = ["F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+",\
                     "M22-", "M23-26", "M27-28", "M29-31", "M32-38", "M39+"])
sns.plt.title("group/target variable")
# showing target variable by is_installed ordered by gender and age
sns.FacetGrid(train, col = "is_installed", size = 7)\
.map(sns.countplot, "group", \
    order = ["F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+",\
             "M22-", "M23-26", "M27-28", "M29-31", "M32-38", "M39+"])
# showing target variable by is_active
plt.figure(figsize = (14, 8))
sns.countplot("group", data = train, hue = "is_active", \
             order = ["F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+",\
                     "M22-", "M23-26", "M27-28", "M29-31", "M32-38", "M39+"])
sns.factorplot(y = "group", data = train, col = "is_active", kind = "count", size = 4.5, \
              order = ["F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+",\
                     "M22-", "M23-26", "M27-28", "M29-31", "M32-38", "M39+"])
# showing topk brand and group/target
def topk_vs_rest(column, k = 5):
    tops = column.value_counts().nlargest(k)
    column.loc[~column.isin(tops.index)] = "other"
    return(column)

train["brand_group"] = topk_vs_rest(train["phone_brand"].copy(), k = 5)
print(train["brand_group"].value_counts())
# topk brand and group ordered by gender and age
sns.factorplot(y = "group", data = train, col = "brand_group", kind = "count",\
               col_wrap = 3,\
              order = ["F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+",\
                     "M22-", "M23-26", "M27-28", "M29-31", "M32-38", "M39+"])
# showing topk brands and gender
sns.countplot("brand_group", data = train, hue = "gender")
# group/target variable and topk brands ordered by gender and age
plt.figure(figsize = (16, 8))
sns.countplot("group", data = train, hue = "brand_group",\
             order = ["F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+",\
                     "M22-", "M23-26", "M27-28", "M29-31", "M32-38", "M39+"])
# showing topk category group and group/target
train["category_group"] = topk_vs_rest(train["category"].copy(), k = 7)
print(train["category_group"].value_counts())
sns.factorplot(y = "group",\
               data = train.loc[np.logical_and(train["category_group"] != "NotAvailable",\
                                              train["category_group"] != "other"), :],\
               col = "category_group", kind = "count", col_wrap = 3,\
              order = ["F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+",\
                     "M22-", "M23-26", "M27-28", "M29-31", "M32-38", "M39+"])
