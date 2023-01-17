import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv("../input/train.csv", index_col=0)

df.head()
df.describe()
df.info()
def processing(dataset):

    

    #Extracing Car brand (to be converted into dummies later) and removing typos 

    a = dataset.CarName.str.extract("(\w+)[ -]*")

    a[0] = a[0].replace("toyouta", "toyota")

    a[0] = a[0].replace("vokswagen","volkswagen")

    a[0] = a[0].replace("vw","volkswagen")

    a[0] = a[0].replace("maxda","mazda")

    a[0] = a[0].replace("porcshce","porsche")

    

    dataset.CarName = a[0].str.lower()

    

    #processing typos in data (hopefully correct)

    dataset.drivewheel = dataset.drivewheel.replace("4wd", "fwd")

    

    #replacing text numbers into integers (helpful in Linear regression??)

    dic = {'one': 1,

 'two': 2,

 'three': 3,

 'four': 4,

 'five': 5,

 'six': 6,

 'seven': 7,

 'eight': 8,

 'nine': 9,

 'zero': 0,

 'ten': 10,

 'eleven':11,

      'twelve':12,

      'thirteen':13}

    dataset.doornumber = dataset.doornumber.replace(dic).astype("float")

    dataset.cylindernumber = dataset.cylindernumber.replace(dic).astype("float")





    # dataset.groupby("fuelsystem")['price'].agg({"means":'mean',"medians":"median","sizes":"size"}).sort_values("means")

    #could I treat 2bbl and 1bbl as same?

    

    #Ceating dummies for categorical variables

    def dummifier(df, col):    

        title_dummies = pd.get_dummies(df[col], prefix=col)

        df = pd.concat([df, title_dummies], axis=1)

        

        df.drop([col], axis=1, inplace=True)

        return df

    

    

    columns_to_dummify = ["CarName", "fueltype", "aspiration", "carbody", "drivewheel", "enginelocation", "enginetype", "fuelsystem"]

    for i in columns_to_dummify:

        dataset = dummifier(dataset, i)

        print("{} dummified".format(i))

    return dataset
combined = processing(df)
combined.head()
test = pd.read_csv("../input/test.csv", index_col=0)

test.head()
combined_test = processing(test)
#finding mutually exclusive columns and drop them from both datasets 

[i for i in combined.columns if i not in combined_test.columns], [i for i in combined_test.columns if i not in combined.columns]
#saving price column to be used train_test_split later

y = combined.price
#dropping mutually exclusive columns from dummified versions of training and testing datasets

def dont_know_what_to_name(dataset, cols):

    return dataset.drop(columns=cols)

combined = dont_know_what_to_name(combined, [i for i in combined.columns if i not in combined_test.columns])

combined_test = dont_know_what_to_name(combined_test, [i for i in combined_test.columns if i not in combined.columns])
#checking if still any columns are left. Should return ([], [])

[i for i in combined.columns if i not in combined_test.columns], [i for i in combined_test.columns if i not in combined.columns]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(combined, y, test_size=0.3)



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt



clf = LinearRegression()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(sqrt(mean_squared_error(pred, y_test)))
final_pred = clf.predict(combined_test)

combined_test["price"] = final_pred

combined_test.head()
# combined_test[["car_ID", "price"]].to_csv("submission.csv", index=False)