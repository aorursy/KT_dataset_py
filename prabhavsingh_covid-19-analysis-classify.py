import pandas as pd

import numpy as np



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV



from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
del df["case_in_country"]

del df['Unnamed: 3']

del df["location"]

del df["If_onset_approximated"]

del df['hosp_visit_date']

del df["symptom"]

del df["source"]

del df["link"]
del df['Unnamed: 21']

del df['Unnamed: 22']

del df['Unnamed: 23']

del df['Unnamed: 24']

del df['Unnamed: 25']

del df['Unnamed: 26']
del df["summary"]
df.head()
def gender(str):

    if str == "male":

        return 1

    else:

        return 2

    

df["gender"] = df["gender"].apply(gender)
def location(str):

    if str == "Afghanistan":

        return 1

    elif str == "Algeria":

        return 2

    elif str == "Australia":

        return 3

    elif str == "Austria":

        return 4

    elif str == "Cambodia":

        return 5

    elif str == "Bahrain":

        return 6

    elif str == "Belgium":

        return 7

    elif str == "Canada":

        return 8

    elif str == "China":

        return 9

    elif str == "Croatia":

        return 10

    elif str == "Egypt":

        return 11

    elif str == "France":

        return 12

    elif str == "Germany":

        return 13

    elif str == "Hong Kong":

        return 14

    elif str == "India":

        return 15

    elif str == "Israel":

        return 16

    elif str == "Iran":

        return 17

    elif str == "Italy":

        return 18

    elif str == "Kuwait":

        return 19

    elif str == "Japan":

        return 20

    elif str == "Lebanon":

        return 21

    elif str == "Malaysia":

        return 22

    elif str == "Nepal":

        return 23

    elif str == "Phillipines":

        return 24

    elif str == "Russia":

        return 25

    elif str == "Singapore":

        return 26

    elif str == "Spain":

        return 27

    elif str == "Sri Lanka":

        return 28

    elif str == "South Korea":

        return 29

    elif str == "Switzerland":

        return 30

    elif str == "Taiwan":

        return 31

    elif str == "Thailand":

        return 32

    elif str == "UAE":

        return 33

    elif str == "UK":

        return 34

    elif str == "USA":

        return 35

    elif str == "Finland":

        return 36

    else:

        return 37
df["country"] = df["country"].apply(location)
locs = np.array(df.country)

print(np.unique(locs))

# 38 different Locations
a = [30 for i in range(100)]

b = [20 for i in range(100)]

c = [60 for i in range(100)]

plt.figure(figsize=(27,6))



plt.title("Distribution of Age for first 100 patients.")

sns.barplot(x=df.index[:100], y=df['age'][:100])



plt.plot(a)

plt.plot(b)

plt.plot(c)

plt.ylabel("Age")
plt.figure(figsize=(15,3))



plt.title("Distribution of Gender for first 50 patients.")

sns.barplot(x=df.index[:50], y=df['gender'][:50])



plt.ylabel("Age")
del df["exposure_end"]

del df["exposure_start"]
del df["symptom_onset"]
df["reporting date"].fillna("1/21/2020", inplace = True)
def dates(a):

    li = a.split('/')

    x1 = float(li[0])

    x2 = float(li[1])

    ans = (x1 * (x2 ** 2)) ** 0.5

    return ans
df['reporting date']= df['reporting date'].apply(dates)
plt.figure(figsize=(20,4))



plt.title("Patients who visited Wuhan and are infected")

sns.barplot(x=df.index[:50], y=df['visiting Wuhan'][:50])



plt.ylabel("1 : Visited Wuhan")
plt.figure(figsize=(20,4))



plt.title("Patients who are from Wuhan and are infected")

sns.barplot(x=df.index[:50], y=df['from Wuhan'][:50])



plt.ylabel("1 : From Wuhan")
df.reset_index(inplace = True) 

df.head()
del df["id"]
df["reporting date"].fillna(0, inplace = True)

df["country"].fillna(9, inplace = True)

df["age"].fillna(45, inplace = True)

df["from Wuhan"].fillna(1, inplace = True)

df["visiting Wuhan"].fillna(1, inplace = True)

df["death"].fillna(1, inplace = True)

df["recovered"].fillna(1, inplace = True)
df["Age_Gender"] = df["age"]*df["gender"]
df.head()
def change(str):

    if str == '0':

        return 0

    elif str == '1':

        return 1

    else:

        return 1
df["death"] = df["death"].apply(change)
df["recovered"] = df["recovered"].apply(change)
Y1 = df["death"]

Y2 = df["recovered"]

del df["death"]

del df["recovered"]
Y1 = np.array(Y1)

Y2 = np.array(Y2)



# for i in range(len(Y1)):

#     if type(Y1[i] == str):

#         Y1[i] = 1

        

# for i in range(len(Y2)):

#     if type(Y2[i] == str):

#         Y2[i] = 1
X = df.values
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y1)

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y2)
scaler = MinMaxScaler(feature_range=(0,1))

X_train1 = scaler.fit_transform(X_train1)

X_test1 = scaler.transform(X_test1)



X_train2 = scaler.fit_transform(X_train2)

X_test2 = scaler.transform(X_test2)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout

from tensorflow.keras.utils import to_categorical  
model = Sequential()
model.add(Dense(32, activation = "sigmoid"))

model.add(Dense(64, activation = "tanh"))

model.add(Dense(128, activation = "relu"))

model.add(Dense(512, activation = "relu"))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train1, Y_train1, epochs = 30, batch_size = 10)
predictions = model.predict(X_test1)

score = model.evaluate(X_test1, Y_test1)

print(score)
model.fit(X_train1, Y_train2, epochs = 50, batch_size = 10)
predictions2 = model.predict(X_test2)

score2 = model.evaluate(X_test2, Y_test2)

print(score2)