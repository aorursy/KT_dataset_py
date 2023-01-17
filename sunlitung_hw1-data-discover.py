import pandas as pd
import numpy as np
train = pd.read_csv(r"../input/qbs-hw1/training_data.csv")
#train = pd.read_csv(r"C:\Users\Tyller Sun\OneDrive\桌面\training.csv")
import numpy as np
import matplotlib.pyplot as plt

train.head()
train = train.fillna(0)
train.head()
def find_1(data , word):
    count = 0
    print(type(word))
    print(type(data[1]))
    for i in range(len(data)):
        print(str(data[i]))
        if str(data[i]) == "Male":
            print("aaa")
            count += train["target"][i]
    return count
import seaborn as sns
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(train)
import pandas as pd
train = pd.read_csv(r"../input/qbs-hw1/new_train_data.csv")
import seaborn as sns
sns.set(style="ticks", color_codes=True)
s = sns.pairplot(train)

#s.show()
train.head()
def get_num(data):
    #print(data)
    #data = data.type(string)
    print(data)
    trains = pd.read_csv(r"../input/qbs-hw1/train.csv")
    trains_ = trains[data]
    trains_ = list(set(trains_)) 
    #print(train_type)
    trains_ = list(enumerate(trains_))
    print(trains_)
   # return trains

#train = train.drop(columns = ["index", "ID"])
#train = train.drop(columns = "Unnamed: 0")
train_type = list(set(train))
#print(train_type)
#train_type = train_type.remove(Unnamed: 0)
#print(train_type)
for i in range(len(train_type)):
    print(train_type[i])
    get_num(train_type[i])
print("finish")
import matplotlib.pyplot as plt

import pandas as pd
train = pd.read_csv(r"../input/qbs-hw1/training_data.csv")
plt.bar(train["Education"], train["Target"])
Education = train.groupby('Education')
#target["Education"]
Education["Target"].value_counts()/Education["Target"].count()
def get_pro(word):
    word = train.groupby(word)
    print(word["Target"].value_counts()/word["Target"].count())
for i in range(len(train_type)):
    get_pro(train_type[i])
    print("\n")
print(train_type)
#target.groups
import matplotlib.pyplot as plt
import random
data = train["fnlwgt"]
plt.hist(data)
data = np.log(train["fnlwgt"])
plt.hist(data)        
data = train["Age"]
plt.hist(data)
data = np.log(train["Age"])
plt.hist(data)
plt.hist(np.square(data))
train.head()
data = train["Hours_per_week"]
plt.hist(data)
plt.hist(np.sqrt(data))
data = train["Capital_Gain"] - train["Capital_Loss"]
plt.hist(data)
data = np.log(train["Capital_Gain"] - train["Capital_Loss"]+10000)
plt.hist(data)
plt.hist(np.sqrt(data))
data = (train["Education_Num"])
plt.hist(data)
plt.hist((np.sqrt(data)))
