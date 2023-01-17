import numpy

import pandas

import matplotlib.pyplot as plt



%matplotlib inline
data_train = pandas.read_csv("../input/train.csv")

data_test = pandas.read_csv("../input/test.csv")
print(data_train.shape)

print(data_test.shape)
print(data_train.head())

print(data_test.head())
def create_vector(line):

    return line.split(",")



#create a vector from sequences

data_train["vector"] = data_train["Sequence"].apply(create_vector)

data_test["vector"] = data_test["Sequence"].apply(create_vector)



#create a column with the len of the vectors

data_train["len"] = data_train["vector"].apply(len)

data_test["len"] = data_test["vector"].apply(len)

data_train["len"] = data_train["len"]-1
data_train["len"].hist(bins=100)

plt.show()

data_test["len"].hist(bins=100)

plt.show()
#plt.figure(figsize=(8,8))

#plt.suptitle("Some sequences graphs", fontsize="x-large")

#for k in range(100):

#    y = data_train["vector"][k]

#    x = [x for x in range(len(y))]

#    plt.subplot(10,10,1+k)    

#    plt.scatter(x, y, s=20, c='blue', alpha=0.4)    

#    plt.scatter(x[-1], y[-1], c='red')

#    plt.plot(x, y, c='gray', alpha=0.3)

#    plt.axis('off')
plt.figure(figsize=(8,8))

plt.suptitle("Some sequences logistic graphs,xn+1=f(xn)", fontsize="x-large")

for k in range(100):

    y = data_train["vector"][k][:-1]

    x = data_train["vector"][k][1:]

    plt.subplot(10,10,1+k)

    

    plt.scatter(x, y, s=20, c='blue', alpha=0.4)

    

    plt.scatter(x[-1], y[-1], c='red')

    plt.plot(x, y, c='gray', alpha=0.3)

    plt.axis('off')
#plt.figure(figsize=(8,8))

#plt.suptitle("Some sequences graphs (log)", fontsize="x-large")

#for k in range(100):

#    y = [numpy.log(float(x)) for x in list(data_train["vector"][k])]

#    x = [x for x in range(len(y))]

#    plt.subplot(10,10,1+k)    

#    plt.scatter(x, y, s=20, c='blue', alpha=0.4)    

#    plt.scatter(x[-1], y[-1], c='red')

#    plt.plot(x, y, c='gray', alpha=0.3)

#    plt.axis('off')
def last_int(vector):

    if len(str(vector[-1])) < 5:

        return int(vector[-1])

    return 0



def second_last_int(vector):

    if len(vector) > 1:

        if len(str(vector[-2])) < 5:

            return int(vector[-2])

    return 0



def before_second_last(vector):

    if len(vector) > 2:

        if len(str(vector[-3])) < 5:

            return int(vector[-3])

    return 0

    



data_train["prediction"] = data_train["vector"].apply(last_int)

data_train["last_int"] = data_train["vector"].apply(second_last_int)

data_train["second_last"] = data_train["vector"].apply(before_second_last) 



data_test["last_int"] = data_test["vector"].apply(last_int)

data_test["second_last"] = data_test["vector"].apply(second_last_int)
data_train.info()
def len_int(num):

    return len(str(num))





data_train["len_sc"] = data_train["second_last"].apply(len_int)

data_train["len_li"] = data_train["last_int"].apply(len_int)



data_test["len_sc"] = data_test["second_last"].apply(len_int)

data_test["len_li"] = data_test["last_int"].apply(len_int)
data_train = data_train.drop(["Sequence","vector"],axis=1)

data_test = data_test.drop(["Sequence","vector"],axis=1)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
train = data_train[(data_train['last_int']!=0) &

                   (data_train['second_last']!=0) &

                   (data_train['prediction']!=0)].copy()



del data_train



train.shape
col = ['last_int','second_last','len_sc','len_li']



X = train.loc[:,col]

y = train.loc[:,'prediction']
#model.fit(X,y)
#model.score(X,y)
#data_test["prediction"] = model.predict(data_test[col])
#data_test["prediction"].value_counts()