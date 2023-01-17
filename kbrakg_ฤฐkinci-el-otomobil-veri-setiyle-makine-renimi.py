# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/autos.csv', sep=',', header=0, encoding='cp1252')
data.head()
data.tail()
data.info()
data.describe()
data.shape
data.columns #Kolon isimlerini görüntüledik.
print("Seller value counts:\n",data.seller.value_counts(),"\n")
print("Offer Type value counts:\n", data.offerType.value_counts(),"\n")
print("nrOfPictures value counts:\n",data.nrOfPictures.value_counts(),"\n")
print("Vehicle Type value counts:\n", data.vehicleType.value_counts(),"\n")
print("Abtest value counts:\n",data.abtest.value_counts(),"\n")
del data["seller"]
del data["offerType"]
del data["nrOfPictures"]
del data["dateCrawled"]
del data["dateCreated"]
del data["lastSeen"]
del data["name"]
del data["postalCode"]
data.isnull().sum()
print("Gearbox value counts:\n",data.gearbox.value_counts(),"\n")
print("brand NULL value counts:",data.brand.isnull().sum())
data.groupby("brand")["gearbox"].value_counts()
gearbox = data["gearbox"].unique()
brand = data["brand"].unique()
willFill = {}

for i in brand :
    m = 0
    for j in gearbox :
        if data[(data.gearbox == j) & (data.brand == i)].shape[0] > m :
            m = data[(data.gearbox == j) & (data.brand == i)].shape[0]
            willFill[i] = j
for i in brand :
    data.loc[(data.brand == i) & (data.gearbox.isnull()) ,"gearbox" ] = willFill[i]

data.gearbox.isnull().sum()
print(data.fuelType.isnull().sum())
data.fuelType.fillna('benzin',inplace=True)
data.fuelType.isnull().sum()
print("notRepairedDamage Null counts:",data.notRepairedDamage.isnull().sum())
print("value counts:\n",data.notRepairedDamage.value_counts())
data.notRepairedDamage.fillna("nein",inplace=True) #inplace=true ile yeni değerler eklenip data ya bind edilir
print("notRepairedDamage Null counts",data.notRepairedDamage.isnull().sum())
data = data.dropna()
data.vehicleType.isnull().sum()
data.isnull().sum()
data.describe()
print("price value higher than 900.000 counts:",data[data.price>900000].shape[0])
print("price value lower than 500 counts:",data[data.price<500].shape[0])

data = data[(data.price>=500) & (data.price<=900000)]
print("yearOfRegistration higher than 1940 counts:",data[data.yearOfRegistration<1940].shape[0])
data = data[data.yearOfRegistration>=1940]
from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()
#le.fit(data.gearbox.drop_duplicates()) 
#data.gearbox = le.transform(data.gearbox)
#le.fit(data.fuelType.drop_duplicates())
#data.fuelType = le.transform(data.fuelType)
#le.fit(data.vehicleType.drop_duplicates())
#data.vehicleType = le.transform(data.vehicleType)
#data.head()

labels = ['gearbox','abtest', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
les = {}

for l in labels:
    le = LabelEncoder()
    le.fit(data[l].drop_duplicates())
    data[l] = le.transform(data[l]) 

data.head()

fig,axes = plt.subplots(nrows=3,ncols=1)
data.plot(kind="hist", y="yearOfRegistration",normed=False,ax=axes[0])
data.plot(kind="hist", y="fuelType", normed=False,ax=axes[1])
data.plot(kind="hist", y="vehicleType", normed=False,ax=axes[2])
plt.show()
data.corr() #düz metin korelasyon
#seaborn ısı haritasında korelasyon

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
    
plot_correlation_map(data)
plotingData = data.loc[:,["yearOfRegistration","price"]]
plotingData.plot(x='price', y='yearOfRegistration', style='o')
plotingData = data.loc[:,["kilometer","price"]]
plotingData.plot(x='price', y='kilometer', style='o')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
x = data.drop(['price'], axis = 1)
y = data.price
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)

forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

forest.score(x_test,y_test)
data["index"] = range(0,len(data.price)) 
#silinen veriler olduğu için indexler değişti.Yeniden index sıralaması oluşturuyoruz.
#For döngüsünde hata almamak için
data = data.set_index("index")
data.head()
results = []
for i in range (len(data['price'])):
    if data['price'][i] >= 200000:
        results.append("expensive class")
    elif data['price'][i] < 200000 and data['price'][i]>=90000:
        results.append("middle class")
    else:
        results.append('cheap class')
data["result"] = results
data.sample(20)
print("Expensive class counts:",data[data.result=="expensive class"].shape[0])
print("Middle class counts:",data[data.result=="middle class"].shape[0])
print("Cheap class counts:",data[data.result=="cheap class"].shape[0])
def run_model(model):
    from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score
    from sklearn.preprocessing import scale 
    
    x = data.drop(['result','price'], axis = 1)
    x = scale(x)
    y = data.result
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 100)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy =  accuracy_score(y_test, y_pred) * 100

    print("accuracy: ",accuracy)
    print("confusion matrix:\n",confusion_matrix(y_test,y_pred))
    print("precision:\n",precision_score(y_test, y_pred, average='weighted'))
    print("recall:\n",recall_score(y_test, y_pred, average='weighted'))
# ---- Decision Tree -----------
from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
run_model(model)
# ------ SVM Classifier ----------------
from sklearn.svm import SVC
model = SVC()
run_model(model)
# -------- Nearest Neighbors ----------
from sklearn import neighbors
model = neighbors.KNeighborsClassifier()
run_model(model)
# --------- Gaussian Naive Bayes ---------
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
run_model(model)

