# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import Imputer

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn import linear_model



amazon_path = "../input/forest-fires-in-brazil/amazon.csv"

data = pd.read_csv(amazon_path, thousands=".", dtype={"year" : int, "state": str, "month": str, "number": int},

                delimiter = ',',  encoding="ISO-8859-1")



str_to_int = {'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6, 'Julho': 7, 'Agosto': 8, 

              'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12}



data['month_int'] = data['month'].apply(lambda x: str_to_int[x])

int_data = data[['year', 'month_int', 'number']].values



str_to_int = {'Acre': 1, 'Alagoas': 2, 'Amapa': 3,'Amazonas': 4, 

              'Bahia': 5, 'Ceara': 6 ,'Distrito Federal': 7,'Espirito Santo': 8, 

              'Goias': 9, 'Maranhao': 10,'Mato Grosso': 11, 'Minas Gerais': 12, 

              'Pará': 13, 'Paraiba' : 14, 'Pernambuco': 15, 'Piau': 16, 

              'Rio': 17,'Rondonia': 18, 'Roraima': 19, 'Santa Catarina': 20, 

              'Sao Paulo': 21, 'Sergipe': 22, 'Tocantins': 23}



data['state_int'] = data['state'].apply(lambda x: str_to_int[x])

data_np = np.array(data)

#type(data)



###################### draw ################################



from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

from matplotlib import style



int_data = int_data[240:479]



style.use('ggplot')



fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')



x3 = int_data[:, 0]

y3 = int_data[:, 1]

z3 = np.zeros(int_data.shape[0])

#z3 = z3[0:239]



dx = np.ones(int_data.shape[0])

dy = np.ones(int_data.shape[0])

dz = int_data[:, 2]



ax1.bar3d(x3, y3, z3, dx, dy, dz)



ax1.set_xlabel('year [#]')

ax1.set_ylabel('month [#]')

ax1.set_zlabel('number of fires [#]')



plt.show()



print("done")





###################### 2d graph ################################



import matplotlib.pyplot as plt



#print(data.number.sum())



#creating a list of years we have

years=list(data.year.unique())

#print(years)



#creating an empty list, which will be populated later with amount of fires reported

sub_fires_per_year=[]

#using for loop to extract sum of fires reported for each year and append list above

for i in years:

   y=data.loc[data['year']==i].number.sum()

   sub_fires_per_year.append(y)



plt.scatter(years, sub_fires_per_year)

plt.plot(years, sub_fires_per_year)

plt.grid()

plt.title('Brazil Fires per 1998-2017 Years')

#plt.legend(loc="upper right")



plt.show()



#print(sub_fires_per_year)



###################### month graph ################################

###################### PCA ################################



x_data=data[['year','month_int','number']].values

y_data=data['state_int'].values



type(x_data)



mean_vector=np.mean(x_data,axis=0)



centered_x=x_data-mean_vector #centered_x



cov_matrix=np.dot(centered_x.T,centered_x)/(x_data.shape[0]-1)



w, v = np.linalg.eig(cov_matrix)



print("w:{} \n v:{}".format(w,v))



Z=np.dot(centered_x,v[:,:2])



plt.figure()

plt.scatter(Z[y_data==5][:,0], Z[y_data==5][:,1], c='b')

plt.scatter(Z[y_data==11][:,0], Z[y_data==11][:,1], c='r')

plt.scatter(Z[y_data==18][:,0], Z[y_data==18][:,1], c='y')

plt.grid()

plt.show()

print("Explained variance of Z[:,0]:{:.4f}\nExplained variance of Z[:,1]:{:.4f}\nExplained variance Ratio: {:.4f} ".

     format(np.var(Z[:,0]),

            np.var(Z[:,1]),

            (np.var(Z[:,0])+np.var(Z[:,1]))/np.sum(np.var(x_data,axis=0))))

data
data = data.drop(['state','date','month'], axis=1)

data
totalTrainingSet, totalTestSet = train_test_split(data,test_size=.2)

len(totalTrainingSet)

len(totalTestSet)
fold = KFold(n_splits = 6)

arr = []



############ KNN #########



from sklearn.neighbors import KNeighborsClassifier  

classifier = KNeighborsClassifier(n_neighbors=6)  



for i in range(0,5):

    split = next(fold.split(totalTrainingSet), None)

    train = totalTrainingSet.iloc[split[0]]

    validate = totalTrainingSet.iloc[split[1]]

      

    ##training/testing

    X_train=train.iloc[:,0:3]

    y_train=train.iloc[:,3]

    

    X_validate = validate.iloc[:,0:3]

    y_validate=validate.iloc[:,3]

    model = classifier.fit(X_train, y_train)

    model.predict_proba(X_validate)

    score=model.score(X_validate, y_validate)

    ##end

    arr.append(score)

    

arr

class_names=['Acre', 'Alagoas', 'Amapa','Amazonas','Bahia','Ceara','Distrito Federal',

             'Espirito Santo','Goias', 'Maranhao','Mato Grosso', 'Minas Gerais',

             'Pará', 'Paraiba', 'Pernambuco', 'Piau','Rio','Rondonia', 

             'Roraima', 'Santa Catarina', 'Sao Paulo','Sergipe','Tocantins']



X_test=totalTestSet.iloc[:,0:3]

y_test=totalTestSet.iloc[:,3]

y_pred = classifier.predict(X_test)



score=model.score(X_test, y_test)

score


from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

plt.matshow(cm, cmap=plt.cm.Blues_r)

plt.title('Confusion matrix')

plt.colorbar()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.matshow(cm, cmap=plt.cm.Blues_r)

plt.show()



classes = ['Acre', 'Alagoas', 'Amapa','Amazonas','Bahia','Ceara','Distrito Federal',

           'Espirito Santo','Goias', 'Maranhao','Mato Grosso', 'Minas Gerais',

           'Pará', 'Paraiba', 'Pernambuco', 'Piau','Rio','Rondonia', 

           'Roraima', 'Santa Catarina', 'Sao Paulo','Sergipe','Tocantins']



plt.rcParams["figure.figsize"] = (23,20)

normalize = True

cm =confusion_matrix(y_test,y_pred)



fig, ax = plt.subplots()

im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

ax.figure.colorbar(im, ax=ax)

# We want to show all ticks...

ax.set(xticks=np.arange(cm.shape[1]),

       yticks=np.arange(cm.shape[0]),

       # ... and label them with the respective list entries

       xticklabels=classes, yticklabels=classes,

       title= "Number of Incidents",

       ylabel='True label',

       xlabel='Predicted label')

fmt = '.2f' if normalize else 'd'

thresh = cm.max() / 2.

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(j, i, format(cm[i, j], fmt),

                ha="center", va="center",

                color="white" if cm[i, j] > thresh else "black")

fig.autofmt_xdate()

plt.savefig('MLR_Confusion_Matrix')
