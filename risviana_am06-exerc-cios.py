# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
# import category encoders
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing dataset
data_student= pd.read_csv("/kaggle/input/student-dataset/student-mat.csv")

#formatting dataset
name='school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3'

data_student[['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian',
              'traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet',
              'romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']]=data_student[name].str.split(";", expand = True)
del data_student[name]
data_student.head()
# this will replace "GP"with 0 and "MS" with 1 
data_student=data_student.replace(to_replace =["GP"],value =0) 
data_student=data_student.replace(to_replace =["MS"],value =1)

# this will replace "F"with 0 and "M" with 1 
data_student=data_student.replace(to_replace =['"F"'],value =0) 
data_student=data_student.replace(to_replace =['"M"'],value =1)

# this will replace "U"with 0 and "R" with 1 
data_student=data_student.replace(to_replace =['"U"'],value =0) 
data_student=data_student.replace(to_replace =['"R"'],value =1)

# this will replace "GT3" with 0 and "LE3" with 1 
data_student=data_student.replace(to_replace =['"GT3"'],value =0) 
data_student=data_student.replace(to_replace =['"LE3"'],value =1)

# this will replace "A" with 0 and "T" with 1 
data_student=data_student.replace(to_replace =['"A"'],value =0) 
data_student=data_student.replace(to_replace =['"T"'],value =1) 

# this will replace "yes" with 0 and "no" with 1 
data_student=data_student.replace(to_replace =['"yes"'],value =0) 
data_student=data_student.replace(to_replace =['"no"'],value =1) 

# this will convert str G1 and G2 to int
data_student=data_student.replace(to_replace =['"0"'],value =0) 
data_student=data_student.replace(to_replace =['"1"'],value =1) 
data_student=data_student.replace(to_replace =['"2"'],value =2) 
data_student=data_student.replace(to_replace =['"3"'],value =3) 
data_student=data_student.replace(to_replace =['"4"'],value =4) 
data_student=data_student.replace(to_replace =['"5"'],value =5) 
data_student=data_student.replace(to_replace =['"6"'],value =6) 
data_student=data_student.replace(to_replace =['"7"'],value =7) 
data_student=data_student.replace(to_replace =['"8"'],value =8) 
data_student=data_student.replace(to_replace =['"9"'],value =9) 
data_student=data_student.replace(to_replace =['"10"'],value=10) 
data_student=data_student.replace(to_replace =['"11"'],value =11) 
data_student=data_student.replace(to_replace =['"12"'],value =12) 
data_student=data_student.replace(to_replace =['"13"'],value =13) 
data_student=data_student.replace(to_replace =['"14"'],value =14) 
data_student=data_student.replace(to_replace =['"15"'],value =15) 
data_student=data_student.replace(to_replace =['"16"'],value =16) 
data_student=data_student.replace(to_replace =['"17"'],value =17) 
data_student=data_student.replace(to_replace =['"18"'],value =18) 
data_student=data_student.replace(to_replace =['"19"'],value =19) 
data_student=data_student.replace(to_replace =['"20"'],value =20) 

def compare_Mjob_col(data_student, dado,k):
      if dado=='"teacher"':
            data_student.iloc[k:k+1,33:34]=0
            
      elif dado=='"services"':
            data_student.iloc[k:k+1,34:35]=0
                
      elif dado=='"health"':
            data_student.iloc[k:k+1,35:36]=0
                    
      elif dado=='"at_home"':
            data_student.iloc[k:k+1,36:37]=0
                
      elif dado=='"other"':
            data_student.iloc[k:k+1,37:38]=0
      return data_student

def compare_Fjob_col(data_student, dado,k):
      if dado=='"teacher"':
            data_student.iloc[k:k+1,38:39]=0
            
      elif dado=='"services"':
            data_student.iloc[k:k+1,39:40]=0
                
      elif dado=='"health"':
            data_student.iloc[k:k+1,40:41]=0
                    
      elif dado=='"at_home"':
            data_student.iloc[k:k+1,42:43]=0
                
      elif dado=='"other"':
            data_student.iloc[k:k+1,43:44]=0
      return data_student

def compare_reason_col(data_student, dado,k):
      if dado=='"home"':
            data_student.iloc[k:k+1,39:40]=0
            
      elif dado=='"reputation"':
            data_student.iloc[k:k+1,40:41]=0
                
      elif dado=='"course"':
            data_student.iloc[k:k+1,41:42]=0
                    
      elif dado=='"at_home"':
            data_student.iloc[k:k+1,42:43]=0
                
      elif dado=='"other"':
            data_student.iloc[k:k+1,43:44]=0
      return data_student

def compare_guardian_col(data_student, dado,k):
      if dado=='"mother"':
            data_student.iloc[k:k+1,43:44]=0
            
      elif dado=='"father"':
            data_student.iloc[k:k+1,44:45]=0
                
      
      elif dado=='"other"':
            data_student.iloc[k:k+1,45:46]=0
      return data_student
def convert_nominal_atrb_in_columns(data_student):
    
    data_student['Mjob_teacher']=1
    data_student['Mjob_services']=1
    data_student['Mjob_health']=1
    data_student['Mjob_at_home']=1
    data_student['Mjob_other']=1
    
    data_student['Fjob_teacher']=1
    data_student['Fjob_services']=1
    data_student['Fjob_health']=1
    data_student['Fjob_at_home']=1
    data_student['Fjob_other']=1
    
    data_student['reason_home']=1
    data_student['reason_reputation']=1
    data_student['reason_course']=1
    data_student['reason_other']=1
    
    data_student['guardian_mother']=1
    data_student['guardian_father']=1
    data_student['guardian_other']=1
    
    
    l1=data_student['Mjob'].values
    l2=data_student['Fjob'].values
    l3=data_student['reason'].values
    l4=data_student['guardian'].values
    
    del data_student['Mjob']
    del data_student['Fjob']
    del data_student['reason']
    del data_student['guardian']
    
    for i in range(len(data_student)):
       
       data_student=compare_Mjob_col(data_student, l1[i],i)
       data_student=compare_Fjob_col(data_student, l2[i],i)
       data_student=compare_reason_col(data_student, l3[i],i)
       data_student=compare_guardian_col(data_student, l4[i],i)
    
    return data_student
                
data_student=convert_nominal_atrb_in_columns(data_student)
data_student.head(10)
#importing dataset
data_forest= pd.read_csv("/kaggle/input/forestfires-dataset/forestfires.csv")
data_forest.head()
def convert_month_to_number(data_forest):
    lista=data_forest['month'].values
    data_forest['month_sen']=0
    data_forest['month_cos']=0
    for i in range(len(lista)):
        
        if(lista[i]== "jan"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*1/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*1/12)
            
        elif(lista[i]== "feb"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*2/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*2/12)
            
        elif(lista[i]== "mar"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*3/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*3/12)
            
        elif(lista[i]== "abr"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*4/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*4/12)
            
        elif(lista[i]== "mai"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*5/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*5/12)
            
        elif(lista[i]== "jun"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*6/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*6/12)
            
        elif(lista[i]== "jul"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*7/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*7/12)
            
        elif(lista[i]== "agu"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*8/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*8/12)
            
        elif(lista[i]== "set"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*9/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*9/12)
            
        elif(lista[i]== "oct"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*10/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*10/12)
            
        elif(lista[i]== "nov"):
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*11/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*11/12)
            
        elif(lista[i]== "dec"):
            
            data_forest.iloc[i:i+1,13:14]=np.sin(2*np.pi*12/12)
            data_forest.iloc[i:i+1,14:15]=np.cos(2*np.pi*12/12)
            
    return  data_forest
data_forest=convert_month_to_number(data_forest)
def convert_day_to_number(data_forest):
    lista=data_forest['day'].values
    data_forest['day_sen']=0
    data_forest['day_cos']=0
    for i in range(len(lista)):
        
        if(lista[i]== "sun"):
            data_forest.iloc[i:i+1,15:16]=np.sin(2*np.pi*1/7)
            data_forest.iloc[i:i+1,16:17]=np.cos(2*np.pi*1/7)
            
        elif(lista[i]== "mon"):
            data_forest.iloc[i:i+1,15:16]=np.sin(2*np.pi*2/7)
            data_forest.iloc[i:i+1,16:17]=np.cos(2*np.pi*2/7)
            
        elif(lista[i]== "tue"):
            data_forest.iloc[i:i+1,15:16]=np.sin(2*np.pi*3/7)
            data_forest.iloc[i:i+1,16:17]=np.cos(2*np.pi*3/7)
            
        elif(lista[i]== "wed"):
            data_forest.iloc[i:i+1,15:16]=np.sin(2*np.pi*4/7)
            data_forest.iloc[i:i+1,16:17]=np.cos(2*np.pi*4/7)
            
        elif(lista[i]== "thu"):
            data_forest.iloc[i:i+1,15:16]=np.sin(2*np.pi*5/7)
            data_forest.iloc[i:i+1,16:17]=np.cos(2*np.pi*5/7)
            
        elif(lista[i]== "fri"):
            data_forest.iloc[i:i+1,15:16]=np.sin(2*np.pi*6/7)
            data_forest.iloc[i:i+1,16:17]=np.cos(2*np.pi*6/7)
            
        elif(lista[i]== "sat"):
            data_forest.iloc[i:i+1,15:16]=np.sin(2*np.pi*7/12)
            data_forest.iloc[i:i+1,16:17]=np.cos(2*np.pi*7/12)
    return data_forest
data_forest=convert_day_to_number(data_forest)
del data_forest['month']
del data_forest['day']
data_forest.head(20)

#importing dataset
data_car= pd.read_csv("/kaggle/input/dataset-car/data_car.csv")
data_car.head()
#Pré-processamento

encoder = ce.OrdinalEncoder(cols=['buying','maint','doors','persons','lug_boot','safety', 'class   '])
data_car = encoder.fit_transform(data_car)
data_car.head()


def holdout(x,y,frac):
    X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y,test_size=frac)
    return X_train, X_test, y_train, y_test
x_train, x_test, y_train, y_test=holdout(data_car.iloc[:,1:],data_car['class   '],0.5)


def score_knn_without_weight(k,x_train, x_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
    knn=neigh.fit(x_train, y_train)
    y_pred=knn.predict(x_test)
    acc=accuracy_score(y_test,y_pred)*100
    return acc
score_knn_without_weight(1,x_train, x_test, y_train, y_test)
#importing dataset
data_iris= pd.read_csv("/kaggle/input/iris/Iris.csv")
data_iris=data_iris.iloc[:,1:]
data_iris
SepalLength_min_index=data_iris[(data_iris['SepalLengthCm']< 5.84)].index
SepalLength_max_index=data_iris[(data_iris['SepalLengthCm']>= 7.9)].index
SepalLength_mean_index=data_iris[(data_iris['SepalLengthCm']>= 5.84) & (data_iris['SepalLengthCm']< 7.9)].index


data_iris.loc[SepalLength_min_index,'SepalLengthCm']='min'
data_iris.loc[SepalLength_max_index,'SepalLengthCm']='max'
data_iris.loc[SepalLength_mean_index,'SepalLengthCm']='mean'



SepalWidthCm_min_index=data_iris[(data_iris['SepalWidthCm']< 3.05)].index
SepalWidthCm_max_index=data_iris[(data_iris['SepalWidthCm']>= 4.4)].index
SepalWidthCm_mean_index=data_iris[(data_iris['SepalWidthCm']>= 3.05) & (data_iris['SepalWidthCm']< 4.4)].index


data_iris.loc[SepalWidthCm_min_index,'SepalWidthCm']='min'
data_iris.loc[SepalWidthCm_max_index,'SepalWidthCm']='max'
data_iris.loc[SepalWidthCm_mean_index,'SepalWidthCm']='mean'


PetalLengthCm_min_index=data_iris[(data_iris['PetalLengthCm']< 3.76)].index
PetalLengthCm_max_index=data_iris[(data_iris['PetalLengthCm']>= 6.9)].index
PetalLengthCm_mean_index=data_iris[(data_iris['PetalLengthCm']>= 3.76 ) & (data_iris['PetalLengthCm']< 6.9 )].index


data_iris.loc[PetalLengthCm_min_index,'PetalLengthCm']='min'
data_iris.loc[PetalLengthCm_max_index,'PetalLengthCm']='max'
data_iris.loc[PetalLengthCm_mean_index,'PetalLengthCm']='mean'

PetalWidthCm_min_index=data_iris[(data_iris['PetalWidthCm']< 1.20)].index
PetalWidthCm_max_index=data_iris[(data_iris['PetalWidthCm']>= 2.5)].index
PetalWidthCm_mean_index=data_iris[(data_iris['PetalWidthCm']>= 1.20) & (data_iris['PetalWidthCm']< 2.5)].index


data_iris.loc[PetalWidthCm_min_index,'PetalWidthCm']='min'
data_iris.loc[PetalWidthCm_max_index,'PetalWidthCm']='max'
data_iris.loc[PetalWidthCm_mean_index,'PetalWidthCm']='mean'



csv = data_iris.to_csv(r'/kaggle/output\iris.csv')
#importing dataset
data_adult= pd.read_csv("/kaggle/input/adult-dataset/adult.data")
data_adult.head()

x_train, x_test, y_train, y_test=holdout(data_adult.iloc[:,:14],data_adult['class'],0.5)

def fill_data_adult_missing_values(data):

    lista1=data['workclass'].values
    lista2=data['occupation'].values
    lista3=data['native-country'].values

    dt_workclass=pd.DataFrame(data['workclass'].value_counts())
    dt_occupation=pd.DataFrame(data['occupation'].value_counts())
    dt_native_country=pd.DataFrame(data['native-country'].value_counts())

    for k in range(len(lista1)):
        if ('?' in lista1[k].strip()):
            data.iloc[k:k+1,1:2]=dt_workclass.index[0]

        if ('?' in lista2[k].strip()):
            data.iloc[k:k+1,6:7]=dt_occupation.index[0]

        if ('?' in lista3[k].strip()):
            data.iloc[:,13:14]=dt_native_country.index[0]
    

    return data
x_train=fill_data_adult_missing_values(x_train)
x_train.head()
x_test=fill_data_adult_missing_values(x_test)
x_test.head()
#Pré-processamento
def encode(x_train,x_test,c):
    
    encoder = ce.OrdinalEncoder(cols=c)
    x_train = encoder.fit_transform(x_train)
    x_test = encoder.fit_transform(x_test)
    return x_train,x_test
    
def score_decision_tree(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    acc=accuracy_score(y_test,y_pred)*100
    return acc

x_train, x_test=encode(pd.DataFrame(x_train),pd.DataFrame( x_test),['workclass', 'education', 'marital-status','occupation','relationship','race','sex',
                                                                       'hours-per-week','native-country'])
score_decision_tree(x_train, x_test, y_train, y_test)
#importing dataset
data_heart= pd.read_csv("/kaggle/input/heart-dataset/processed.hungarian.data")
#eliminando colunas com muitos zeros
del data_heart['fbs']
del data_heart['restecg']
del data_heart['exang']
del data_heart['ca']
data_heart.head()

x_train, x_test, y_train, y_test=holdout(data_heart.iloc[:,:9],data_heart['num'],0.5)

def fill_data_heart_missing_values(data):
    i=0
    for j in data.columns:   
        lista=data[j].values
        dt=data[data[j]!='?']
        mediana=dt[j].median()
        for k in range(len(lista)):
            if(type(lista[k]==str)):
                if(lista[k]=='?'):
                    data.iloc[k:k+1,i:i+1]=mediana
        i+=1
                         
    return data
x_train=fill_data_heart_missing_values(x_train)
x_train.head()
#data test
x_test=fill_data_heart_missing_values(x_test)
x_test.head()

score_decision_tree(x_train, x_test, y_train, y_test)