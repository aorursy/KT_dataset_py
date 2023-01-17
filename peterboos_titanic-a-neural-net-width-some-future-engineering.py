# if you want to know what kaggle comes whidth 
! pip list  
folderpath =" "
try:
  # Load the Drive helper and mount
  from google.colab import drive
  # This will prompt for authorization.
  drive.mount('/content/drive')
  folderpath="drive/My Drive/"
except:
  import os
  folderpath ="../input/"
print(os.listdir(folderpath))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#header =['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
trainpath = folderpath+"train.csv"
testpath = folderpath + "test.csv"
df = pd.read_csv(trainpath)  #to learn from
dt = pd.read_csv(testpath)   #to test upon later


DataSet_Train = df.values

y = DataSet_Train[:,1].astype(int)  #the result truth to learn 
Dataset_Final = dt.values
ids = Dataset_Final[:,0].astype(int)# the passsenger ID column of the chalange



# helper functions for viewing and testing the data :

def show_table(data,x):
    print('Sample data entry :',x)
    for col in df.columns:
        dol = col #+ "            "
       # dol = dol[:16] 
        print (dol, data[col].values[x-1])
    print() 

def find_Nan(data,name):
    print ('Missing values:',name)
    print(data.isnull().sum(axis=0)) # list columns with Nan values. (axis=1 will show rows)
    print(data.isnull().sum(axis=1).sum()) 
    print()

    

##--------------- Helper functions for data Engineering------------------------
# helper functions 
def checkforindex(text,array): # return array index of match or else -1 in case nan -1 as well
    i=-1
    if (pd.isna(text)):
        #print("empty")
        return i
    else:
        #print("found",text)
        for item in array:
            i=i+1
            if (text.find(item)>=0):
                return i
    return -1

def checkfor(text,array):  # a simple check does a string contain a word from the array
    if (pd.isna(text)):
        return False    
    for item in array:
        if (text.find(item)>=0):
            return True
    return False

def containsAll(str, set):
    for c in set:
        if c not in str: return False;
    return True 
  
  
def stringbetween(s,a,b):
  try:
    named= (s.split(a))[1].split(b)[0]
    named = named.replace(' ', '')+'.'
    return named
  except:
    return "?"


##--------------------Data engineering (can we exctract more information from the data..
#data enginering functions
def dating_rank(x): 
    title=x['Name']
    if (checkfor(title,['Mr.','Mrs.'])):
        return 2
    if (checkfor(title,['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col'])):
        return 3
    if (checkfor(title,['Master','Sir'])):
        return 4
    if (checkfor(title,['Countess', 'Mme'])):  
        return 5
    if (checkfor(title,['Mlle', 'Ms','Miss'])):       #non maried
        return 6
    if (checkfor(title,['Dr'])):
        return 3
    else:
        return 4
      
def Named_Title(x):
    title=x['Name']
    try:
      named = stringbetween(title,',','.')
      return named
    except:
      print(title)
    return '?'


def relations_onboard(x):
    result = x['SibSp']+x['Parch']
    return result

def fam_ideal_team_work(x):
    result = x['SibSp']+x['Parch']
    if ((result>1)&(result<5)):
        return 1
    return 0


def getdeck(x):
    deck = x['Cabin']
    return checkforindex(deck,['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])

def hascabin(x):
    xn=x['Cabin']
    c = checkforindex(xn,['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
#     if (c==0):
#         return 0
    if (c==-1):
        return 0
    return 1


def getembarkedA(x):
    harbor=x['Embarked']
    if (checkforindex(harbor,['C','S','Q'])==0):
        return 1
    return 0
def getembarkedB(x):
    harbor=x['Embarked']
    if (checkforindex(harbor,['C','S','Q'])==1):
        return 1
    return 0
def getembarkedC(x):
    harbor=x['Embarked']
    if (checkforindex(harbor,['C','S','Q'])==2):
        return 1
    return 0




def getsex(x):
    sex = x['Sex']
    
    if (sex=="female"):
        return 1 
    else:
        return 0
    return 1

def get_survived(x):
        xo=x['Survived']
        if (pd.isna(xo)):
            return 0
        if (xo==0):
            return 0
        return 1
def upperclass(x):
    xo=x['Pclass']
    if (xo==1):
        return 1
    return 0

def midclass(x):
    xo=x['Pclass']
    if (xo==2):
        return 1
    return 0

def lowerclass(x):
    xo=x['Pclass']
    if (xo==3):
        return 1
    return 0

def age_fix(x):
    xn=x['Age']
    if (xn==0):
        return 1
    else:      
        return 0
    return 0
    
def cabin3thclass(x):
    xx=x['Fare_Per_Person']  
    if (xx<12):
        return 1
    return 0

def cabin2ndclass(x):
    xx= x['Fare_Per_Person']  
    if ((xx>=12)&(xx<57)):
        return 1
    return 0
def cabin1stclass(x):
    xx= x['Fare_Per_Person']  
    if (xx>=57):
        return 1
    return 0
def cabinluxclass(x):
    xx= x['Fare_Per_Person']  
    if (xx>=100):
        return 1
    return 0        

def Unknown_Fare(x):
    xx =x['Fare']
    if (pd.isna(xx)):
      return 0
    if (xx>0):
      return 0
    return 1
  
def UnKnownAge(x):
    xx =x['Age']
    zz=x['female']
    try:   #NAN
      if (xx>0):
        if(zz==1):
          return 0
        else:
          return -1
    except:
      nothing =True
      
    if (zz!=1):
      return 1 
    else:
      return 0 # male 'sure dead' is less effective on female
      
  
def SureName(x):
  xx=x['Name']
  sep = ','
  xx = xx.split(sep, 1)[0]
  return xx
  
def Survivingfam(x):
  # https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818  (Here I converted it to python logic)
  female = x['female']
  alive_list = ['Baclini', 'Becker', 'Brown', 'Caldwell', 'Collyer', 'Coutts', 'Doling', 'Fortune', 'Goldsmith', 'Graham', 'Hamalainen',
             'Harper', 'Hart', 'Hays', 'Herman', 'Hippach', 'Johnson', 'Kelly', 'Laroche', 'Mellinger', 'Moor', 'Moubarek', 'Murphy',
             'Navratil', 'Newell', 'Nicola-Yarred', 'Peter', 'Quick', 'Richards', 'Ryerson', 'Sandstrom', 'Taussig', 'West', 'Wick']
  
  dead_list = ['Barbara' ,'Boulos', 'Bourke', 'Ford', 'Goodwin', 'Jussila', 'Lefebre', 'Palsson',
             'Panula', 'Rice', 'Sage' 'Skoog', 'Strom', 'Van Impe', 'Vander Planke', 'Zabour']

  if (female!=1):        #male
    xx=x['Title']
    if (xx=='Master'):
      xx=x['SureName']
      if xx in alive_list:
        return 1
  else:                  #female
    xx=x['SureName']       
    if xx in dead_list:
      return 0
    else:
      return 1
  return 0

def DeadFam(x):
  dead_list = ['Barbara' ,'Boulos', 'Bourke', 'Ford', 'Goodwin', 'Jussila', 'Lefebre', 'Palsson',
             'Panula', 'Rice', 'Sage' 'Skoog', 'Strom', 'Van Impe', 'Vander Planke', 'Zabour']
  xx=x['SureName']
  if xx in dead_list:
    return 1
  return 0

def travel_alone(x):
  xx=x['SibSp']
  xy=x['Parch']
  if (xx==0):
    if(xy==0):
      return 1
  return 0

def inverseAge(x):  # just in doubt how a neural net would handle this, a stronger signal for low values maybe?
  xx=x['Age']
  inv = 1-xx
  return inv


  
  
def prepare_data(data):
    data['RelationsOnBoard']=  data.apply(relations_onboard, axis=1)
    data['DatingRank']= data.apply(dating_rank, axis=1)
    data['Deck'] = data.apply(getdeck, axis=1)
    data['female'] = data.apply(getsex,axis=1)
  
    data['harborA'] = data.apply(getembarkedA,axis=1)
    data['harborB'] = data.apply(getembarkedB,axis=1)
    data['harborC'] = data.apply(getembarkedC,axis=1)
    data['ClassUp'] = data.apply(upperclass,axis=1)
    data['ClassMid'] = data.apply(midclass,axis=1)
    data['ClassLow'] = data.apply(lowerclass,axis=1)
    data['HasCabin'] = data.apply(hascabin,axis=1)
    
    #there are 1 or 2 fare prices missing, i assume they travel cheap

    try :
        data['Survived'] = data.apply(get_survived,axis=1)
    except:
        noSurvivaldata=True
        
    data['KnownAge']=data.apply(UnKnownAge,axis=1) 
    data['Parch'].astype('int64')
    data['SibSp'].astype('int64')
    
    data['Family_Size']= data['Parch']+ data['SibSp']
    
    data['Fare_Per_Person']=data['Fare']/(data['Family_Size']+1)
    
    #https://www.dummies.com/education/history/suites-and-cabins-for-passengers-on-the-titanic/
    data['Cabin3thclass']= data.apply(cabin3thclass,axis=1)
    data['Cabin2ndClass']= data.apply(cabin2ndclass,axis=1)
    data['Cabin1stClass']= data.apply(cabin1stclass,axis=1)
    data['CabinLuxeryClass']=data.apply(cabinluxclass,axis=1)
    data['IdealFamSize'] = data.apply(fam_ideal_team_work ,axis=1)    
    #remove any NAN values left out of safity (shouldnt be there by now)
    #data.replace(np.nan, 0, inplace=True)
    
    #normalization shouldnt cause difference between train and test data, so lets do it manual   >> Later i normalized certain columns (you can remove that if you whish)
    
    data['Age']=data['Age']*0.01
    data['Ageinv']=data.apply(inverseAge,axis=1)
    
    data['Fare_Per_Person']=data['Fare_Per_Person']*0.001
    data['DatingRank']=data['DatingRank']*0.15
    data['Family_Size']=data['Family_Size']*0.02  # max 50
    data['Deck']=data['Deck']*0.025 + 0.025
    data['Fare']=data['Fare']*0.0001
    data['RelationsOnBoard']=data['RelationsOnBoard']*0.02
    data['SibSp']=data['SibSp']*0.02
    data['Parch']=data['Parch']*0.02
    
    data['Title'] = data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
    
    data['SureName']= data.apply(SureName,axis=1)
    
    data['Survivingfam']=data.apply(Survivingfam,axis=1)
    data['DeadFam']=data.apply(DeadFam,axis=1)
    data['Travel_Alone']=data.apply(travel_alone,axis=1) 
    data['Unknown_Fare']= data.apply(Unknown_Fare,axis=1)
    
    data.drop(columns=['Name','Sex','Cabin','Embarked','Pclass'], inplace=True)
    data.drop(columns=['Ticket'], inplace=True)
    
    data['Deck']=data['Deck'].round(4)
    data['Age']=data['Age'].round(3)    
    return data
##

dt= prepare_data(data=dt)        
df =prepare_data(data=df)
show_table(df,2)
df.dtypes
#find_Nan(df,'dc nan entries')

try:
    df.drop('Survived', axis=1, inplace=True)
    dc =  df.append(dt) 
except:
    print('Survived likely deleted in previou run of this cell')


dc['Age'].replace(0,np.NaN)
dc['Age']  = dc.groupby(['female','ClassUp','ClassMid','ClassLow','SibSp','Parch','harborA','harborB','harborC','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
dc['Age'].replace(0,np.NaN)
dc['Age']  = dc.groupby(['female','ClassUp','ClassMid','ClassLow','SibSp','Parch','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
dc['Age'].replace(0,np.NaN)
dc['Age']  = dc.groupby(['female','ClassUp','ClassMid','ClassLow','SibSp','Parch'])['Age'].transform(lambda x: x.fillna(x.median()))
dc['Age'].replace(0,np.NaN)
dc['Age']  = dc.groupby(['female','Title'])['Age'].transform(lambda x: x.fillna(x.median()))

# The fact that we dont know age information from some people, and not from others, is that fact itself informative too ?.
# I made allready an AgeUnknown column, now where that is 1 or 0 i'll move value of dc['Age'] into it and make age zero.
# The thinking is some people who survived we know age of, so knowing age might be also hide a survival factor information.

# for index, row in dc.iterrows():
#   u = row['AgeUnknown']
#   if (u>0):
#     w = dc.at[index,'Age']
#     dc.at[index,'AgeUnknown']=w
#     dc.at[index,'Age']=0

# for index in dc.index:
#   rr = dc.at[index,'AgeUnknown']
#   r=rr.item(0)
#   if (r>0):
#       w= dc.at[index,'Age'].item(0)
#       dc.at[index,'AgeUnknown']=w
#       dc.at[index,'Age']=0

# # indexes where column AgeUnknown is >0
# inds = dc[dc['AgeUnknown'] > 0].index.tolist()
# # change the indexes of AgeUnknown to to the Age column
# df.loc[inds, 'AgeUnknown'] = dc.loc[inds, 'Age']
# # change the Age to 0 at those indexes
# dc.loc[inds, 'Age'] = 0

dc['Ageinv']=dc.apply(inverseAge,axis=1)


dc['Fare'].replace(0,np.NaN)
dc['Fare'] =            dc.groupby(['female','DatingRank','harborA','harborB','harborC',
                                    'ClassUp','ClassMid','ClassLow','Family_Size','Title' ])['Fare'].transform(lambda x: x.fillna(x.median()))

dc['Fare_Per_Person'].replace(0,np.NaN)
dc['Fare_Per_Person'] = dc.groupby(['female','DatingRank','harborA','harborB','harborC',
                                    'ClassUp','ClassMid','ClassLow','Family_Size','Title' ])['Fare_Per_Person'].transform(lambda x: x.fillna(x.median()))

dc['Fare'].replace(0,np.NaN)
dc['Fare'] =            dc.groupby(['ClassUp','ClassMid','ClassLow','Family_Size' ])['Fare'].transform(lambda x: x.fillna(x.median()))

dc['Fare_Per_Person'].replace(0,np.NaN)
dc['Fare_Per_Person'] = dc.groupby(['ClassUp','ClassMid','ClassLow','Family_Size'])['Fare_Per_Person'].transform(lambda x: x.fillna(x.median()))


#dc['Fare_Per_Person_Rounded'] = dc['Fare_Per_Person'].round(3)  

for index in dc.index:
    rr = dc.at[index,'Unknown_Fare']
    r=rr.item(0)
    if (r>0):
      w= dc.at[index,'Fare_Per_Person']
      dc.at[index,'Unknown_Fare']=w
      dc.at[index,'Fare_Per_Person']=0

dc.drop(['PassengerId'], axis=1, inplace=True) #they're kinda redundant now    
    
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
    
dc.Title = dc.Title.map(normalized_titles)

# turn titles into boolean columns
# https://stackoverflow.com/questions/36544089/pandas-create-boolean-columns-from-categorical-column/36544125

def normalize_column (data, name):
  data[name]=((data[name]-data[name].min())/(data[name].max()-data[name].min()))
  return data

dc =normalize_column(dc,'Fare')
dc =normalize_column(dc,'Age')
dc =normalize_column(dc,'Deck')
dc =normalize_column(dc,'Fare_Per_Person')
dc =normalize_column(dc,'SibSp')
dc =normalize_column(dc,'Parch')
dc =normalize_column(dc,'RelationsOnBoard')

dr = dc.Title.str.get_dummies()
dr.columns = ['is_'+col for col in dr.columns]

dc.reset_index(drop=True, inplace=True)
dr.reset_index(drop=True, inplace=True)
dc = pd.concat([dc,dr],axis=1)

import math
def age_inverted(x):
  a = x['Age']
  return 1-math.sqrt(a)
  
  

dc['Age']= dc.apply(age_inverted,axis=1)

plotdata =dc['Age']
plt.hist(plotdata*100)
plt.title("Gaussian Histogram")
plt.xlabel("Age-group")
plt.ylabel("Frequency")
plt.hist(plotdata*100, bins=40)
plt.hist(plotdata*100, bins=160)
plt.show()

# creating all family names (is a bit to much try to do it only for certain family size.)


# fam = pd.get_dummies(dc.SureName)
# dc.reset_index(drop=True, inplace=True)
# fam.reset_index(drop=True, inplace=True)
# dc = pd.concat([dc,fam],axis=1)

# Nice function but it resulted in a terrible score..
# dc['CatFare']=pd.qcut(dc.Fare_Per_Person,q=8,labels=False)
# dc['CatAge']=pd.qcut(dc.Age,q=15,labels=False)


dc.drop(['Title','DatingRank','SureName'], axis=1, inplace=True) 

X = dc.iloc[:891]
T  = dc.iloc[891:]
# y is allready created in the first jupyter cell.

try:
    dc.to_csv("drive/My Drive/corrected.csv",index=False)  # to see all missing files
except:
    print('Warning its likely corrected.csv file is in use and so it cannot be updated')

find_Nan(dc,'dc nan entries')
print(dc.shape)
from IPython.core.display import HTML
display(HTML(df[0:8].to_html()))
dc.drop(['SibSp','RelationsOnBoard'], axis=1, inplace=True) #if you want you can drop more columns here, droping Age is no problem !

Lzero =[len((dc.columns))]
print (Lzero)

X = dc.iloc[:891]
T  = dc.iloc[891:]
from sklearn.model_selection import train_test_split

# Despite that i wont use this, essentially i will choose best network by as much known data (to fit the best classifier).
# I keep it in here in case you would like to use it.
X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=41)   
print (X_train.shape)
print (X_test.shape)
print(y_train.shape)
print(y_test.shape)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 400)
display(dc[:17])
from keras.models import Sequential
from keras.layers import Dense,Dropout,advanced_activations
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV, KFold

# Define a random seed, so all nets start the same (so you can compare them better)
seed = 6  
np.random.seed(seed)

# Start defining the model
def create_model(L0,L1,L2,loss,optimizer,dropout_rate,activationfunction):
    # create model
    model = Sequential()
    model.add(Dense(L1, input_dim =L0, kernel_initializer='normal', activation='relu'))
    
#     act = advanced_activations.PReLU(weights=None, alpha_initializer="zero")
#     model.add(Dense(L1, input_dim=31, kernel_initializer="uniform"))
#     model.add(act)
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(L2, input_dim = L1, kernel_initializer='normal', activation='relu'))
#     act = advanced_activations.PReLU(weights=None, alpha_initializer="zero")
#     model.add(Dense(L2, input_dim=31, kernel_initializer="uniform"))
#     model.add(act)
    
    model.add(Dense(1, activation='sigmoid')) # Use sigmoid for classification answers (ea 1 or 0 )
    
    # compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model)


#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='binary_crossentropy', optimizer='rmsprop',  metrics=['accuracy'],n_jobs=-1)   #n-jobs-1  seems to crash google colab, unsure for kaggle

# in an array diffrent parameters can be givven to compare different neural net settings

Lzero =[len((dc.columns))]
parameters_set2 ={'L0': Lzero,             # dont alter it automatically adjust to the data column size
                  'L1': [17],              # [7,11,15,17,25]
                  'L2': [7],               # [3,5,7,9,11]
                  'dropout_rate':[0.250],  #[0.0,2,2.25,0.5],
                  'batch_size': [32],      #[32,64,128 ], 
                  'epochs': [200,50],       #[2000,1000,800000]
                  'loss': ['binary_crossentropy'],   #['rmsprop','Nadam','Adamax','Adadelta','Adam'], 
                  'optimizer': ['rmsprop'],          #note usually relu is best.
                  'activationfunction' : ['relu']    # ['relu','tanh','logistic'] 
                  
                  #'verbose': [3]
                 }

#if you got something good you can create new parameter sets easily 
grid = GridSearchCV(model, parameters_set2)  

# how long is your coding running... you might want to know so display the current time
import time
localtime = time.asctime( time.localtime(time.time()) )
print ("Local current time :", localtime)
grid.fit(X,y,verbose=0)   # verbose info is nice, but it slows down a lot as well.

best_est = grid.best_estimator_

localtime = time.asctime( time.localtime(time.time()) )
print ("Local current time :", localtime)
print(best_est)

trainscore = best_est.predict(X)
predictions = best_est.predict(T)  #remind that T is our Kaggle chalange our unknown set...

submissions = pd.DataFrame({'PassengerId' : ids })
submissions['Survived']=predictions[:,0]

i=1
# Incremental file saving of results
import os
outputname = folderpath+"submit_v"
fname = outputname+str(i)+".csv"
print("prepare saving..")
while os.path.exists(fname):
    i += 1
    fname = outputname+str(i)+".csv"
submissions.to_csv(fname,index=False)


# Showing results
print ("Done, results are in ",fname)
print(grid.best_params_)   #showing best parameters so you can refine it later.
print()
kk=dc.columns.get_values().tolist()
makeitastring = ' '.join(map(str, kk))
print("Used :",makeitastring)
print()
print(submissions)
