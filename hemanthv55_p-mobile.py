# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
def data_clean(data):

    # Nan Available

    columns_to_remove = ['Also Known As','Applications','Audio Features','Bezel-less display',

                         'Build Material','Co-Processor',

                         'Display Colour','Mobile High-Definition Link(MHL)',

                         'Music', 'Email','Fingerprint Sensor Position',

                         'Games','HDMI','Heart Rate Monitor','IRIS Scanner', 

                         'Optical Image Stabilisation','Other Facilities',

                         'Phone Book','Physical Aperture','Quick Charging',

                         'Ring Tone','Ruggedness','SAR Value','SIM 3','SMS',

                         'Screen Protection','Screen to Body Ratio (claimed by the brand)',

                         'Sensor','Software Based Aperture', 'Special Features',

                         'Standby time','Stylus','TalkTime', 'USB Type-C',

                         'Video Player', 'Video Recording Features','Waterproof',

                         'USB OTG Support', 'Video Recording','Java']

    columns_to_retain=list(set(data.columns)-set(columns_to_remove))

    data=data[columns_to_retain]

    # Low Varaince

    columns_to_remove=['Architecture','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE']

    columns_to_retain=list(set(data.columns)-set(columns_to_remove))

    data=data[columns_to_retain]

    # Multi Valued

    columns_to_remove=['Architecture','Launch Date','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE','Custom UI']

    columns_to_retain=list(set(data.columns)-set(columns_to_remove))

    data=data[columns_to_retain]

    columns_to_remove=['Bluetooth','Settings','Wi-Fi','Wi-Fi Features']

    columns_to_retain=list(set(data.columns)-set(columns_to_remove))

    data=data[columns_to_retain]

    return data
train=data_clean(train)

test=data_clean(test)
train.shape,test.shape
train=train[(train.isnull().sum(axis=1)<=25)]
train.shape,test.shape
def for_integer(test):

    try:

        test = test.strip()

        return int(test.split(' ')[0])

    except IOError:

           pass

    except ValueError:

        pass

    except:

        pass



def for_string(test):

    try:

        test = test.strip()

        return (test.split(' ')[0])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass



def for_float(test):

    try:

        test = test.strip()

        return float(test.split(' ')[0])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass

def find_freq(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[2][0] == '(':

            return float(test[2][1:])

        return float(test[2])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass



    

def for_Internal_Memory(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[1] == 'GB':

            return int(test[0])

        if test[1] == 'MB':

#             print("here")

            return (int(test[0]) * 0.001)

    except IOError:

           pass

    except ValueError:

        pass

    except:

        pass

    

def find_freq(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[2][0] == '(':

            return float(test[2][1:])

        return float(test[2])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass

def data_clean_2(x):

    data = x.copy()

    

    data['Capacity'] = data['Capacity'].apply(for_integer)



    data['Height'] = data['Height'].apply(for_float)

    data['Height'] = data['Height'].fillna(data['Height'].mean())



    data['Internal Memory'] = data['Internal Memory'].apply(for_Internal_Memory)



    data['Pixel Density'] = data['Pixel Density'].apply(for_integer)



    data['Internal Memory'] = data['Internal Memory'].fillna(data['Internal Memory'].median())

    data['Internal Memory'] = data['Internal Memory'].astype(int)



    data['RAM'] = data['RAM'].apply(for_Internal_Memory)

    data['RAM'] = data['RAM'].fillna(data['RAM'].median())

    data['RAM'] = data['RAM'].astype(int)



    data['Resolution'] = data['Resolution'].apply(for_integer)

    data['Resolution'] = data['Resolution'].fillna(data['Resolution'].median())

    data['Resolution'] = data['Resolution'].astype(int)



    data['Screen Size'] = data['Screen Size'].apply(for_float)



    data['Thickness'] = data['Thickness'].apply(for_float)

    data['Thickness'] = data['Thickness'].fillna(data['Thickness'].mean())

    data['Thickness'] = data['Thickness'].round(2)



    data['Type'] = data['Type'].fillna('Li-Polymer')



    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].apply(for_float)

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].fillna(data['Screen to Body Ratio (calculated)'].mean())

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].round(2)



    data['Width'] = data['Width'].apply(for_float)

    data['Width'] = data['Width'].fillna(data['Width'].mean())

    data['Width'] = data['Width'].round(2)



    data['Flash'][data['Flash'].isna() == True] = "Other"



    data['User Replaceable'][data['User Replaceable'].isna() == True] = "Other"



    data['Num_cores'] = data['Processor'].apply(for_string)

    data['Num_cores'][data['Num_cores'].isna() == True] = "Other"





    data['Processor_frequency'] = data['Processor'].apply(find_freq)

    #because there is one entry with 208MHz values, to convert it to GHz

    data['Processor_frequency'][data['Processor_frequency'] > 200] = 0.208

    data['Processor_frequency'] = data['Processor_frequency'].fillna(data['Processor_frequency'].mean())

    data['Processor_frequency'] = data['Processor_frequency'].round(2)



    data['Camera Features'][data['Camera Features'].isna() == True] = "Other"



    #simplifyig Operating System to os_name for simplicity

    data['os_name'] = data['Operating System'].apply(for_string)

    data['os_name'][data['os_name'].isna() == True] = "Other"



    data['Sim1'] = data['SIM 1'].apply(for_string)



    data['SIM Size'][data['SIM Size'].isna() == True] = "Other"



    data['Image Resolution'][data['Image Resolution'].isna() == True] = "Other"



    data['Fingerprint Sensor'][data['Fingerprint Sensor'].isna() == True] = "Other"



    data['Expandable Memory'][data['Expandable Memory'].isna() == True] = "No"



    data['Weight'] = data['Weight'].apply(for_integer)

    data['Weight'] = data['Weight'].fillna(data['Weight'].mean())

    data['Weight'] = data['Weight'].astype(int)



    data['SIM 2'] = data['SIM 2'].apply(for_string)

    data['SIM 2'][data['SIM 2'].isna() == True] = "Other"

    

    return data
train=data_clean_2(train)

test=data_clean_2(test)
def data_clean_3(x):

    

    data = x.copy()



    columns_to_remove = ['User Available Storage','SIM Size','Chipset','Processor','Autofocus','Aspect Ratio','Touch Screen',

                        'Bezel-less display','Operating System','SIM 1','USB Connectivity','Other Sensors','Graphics','FM Radio',

                        'Shooting Modes','Display Colour' ]



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]





    columns_to_remove = [ 'Screen Resolution','Camera Features',

                        'Display Type']



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]





    columns_to_remove = ['Rating Count', 'Review Count','Image Resolution','Type','Expandable Memory',\

                        'Colours','Width','Model', 'Screen Size','Browser']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    return data
train=data_clean_3(train)

test=data_clean_3(test)
train['Num_cores'].unique()
train['Pixel Density'].median()
def data_binarize(data):

    data['os_name']= data['os_name'].map(lambda x: 1 if x in ['Android', 'iOS', 'Nokia'] else 0)

    data['Num_cores']= data['Num_cores'].map(lambda x: 1 if x in ['Octa', 'Quad', 'Deca', 'Hexa', 'Tru-Octa'] else 0)

    data['Weight'] = data['Weight'].map(lambda x: 0 if x > 165 else 1)

    data['RAM'] = data['RAM'].map(lambda x: 1 if x >= 3 else 0)

    data['Resolution'] = data['Resolution'].map(lambda x: 1 if x >= 8 else 0)

    data['Processor_frequency']= data['Processor_frequency'].map(lambda x: 1 if x>=1.4 else 0)  #mode reference

    data['Pixel Density']= data['Pixel Density'].map(lambda x: 1 if x>=308 else 0) # median refrence

    data['Sim1']= data['Sim1'].map(lambda x: 1 if x=='4G' or x=='3G' else 0)

    data['SIM 2']= data['SIM 2'].map(lambda x: 1 if x=='4G' or x=='3G' else 0)

    data['Internal Memory']= data['Internal Memory'].map(lambda x: 1 if x>=32 else 0)

    #data['Capacity']= data['Capacity'].map(lambda x: 1 if x>=3500 else 0)

    data['User Replaceable']= data['User Replaceable'].map(lambda x: 1 if x=='No' else 0)

    data['NFC']= data['NFC'].map(lambda x: 1 if x=='yes' else 0)

    data['Fingerprint Sensor']=data['Fingerprint Sensor'].map(lambda x: 1 if x=='yes' else 0)

    data['Flash']=data['Flash'].map(lambda x: 0 if x in ['No', "Other"] else 1)

    

    data['Thickness']=data['Thickness'].map(lambda x: 0 if x>=8 else 1)

    

    #data['Screen Size']= data['Screen Size'].map(lambda x: 1 if x>=4 else 0)

    

    return data

    
train=data_binarize(train)

test=data_binarize(test)
def data_remove(data):

    data=data.drop('SIM Slot(s)', axis=1)

    data=data.drop('Height',axis=1)

    data=data.drop('Screen to Body Ratio (calculated)',axis=1)

    return data
train=data_remove(train)

test=data_remove(test)
train['Rating']=train['Rating'].map(lambda x: 1 if x>=4 else 0)
train.head()
train.shape,test.shape
X = train.drop('Rating', axis=1)

X = X.drop('PhoneId', axis=1)

Y = train['Rating']

test_new= test.drop('PhoneId', axis=1)
columns_new=X.columns

bd=columns_new.get_indexer(['Brand'])[0]

cp=columns_new.get_indexer(['Capacity'])[0]

wc=columns_new.get_indexer(['Wireless Charging'])[0]

columns_new=test_new.columns

t_bd=columns_new.get_indexer(['Brand'])[0]

t_cp=columns_new.get_indexer(['Capacity'])[0]

t_wc=columns_new.get_indexer(['Wireless Charging'])[0]
X=X.values

Y=Y.values

test_new=test_new.values
def data_special(data,bd,cp,wc):

    m=data.shape[0]

    for i in range(m):

        if data[i][bd]=='Apple':

            data[i,:]=1

        if data[i][cp]<=1200:

            data[i,:]=1

        elif data[i][cp]>=3000:

            data[i][cp]=1

        else:

            data[i][cp]=0

        if data[i][wc]=='yes':

            data[i,:]=1

    return data
X=data_special(X,bd,cp,wc)

test_new=data_special(test_new,t_bd,t_cp,t_wc)
X[1,:]
test_new[1,:]
def data_special1(data,bd,wc):

    m=data.shape[0]

    brand_high=['Xiaomi', 'Realme', 'Samsung', 'Vivo', 'OPPO',

        'Xiaomi Poco', 'OnePlus', 'Asus', 'Huawei','Apple', 'Infinix', 'Google', 'Oppo', 

        'LeEco', 'Lava', 'Moto', 'Motorola','Tecno', 'Itel',1]

    for i in range(m):

        if data[i][bd] in brand_high:

            data[i][bd]=1

        else:

            data[i][bd]=0



        if data[i][wc]!=1:

            data[i][wc]=0

    return data
X=data_special1(X,bd,wc)

test_new=data_special1(test_new,t_bd,t_wc)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y,random_state=1)
X_train[1,:]
class Perceptron:

  

  def __init__ (self):

    self.w = None

    self.b = None

    

  def model(self, x):

    return 1 if (np.dot(self.w, x) >= self.b) else 0

    

  def predict(self, X):

    Y = []

    for x in X:

      result = self.model(x)

      Y.append(result)

    return np.array(Y)

    

  def fit(self, X, Y, epochs = 1, lr = 1):

    

    self.w = np.ones(X.shape[1])

    self.b = 0

    

    accuracy = {}

    max_accuracy = 0

    

    wt_matrix = []

    

    for i in range(epochs):

      for x, y in zip(X, Y):

        y_pred = self.model(x)

        if y == 1 and y_pred == 0:

          self.w = self.w + lr * x

          self.b = self.b - lr * 1

        elif y == 0 and y_pred == 1:

          self.w = self.w - lr * x

          self.b = self.b + lr * 1

          

      wt_matrix.append(self.w)    

          

      accuracy[i] = accuracy_score(self.predict(X), Y)

      if (accuracy[i] > max_accuracy):

        max_accuracy = accuracy[i]

        chkptw = self.w

        chkptb = self.b

        

    self.w = chkptw

    self.b = chkptb

        

    print(max_accuracy)

    

    plt.plot(accuracy.values())

    plt.ylim([0, 1])

    plt.show()

    

    return np.array(wt_matrix)

perceptron = Perceptron()
wt_matrix = perceptron.fit(X_train, Y_train, 1000, 0.5)
Y_pred_test = perceptron.predict(X_test)

print(accuracy_score(Y_pred_test, Y_test))
test_pred=perceptron.predict(test_new)
submission = pd.DataFrame({'PhoneId':test['PhoneId'], 'Class':test_pred})

submission = submission[['PhoneId', 'Class']]

submission.to_csv("submission.csv",index=False)
wt_matrix = perceptron.fit(X, Y, 1000, 0.5)
test_pred1=perceptron.predict(test_new)
submission1 = pd.DataFrame({'PhoneId':test['PhoneId'], 'Class':test_pred1})

submission1 = submission1[['PhoneId', 'Class']]

submission1.to_csv("submission1.csv",index=False)