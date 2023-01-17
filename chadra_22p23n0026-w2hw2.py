#for kagle

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd


if 'google.colab' in str(get_ipython()):
  print('Running on CoLab')
  df_train = pd.read_csv('/content/drive/My Drive/DataFromKaggle/titanic/train.csv')
  df_test = pd.read_csv('/content/drive/My Drive/DataFromKaggle/titanic/test.csv')
else:
  print('Running on Something else')

  df_train = pd.read_csv('../input/titanic/train.csv')
  df_test = pd.read_csv('../input/titanic/test.csv')
  #df_train = pd.read_csv('Google Drive/DataFromKaggle/titanic/train.csv')
  #df_test = pd.read_csv('Google Drive/DataFromKaggle/titanic/test.csv')


print('train shape: ',df_train.shape)
print('test shape : ',df_test.shape)

df_train.head()
df_test.head()
#see the information
df_train.info()
df_test.info()
#Check null value in the test set
df_train.isnull().sum()
#Check null value in the train set
df_test.isnull().sum()
#ลองหา คนที่ค่า fare = null
np.argmax(df_test['Fare'].isnull())
df_test.loc[152]
#ตรวจความสัมพันธ์ของ Pclass กับ Cabin
df_train.groupby(['Pclass'])[['Cabin']].count()
df_test.groupby(['Pclass'])[['Cabin']].count()
df_train.groupby(['Pclass','Embarked'])[['Embarked']].count()
df_test.groupby(['Pclass','Embarked'])[['Embarked']].count()
#เราพบว่า คนที่ 152 ใน test set อยู่ Pclass = 3 จึงกำหนดให้ ค่า fair ของคนนั้น = ค่าเฉลี่ยของ class 3 
df_train.groupby(['Pclass'])[['Fare']].mean()
df_test.groupby(['Pclass'])[['Fare']].mean()
# Imports needed for the script
import numpy as np
import pandas as pd
import re
import xgboost as xgb #Ensemble meathod -> boosting เอาโมเดลมาต่อๆกัน
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
full_data = [df_train, df_test]

# Create new feature FamilySize เพราะขนาดของครอบครัวที่มาด้วยกันอาจมีส่วน
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Replace null ด้วย S (Southampton) เนื่องจากพบว่าคนส่วนใหญ่ในทุกคลาส ขึ้นจาก Southampton
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# เนื่องจากมีแค่คนเดียวที่ขอมูล fare และตรวจสอบพบว่าเขาอยุ่ Pclass 3 จึงกำหนดให้ far = np.mean([Pclass = 3 trainset], [Pclass = 3 testset])
for dataset in full_data:
       dataset['Fare'] = dataset['Fare'].fillna(np.mean([13.675550,12.459678]))

# แทนที่ค่าอายุที่หายไปด้วยการสุ่ม
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)


for dataset in full_data:
    # Mapping Sex femal
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare กำหนดเป็น 4 label
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						                  = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							                    = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age กำหนดเป็น 4 label 
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					               = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] ;
# เก็บ
PassengerId = df_test['PassengerId']
# ตัดแถวที่จะไม่ถูกมาใช้คำนวณออก
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_train = df_train.drop(drop_elements, axis = 1)
df_test  = df_test.drop(drop_elements, axis = 1)
print(df_train.head(3))
print(df_test.head(3))
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# Define function to calculate Gini Impurity
def get_gini_impurity(survived_count, total_count):
    survival_prob = survived_count/total_count
    not_survival_prob = (1 - survival_prob)
    random_observation_survived_prob = survival_prob
    random_observation_not_survived_prob = (1 - random_observation_survived_prob)
    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob
    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob
    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob
    return gini_impurity
df_train.groupby(['Survived'])[['Survived']].count()
df_train['Survived'].count()
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])
# แบ่งข้อมูล train เป็น 5 folds เพื่อหา ความลึกของ tree ที่ดีที่สุด โดยใช้ crossvalidation
# ซึ่งความลึกที่จะถูกทดสอบขึ้นอยู่กับจำนวนณ Features ในที่นี้เรามี 8 feature
cv = KFold(n_splits=5)            
accuracies = list()
max_attributes = len(list(df_test))
depth_range = range(1, max_attributes + 1)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth,criterion ="gini")
    #print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(df_train):
        f_train = df_train.loc[train_fold] # Extract train data with cv indices
        f_valid = df_train.loc[valid_fold] # Extract valid data with cv indices

        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 
                               y = f_train["Survived"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 
                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n")
    
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))
#Test DecisionTree
def DecitionTreeCF(df_train,df_test,max_depth):
    # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    y_train = df_train['Survived']
    x_train = df_train.drop(['Survived'], axis=1).values 
    x_test = df_test.values
    
    decision_tree = tree.DecisionTreeClassifier(max_depth = max_depth,criterion ="gini")
    decision_tree.fit(x_train, y_train)

    # Predicting results for test dataset
    y_pred = decision_tree.predict(x_test)
    submission = pd.DataFrame({
            "PassengerId": PassengerId,
            "Survived": y_pred
        })
    submission.to_csv('submission.csv', index=False)
    return decision_tree,x_train,y_train

# Create Decision Tree with max_depth = 5
decision_tree,x_train,y_train = DecitionTreeCF(df_train,df_test,5)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print('Accuracy = ',acc_decision_tree ,'%')


# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 5,
                              impurity = True,
                              feature_names = list(df_train.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )

#Convert .dot to .png to allow display in web notebook

import os
from subprocess import check_call
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
    
# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('sample-out.png')
PImage("sample-out.png")



# Code to check available fonts and respective paths
# import matplotlib.font_manager
# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')


def DecitionTreeCF(df_train,df_test,max_depth):
    #print(df_train)
    #print(df_test)
    
    y_train = df_train['Survived']
    x_train = df_train.drop(['Survived'], axis=1).values 

    y_true = df_test['Survived']
    x_test = df_test.drop(['Survived'], axis=1).values 
    #print(x_test)
     
    
    decision_tree = tree.DecisionTreeClassifier(max_depth = max_depth,criterion ="gini")
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)
    return y_true,y_pred
    
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

def NB(df_train,df_test):  

    y_train = df_train['Survived']
    x_train = df_train.drop(['Survived'], axis=1).values 

    y_true = df_test['Survived']
    x_test = df_test.drop(['Survived'], axis=1).values 

    gnb = GaussianNB()
 
    gnb.fit(x_train,y_train)
    
    y_pred = gnb.predict(x_test)
    return y_true,y_pred
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
'''
def ScaledData(df_train,df_test):
    scaler_train = MinMaxScaler()
    scaler_test  = MinMaxScaler()
    scaled_train = scaler_train.fit_transform(df_train[:,1:])
    scaled_test = scaler_y.fit_transform(df_test)
    return scaler_train,scaler_test,scaled_train,scaled_test
'''

def MLP(df_train,df_test):


    y_train = np.array(df_train['Survived'])
    x_train = np.array(df_train.drop(['Survived'], axis=1).values)

    y_true = np.array(df_test['Survived'])
    x_test = np.array(df_test.drop(['Survived'], axis=1).values) 

    d_in    = np.array(x_train)[0].shape

    #print("training            | target")
    #print(print(x_train[0,:],'  |',y_train[0]))
    
    #print(d_in)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, input_shape=d_in))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(tf.keras.layers.Dense(20))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                  metrics=['accuracy']
                  )

    #model.summary()
    
    history = model.fit(x_train, y_train, epochs=500,verbose=0)

    '''
    plt.plot(history.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    '''
    y_pred = model.predict(x_test).round()

    return y_true,y_pred
    
    
    
from sklearn.metrics import recall_score,precision_score
def evaluate(y_true, y_pred, modelname):
  #Recall ค่าความแม่นยำ
  print(modelname)
  re_scr = recall_score(y_true, y_pred, average='micro')
  print('   ','Recall score = ',re_scr)

  #Precision
  preci_scr = precision_score(y_true, y_pred, average='micro')
  print('   ','Precision score = ',preci_scr)

  #F-Measure = 2 x Precision x Recall / (Precision + Recall)
  FM_scr = (2 * preci_scr * re_scr) / (preci_scr + re_scr)
  print('   ','F-Measure = ',FM_scr)
  #all_fm.append(FM_scr)
  return FM_scr
from sklearn.model_selection import KFold#
cv = KFold(n_splits=5)            
accuracies = list()
fold_accuracy = []
i_flod = 1
FM_deci_all = []
FM_naiv_all = []
FM_mlp_all =  []


for train_fold, valid_fold in cv.split(df_train):
    f_train = df_train.loc[train_fold] # Extract train data with cv indices
    f_valid = df_train.loc[valid_fold] # Extract valid data with cv indices
    
    print('Flod #',i_flod)
    #Decition Tree
    y_true,y_pred = DecitionTreeCF(f_train,f_valid,5)
    FM_deci_all.append(evaluate(y_true,y_pred,'DecitionTreeCF'))

    #Naïve Bayes
    y_true,y_pred = NB(f_train,f_valid)
    FM_naiv_all.append(evaluate(y_true,y_pred,'Naïve Bayes'))
    
    #Multilayer perceptron
    y_true,y_pred = MLP(f_train,f_valid)
    FM_mlp_all.append(evaluate(y_true,y_pred,'Multilayer perceptron'))

    i_flod += 1
    print('---------------------------------------------')

    #avg = sum(fold_accuracy)/len(fold_accuracy)
#print(FM_deci_all)
#print(FM_naiv_all)
#print(FM_mlp_all)
print('\n')
print('Average F-Measure')
print('Decition Tree : ',np.mean(FM_deci_all))
print('Naïve Bayes : ',np.mean(FM_naiv_all))
print('Multilayer perceptron : ',np.mean(FM_mlp_all))
