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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

from sklearn.neural_network import MLPClassifier

from sklearn import metrics
df_train = pd.read_csv ( "/kaggle/input/titanic/train.csv" ) 



df_test = pd.read_csv ( "/kaggle/input/titanic/test.csv" ) 

df_gener_submission = pd.read_csv ( "/kaggle/input/titanic/gender_submission.csv" ) 
df_test = pd.merge ( df_test, df_gener_submission, on = [ "PassengerId" ], how = "inner" )

df_train = pd.concat ( [ df_train, df_test ] )

df_train
df_train.groupby ( [ "Pclass", "Embarked" ] ) [ "Fare" ].agg ( [ "min", "max", "mean", "std" ] )
# ทดสอบด้วยการไม่นำข้อมูลที่มีค่าตั๋วโดยสารเป็น 0 

# เนื่องจากเป็นไปได้ว่า ผู้โดยสารนั้น ๆ อาจได้รับสิทธิพิเศษ ให้ขึ้นเรือแบบฟรี ๆ



filter = df_train [ "Fare" ] != 0 

df_train.loc [ filter ].groupby ( [ "Pclass", "Embarked" ] ) [ "Fare" ].agg ( [ "min", "max", "mean", "std" ] )
df_train = df_train.drop ( [ "Fare" ], axis = "columns" )
df_consider_embarked = df_train.groupby ( [ "Pclass", "Embarked", "Survived" ] ) [ "PassengerId" ].agg ( [ "count" ] )

df_consider_embarked
df_consider_embarked = df_consider_embarked.reset_index ( )



filter = df_consider_embarked [ "Survived" ] == 0

df_consider_embarked_died = df_consider_embarked [ filter ]



filter = df_consider_embarked [ "Survived" ] == 1

df_consider_embarked_alive = df_consider_embarked [ filter ]



df_consider_embarked = pd.merge ( df_consider_embarked_died, df_consider_embarked_alive, on = [ "Pclass", "Embarked" ], how = "inner", suffixes = ( "_died", "_alive" ) )



df_consider_embarked [ "ratio_died" ] = df_consider_embarked [ "count_died" ] / ( df_consider_embarked [ "count_died" ] + df_consider_embarked [ "count_alive" ] )

df_consider_embarked [ "ratio_alive" ] = df_consider_embarked [ "count_alive" ] / ( df_consider_embarked [ "count_died" ] + df_consider_embarked [ "count_alive" ] )

df_consider_embarked = df_consider_embarked.drop ( [ "Survived_died", "Survived_alive" ], axis = "columns" )

df_consider_embarked
# จากกราฟที่แสดงผลด้านล่าง เมื่อเทียบใน Ticket Class เดียวกัน จะเห็นว่า ผู้โดยสารที่ขึ้นท่าเรือ Southampton

# มีอัตราการตายที่สูงกว่าเสมอ (จะไม่พิจารณาท่าเรือ Queenstown ของ Ticket Class 1st และ 2nd เนื่องจากมีข้อมูลน้อยเกินไป)



filter = df_consider_embarked [ "Pclass" ] == 1

pclass_1_y = df_consider_embarked.loc [ filter ] [ "ratio_died" ]

pclass_1_x = df_consider_embarked.loc [ filter ] [ "Embarked" ]



filter = df_consider_embarked [ "Pclass" ] == 2

pclass_2_y = df_consider_embarked.loc [ filter ] [ "ratio_died" ]

pclass_2_x = df_consider_embarked.loc [ filter ] [ "Embarked" ]



filter = df_consider_embarked [ "Pclass" ] == 3

pclass_3_y = df_consider_embarked.loc [ filter ] [ "ratio_died" ]

pclass_3_x = df_consider_embarked.loc [ filter ] [ "Embarked" ]





plt.figure ( figsize = ( 15, 5 ) )

plt.plot ( pclass_1_x, pclass_1_y, label = "Ticket Class 1st" )

plt.plot ( pclass_2_x, pclass_2_y, label = "Ticket Class 2nd" )

plt.plot ( pclass_3_x, pclass_3_y, label = "Ticket Class 3rd" )

plt.legend ( loc = "upper right" )

plt.show ( )
df_train [ "Embarked" ].unique ( )
df_train [ "embarked_s" ] = ( df_train [ "Embarked" ] == "S" ).astype ( int )

df_train [ "embarked_c" ] = ( df_train [ "Embarked" ] == "C" ).astype ( int )

df_train [ "embarked_q" ] = ( df_train [ "Embarked" ] == "Q" ).astype ( int )

df_train = df_train.drop ( [ "Embarked" ], axis = "columns" )

df_train
df_train.shape
df_train.isna ( ).sum ( )
df_train = df_train.drop ( [ "Cabin" ], axis = "columns" )
df_consider_sex = df_train.groupby ( [ "Sex", "Pclass", "Survived" ] ) [ "PassengerId" ].agg ( [ "count" ] )

df_consider_sex
df_consider_sex = df_consider_sex.reset_index ( )



filter = df_consider_sex [ "Survived" ] == 0

df_consider_sex_died = df_consider_sex [ filter ]



filter = df_consider_sex [ "Survived" ] == 1

df_consider_sex_alive = df_consider_sex [ filter ]



df_consider_sex = pd.merge ( df_consider_sex_died, df_consider_sex_alive, on = [ "Pclass", "Sex" ], how = "inner", suffixes = ( "_died", "_alive" ) )



df_consider_sex [ "ratio_died" ] = df_consider_sex [ "count_died" ] / ( df_consider_sex [ "count_died" ] + df_consider_sex [ "count_alive" ] )

df_consider_sex [ "ratio_alive" ] = df_consider_sex [ "count_alive" ] / ( df_consider_sex [ "count_died" ] + df_consider_sex [ "count_alive" ] )

df_consider_sex = df_consider_sex.drop ( [ "Survived_died", "Survived_alive" ], axis = "columns" )

df_consider_sex
filter = df_consider_sex [ "Pclass" ] == 1

pclass_1_y = df_consider_sex.loc [ filter ] [ "ratio_died" ]

pclass_1_x = df_consider_sex.loc [ filter ] [ "Sex" ]



filter = df_consider_sex [ "Pclass" ] == 2

pclass_2_y = df_consider_sex.loc [ filter ] [ "ratio_died" ]

pclass_2_x = df_consider_sex.loc [ filter ] [ "Sex" ]



filter = df_consider_sex [ "Pclass" ] == 3

pclass_3_y = df_consider_sex.loc [ filter ] [ "ratio_died" ]

pclass_3_x = df_consider_sex.loc [ filter ] [ "Sex" ]





plt.figure ( figsize = ( 15, 5 ) )

plt.plot ( pclass_1_x, pclass_1_y, label = "Ticket Class 1st" )

plt.plot ( pclass_2_x, pclass_2_y, label = "Ticket Class 2nd" )

plt.plot ( pclass_3_x, pclass_3_y, label = "Ticket Class 3rd" )

plt.legend ( loc = "upper right" )

plt.show ( )
df_train [ "Sex" ].unique ( )
df_train [ "sex_male" ] = ( df_train [ "Sex" ] == "male" ).astype ( int )

df_train [ "sex_female" ] = ( df_train [ "Sex" ] == "female" ).astype ( int )

df_train = df_train.drop ( [ "Sex" ], axis = "columns" )

df_train
df_consider_pclass = df_train.groupby ( [ "Pclass", "Survived" ] ) [ "PassengerId" ].agg ( [ "count" ] )

df_consider_pclass
df_consider_pclass = df_consider_pclass.reset_index ( )



filter = df_consider_pclass [ "Survived" ] == 0

df_consider_pclass_died = df_consider_pclass [ filter ]



filter = df_consider_pclass [ "Survived" ] == 1

df_consider_pclass_alive = df_consider_pclass [ filter ]



df_consider_pclass = pd.merge ( df_consider_pclass_died, df_consider_pclass_alive, on = [ "Pclass" ], how = "inner", suffixes = ( "_died", "_alive" ) )



df_consider_pclass [ "ratio_died" ] = df_consider_pclass [ "count_died" ] / ( df_consider_pclass [ "count_died" ] + df_consider_pclass [ "count_alive" ] )

df_consider_pclass [ "ratio_alive" ] = df_consider_pclass [ "count_alive" ] / ( df_consider_pclass [ "count_died" ] + df_consider_pclass [ "count_alive" ] )

df_consider_pclass = df_consider_pclass.drop ( [ "Survived_died", "Survived_alive" ], axis = "columns" )

df_consider_pclass
data_y = df_consider_pclass [ "ratio_died" ]

data_x = df_consider_pclass [ "Pclass" ].astype ( str )



plt.figure ( figsize = ( 15, 5 ) )

plt.plot ( data_x, data_y )

plt.show ( )
df_train [ "Pclass" ].unique ( )
# เนื่องจาก ข้อมูลที่อยู่ในคอลัมน์ Pclass เป็นการบอกว่า 1 คือ 1st, 2 คือ 2st และ 3 คือ 3rd

# ซึ่งตัวเลขนี้เปรียบเสมือนเป็นเพียง label

# ดังนั้นต้องแปลงข้อมูล label นี้ให้อยู่ในรูปแบบตัวเลข



df_train [ "ticketclass_1st" ] = ( df_train [ "Pclass" ] == 1 ).astype ( int )

df_train [ "ticketclass_2nd" ] = ( df_train [ "Pclass" ] == 2 ).astype ( int )

df_train [ "ticketclass_3rd" ] = ( df_train [ "Pclass" ] == 3 ).astype ( int )

df_train = df_train.drop ( [ "Pclass" ], axis = "columns" )

df_train
df_consider_sibsp = df_train.groupby ( [ "SibSp", "Survived" ] ) [ "PassengerId" ].agg ( [ "count" ] )

df_consider_sibsp
df_consider_sibsp = df_consider_sibsp.reset_index ( )



filter = df_consider_sibsp [ "Survived" ] == 0

df_consider_sibsp_died = df_consider_sibsp [ filter ]



filter = df_consider_sibsp [ "Survived" ] == 1

df_consider_sibsp_alive = df_consider_sibsp [ filter ]



df_consider_sibsp = pd.merge ( df_consider_sibsp_died, df_consider_sibsp_alive, on = [ "SibSp" ], how = "inner", suffixes = ( "_died", "_alive" ) )



df_consider_sibsp [ "ratio_died" ] = df_consider_sibsp [ "count_died" ] / ( df_consider_sibsp [ "count_died" ] + df_consider_sibsp [ "count_alive" ] )

df_consider_sibsp [ "ratio_alive" ] = df_consider_sibsp [ "count_alive" ] / ( df_consider_sibsp [ "count_died" ] + df_consider_sibsp [ "count_alive" ] )

df_consider_sibsp = df_consider_sibsp.drop ( [ "Survived_died", "Survived_alive" ], axis = "columns" )

df_consider_sibsp
data_y = df_consider_sibsp [ "ratio_died" ]

data_x = df_consider_sibsp [ "SibSp" ]



plt.figure ( figsize = ( 15, 5 ) )

plt.bar ( data_x, data_y )

plt.show ( )
# เมื่อพิจารณาจากข้อมูลแล้วจะพบว่า 

# ผู้โดยสารที่มีค่า SibSp เป็น 0 จะมีอัตราการเสียชีวิต มากกว่า ผู้โดยสารที่มีค่า SibSp เป็น 1 หรือ 2

# และผู้โดยสารที่มีค่า SibSp ตั้งแต่ 3 ขึ้นไป จะมีอัตราการเสียชีวิตสูงมาก

# ดังนั้นจึงมีการปรับค่า ดังนี้



df_train [ "sibsp_zero" ] = ( df_train [ "SibSp" ] == 0 ).astype ( int )

df_train [ "sibsp_little" ] = ( ( df_train [ "SibSp" ] == 1 ) | ( df_train [ "SibSp" ] == 2 ) ).astype ( int )

df_train [ "sibsp_much" ] = ( df_train [ "SibSp" ] > 2 ).astype ( int )

df_train = df_train.drop ( [ "SibSp" ], axis = "columns" )

df_train
df_consider_parch = df_train.groupby ( [ "Parch", "Survived" ] ) [ "PassengerId" ].agg ( [ "count" ] )

df_consider_parch
df_consider_parch = df_consider_parch.reset_index ( )



filter = df_consider_parch [ "Survived" ] == 0

df_consider_parch_died = df_consider_parch [ filter ]



filter = df_consider_parch [ "Survived" ] == 1

df_consider_parch_alive = df_consider_parch [ filter ]



df_consider_parch = pd.merge ( df_consider_parch_died, df_consider_parch_alive, on = [ "Parch" ], how = "inner", suffixes = ( "_died", "_alive" ) )



df_consider_parch [ "ratio_died" ] = df_consider_parch [ "count_died" ] / ( df_consider_parch [ "count_died" ] + df_consider_parch [ "count_alive" ] )

df_consider_parch [ "ratio_alive" ] = df_consider_parch [ "count_alive" ] / ( df_consider_parch [ "count_died" ] + df_consider_parch [ "count_alive" ] )

df_consider_parch = df_consider_parch.drop ( [ "Survived_died", "Survived_alive" ], axis = "columns" )

df_consider_parch
data_y = df_consider_parch [ "ratio_died" ]

data_x = df_consider_parch [ "Parch" ]



plt.figure ( figsize = ( 15, 5 ) )

plt.bar ( data_x, data_y )

plt.show ( )
# เมื่อพิจารณาจากข้อมูลแล้วจะพบว่า 

# ผู้โดยสารที่มีค่า Parch เป็น 0 จะมีอัตราการเสียชีวิต มากกว่า ผู้โดยสารที่มีค่า Parch เป็น 1 หรือ 2

# (ผู้โดยสารที่มีค่า Parch ตั้งแต่ 3 ขึ้นไป มีข้อมูลตัวอย่างน้อยมาก จึงไม่สามารถสรุปอะไรได้)

# ดังนั้นจึงมีการปรับค่า ดังนี้



df_train [ "parch_zero" ] = ( df_train [ "Parch" ] == 0 ).astype ( int )

df_train [ "parch_nonzero" ] = ( df_train [ "Parch" ] != 0 ).astype ( int )

df_train = df_train.drop ( [ "Parch" ], axis = "columns" )

df_train
df_train = df_train.drop ( [ "PassengerId", "Name", "Ticket" ], axis = "columns" )
df_train.isna ( ).sum ( )
df_train = df_train.dropna ( )
df_train
def fold_df ( df_data ) :



  df_fold_container = [ ]



  for round in range ( 5 ) : 

      filter = df_data.index % 5 == round

      df_fold_container.append ( df_data [ filter ] )



  return df_fold_container
def traintest_model ( model, df_fold ) : 



  result = { }



  for round_test in range ( len ( df_fold ) ) :

    

    each_model = model



    df_traindata = pd.DataFrame ( )

    for round_train in range ( len ( df_fold ) ) :

      if round_train != round_test :

        df_traindata = pd.concat ( [ df_traindata, df_fold [ round_train ] ] )

    

    each_model.fit ( df_traindata.drop ( [ "Survived" ], axis = "columns" ), df_traindata [ "Survived" ] )

    

    SURVIVED_PREDICT = each_model.predict ( df_fold [ round_test ].drop ( [ "Survived" ], axis = "columns" ) )

    SURVIVED_REAL = df_fold [ round_test ] [ "Survived" ]

    

    recall_score = metrics.recall_score ( SURVIVED_REAL, SURVIVED_PREDICT )

    precision_score = metrics.precision_score ( SURVIVED_REAL, SURVIVED_PREDICT )

    f1_score = metrics.f1_score ( SURVIVED_REAL, SURVIVED_PREDICT )

    accuracy_score = metrics.accuracy_score ( SURVIVED_REAL, SURVIVED_PREDICT )

    

    result [ "FOLD " + str ( round_test ) ] = list ( [ recall_score, precision_score, f1_score, accuracy_score ] )



  df_result = pd.DataFrame ( result, ).rename ( index = { 0 : "RECALL", 1 : "PRECISION", 2 : "F1_MEASURE", 3 : "ACCURACY" } ).T

  result_mean = df_result.mean ( ).reset_index ( ) [ 0 ]



  return df_result, result_mean
result_model = { }
df_fold_container = fold_df ( df_train )

df_fold_container
result, result_mean = traintest_model ( DecisionTreeClassifier ( criterion = "gini", max_depth = 4 ), df_fold_container )

result_model [ "DecisionTree" ] = result_mean



result
result, result_mean = traintest_model ( GaussianNB ( ), df_fold_container )

result_model [ "GaussianNB" ] = result_mean



result
result, result_mean = traintest_model ( MultinomialNB ( ), df_fold_container )

result_model [ "MultinomialNB" ] = result_mean



result
result, result_mean = traintest_model ( ComplementNB ( ), df_fold_container )

result_model [ "ComplementNB" ] = result_mean



result
result, result_mean = traintest_model ( BernoulliNB ( ), df_fold_container )

result_model [ "BernoulliNB" ] = result_mean



result
result, result_mean = traintest_model ( MLPClassifier ( ), df_fold_container )

result_model [ "MLPClassifier" ] = result_mean



result
pd.DataFrame ( result_model, ).rename ( index = { 0 : "RECALL", 1 : "PRECISION", 2 : "F1_MEASURE", 3 : "ACCURACY" } ).T