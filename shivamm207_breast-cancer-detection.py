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
import pandas as pd

df = pd.read_csv('../input/breast-cancer-csv/breastCancer.csv')

df.head()
df.info()
df = df.drop(columns=["id"])

df.head()
df['bare_nucleoli'].value_counts()
import numpy as np

df_absent = df[df['bare_nucleoli']=='?']

df_absent = df_absent.reset_index()

df_absent = df_absent.drop(columns=['index'])

df_absent.head()
df_present = df[df['bare_nucleoli']!='?']

df_present = df_present.reset_index()

df_present = df_present.drop(columns=["index"])

df_present = df_present.astype(np.float64)

df_present.head()
df_present_temp = df_present.drop(columns=['bare_nucleoli'])

xm = df_present_temp.values



ym = df_present['bare_nucleoli'].values



from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(xm, ym, test_size=0.2, random_state=4)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt



k_min = 2

test_MAE_array = []

k_array = []

MAE = 10^12



for k in range(2, 20):

    model = KNeighborsRegressor(n_neighbors=k).fit(train_x, train_y)

    

    y_predict = model.predict(test_x)

    y_true = test_y



    test_MAE = mean_absolute_error(y_true, y_predict)

    if test_MAE < MAE:

        MAE = test_MAE

        k_min = k



    test_MAE_array.append(test_MAE)

    k_array.append(k)



plt.plot(k_array, test_MAE_array,'r')

plt.show()



print("Best k parameter is ",k_min )
final_model = KNeighborsRegressor(n_neighbors=16).fit(xm,ym)



df_absent_temp = df_absent.drop(columns=['bare_nucleoli'])

df_absent_temp = df_absent_temp.astype(np.float64)

df_absent_temp.head()
x_am = df_absent_temp.values

y_am = final_model.predict(x_am)

y_am
y_am = np.round(y_am)

y_am = y_am.astype(np.int64)

y_am
df_pred = pd.DataFrame({'bare_nucleoli':y_am})

data_frame_1 = df_absent_temp.join(df_pred)

data_frame_1 = data_frame_1.astype(np.int64)

data_frame_1
df_join_2 = df_present['bare_nucleoli']

data_frame_2 = df_present_temp.join(df_join_2)

data_frame_2 = data_frame_2.astype(np.int64)

data_frame_2.head()
data_frame = [data_frame_1, data_frame_2]

data_frame = pd.concat(data_frame)

data_frame.head()
df['class'].value_counts()
import matplotlib.pyplot as plt



fig, ax = plt.subplots(1,1)



ax.pie(df['class'].value_counts(),explode=(0,0.1), autopct='%1.1f%%', labels = ['Benign', 'Malignant'], colors=['g','r'])

plt.axis = 'equal'
data_frame.hist(figsize = (10, 10))

plt.show()
data_frame_1 = data_frame



def num_to_class(x):

    if x==2:

        return 'Benign'

    elif x==4:

        return 'Malignant'



data_frame_1['class'] = data_frame_1['class'].apply(lambda x: num_to_class(x))

data_frame_1['class'].value_counts()
import seaborn as sns

for i in range(8):

    x = data_frame.iloc[:,i]

    for j in range(i+1,8):

        y = data_frame.iloc[:,j]

        hue_parameter = data_frame['class']

        ax = sns.scatterplot(x=x, y=y, hue=hue_parameter)

        plt.show()
X = data_frame_1.drop(columns='class').values

Y = data_frame_1['class'].values



from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=4)



from sklearn.svm import SVC

model = SVC()

model.fit(train_x, train_y)



y_true = test_y

y_predict = model.predict(test_x)
from sklearn.metrics import confusion_matrix

confusion_matrix_1 = confusion_matrix(y_true, y_predict)

print(confusion_matrix_1)
sns.heatmap(confusion_matrix_1, annot=True)
from sklearn.metrics import classification_report

print(classification_report(y_true, y_predict))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_true, y_predict))