import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from xgboost import XGBClassifier

from heapq import nsmallest



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/shopping_new.csv")

data.shape
data.head()
X = data.drop('102', axis=1)

Y = data['102']
X.head()
Y.head()
plt.figure(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, center=0, cmap='YlGnBu')
print(data['102'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(data['102'], color='#32A8A2', bins=100, hist_kws={'alpha': 0.4});
seed = 2000

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
print("Total Number of Test Smaples : ", len(Y_test))
# Model Training 

model = XGBClassifier(num_class = len(Y.unique()),objective='multi:softmax',)

model.fit(X_train, Y_train,  verbose=True)
predicted = model.predict(X_test)

predictions = [round(value) for value in predicted]

print(classification_report(Y_test, predicted))
data2 = pd.DataFrame(columns=['Y_test', 'predicted'])

data2['Y_test'] = Y_test

data2['predicted'] = predicted

result_unique = pd.DataFrame()

column_val = data[X.columns.tolist()].values.ravel()

unique_values = pd.unique(column_val)

UNfb = pd.DataFrame(unique_values)

result_unique['u_val'] = unique_values
from heapq import nsmallest, nlargest



result = pd.DataFrame()

result['predicted'] = predicted

result['Y_test'] = Y_test.tolist()

result[result['predicted'] == result['Y_test']]



flag_counter = 0



encoded_list = []

for i in range(len(result)):

    encoded_list.append(nsmallest(5, result_unique['u_val'], key=lambda x: abs(x-result['predicted'].iloc[i])))

counter = 0

label_list = []

p_list = []



for i in encoded_list:

    flag = 0

    p_val = result['Y_test'].iloc[counter]

    list_item = []

    for j in range(len(i)):

        if p_val == i[j]:

            list_item.extend((i[j - 4], i[j - 3], i[j - 2], i[j - 1], i[j]))

            list_item.sort()

            p_list.append(list_item)

            flag = 1

    if flag==1:

        label_list.append("True")

    else:

        f_val = 5

        list_item.extend((i[j - 4], i[j - 3], i[j - 2], i[j - 1], i[j]))

        list_item.sort()

        p_list.append(list_item)

        label_list.append("False")

    counter += 1

count = 0

    

for k in label_list:

    if k == "True":

        count += 1

result["Probabilities"] = p_list

result["True_False"] = label_list



column_values_result = result[["Y_test", "Probabilities"]]

print(column_values_result)

column_values_result = result[["Y_test", "Probabilities", "True_False"]]

print(column_values_result[(column_values_result['True_False'] == 'True')])

print("Patterns Recoginition Rate (Accuracy):",len(result[result['True_False']=='True'])/len(result)*100)