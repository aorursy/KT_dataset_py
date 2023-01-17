import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Load the datasets

train = pd.read_csv('../input/healthcare-dataset-stroke-data/train_2v.csv')

holdout = pd.read_csv('../input/healthcare-dataset-stroke-data/test_2v.csv')
print(train.shape)

train.head()
train.isna().sum()
print(len(train['id'].value_counts()) == train.shape[0])
gender_group = train.groupby(['gender'], as_index=False)

gender_group_count = gender_group.count()['stroke']

gender_group_sum = gender_group.sum()['stroke']

gender_group_percentage = gender_group_sum / gender_group_count * 100





plt.bar(x=[0,1,2], height=gender_group_percentage, tick_label=['Female', 'Male', 'Other'])

plt.title("Gender vs Stroke Risk")

plt.xlabel("Gender")

plt.ylabel("Stoke Risk %")

plt.show()
def grouped_graph(column, labels):

    group = train.groupby([column], as_index=False)

    group_count = group.count()['stroke']

    group_sum = group.sum()['stroke']

    group_percentage = group_sum / group_count * 100



    group_percentage

    

    plt.bar(x=range(0, len(labels)), height=group_percentage, tick_label=labels)

    plt.title("{} vs Stroke Risk".format(column))

    plt.xlabel("{}".format(column))

    plt.ylabel("Stoke Risk %")

    plt.show()
grouped_graph('hypertension', ['no hypertension', 'hypertension'])
grouped_graph('heart_disease', ['no heart_disease', 'heart_disease'])
grouped_graph('ever_married', ['no', 'yes'])
def get_bmi_groups(bmi):

    if bmi >= 16 and bmi <18.5:

        return "Underweight"

    elif bmi >= 18.5 and bmi < 25 :

        return "Normal weight"

    elif bmi >= 25 and bmi < 30:

        return "Overweight"

    elif bmi >= 30 and bmi < 35:

        return "Obese Class I (Moderately obese)"

    elif bmi >= 35 and bmi < 40:

        return "Obese Class II (Severely obese)"

    elif bmi >= 40 and bmi < 45:

        return "Obese Class III (Very severely obese)"

    elif bmi >= 45 and bmi < 50:

        return "Obese Class IV (Morbidly Obese)"

    elif bmi >= 50 and bmi < 60:

        return "Obese Class V (Super Obese)"

    elif bmi >= 60:

        return "Obese Class VI (Hyper Obese)"

    

    

train['bmi_group'] = train['bmi'].apply(get_bmi_groups)

labels = ["Normal Weight", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Overweight", "Underweight"]

grouped_graph('bmi_group', labels)
group = train.groupby(['work_type'], as_index=False).sum()

labels = ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children']

grouped_graph('work_type', labels)
grouped_graph('Residence_type', ['Rural', 'Urban'])
train['smoking_status'] = train['smoking_status'].fillna(-1)

train.groupby(['smoking_status'], as_index=False).sum()

labels = ['-1', 'formerly smoked', 'never smoked', 'smokes']

grouped_graph('smoking_status', labels)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



clf = RandomForestClassifier()



X = train[['age', 'hypertension', 'heart_disease',

           'avg_glucose_level']]

y = train['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

f1 = f1_score(y_test, predictions) 



print(f1)