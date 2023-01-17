import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import warnings  

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data
accuracy = 0

k = 3

power_1_num = 0

power_2_num = 0

power_3_num = 0

power_4_num = 0

const_1_num = 0

const_2_num = 0

const_3_num = 0

const_4_num = 0

for power_1 in range(-k,k):

    for power_2 in range(-k,k):

        for power_3 in range(-k,k):

            for power_4 in range(-k,k):

                for const_1 in range(-k,k):

                    for const_2 in range(-k,k):

                        for const_3 in range(-k,k):

                            for const_4 in range(-k,k):

            

                                data['score'] = const_1 * data['SepalLengthCm']**power_1 + const_2 * data['SepalWidthCm']**power_2 + const_3 * data['PetalLengthCm']**power_3 + const_4 * data['PetalWidthCm']**power_4

                                X = data['score'].values

                                y = data['Species']

                                clf = LogisticRegression(random_state=42)

                                clf.fit(X.reshape(-1,1),y)

                                predictions = clf.predict(X.reshape(-1,1))

                                if accuracy_score(predictions,y) > accuracy:

                                    accuracy = accuracy_score(predictions,y)

                                    power_1_num = power_1

                                    power_2_num = power_2

                                    power_3_num = power_3

                                    power_4_num = power_4

                                    const_1_num = const_1

                                    const_2_num = const_2

                                    const_3_num = const_3

                                    const_4_num = const_4

                                    print(accuracy)



print(power_1_num,power_2_num,power_3_num,power_4_num,const_1_num,const_2_num,const_3_num,const_4_num)