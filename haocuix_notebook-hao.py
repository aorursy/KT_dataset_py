import pandas as pd

import numpy as np

from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder



dataset = pd.read_csv('../input/dbforgenerator/df_test2.csv')

anomalies = dataset.loc[:,"anomalieIns"]

print(dataset)
object_cols = [col for col in dataset.columns if dataset[col].dtype == "object"]

object_nunique = list(map(lambda col: dataset[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))
#dataset['code-postal']=dataset['code-postal'].map(lambda x:x[0:2])

#print(dataset[['code-postal']])

#print(dataset[['code-postal']])

#dataset[['code-postal']]=((dataset[['code-postal']]//10)//10)//10

#print(dataset[['code-postal']].head())
#print(dataset.loc[0:100,"code-postal"])
# Print number of unique entries by column, in ascending order

print(sorted(d.items(), key=lambda x: x[1]))

# Columns that will be one-hot encoded

#low_cardinality_cols = [col for col in object_cols if dataset[col].nunique() < 10]





#####################change 

#1 for numbers  2 for all the categorials



low_cardinality_cols = dataset[['age', 'bonus','capital_gain','capital_loss','hours_per_week','education_num']]

#low_cardinality_cols = dataset[['workclass', 'education','marital_status','occupation','relationship','race','gender','income_bracket.']]

#low_cardinality_cols = dataset[['age', 'bonus','hours_per_week']]



######################comment these lines when testing numbers

# Columns that will be dropped from the dataset

# high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

# #

# OH_encoder=OneHotEncoder(handle_unknown='ignore', sparse=False)

# #

# OH_cols_train=pd.DataFrame(OH_encoder.fit_transform(low_cardinality_cols))



########################



#######################1 for numbers 2 for all the categorials

dataset=low_cardinality_cols

#dataset=OH_cols_train
#set variable

rs = np.random.RandomState(169);

outliers_fraction = 0.05;lendata = dataset.shape[0]

#label

anomaly = [];test_data = []

#sit normalize limited

nmlz_a = -1;nmlz_b = 1;



#some function is useful but not here

#def normalize(dataset,a,b):

#    scaler = MinMaxScaler(feature_range=(a, b))

#    normalize_data = scaler.fit_transform(dataset)

#    return normalize_data


ifm = IsolationForest(n_estimators=100, verbose=2,contamination=0.05, n_jobs=2,

                      max_samples="auto")



if __name__ == '__main__':

    Iso_train_dt = dataset

    ifm.fit(Iso_train_dt)

#######################predict  

    y=ifm.predict(Iso_train_dt)

    

    

    for i in range(len(y)):

        if (y[i]==-1):

            y[i]=1

        else:

            y[i]=0

    plt.show()



#    ifm.fit(Iso_train_dt)

    scores_pred = ifm.decision_function(Iso_train_dt)

    

    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)





    for i in scores_pred:

        if i <= threshold:

            #print(i)

            test_data.append(1)

            anomaly.append(i)

        else:

            test_data.append(0)





    print("number data：",len(dataset),"个")

    print("number of anomaly：",len(anomaly),"个")

    

    for i in range(3000,3100):

        if (test_data[i]==1):

            print('anomaly by decision',i)

    plt.show()
print(sum(anomalies))

print(type(anomalies),type(test_data))
counter = 0

sample = range(10000)



#Percentage de anomalies détectés

for i in sample:

        if (y[i]==1):

            counter=counter+1

print("%detected :",100*counter/len(sample))



#Percentage de anomalies inserées

counter = 0

for i in sample:

        if (anomalies[i]==1):

            counter=counter+1

print("%inserted :",100*counter/len(sample))



#Percentage de anomalies inserted non detectes:

counter = 0

for i in sample:

        if ((anomalies[i]-y[i])==1):

            counter=counter+1

print("%inserted not detected :",100*counter/sum(anomalies[list(sample)]))



#Percentage de anomalies not inserted detectes:

counter = 0

sum_detected = 0

for i in sample:

        if (y[i]==1):

            sum_detected=sum_detected+1

        if ((y[i]-anomalies[i])==1):

            counter=counter+1

print("%not inserted detected :",100*counter/sum_detected)





#Erreurs dans la detection de inserted:

counter = 0

for i in sample:

        if (np.abs(y[i]-anomalies[i])==1):

            counter=counter+1

print("%totalErreur : ",100*counter/len(sample))

dbanalyse = dataset.copy()

dbanalyse.at[:,'score'] = 0

for i in range(len(dbanalyse)):

    dbanalyse.at[i,'score'] = scores_pred[i];