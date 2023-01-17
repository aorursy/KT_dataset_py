"""

load in the training data



"""

import pandas as pd

import numpy as np



df_training_full = pd.read_csv("../input/kernel52a784bb29-files/training.csv", na_values=['na'])



df_training_full.head()
df_test = pd.read_csv("../input/kernel52a784bb29-files/test.csv", na_values=['na'])

df_test.head()
df_training_full['Type'] = 'Train' 

df_test['Type'] = 'Test'

all_data = pd.concat([df_training_full,df_test],axis=0)

all_data.dtypes
all_data
numeric_cols = list()

for k,v in all_data.iteritems():

    if v.dtype in ['int64', 'float64']:

        numeric_cols.append(k)





#fullData[numeric_cols] = fullData[numeric_cols].fillna(fullData[numeric_cols].mean())

#fullData.head()
print("numeric columns:", numeric_cols)

# this spams
#import some nice packages that everyone likes





import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt 

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn





from scipy import stats





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))



print("done")
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(100)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

plt.show()
all_data1 = all_data.replace({'0':np.nan, 0:np.nan})

all_data_zero = (all_data1.isnull().sum() / len(all_data1)) * 100

all_data_zero = all_data_zero.drop(all_data_zero[all_data_zero == 0].index).sort_values(ascending=False)[:30]

zero_data = pd.DataFrame({'Zero Density' :all_data_zero})

zero_data.head(100)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_zero.index, y=all_data_zero)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of zero or nan', fontsize=15)

plt.title('Percent zero or nan by feature', fontsize=15)

plt.show()
corrmat = df_training_full.corr()

plt.subplots(figsize=(60,60))

sns.heatmap(corrmat, vmax=0.9, square=True)

plt.show()
# lots of exponential distributions... lets log transform



for c in numeric_cols:

    data = df_training_full[c].replace({0:np.nan}).dropna()

    try:

        if data.mean(axis=0)/data.median() > 2 or (data.max(axis=0)-data.median())/(data.median()-data.min(axis=0))>5:



            # ==== you can comment out this line with a '#' to see the difference in the graphs below

            df_training_full[c] = np.log(1+data)

            # ====



            all_data[c] = np.log(1+data)

            print(c)

    except ZeroDivisionError:

        pass
corrmat = all_data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)

plt.show()
for c in numeric_cols:

    f, ax = plt.subplots(figsize=(15, 12))

    plt.xticks(rotation='90')

    data = df_training_full[c].replace({0:np.nan}).dropna()

    plt.hist(data)

    plt.hist(df_training_full[df_training_full['target'] == 1][c].replace({0:np.nan}).dropna(), range=(min(data), max(data)))

    plt.title(c, fontsize=20)

    value = 0

    value2 = 0

    try:

        value = missing_data.loc[c,"Missing Ratio"]

    except KeyError:

        value = 0

    try:

        value2 = zero_data.loc[c, "Zero Density"]

    except KeyError:

        value2 = 0

    print("Missing %s%%, Zero %s%%\n%s" % (value,value2,data.describe()))

    plt.show()
# much better. now that our data makes sense, lets actually do a thing on it

#



from sklearn.model_selection import train_test_split



# temp



#

from IPython.display import display



# temp



if "target" in numeric_cols:

    numeric_cols.remove("target") #no answers kek

if "id" in numeric_cols:

    numeric_cols.remove("id")

    

# replace na with mean value of column

all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())

# normalize data

all_data[numeric_cols] = all_data[numeric_cols]/all_data[numeric_cols].max()





train=all_data[all_data['Type']=="Train"]

test=all_data[all_data['Type']=="Test"]

X = train[numeric_cols].values

y = train["target"].values







training_set, validation_set, training_sols, validation_sols = train_test_split(X, y, test_size=0.30, random_state=42)





display(train[numeric_cols])
def rn(n, cutoff=0.5):

    if n > cutoff:

        return 1

    else:

        return 0



def calc_stats(Vx, Vy, cutoff=0.5):

    """

    Vx: predictions

    Vy: actual

    """

    assert len(Vx) == len(Vy), "length mismatch: %s/%s" % (len(Vx), len(Vy))

    cp=0;cn=0;fp=0;fn=0

    for i in range(len(Vx)):

        if rn(Vx[i], cutoff=cutoff) == 1 and rn(Vy[i], cutoff=cutoff) == 1:

            cp+=1

        elif rn(Vx[i], cutoff=cutoff) == 0 and rn(Vy[i], cutoff=cutoff) == 0:

            cn+=1

        elif rn(Vx[i], cutoff=cutoff) == 1 and rn(Vy[i], cutoff=cutoff) == 0:

            fp+=1

        elif rn(Vx[i], cutoff=cutoff) == 0 and rn(Vy[i], cutoff=cutoff) == 1:

            fn+=1

        else:

            raise Exception("wtf")

    return cp, cn, fp, fn
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=15, verbose=2, n_jobs=16)

rf.fit(training_set, training_sols)



predict_training = rf.predict(training_set)

tp, tn, fp, fn = calc_stats(predict_training, training_sols, cutoff=0.5)

print("Correct positives: %s\nCorrect negatives: %s\nFalse positives: %s\nFalse negatives: %s" % (tp, tn, fp, fn))





predict_validation = rf.predict(validation_set)

tp, tn, fp, fn = calc_stats(predict_validation, validation_sols, cutoff=0.5)

print("Correct positives: %s\nCorrect negatives: %s\nFalse positives: %s\nFalse negatives: %s" % (tp, tn, fp, fn))
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer



my_imputer = Imputer()

training_set = my_imputer.fit_transform(training_set)

validation_set = my_imputer.transform(validation_set)



from xgboost import XGBRegressor



my_model = XGBRegressor()



train_in, feedback_in, train_out, feedback_out = train_test_split(training_set, training_sols, test_size=0.25)



my_model.fit(train_in, train_out, early_stopping_rounds=5, 

             eval_set=[(feedback_in, feedback_out)], verbose=True)
# computes statistics for the models

# accuracy is not bad



predictions = my_model.predict(validation_set)

c_set = test[numeric_cols].values

tp, tn, fp, fn=calc_stats(predictions, validation_sols)

print("Correct positives: %s\nCorrect negatives: %s\nFalse positives: %s\nFalse negatives: %s" % (tp, tn, fp, fn))



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, validation_sols)))
from sklearn.linear_model import LinearRegression

my_model = LinearRegression()

my_model.fit(training_set, training_sols)

predictions = my_model.predict(validation_set)

tp, tn, fp, fn=calc_stats(predictions, validation_sols)

print("Correct positives: %s\nCorrect negatives: %s\nFalse positives: %s\nFalse negatives: %s" % (tp, tn, fp, fn))
c_set = test[numeric_cols].values

from IPython.display import display

display(test[numeric_cols])



predicts = my_model.predict(c_set)



for i in range(len(predicts)):

    predicts[i] = rn(predicts[i], cutoff=0.5)



idx=[i for i in range(1,16002)]

df = pd.DataFrame(data={"id": idx,"target":predicts}).to_csv("test3.csv", index=False)

print("total targets predicted: %s" % sum(predicts))