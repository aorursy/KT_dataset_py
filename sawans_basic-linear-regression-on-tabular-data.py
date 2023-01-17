import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import pydicom

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,PowerTransformer
main_dir = '../input/osic-pulmonary-fibrosis-progression'



train_files = tf.io.gfile.glob(main_dir+"/train/*/*")

test_files = tf.io.gfile.glob(main_dir+"/test/*/*")



sample_sub = pd.read_csv(main_dir+'/sample_submission.csv')

train = pd.read_csv(main_dir + "/train.csv")

test = pd.read_csv(main_dir + "/test.csv")



print ("Number of train patients: {}\nNumber of test patients: {:4}"

       .format(train.Patient.nunique(), test.Patient.nunique()))



print ("\nTotal number of Train patient records: {}\nTotal number of Test patient records: {:6}"

       .format(len(train_files), len(test_files)))



train.shape, test.shape, sample_sub.shape
def laplace_log_likelihood(y_true, y_pred, sigma=70):

    # values smaller than 70 are clipped

    sigma_clipped = tf.maximum(sigma, 70)



    # errors greater than 1000 are clipped

    delta_clipped = tf.minimum(tf.abs(y_true - y_pred), 1000)

    

    # type cast them suitably

    delta_clipped = tf.cast(delta_clipped, dtype=tf.float32)

    sigma_clipped = tf.cast(sigma_clipped, dtype=tf.float32)

    

    # score function

    score = - tf.sqrt(2.0) * delta_clipped / sigma_clipped - tf.math.log(tf.sqrt(2.0) * sigma_clipped)

    

    return tf.reduce_mean(score)
# This will be the perfect score when actual and predicted values are exactly same

laplace_log_likelihood(train['FVC'], train['FVC'], 70)
train.head()
test
# Using Weeks, Age, Sex and Smoking Status columns from train data

X = train[['Weeks','Age','Sex','SmokingStatus']].copy()

y = train['FVC'].copy()



# save the stats for future use

stats = X.describe().T



# One hot encoding on Sex and SmokingStatus columns

X = pd.get_dummies(X, columns =['Sex','SmokingStatus'],drop_first=True)



#Scaling numeric features 

# scaling the numeric features

for col in ['Weeks', 'Age']:

    X[col] = (X[col] - stats.loc[col, 'min']) / (stats.loc[col, 'max'] - stats.loc[col, 'min'])
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.metrics import make_scorer
sigma = 250
# Creating a scorer function

l1 = (make_scorer(

    lambda X,y : laplace_log_likelihood(X,y,sigma=sigma).numpy(),

    greater_is_better=False))



cross_val_score(LinearRegression(),X,y,cv=3,scoring=l1)
X = train.copy()

y = train['FVC'].copy()



X['base_week'] = X.groupby('Patient')['Weeks'].transform('min')

X['base_FVC'] = X.groupby('Patient')['FVC'].transform('first')



# save the stats for future use

stats = X.describe().T



# one hot encoding for categorcial features

X = pd.get_dummies(data=X, columns=['Sex','SmokingStatus'], drop_first=True)



# Scaling numeric columns

num_cols = ['Age','Weeks','base_week','base_FVC']



# Min-max scaling

for col in num_cols:

    X[col] = (X[col]-stats.loc[col,'min']) / (stats.loc[col,'max'] - stats.loc[col,'min'])

    

# printing the correlation of all features with FVC

print(X.corr()['FVC'].abs().sort_values(ascending=False)[1:])
# removing unnecesary columns after transformations

X.drop(['Patient','Percent','FVC'], axis=1, inplace=True)

X.head()
# Checking the score on transformed data now

cross_val_score(LinearRegression(),X,y,cv=3,scoring=l1)
# fit on the train dataset

lr = LinearRegression().fit(X, y)
# Processing submission file

sub = sample_sub.Patient_Week.str.extract("(ID\w+)_(\-?\d+)").rename({0: "Patient", 1: "Weeks"}, axis=1)

sub['Weeks'] = sub['Weeks'].astype(int)

sub = pd.merge(sub, test[['Patient', 'Sex', 'SmokingStatus']], on='Patient')

sub.head()
week_temp = train.groupby(["Weeks", 'Sex'])['FVC'].median()

sex_temp = train.groupby(['Sex'])['FVC'].median()



for index, week, sex in sub.iloc[:, 1:3].itertuples():

    if (week, sex) in week_temp:

        # we assume we are more accurate here

        sub.loc[index, 'FVC'] = week_temp[week, sex]

        sub.loc[index, 'Confidence'] = sigma

    else:

        # we assume we are less accurate here, boost confidence

        sub.loc[index, 'FVC'] = sex_temp[sex]

        sub.loc[index, 'Confidence'] = sigma + 100

        

sub.sample(5)
# swelling confidence as progress in the weeks

sub["Patient_Week"] = sub.Patient + "_" + sub.Weeks.astype(str)

sub.head()
x = (sub.drop(['Confidence', 'Patient_Week'], 1)

     .merge(test[['Patient', 'Weeks', 'FVC', 'Age']], on='Patient')

     .rename({"Weeks_y": "base_Week", "FVC_y": "Base_FVC", "Weeks_x": "Weeks"}, axis=1)

     .drop(['Patient', 'FVC_x'], axis=1))



# one hot encoding, We set drop_first as 

# false to ensure the test is same as train

x = pd.get_dummies(x, columns=['Sex', 'SmokingStatus'])



# # scaling the numeric features

#for col in ['Weeks', 'Age', 'base_Week', 'Base_FVC']:

#    x[col] = (x[col] - stats.loc[col, 'min']) / (stats.loc[col, 'max'] - stats.loc[col, 'min'])

    

num_cols = ['Weeks', 'Age', 'base_Week', 'Base_FVC']

scaler = StandardScaler()

scaler.fit(x[num_cols])



x = pd.concat([x[['Sex_Male','SmokingStatus_Ex-smoker', 'SmokingStatus_Never smoked']].reset_index(drop=True),

                pd.DataFrame(scaler.transform(x[num_cols]),columns=num_cols)],axis=1)

    



x = x[['Weeks', 'Age', 'base_Week', 'Base_FVC', 'Sex_Male',

   'SmokingStatus_Ex-smoker', 'SmokingStatus_Never smoked']]



x.head()
sub['FVC'] = lr.predict(x)

sub.head()
# LR submission

#sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("submission.csv", index=False)
x = train.copy()



# Create base_Week, Base_FVC and Base_Percent for train

temp = (x.groupby("Patient")

        .apply(lambda x: x.loc[int(

            np.percentile(x['Weeks'].index, q=25)

        ), ["Weeks", "FVC", "Percent"]]))



temp.rename(

    {"Weeks": "Base_Week", 

     "FVC": "Base_FVC", 

     "Percent": "Base_Percent"}, 

    axis=1, inplace=True)



# merge it with train data

x = x.merge(temp, on='Patient')

x['Where'] = 'train'



# merge the test dataset as well to be able to handle 1hC

temp = sub[['Patient', 'Weeks']].merge(

    test.rename({"Weeks": "Base_Week", 

                 "FVC": "Base_FVC", 

                 "Percent": "Base_Percent"}, axis=1), 

    on='Patient')



# concatente to the train dataset

temp['Where'] = 'test'

x = pd.concat([x, temp], axis=0)



# create week offsets

x['Week_Offset'] = x['Weeks'] - x['Base_Week']



# oridinal encode categorical values

x['Sex'] = x['Sex'].map({"Male": 1, "Female": 0})

x['SmokingStatus'] = x['SmokingStatus'].map({"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2})



# one hot encoding

x = pd.get_dummies(x, columns=['Sex', 'SmokingStatus'], drop_first=True)



# binned FVC does better?

x['Bin_base_FVC'] = pd.cut(x['Base_FVC'], bins=range(0, 7501, 500)).cat.codes / 15



# lets scale the numeric columns (We scale it with max possibe values)

num_cols = ['Weeks', 'Week_Offset', 'Base_Week', 'Age', 'Base_FVC', 'Percent', 'Base_Percent']

for col in num_cols:

    x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())



to_drop = (

    ["FVC", 'Percent']

    

    + [

#         "Base_FVC", 

#         'Base_Week', 

#         'Weeks', 

#         'Bin_base_FVC', 

#         'Base_Percent'

    ] + 

    

    ['Patient']

)



# print out how well our features would do

print (x[x.Where == 'train'].corr()['FVC'].abs().sort_values(ascending=False).drop(to_drop[:-1]))



y = x['FVC'].dropna()

x = x.drop(to_drop, axis=1)



x.head()
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



grid_params = {"SFromModel__k": range(10)}



temp = Pipeline(

    [("SFromModel", SelectKBest(score_func=f_regression)),

    ("Model", LinearRegression())])



grid = GridSearchCV(temp, param_grid=grid_params, n_jobs=-1, cv=3, scoring=l1)

grid.fit(x[x.Where == 'train'].drop('Where', 1), y)



print (grid.best_params_, grid.best_score_)

model = grid.best_estimator_
best_score = (0, np.inf, np.inf)

for i in range(50, 1500, 50):

    sigma=i

    temp = cross_val_score(model, x[x.Where == 'train'].drop('Where', 1), y, cv=3, scoring=l1)

    if best_score[1] > temp.mean():

        best_score = i, temp.mean(), temp.std()

        

sigma = best_score[0]

best_score
lr = LinearRegression().fit(x[x.Where == 'train'].drop('Where', 1), y)

sub['FVC'] = lr.predict(x[x.Where == 'test'].drop('Where', 1))

sub.head()
# LR submission

sub['Confidence'] = best_score[0]

sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("submission.csv", index=False)