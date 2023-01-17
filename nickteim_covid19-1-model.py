from fastai.tabular import *
path = Path('/kaggle/input/covid19-global-forecasting-week-1/')

path.ls()
import pandas as pd

sample_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
len(train)
sample_submission.head()
test.head()
train.head()
#merge test set and training set and rename, som columns

Full_data = pd.merge(test, train, on=['Lat','Long','Date','Country/Region','Province/State'])

Full_data.rename(columns={'Province/State':'Province'}, inplace=True)

Full_data.rename(columns={'Country/Region':'Country'}, inplace=True)

Full_data.rename(columns={'ConfirmedCases':'Confirmed'}, inplace=True)

Full_data.head()
len(Full_data)
#rename therefor the data columns

train.rename(columns={'Province/State':'Province'}, inplace=True)

train.rename(columns={'Country/Region':'Country'}, inplace=True)

train.rename(columns={'ConfirmedCases':'Confirmed'}, inplace=True)
#and we do the same for test set

test.rename(columns={'Province/State':'Province'}, inplace=True)

test.rename(columns={'Country/Region':'Country'}, inplace=True)
from sklearn.preprocessing import LabelEncoder

# creating initial dataframe

bridge_types = ('Lat', 'Date', 'Province', 'Country', 'Long', 'Confirmed',

       'ForecastId', 'Id')

countries = pd.DataFrame(train, columns=['Country'])

# creating instance of labelencoder

labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column

train['Countries'] = labelencoder.fit_transform(train['Country'])

train['Countries'].head()

# train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

# train["Date"]  = train["Date"].astype(int)
train['Date']= pd.to_datetime(train['Date']) 

test['Date']= pd.to_datetime(test['Date']) 
train = train.set_index(['Date'])

test = test.set_index(['Date'])
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
create_time_features(train).head()

create_time_features(test).head()
train.head()
train.drop("date", axis=1, inplace=True)

test.drop("date", axis=1, inplace=True)
#do the same for test set

test['Countries'] = labelencoder.fit_transform(test['Country'])



# test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

# test["Date"]  = test["Date"].astype(int)
train.head()
train.isnull().sum()
#drop useless columns for train and test set

train.drop(['Country'], axis=1, inplace=True)

train.drop(['Province'], axis=1, inplace=True)
test.drop(['Country'], axis=1, inplace=True)

test.drop(['Province'], axis=1, inplace=True)
#slpit the data set in to from the merge dataframe called Full_data

train_procent=int(((len(Full_data))/100)*50)

test_procent=int(((len(Full_data))/100)*50)



train_df=Full_data.loc[train_procent:]

test_df=Full_data.loc[:test_procent]
len(test_df)
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X = train_df.copy()

X_test_full = test_df.copy()



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['Fatalities'], inplace=True)

y = X.Fatalities              

X.drop(['Fatalities'], axis=1, inplace=True)

   

    

    # Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

### for cname (every value, one at the time) in dataframe for columns return a value to 'numeric_cols' if the 

### dtype= int64 or float64. 







# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from xgboost import XGBRegressor







model2 = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=1)

model = GradientBoostingClassifier(random_state=1)

model3 = DecisionTreeClassifier(random_state=1)

#model=SGDClassifier(random_state=1)

#model=ExtraTreesClassifier(random_state=1)

model = XGBRegressor()

# Define the models

model_1 = RandomForestClassifier(n_estimators=50, random_state=0)

model_2 = RandomForestClassifier(n_estimators=100, random_state=0)

model_3 = RandomForestClassifier(n_estimators=200, min_samples_split=20, random_state=0)

model_4 = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=1)







model.fit(X_train, y_train)

y_predictions = model.predict(X_valid)



print('model accuracy score',model.score(X_valid,y_valid))
y_test=y_valid

X_test=X_valid
model2.fit(X_train,y_train)

print(f'Model test accuracy: {model2.score(X_test, y_test)*100:.3f}%')

model3.fit(X_train,y_train)

print(f'Model test accuracy: {model3.score(X_test, y_test)*100:.3f}%')
model_1.fit(X_train,y_train)

print(f'Model test accuracy: {model_1.score(X_test, y_test)*100:.3f}%')

model_2.fit(X_train,y_train)

print(f'Model test accuracy: {model_2.score(X_test, y_test)*100:.3f}%')

model_3.fit(X_train,y_train)

print(f'Model test accuracy: {model_3.score(X_test, y_test)*100:.3f}%')

model_4.fit(X_train,y_train)

print(f'Model test accuracy: {model_4.score(X_test, y_test)*100:.3f}%')
from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state = 0) 
train.head()
#train part 2, start over for having enought rows for the submussion

x = train[['Lat', 'Long','Countries','dayofweek','month','dayofyear','weekofyear']]

y1 = train[['Confirmed']]

y2 = train[['Fatalities']]

x_test = test[['Lat', 'Long','Countries','dayofweek','month','dayofyear','weekofyear']]
x.head()
# import numpy as np

# y1=np.ravel(y1)

# y1
regressor.fit(x,y1)

predict_1 = regressor.predict(x_test)

predict_1 = pd.DataFrame(predict_1)

predict_1.columns = ["Confirmed_predict"]
predict_1.head()
# y2=np.ravel(y2)
regressor.fit(x,y2)

predict_2 = regressor.predict(x_test)

predict_2 = pd.DataFrame(predict_2)

predict_2.columns = ["Death_prediction"]

predict_2.head()
Samle_submission = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

Samle_submission.columns

submission = Samle_submission[["ForecastId"]]
Final_submission = pd.concat([predict_1,predict_2,submission],axis=1)

Final_submission.head()
Final_submission.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

Final_submission = Final_submission[['ForecastId','ConfirmedCases', 'Fatalities']]



Final_submission["ConfirmedCases"] = Final_submission["ConfirmedCases"].astype(int)

Final_submission["Fatalities"] = Final_submission["Fatalities"].astype(int)
Final_submission.head()
Final_submission.to_csv("submission.csv",index=False)

print('Model ready for submission!')
# procs = [FillMissing, Categorify, Normalize]



# dep_var = 'Fatalities'

# cat_names = ['Country', 'Province']

# cont_names = ['Long','Lat', 'ForecastId']

# data = (TabularList.from_df(train_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

#         .random_split_by_pct(0.2, seed=42)

#         .label_from_df(cols=dep_var)

#         .add_test(test_df)

#         .databunch()

# )
# data.show_batch(rows=10)


# data = (TabularList.from_df(train_df, procs=procs, cont_names=cont_names, cat_names=cat_names)

#         .split_by_idx(valid_idx=range(int(len(train_df)*0.9),len(train_df)))

#         .label_from_df(cols=dep_var)

#         .add_test(TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names, procs=procs))

#         .databunch())

# print(data.train_ds.cont_names)

# print(data.train_ds.cat_names)
# WE HAVE TO CHANGE ACC. se below reason and code



# [quote="stephenjohnson, post:11, topic:33778"]

# targs stands for **t** arget **arg** ument **s** 

# It’s the values that are the truth values (the Y values) that are being compared to your model’s predicted values. 

# The accuracy metric above takes two arguments the input (predicted values) and targs (target values) and calculates the accuracy. 

# The error encountered above was due to the fact that the input had Long values but targs had Float values.

# [/quote]





# def accuracy_1(input:Tensor, targs:Tensor)->Rank0Tensor:

# #     “Compute accuracy with targs when input is bs * n_classes.”

#     targs = targs.view(-1).long()

#     n = targs.shape[0]

#     input = input.argmax(dim=-1).view(n,-1)

#     targs = targs.view(n,-1)

#     return (input==targs).float().mean()



# # So use metrics=accuracy_1
# learn = tabular_learner(data, layers=[1000,500],metrics=accuracy,model_dir="/tmp/model/")
#test = TabularList.from_df(train.iloc[800:1000].copy(), cat_names=cat_names, cont_names=cont_names)
#data = (TabularList.from_df(train, cat_names=cat_names, cont_names=cont_names, procs=procs)

#                           .split_by_idx(list(range(800,1000)))

#                           .label_from_df(cols=dep_var)

#                           .add_test(X_test)

#                           .databunch())
#data.show_batch(rows=2)
#learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
# learn.fit(5, 1e-2)
# learn.lr_find()

# learn.recorder.plot()
# learn.unfreeze()
# stop- learn.fit_one_cycle(20, slice(1e-3))
#output = pd.DataFrame({'id': sample_submission.id, 'target': y_predictions})
# preds, _ = learn.get_preds(ds_type=DatasetType.Test)

# pred_prob, pred_class = preds.max(1)
# submission = pd.DataFrame({'id':sample_submission['id'],'target':pred_class})
# submission.to_csv('submission-fastai.csv', index=False)
# submission.id = submission.id.astype(int)
# submission.head()
# submission.to_csv('my_submission.csv', index=False)
# sample_submission = pd.read_csv('my_submission.csv')
#row = train.iloc[0]
#X_test.isnull().sum()
#y_predictions=learn.predict(X_test)
# X_test['bin_0'].fillna(X_test['bin_0'].median(), inplace = True)

# X_test['bin_1'].fillna(X_test['bin_1'].median(), inplace = True)

# # X_test['bin_2'].fillna(X_test['bin_2'].median(), inplace = True)

# X_test['ord_0'].fillna(X_test['ord_0'].median(), inplace = True)

# X_test['day'].fillna(X_test['day'].median(), inplace = True)

# X_test['month'].fillna(X_test['month'].median(), inplace = True)


#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")