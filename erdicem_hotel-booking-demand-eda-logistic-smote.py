import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

title_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }
curve_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }
df=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.head()
df.columns
df['is_canceled'].value_counts() #target variable
sns.countplot(df['is_canceled'])
df.shape
df.isnull().sum()*100/df.shape[0]
df.drop('company',axis=1, inplace=True)
df['children'].unique()
df['country'].unique()
df['agent'].unique()
df[df['children']==10]
df.drop([328],axis=0,inplace=True) # outliers values delete. 
df['children'].unique()
df['country'].replace(np.nan,"Undefined",inplace=True)
df['agent'].replace(np.nan , 0 , inplace=True)
df['children'].replace(np.nan , 0 , inplace=True)
df.isnull().sum()*100/df.shape[0]
df=df.to_csv('Clear_Hotel_Booking.csv',encoding='utf8')
df= pd.read_csv('Clear_Hotel_Booking.csv')
df.drop('Unnamed: 0',axis=1 , inplace=True)
df.head()
df.describe().T
sns.set(style = "darkgrid")
plt.title("Canceled ", fontdict = {'fontsize': 20})
ax = sns.countplot(x = "is_canceled", data = df)
plt.figure(figsize =(13,10))
sns.set(style="darkgrid")
plt.title("Total Customers - Monthly ", fontdict={'fontsize': 20})
ax = sns.countplot(x = "arrival_date_month", hue = 'is_canceled', data = df)
plt.figure(figsize=(15,10))
sns.barplot(x = "market_segment", y = "stays_in_weekend_nights", data = df, hue = "is_canceled", palette = 'Set1')
plt.figure(figsize = (13,10))
sns.set(style = "darkgrid")
plt.title("Countplot Distrubiton of Segment by Deposit Type", fontdict = {'fontsize':20})
ax = sns.countplot(x = "market_segment", hue = 'deposit_type', data = df)
plt.figure(figsize = (13,10))
sns.set(style = "darkgrid")
plt.title("Countplot Distributon of Segments by Cancellation", fontdict = {'fontsize':20})
ax = sns.countplot(x = "market_segment", hue = 'is_canceled', data = df)
df['hotel'] = pd.get_dummies(df['hotel'],drop_first=True)
df['meal'] = pd.get_dummies(df['meal'],drop_first=True)
df['market_segment'] = pd.get_dummies(df['market_segment'],drop_first=True)
df['distribution_channel'] = pd.get_dummies(df['distribution_channel'],drop_first=True)
df['deposit_type'] = pd.get_dummies(df['deposit_type'],drop_first=True)
df['customer_type'] = pd.get_dummies(df['customer_type'],drop_first=True)
df['assigned_room_type'] = pd.get_dummies(df['assigned_room_type'],drop_first=True)
df['country'] = pd.get_dummies(df['country'],drop_first=True)
df.head()
df_corr=df.corr()
df_corr
plt.figure(figsize=(28,20))
sns.heatmap(df_corr, square=True, annot=True, linewidths=.5, vmin=0, vmax=1, cmap='viridis')
plt.title("Correlation Matrix", fontdict=title_font)
plt.show()
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
df.head()
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

X = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type","babies"],axis=1)
y = df['is_canceled']

sc = StandardScaler()
X_scl = sc.fit_transform(X)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=111, stratify = y)


loj = sm.Logit(y_train,X_train)
loj_model = loj.fit()
loj_model.summary()
def create_model(X,y,model,tip):
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=111, stratify = y)
    model.fit(X_train, y_train)
    
    prediction_train=model.predict(X_train)
    prediction_test=model.predict(X_test)
    
    prediction_test_prob = model.predict_proba(X_test)[:,1]
    
    cv = cross_validate(estimator=model,X=X,y=y,cv=10,return_train_score=True)
    
    d = pd.Series({'Accuracy_Train':accuracy_score(y_train,prediction_train),
                   'Precision_Train':precision_score(y_train,prediction_train),
                   'Recall_Train':recall_score(y_train,prediction_train),
                   'F1 Score_Train':f1_score(y_train,prediction_train),
                   'Accuracy_Test':accuracy_score(y_test,prediction_test),
                   'Precision_Test':precision_score(y_test,prediction_test),
                   'Recall_Test':recall_score(y_test,prediction_test),
                   'F1 Score_Test':f1_score(y_test,prediction_test),
                   'AUC Score':roc_auc_score(y_test, prediction_test_prob),
                   "Cross_val_train":cv['train_score'].mean(),
                   "Cross_val_test":cv['test_score'].mean() },name=tip)
    return d
X = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type","babies"],axis=1)

y = df['is_canceled']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

logistic=LogisticRegression()

metrics=pd.DataFrame()
metrics=metrics.append(create_model(X_scl,y,logistic,tip='Logistic_Regr.'))
metrics
from sklearn.model_selection import GridSearchCV

logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X, y)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)
logistic=LogisticRegression(C=1000,penalty='l2',solver='liblinear' )
metrics=metrics.append(create_model(X_scl,y,logistic,tip='Logistic_Regr_tuning'))
metrics
from sklearn.utils import resample
canceled_customer=df[df.is_canceled==1]
not_canceled_customer=df[df.is_canceled==0]

canceled_customer_resample= resample(canceled_customer,
                                     replace = True,
                                     n_samples = len(not_canceled_customer),
                                     random_state = 111)

resample_df = pd.concat([not_canceled_customer, canceled_customer_resample])
resample_df.is_canceled.value_counts()
X_r = resample_df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)
y_r = resample_df['is_canceled']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_r)

logistic=LogisticRegression()

metrics=metrics.append(create_model(X_scl,y_r,logistic,'Resampled_Logistic'))
metrics
logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X_scl, y_r)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)
logistic=LogisticRegression(C= 0.001, penalty= 'l1', solver= 'liblinear')

metrics=metrics.append(create_model(X_scl,y_r,logistic,'Resampled_Logistic_tuning'))
metrics
from imblearn.over_sampling import SMOTE

y_s = df['is_canceled']
X_s = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)

sm = SMOTE(random_state=111)
X_smote, y_smote = sm.fit_sample(X_s, y_s)

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_smote)

logistic=LogisticRegression()

metrics=metrics.append(create_model(X_scl,y_smote,logistic,'SMOTE_Logistic'))
metrics
logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X_scl, y_smote)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)
logistic=LogisticRegression(C= 0.001, penalty= 'l1', solver= 'liblinear')

metrics=metrics.append(create_model(X_scl,y_r,logistic,'SMOTE_Logistic_tuning'))
metrics
from imblearn.over_sampling import ADASYN
y_a = df['is_canceled']
X_a = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)

ad = ADASYN(random_state=111)
X_adasyn, y_adasyn = ad.fit_sample(X_a, y_a)

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_adasyn)

logistic=LogisticRegression()

metrics=metrics.append(create_model(X_scl,y_adasyn,logistic,'ADASYN_Logistic'))
metrics
logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X_scl, y_adasyn)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)
logistic=LogisticRegression(C= 0.001, penalty= 'l1', solver= 'liblinear')

metrics=metrics.append(create_model(X_scl,y_adasyn,logistic,'ADASYN_Logistic_tuning'))
metrics
metrics.iloc[[4], :]
y_s = df['is_canceled']
X_s = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)

sm = SMOTE(random_state=111)
X_smote, y_smote = sm.fit_sample(X_s, y_s)

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_smote)

logistic=LogisticRegression()
logistic_model=logistic.fit(X_scl,y_smote)
logistic_model.coef_
logistic_model.intercept_
print("0.65971861","+ hotel * "+str(-0.14196832),"+ lead_time * "+str(0.34654226),"+ total_of_special_requests * "+str(-0.07286722),"+ reservation_status_date * "+str(-0.00639928),
    
"+ arrival_date_week_number * "+str(-0.05187588),"+ arrival_date_day_of_month * "+str(0.07373981),"+ stays_in_weekend_nights *"+str(0.0303397),
      "+ stays_in_week_nights *"+str(0.05642681), "+ adults *"+str(-0.00788824), "+ children *"+str(0.02600913),"+ babies *"+str(0.07122837), 
     "+ meal *"+str(0.05067837), "+ market_segment *"+str(-0.33731968), "+ distribution_channel *"+str(-0.19108182),"+ is_repeated_guest *"+str(2.31945534),
     "+ previous_cancellations *"+str(-0.55344701), "+ previous_bookings_not_canceled *"+str(-0.06139697), "+ assigned_room_type *"+str(-0.36896794),"+ booking_changes *"+str( 1.90414296),
     "+ deposit_type *"+str(0.08676973), "+ agent *"+str( -0.09250369), "+ days_in_waiting_list *"+str(-0.07363203),"+ customer_type *"+str( 0.34808834),"+ adr *"+str(-0.50650057),sep='\n')
print("= is_canceled")