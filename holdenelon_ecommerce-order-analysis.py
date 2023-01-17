import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
plt.style.use({'figure.figsize':(16, 9)})
data = pd.read_csv('../input/olist-public-dataset/olist_public_dataset.csv')
data.head()
data.order_status.unique()
data.customer_state.unique().shape
data.product_category_name.unique().shape
data.drop(['review_comment_title','Unnamed: 0','id','review_comment_message','customer_zip_code_prefix'],axis=1,inplace=True)
data = data.dropna()
data.info()
time_col = [s for s in data.columns if (('_date' in s) or ('_timestamp' in s)) or ('order_aproved_at' in s)]
print(time_col)
for i in time_col:
    data[i] = pd.to_datetime(data[i])
data['order_purchase_date'] = data.order_purchase_timestamp.dt.date
sales_per_purchase_date = data.groupby('order_purchase_date', as_index=False).order_products_value.sum()
ax = sns.lineplot(x="order_purchase_date", y="order_products_value", data=sales_per_purchase_date,color='seagreen')
ax.set_title('Sales per day')
data['order_purchase_date'] = data.order_purchase_timestamp.dt.date
order_counts_per_date = data.groupby('order_purchase_date',as_index=False).order_status.count()
ax = sns.lineplot(x="order_purchase_date", y="order_status", data=order_counts_per_date,color='orangered')
ax.set_ylabel('order purchase counts')
ax.set_title('Order per day')
data['order_items_qty_date'] = data.order_purchase_timestamp.dt.date
order_items_qty_date = data.groupby('order_purchase_date', as_index=False).order_items_qty.sum()
ax = sns.lineplot(x="order_purchase_date", y="order_items_qty", data=order_items_qty_date,color='purple')
ax.set_title('order items qty per day')
data['order_purchase_week'] = data.order_purchase_timestamp.dt.to_period('W').astype(str)
sales_per_purchase_week = data.groupby('order_purchase_week', as_index=False).order_products_value.sum()
ax = sns.lineplot(x="order_purchase_week", y="order_products_value", data=sales_per_purchase_week,color='seagreen')
ax.set_title('Sales per week')
data['order_purchase_week'] = data.order_purchase_timestamp.dt.to_period('W').astype(str)
order_counts_per_week = data.groupby('order_purchase_week', as_index=False).order_status.count()
ax = sns.lineplot(x="order_purchase_week", y="order_status", data=order_counts_per_week,color='orangered')
ax.set_title('order counts per week')
ax.set_ylabel('order per week')
data['order_items_qty_week'] = data.order_purchase_timestamp.dt.to_period('W').astype(str)
order_items_qty_week = data.groupby('order_purchase_week', as_index=False).order_items_qty.sum()
ax = sns.lineplot(x="order_purchase_week", y="order_items_qty", data=order_items_qty_week,color='purple')
ax.set_title('order items qty per week')
data.head()
data['review_date'] = data['review_creation_date'].dt.date
avg_review_score_date = data.groupby('review_date',as_index=False).review_score.mean()
ax = sns.lineplot(x='review_date',y='review_score',data=avg_review_score_date)
ax.set_title('average review score per day')
data['review_week'] = data.order_purchase_timestamp.dt.to_period('W').astype(str)
avg_review_score_week = data.groupby('review_week',as_index=False).review_score.mean()
ax = sns.lineplot(x='review_week',y='review_score',data=avg_review_score_week)
ax.set_title('average review score per week')
data['order_purchase_week'] = data.order_purchase_timestamp.dt.to_period('W').astype(str)
avg_sales_per_purchase_week = data.groupby('order_purchase_week', as_index=False).order_products_value.mean()
ax = sns.lineplot(x="order_purchase_week", y="order_products_value", data=avg_sales_per_purchase_week,color='seagreen')
ax.set_title('Average vales/order per week')
ax.set_ylabel('avg values/order')
data['order_purchase_week'] = data.order_purchase_timestamp.dt.to_period('W').astype(str)
sales_per_purchase_week = data.groupby('order_purchase_week', as_index=False).order_products_value.mean()
order_freight_value_week = data.groupby('order_purchase_week', as_index=False).order_freight_value.mean()
freight_div_sales = pd.concat([sales_per_purchase_week['order_purchase_week'],
                            (order_freight_value_week['order_freight_value']/sales_per_purchase_week['order_products_value'])],
                              axis=1)
freight_div_sales.columns = ['order_purchase_week','freight_div_sales']
ax = sns.lineplot(x="order_purchase_week", y="freight_div_sales", data=freight_div_sales,color='seagreen')
ax.set_title('Freight/sales per week')
data.head()
data.head()
plt.scatter(data['customer_state'].value_counts().index.values,data['customer_state'].value_counts())
data['customer_state'].value_counts()
customer_state = dict(zip(*np.unique(data['customer_state'], return_counts=True)))
data['customer_state'] = data['customer_state'].apply(lambda x: 'OTHER' if customer_state[x] < 1000 else x)

cs = pd.get_dummies(data['customer_state'],prefix='customer_state')
data = pd.concat((data,cs),axis=1)
data.drop(['customer_state'],axis=1,inplace=True)
data['product_category_name'].value_counts()
plt.scatter(data['product_category_name'].value_counts().index.values,data['product_category_name'].value_counts())
product_category_name = dict(zip(*np.unique(data['product_category_name'], return_counts=True)))
data['product_category_name'] = data['product_category_name'].apply(lambda x: 'OTHER' if product_category_name[x] < 2000 else x)

pcn = pd.get_dummies(data['product_category_name'],prefix='product_category_name')
data = pd.concat((data,pcn),axis=1)
data.drop(['product_category_name'],axis=1,inplace=True)
plt.scatter(data['product_name_lenght'].value_counts().index.values,data['product_name_lenght'].value_counts())
def get_nameLength(dt):
    if (dt>=10) & (dt < 20):
        return '20'
    elif (dt>=20) & (dt<30):
        return '30'
    elif (dt>=30) & (dt<40):
        return '40'
    elif (dt>=40) & (dt<50):
        return '50'
    elif (dt>=50) & (dt<60):
        return '60'
    elif (dt>=60) & (dt<70):
        return '70'
    else:
        return 'others'
data['product_name_lenght_new'] = np.array([get_nameLength(x) for x in data['product_name_lenght']])
pnl = pd.get_dummies(data['product_name_lenght_new'],prefix='product_name_lenght')
data = pd.concat((data,pnl),axis=1)
data.drop(['product_name_lenght','product_name_lenght_new'],axis=1,inplace=True)
plt.scatter(data['product_description_lenght'].value_counts().index.values,data['product_description_lenght'].value_counts())
def get_descriptionLength(dt):
    if (dt>=0) & (dt < 1000):
        return '1000'
    elif (dt>=1000) & (dt<2000):
        return '2000'
    elif (dt>=2000) & (dt<3000):
        return '3000'
    elif (dt>=3000) & (dt<4000):
        return '4000'
    else:
        return 'others'
data['product_description_lenght_new'] = np.array([get_descriptionLength(x) for x in data['product_description_lenght']])
pnl = pd.get_dummies(data['product_description_lenght_new'],prefix='product_description_lenght')
data = pd.concat((data,pnl),axis=1)
data.drop(['product_description_lenght','product_description_lenght_new'],axis=1,inplace=True)
data.head()
data['freight_div_sales'] = np.true_divide(data['order_freight_value'].values,data['order_products_value'].values)
plt.scatter(data['freight_div_sales'].value_counts().index.values,data['freight_div_sales'].value_counts())
def getFreightDivSales(dt):
    if dt == 0:
        return '0'
    elif (dt>0) & (dt<0.25):
        return '25%'
    elif (dt>=0.25) & (dt<0.5):
        return '50%'
    elif (dt>=0.5) & (dt<0.75):
        return '75%'
    elif (dt>=0.75) & (dt<1):
        return '100%'
    else:
        return 'more than 100%'
data['freight_div_sales_new'] = np.array([getFreightDivSales(x) for x in data['freight_div_sales']])
fds = pd.get_dummies(data['freight_div_sales_new'],prefix='freight_div_sales')
data = pd.concat((data,fds),axis=1)
data.drop(['freight_div_sales','freight_div_sales_new'],axis=1,inplace=True)
time_span_odcd_opt = data['order_delivered_customer_date'].subtract(data['order_purchase_timestamp']).dt.days
plt.scatter(time_span_odcd_opt.value_counts().index.values,time_span_odcd_opt.value_counts())
def get_time_span_odcd_opt(dt):
    if (dt <= 7) & (dt >= 0):
        return 'OneWeek'
    elif (dt <= 14) & (dt > 7):
        return 'TwoWeeks'
    elif (dt <= 21) & (dt > 14):
        return 'ThreeWeeks'
    elif (dt <= 28) & (dt > 21):
        return 'FourWeeks'
    elif (dt <= 35) & (dt > 28):
        return 'FiveWeeks'
    elif (dt <= 42) & (dt > 35):
        return 'SixWeeks'
    elif (dt <= 48) & (dt > 42):
        return 'SevenWeeks'
    else:
        return 'More than sevenweeks'
data['time_span_odcd_opt'] = np.array([get_time_span_odcd_opt(x) for x in time_span_odcd_opt])
tsoo = pd.get_dummies(data['time_span_odcd_opt'],prefix='time_span_odcd_opt')
data = pd.concat((data,tsoo),axis=1)
data.drop(['time_span_odcd_opt'],axis=1,inplace=True)
data['review_answer_timestamp_wd'] = np.array([x.isoweekday() for x in data['review_answer_timestamp']])
rat_wd = pd.get_dummies(data['review_answer_timestamp_wd'], prefix = 'review_answer_timestamp_wd')
data = pd.concat((data, rat_wd), axis = 1)
data.drop(['review_answer_timestamp_wd','review_answer_timestamp'],axis=1,inplace=True)
data['order_purchase_timestamp_wd'] = np.array([x.isoweekday() for x in data['order_purchase_timestamp']])
opt_wd = pd.get_dummies(data['order_purchase_timestamp_wd'], prefix = 'order_purchase_timestamp_wd')
data = pd.concat((data, opt_wd), axis = 1)
data.drop(['order_purchase_timestamp','order_purchase_timestamp_wd'],axis=1,inplace=True)
order_status = pd.get_dummies(data['order_status'],prefix='order_status')
data = pd.concat((data,order_status),axis=1)
data.drop(['order_status'],axis=1,inplace=True)
time = ['order_aproved_at','order_estimated_delivery_date',
        'order_delivered_customer_date','review_creation_date']

for i in time:
    data[i + '_year'] = np.array([x.year for x in data[i]])
    data[i + '_month'] = np.array([x.month for x in data[i]])   
    data[i + '_day'] = np.array([x.day for x in data[i]])
    data.drop([i],axis=1,inplace=True)
data.head()
data.drop(['customer_city','order_purchase_date','order_items_qty_date','order_purchase_week','order_items_qty_week',
           'review_date','review_week'],axis=1,inplace=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
train_data_ = data.drop('review_score',axis=1)
target = data['review_score']

n = int(train_data_.shape[0])
train_data_ = train_data_.iloc[:n, :]
target = target.iloc[:n]

print(n)

data_scaler = StandardScaler()
train_data = data_scaler.fit_transform(train_data_)
k_fold = KFold(n_splits=10, shuffle=False, random_state=0)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(train_data,target,test_size=0.33,random_state=0)
model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test,predictions)
print("Accuracy:%2f%%" % (accuracy * 100.0))
clf = KNeighborsClassifier(n_neighbors=5)
scoring = 'accuracy'
score = cross_val_score(clf,train_data,target,cv=k_fold,n_jobs=1,scoring=scoring)
print(score)
#knn socre
round(np.mean(score)*100,2)
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#decision tree socre
round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Random Forest Score
round(np.mean(score)*100, 2)