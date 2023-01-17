import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
Test = pd.read_csv("../input/bigmart-sales-data/Test.csv")
Train = pd.read_csv("../input/bigmart-sales-data/Train.csv")
Train.drop(['Item_Identifier','Outlet_Identifier'],axis=1, inplace = True)

cate_col = [col for col in Train.columns if Train[col].dtypes == 'O']
num_col = [col for col in Train.columns if Train[col].dtype in ['int64','float64']]

from sklearn.preprocessing import LabelEncoder

def simple_encoder (df,column_names=[]):
    label_encoder= LabelEncoder()
    if len(column_names) > 0:
        for i in column_names:
            df[i] = label_encoder.fit_transform(df[i])
    elif len(column_names) == 0:
        for i in cate_col:
            df[i] = label_encoder.fit_transform(df[i])
    return df

def cross_df_encoder(df1,df2, cl = []):
    #a must bigger than b
    a=df1.columns.tolist()
    b=df2.columns.tolist()
    label=LabelEncoder()
    for i in cl:
        #print(i)
        try:
            b.index(i)
            #print('try_success')
            df1[i] = label.fit_transform(df1[i])
            df2[i] = label.transform(df2[i])
        except:
            #print('try_fail')
            df1[i] = label.fit_transform(df1[i])
    return df1, df2
    
    
            
Train.info()
Train.describe()

Train[num_col].head()
#we need to change Outlet_Establishment_Year into Object data type
Train.nunique()
fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.violinplot(data = Train, x='Outlet_Type', y='Item_Outlet_Sales',ax = ax[0])
sns.violinplot(data = Train, x='Outlet_Location_Type', y='Item_Outlet_Sales', ax = ax[1])

#Grocery store only sell low-cost products
#Supermarket has many more products varying in size
fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.violinplot(data = Train, x='Outlet_Size', y='Item_Outlet_Sales',ax = ax[0])
sns.violinplot(data = Train, x='Item_Fat_Content', y='Item_Outlet_Sales', ax = ax[1])
#fig, ax = plt.subplots(1,2, figsize= (20,5))
z=Train['Item_Fat_Content'].value_counts().reset_index()
#x=Train['Outlet_Identifier'].value_counts().reset_index()
plt.bar(z.index,z.Item_Fat_Content, tick_label = z['index'])
plt.xlabel('Item_Fat_Content')
#ax[1].bar(x.index,x.Outlet_Identifier, tick_label = x['index'])
#ax[1].set_xlabel('Outlet_Identifier')

#Item fat content got mis label
z=Train['Item_Type'].value_counts().reset_index()
plt.figure(figsize=(20,5))
plt.bar(z.index,z.Item_Type, tick_label = z['index'])
plt.xlabel('Item_Type')

mis_label={'low fat':'Low Fat','LF': 'Low Fat', 'reg':'Regular', 'Low Fat':'Low Fat', 'Regular': 'Regular'}
Train['Item_Fat_Content']=Train['Item_Fat_Content'].map(mis_label)
Train['Item_Fat_Content'].value_counts()
Train.isnull().sum().sort_values(ascending = False)
print('Percentage of null value in Outlet_Size column %s'%(round(Train['Outlet_Size'].isnull().sum()/len(Train.Outlet_Size),2)))
print('Percentage of null value in Item_Weight column %s'%(round(Train['Item_Weight'].isnull().sum()/len(Train.Item_Weight),2)))
train_sample= Train.copy()
train_sample.dropna(axis=0, inplace = True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for col in cate_col:
    train_sample[col] = label_encoder.fit_transform(train_sample[col])

plt.figure(figsize=(15,5))
train_sample=train_sample.corr()
sns.heatmap(train_sample,annot=True)

#At the beginning, I think about fill na values of Outlet_size column with the most popular item and Item weight with the mean
#However, from the Heatmap below, Outlet_size has a strong correlation with Outlet_Location_Type, Outlet_Identifier, Outlet_Type and Outlet_Establishment_Year
#we can make a prediction on that
#
#Prepare some model

def loss (y_true, y_pred, retu = False):
    pre = precision_score(y_true, y_pred, average='micro')
    rec = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    #log = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    if retu:
        return pre, rec, f1, acc
    else:
        print('   pre : %.3f\n    rec : %.3f\n    f1 : %.3f\n    acc : %.3f\n ' %(pre,rec,f1,acc))
        
def train(X,y,models):
    for name, model in models.items():
        print(name + ' : ')
        result_list=[]
        name_loss = ['pre','rec','f1','acc']
        for train, test in skf.split(X,y):
            model.fit(X.iloc[train], y.iloc[train])
            
            y_predict = model.predict(X.iloc[test])

            result_list.append(loss(y.iloc[test],y_predict, retu = True))
        print(pd.DataFrame(np.array(result_list).mean(axis=0), index= name_loss)[0])
        print('\n')
# because the high correlation between {Outlet_location_type, Outlet_establishment_year} and {Outlet_size}
# I wont preproessing to increase the correlation


train_sample=Train.copy()
train_sample.dropna(axis=0,inplace=True)
for col in cate_col:
    train_sample[col] = label_encoder.fit_transform(train_sample[col])
X=train_sample[['Outlet_Establishment_Year','Outlet_Type','Outlet_Location_Type']]
y=train_sample[['Outlet_Size']]



from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

train_models = {
'Decision Tree Classifer ' :DecisionTreeClassifier(random_state = 42),
'SVC ' : SVC(random_state = 42),
'Random Forest Classifier' : RandomForestClassifier(random_state=42),
'K Neighbors Classifier' : KNeighborsClassifier()
}



from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold (n_splits = 5, random_state =42, shuffle =True)

def loss (y_true, y_pred, retu = False):
    pre = precision_score(y_true, y_pred, average='micro')
    rec = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    #log = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    if retu:
        return pre, rec, f1, acc
    else:
        print('   pre : %.3f\n    rec : %.3f\n    f1 : %.3f\n    acc : %.3f\n ' %(pre,rec,f1,acc))
        
def train(X,y,models):
    for name, model in models.items():
        print(name + ' : ')
        result_list=[]
        name_loss = ['pre','rec','f1','acc']
        for train, test in skf.split(X,y):
            model.fit(X.iloc[train], y.iloc[train])
            
            y_predict = model.predict(X.iloc[test])

            result_list.append(loss(y.iloc[test],y_predict, retu = True))
        print(pd.DataFrame(np.array(result_list).mean(axis=0), index= name_loss)[0])
        print('\n')

        
        
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
train(X,y, train_models)
#At this point, I am not really sure whether I overfit the model or not as the result was so high lol
Train_sample = Train.copy()

#due to data leakage, I need to map 'Outlet_Establishment_Year' by this way
year_dict = {1985:1 , 1987:2 , 1997:3 , 1998:4 , 1999:5 , 2002:6 , 2004:7 , 2007:8 , 2009:9}
Train_sample['Outlet_Establishment_Year'] = Train_sample['Outlet_Establishment_Year'].map(year_dict)

non_null_values = Train_sample[~Train_sample.Outlet_Size.isnull()][['Outlet_Establishment_Year', 'Outlet_Location_Type','Outlet_Type', 'Outlet_Size']]
null_values = Train_sample[Train_sample.Outlet_Size.isnull()][['Outlet_Establishment_Year', 'Outlet_Location_Type','Outlet_Type']]

non_null_values, null_values = cross_df_encoder(df1=non_null_values,df2=null_values, cl = ['Outlet_Location_Type','Outlet_Type', 'Outlet_Size'])

model = DecisionTreeClassifier(random_state = 42)

model.fit(non_null_values[['Outlet_Establishment_Year', 'Outlet_Location_Type','Outlet_Type']],
          non_null_values[['Outlet_Size']])

z=model.predict(null_values[['Outlet_Establishment_Year', 'Outlet_Location_Type','Outlet_Type']])
after_train=pd.concat([null_values.reset_index(),pd.Series(z,name= 'Outlet_Size')],axis=1)

outlet_df = pd.concat([non_null_values,after_train.set_index('index')],axis=0).sort_index()
label = ['Outlet_Establishment_Year', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size']
Train[label] = outlet_df[label]
corr_df = Train.dropna(axis=0)[['Item_Fat_Content','Item_Type','Item_Weight','Item_Visibility','Item_MRP']]
corr_df = simple_encoder(corr_df, column_names = ['Item_Fat_Content','Item_Type'])
corr = corr_df.corr()
sns.heatmap(corr, annot = True)

#the correlation is quite low, let have a deeper look at data for a way to improve the corelation
fig, ax = plt.subplots(1,3, figsize = (20,5))
sns.distplot(Train.Item_Visibility, ax = ax[0])
sns.distplot(Train.Item_MRP, ax = ax[1])
sns.distplot(Train.Item_Weight, ax = ax[2])
#sns.distplot(Train.Item_Fat_Content, ax = ax[3])


ax[0].set_xlabel('Item Visibility')
ax[1].set_xlabel('Item MRP')
ax[2].set_xlabel('Item Weight')
#ax[3].set_xlabel('Item Fat Content')

#Item Visibility got skewness

fig, ax = plt.subplots(1,5, figsize = (20,5))
sns.distplot(Train.Item_Visibility, ax = ax[0])
sns.distplot(np.sqrt(Train.Item_Visibility), ax=ax[1])
ax[2].hist(Train.Item_Visibility, bins = 4)
sns.distplot(Train.Item_Visibility[Train.Item_Visibility < 0.2], ax=ax[3])
z=(Train.Item_Visibility == 0).value_counts().reset_index()
#z['Item_Visibility'].tolist()
ax[4].bar(['Visible', 'UnVisible'],z['Item_Visibility'].tolist())

ax[0].set_xlabel('Original Item Visibility')
ax[1].set_xlabel('Sqrt')
ax[2].set_xlabel('Binning (4) ')
ax[3].set_xlabel('Drop outliner')
ax[4].set_xlabel('Boolean')
#sns.distplot((Train.Item_Visibility == 0).astype('int'), ax=ax[2])

#Binning method still gets skewness in Data Distribution
#Sqrt method reduce the skeness, however, the 0 value still makes the data distribution abnormal
#we will check how the correlation values 
corr_df['sqrt']= np.sqrt(corr_df['Item_Visibility'])
corr_df['binning4'] = pd.qcut(corr_df['Item_Visibility'], q=4)
corr_df['bool'] = (Train.Item_Visibility == 0)
aa = corr_df[['Item_Weight','Item_Visibility','sqrt', 'binning4', 'bool']]
aa = simple_encoder (aa,column_names=['binning4']).corr()
sns.heatmap(aa, annot=True)

#binning method has the highest correlevance to Item Weight so we gonna implement this method
#check correlation between Item Weight and Item Visibility after droping outliner
corr_df[corr_df['Item_Visibility'] < 0.2][['Item_Weight','Item_Visibility']].corr()
Train_test = Train.copy()[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Fat_Content','Item_Type']]
null_values = Train_test[Train_test.Item_Weight.isnull()][[ 'Item_Visibility', 'Item_MRP', 'Item_Fat_Content', 'Item_Type']]
non_null_values = Train_test[~Train_test.Item_Weight.isnull()][['Item_Visibility', 'Item_MRP', 'Item_Fat_Content','Item_Weight', 'Item_Type']]
non_null_values , null_values = cross_df_encoder(non_null_values, null_values, cl = ['Item_Fat_Content','Item_Type'])
from sklearn.model_selection import train_test_split
X = non_null_values[['Item_Visibility', 'Item_MRP', 'Item_Fat_Content','Item_Type']]
y = non_null_values['Item_Weight']
train_X, val_X, train_y, val_y = train_test_split(X,y ,test_size = 0.3, random_state = 42)

from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error

models_dict = { 'LinearSVR' : LinearSVR(),'Lasso' : Lasso(), 'Linear Regressionn' : LinearRegression() , 'BayesianRidge' : BayesianRidge()}
for name, model in models_dict.items():
    print(name + ' : ')
    model.fit(train_X,train_y)
    predict_result = model.predict(val_X)
    print('MSE : %s' %(mean_squared_error(val_y,predict_result)))
    print('MAE : %s' %(mean_absolute_error(val_y,predict_result)))
    print('\n')
    
#I gonna choos Bayesian Ridge for making prediction.
model = BayesianRidge()
model.fit(X,y)
null_values['Item_Weight'] = model.predict(null_values)
item_col = pd.concat([non_null_values, null_values] ,axis = 0)
label = ['Item_Visibility', 'Item_MRP', 'Item_Fat_Content', 'Item_Weight','Item_Type']
Train[label] = item_col[label]

#Check corr
z= Train.corr()
plt.figure(figsize= (15,8))
sns.heatmap(z, annot = True)
X= Train.drop('Item_Outlet_Sales',axis =1)
y= Train['Item_Outlet_Sales']

models_dict = { 'LinearSVR' : LinearSVR(),'Lasso' : Lasso(), 'Linear Regressionn' : LinearRegression() , 'BayesianRidge' : BayesianRidge()}

from sklearn.model_selection import cross_validate

for name, model in models_dict.items():
    print(name)
    print(pd.DataFrame(cross_validate(model,X,y, cv=5,scoring = ['neg_mean_absolute_error','neg_mean_squared_error'])).mean())
    print('\n')
    