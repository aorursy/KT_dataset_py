import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

%config InlineBackend.figure_format = 'svg'
import pandas as pd    

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pylab import *
data = pd.read_csv("../input/application_record.csv", encoding = 'utf-8') 

record = pd.read_csv("../input/credit_record.csv", encoding = 'utf-8')  
sns.set_style('white') 
# find all users' account open month.

begin_month=pd.DataFrame(record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))

begin_month=begin_month.rename(columns={'MONTHS_BALANCE':'begin_month'}) 

new_data=pd.merge(data,begin_month,how="left",on="ID") #merge to record data
record['dep_value'] = None

record['dep_value'][record['STATUS'] =='2']='Yes' 

record['dep_value'][record['STATUS'] =='3']='Yes' 

record['dep_value'][record['STATUS'] =='4']='Yes' 

record['dep_value'][record['STATUS'] =='5']='Yes' 
cpunt=record.groupby('ID').count()

cpunt['dep_value'][cpunt['dep_value'] > 0]='Yes' 

cpunt['dep_value'][cpunt['dep_value'] == 0]='No' 

cpunt = cpunt[['dep_value']]

new_data=pd.merge(new_data,cpunt,how='inner',on='ID')

new_data['target']=new_data['dep_value']

new_data.loc[new_data['target']=='Yes','target']=1

new_data.loc[new_data['target']=='No','target']=0
print(cpunt['dep_value'].value_counts())

cpunt['dep_value'].value_counts(normalize=True)
new_data.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Car','FLAG_OWN_REALTY':'Reality',

                         'CNT_CHILDREN':'ChldNo','AMT_INCOME_TOTAL':'inc',

                         'NAME_EDUCATION_TYPE':'edutp','NAME_FAMILY_STATUS':'famtp',

                        'NAME_HOUSING_TYPE':'houtp','FLAG_EMAIL':'email',

                         'NAME_INCOME_TYPE':'inctp','FLAG_WORK_PHONE':'wkphone',

                         'FLAG_PHONE':'phone','CNT_FAM_MEMBERS':'famsize',

                        'OCCUPATION_TYPE':'occyp'

                        },inplace=True)
new_data.dropna()

new_data = new_data.mask(new_data == 'NULL').dropna()
ivtable=pd.DataFrame(new_data.columns,columns=['variable'])

ivtable['IV']=None

namelist = ['FLAG_MOBIL','begin_month','dep_value','target','ID']



for i in namelist:

    ivtable.drop(ivtable[ivtable['variable'] == i].index, inplace=True)
sns.set_style("whitegrid") 
# Calculate information value

def calc_iv(df, feature, target, pr=False):

    lst = []

    df[feature] = df[feature].fillna("NULL")



    for i in range(df[feature].nunique()):

        val = list(df[feature].unique())[i]

        lst.append([feature,                                                        # Variable

                    val,                                                            # Value

                    df[df[feature] == val].count()[feature],                        # All

                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)

                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)



    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()

    data['Bad Rate'] = data['Bad'] / data['All']

    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())

    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()

    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})



    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])



    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])

    data.index = range(len(data.index))



    if pr:

        print(data)

        print('IV = ', data['IV'].sum())



    iv = data['IV'].sum()

    print('This variable\'s IV is:',iv)

    print(df[feature].value_counts())

    return iv, data
def convert_dummy(df, feature,rank=0):

    pos = pd.get_dummies(df[feature], prefix=feature)

    mode = df[feature].value_counts().index[rank]

    biggest = feature + '_' + str(mode)

    pos.drop([biggest],axis=1,inplace=True)

    df.drop([feature],axis=1,inplace=True)

    df=df.join(pos)

    return df
def get_category(df, col, binsnum, labels, qcut = False):

    if qcut:

        localdf = pd.qcut(df[col], q = binsnum, labels = labels) # quantile cut

    else:

        localdf = pd.cut(df[col], bins = binsnum, labels = labels) # equal-length cut

        

    localdf = pd.DataFrame(localdf)

    name = 'gp' + '_' + col

    localdf[name] = localdf[col]

    df = df.join(localdf[name])

    df[name] = df[name].astype(object)

    return df
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
new_data['Gender'] = new_data['Gender'].replace(['F','M'],[0,1])

print(new_data['Gender'].value_counts())

iv, data = calc_iv(new_data,'Gender','target')

ivtable.loc[ivtable['variable']=='Gender','IV']=iv

data.head()
new_data['Car'] = new_data['Car'].replace(['N','Y'],[0,1])

print(new_data['Car'].value_counts())

iv, data=calc_iv(new_data,'Car','target')

ivtable.loc[ivtable['variable']=='Car','IV']=iv

data.head()
new_data['Reality'] = new_data['Reality'].replace(['N','Y'],[0,1])

print(new_data['Reality'].value_counts())

iv, data=calc_iv(new_data,'Reality','target')

ivtable.loc[ivtable['variable']=='Reality','IV']=iv

data.head()
new_data['phone']=new_data['phone'].astype(str)

print(new_data['phone'].value_counts(normalize=True,sort=False))

new_data.drop(new_data[new_data['phone'] == 'nan' ].index, inplace=True)

iv, data=calc_iv(new_data,'phone','target')

ivtable.loc[ivtable['variable']=='phone','IV']=iv

data.head()
print(new_data['email'].value_counts(normalize=True,sort=False))

new_data['email']=new_data['email'].astype(str)

iv, data=calc_iv(new_data,'email','target')

ivtable.loc[ivtable['variable']=='email','IV']=iv

data.head()
new_data['wkphone']=new_data['wkphone'].astype(str)

iv, data = calc_iv(new_data,'wkphone','target')

new_data.drop(new_data[new_data['wkphone'] == 'nan' ].index, inplace=True)

ivtable.loc[ivtable['variable']=='wkphone','IV']=iv

data.head()
new_data.loc[new_data['ChldNo'] >= 2,'ChldNo']='2More'

print(new_data['ChldNo'].value_counts(sort=False))
iv, data=calc_iv(new_data,'ChldNo','target')

ivtable.loc[ivtable['variable']=='ChldNo','IV']=iv

data.head()
new_data = convert_dummy(new_data,'ChldNo')
new_data['inc']=new_data['inc'].astype(object)

new_data['inc'] = new_data['inc']/10000 

print(new_data['inc'].value_counts(bins=10,sort=False))

new_data['inc'].plot(kind='hist',bins=50,density=True)
new_data = get_category(new_data,'inc', 3, ["low","medium", "high"], qcut = True)

iv, data = calc_iv(new_data,'gp_inc','target')

ivtable.loc[ivtable['variable']=='inc','IV']=iv

data.head()
new_data = convert_dummy(new_data,'gp_inc')
new_data['Age']=-(new_data['DAYS_BIRTH'])//365	

print(new_data['Age'].value_counts(bins=10,normalize=True,sort=False))

new_data['Age'].plot(kind='hist',bins=20,density=True)
new_data = get_category(new_data,'Age',5, ["lowest","low","medium","high","highest"])

iv, data = calc_iv(new_data,'gp_Age','target')

ivtable.loc[ivtable['variable']=='DAYS_BIRTH','IV'] = iv

data.head()
new_data = convert_dummy(new_data,'gp_Age')
new_data['worktm']=-(new_data['DAYS_EMPLOYED'])//365	

new_data[new_data['worktm']<0] = np.nan 

new_data['DAYS_EMPLOYED']

new_data['worktm'].fillna(new_data['worktm'].mean(),inplace=True) 

new_data['worktm'].plot(kind='hist',bins=20,density=True)
new_data = get_category(new_data,'worktm',5, ["lowest","low","medium","high","highest"])

iv, data=calc_iv(new_data,'gp_worktm','target')

ivtable.loc[ivtable['variable']=='DAYS_EMPLOYED','IV']=iv

data.head()
new_data = convert_dummy(new_data,'gp_worktm')
new_data['famsize'].value_counts(sort=False)
new_data['famsize']=new_data['famsize'].astype(int)

new_data['famsizegp']=new_data['famsize']

new_data['famsizegp']=new_data['famsizegp'].astype(object)

new_data.loc[new_data['famsizegp']>=3,'famsizegp']='3more'

iv, data=calc_iv(new_data,'famsizegp','target')

ivtable.loc[ivtable['variable']=='famsize','IV']=iv

data.head()
new_data = convert_dummy(new_data,'famsizegp')
print(new_data['inctp'].value_counts(sort=False))

print(new_data['inctp'].value_counts(normalize=True,sort=False))

new_data.loc[new_data['inctp']=='Pensioner','inctp']='State servant'

new_data.loc[new_data['inctp']=='Student','inctp']='State servant'

iv, data=calc_iv(new_data,'inctp','target')

ivtable.loc[ivtable['variable']=='inctp','IV']=iv

data.head()
new_data = convert_dummy(new_data,'inctp')
new_data.loc[(new_data['occyp']=='Cleaning staff') | (new_data['occyp']=='Cooking staff') | (new_data['occyp']=='Drivers') | (new_data['occyp']=='Laborers') | (new_data['occyp']=='Low-skill Laborers') | (new_data['occyp']=='Security staff') | (new_data['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'

new_data.loc[(new_data['occyp']=='Accountants') | (new_data['occyp']=='Core staff') | (new_data['occyp']=='HR staff') | (new_data['occyp']=='Medicine staff') | (new_data['occyp']=='Private service staff') | (new_data['occyp']=='Realty agents') | (new_data['occyp']=='Sales staff') | (new_data['occyp']=='Secretaries'),'occyp']='officewk'

new_data.loc[(new_data['occyp']=='Managers') | (new_data['occyp']=='High skill tech staff') | (new_data['occyp']=='IT staff'),'occyp']='hightecwk'

print(new_data['occyp'].value_counts())

iv, data=calc_iv(new_data,'occyp','target')

ivtable.loc[ivtable['variable']=='occyp','IV']=iv

data.head()         
new_data = convert_dummy(new_data,'occyp')
iv, data=calc_iv(new_data,'houtp','target')

ivtable.loc[ivtable['variable']=='houtp','IV']=iv

data.head()
new_data = convert_dummy(new_data,'houtp')
new_data.loc[new_data['edutp']=='Academic degree','edutp']='Higher education'

iv, data=calc_iv(new_data,'edutp','target')

ivtable.loc[ivtable['variable']=='edutp','IV']=iv

data.head()
new_data = convert_dummy(new_data,'edutp')
new_data['famtp'].value_counts(normalize=True,sort=False)
iv, data=calc_iv(new_data,'famtp','target')

ivtable.loc[ivtable['variable']=='famtp','IV']=iv

data.head()
new_data = convert_dummy(new_data,'famtp')
ivtable=ivtable.sort_values(by='IV',ascending=False)

ivtable.loc[ivtable['variable']=='DAYS_BIRTH','variable']='agegp'

ivtable.loc[ivtable['variable']=='DAYS_EMPLOYED','variable']='worktmgp'

ivtable.loc[ivtable['variable']=='inc','variable']='incgp'

ivtable
new_data.columns
Y = new_data['target']

X = new_data[['Gender','Reality','ChldNo_1', 'ChldNo_2More','wkphone',

              'gp_Age_high', 'gp_Age_highest', 'gp_Age_low',

       'gp_Age_lowest','gp_worktm_high', 'gp_worktm_highest',

       'gp_worktm_low', 'gp_worktm_medium','occyp_hightecwk', 

              'occyp_officewk','famsizegp_1', 'famsizegp_3more',

       'houtp_Co-op apartment', 'houtp_Municipal apartment',

       'houtp_Office apartment', 'houtp_Rented apartment',

       'houtp_With parents','edutp_Higher education',

       'edutp_Incomplete higher', 'edutp_Lower secondary','famtp_Civil marriage',

       'famtp_Separated','famtp_Single / not married','famtp_Widow']]
Y = Y.astype('int')

from imblearn.over_sampling import SMOTE

X_balance,Y_balance = SMOTE().fit_sample(X,Y)

X_balance = pd.DataFrame(X_balance, columns = X.columns)
#conda install -c conda-forge imbalanced-learn
#?SMOTE.fit_sample
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_balance,Y_balance, 

                                                    stratify=Y_balance, test_size=0.3,

                                                    random_state = 10086)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression



logit = LogisticRegression(random_state=0, solver='lbfgs')

clf = logit.fit(X_train, y_train)

y_predict = clf.predict(X_test)

print('Accuracy Score is',accuracy_score(y_test, y_predict))

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))



sns.set_style("white") 

class_names = ['0','1']

plot_confusion_matrix(confusion_matrix(y_test,y_predict),

                      classes= class_names, normalize = True, 

                      title='Normalized Confusion Matrix: Logistic Regression')

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dtfit = dt.fit(X_train, y_train)

y_predict = dtfit.predict(X_test)

print('Accuracy Score is',accuracy_score(y_test, y_predict))

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))



plot_confusion_matrix(confusion_matrix(y_test,y_predict),

                      classes=class_names, normalize = True, 

                      title='Normalized Confusion Matrix: CART')
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50)

rffit = rf.fit(X_train, y_train)

y_predict = rffit.predict(X_test)

print('Accuracy Score is',accuracy_score(y_test, y_predict))

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))



plot_confusion_matrix(confusion_matrix(y_test,y_predict),

                      classes=class_names, normalize = True, 

                      title='Normalized Confusion Matrix: Ramdom Forests')
from sklearn import svm

model = svm.SVC(kernel='linear', C = 1)

svmc = model.fit(X_train, y_train)

y_predict = svmc.predict(X_test)

print('Accuracy Score is',accuracy_score(y_test, y_predict))

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))



plot_confusion_matrix(confusion_matrix(y_test,y_predict),

                      classes=class_names, normalize = True, 

                      title='Normalized Confusion Matrix: SVM')
from lightgbm import LGBMClassifier

lgb = LGBMClassifier()

lgbfit = lgb.fit(X_train, y_train)

y_predict = lgbfit.predict(X_test)

print('Accuracy Score is',accuracy_score(y_test, y_predict))

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
import seaborn as sns

def plot_importance(classifer, x_train, point_size = 25):

    '''plot feature importance'''

    values = sorted(zip(x_train.columns, classifer.feature_importances_), key = lambda x: x[1] * -1)

    imp = pd.DataFrame(values,columns = ["Name", "Score"])

    imp.sort_values(by = 'Score',inplace = True)

    sns.scatterplot(x = 'Score',y='Name', linewidth = 0,

                data = imp,s = point_size, color='red').set(

    xlabel='importance', 

    ylabel='features')

    

plot_importance(lgb, X_train,20)   
lgb.booster_.feature_importance(importance_type='gain')
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgbfit = xgb.fit(X_train, y_train)

y_predict = xgbfit.predict(X_test)

print('Accuracy Score is',accuracy_score(y_test, y_predict))

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
plot_importance(xgb, X_train, 20)   
from keras import models

from keras import layers

model = models.Sequential()

model.add(layers.Dense(64, activation = 'relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(32, activation = 'relu'))

model.add(layers.Dense(32, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.summary()
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),

    loss='binary_crossentropy',

    metrics=['accuracy'])
history = model.fit(X_train, y_train,

    epochs = 64,

    batch_size = 128,

    validation_data = (X_test, y_test))
history_dict = history.history

loss_values = history_dict['accuracy']

val_loss_values = history_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)

cv_info = pd.DataFrame({'epochs':epochs,'loss_values':loss_values,'val_loss_values':val_loss_values})



cv_info = pd.melt(cv_info, id_vars=['epochs'], value_vars=['loss_values', 'val_loss_values'])



from plotnine import *



(

ggplot(cv_info,aes('epochs','value',color = 'variable')) +

geom_line() +

geom_point() +

labs(y = 'accuracy')

)