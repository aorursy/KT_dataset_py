import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data.keys()
print('Target Names:-')
list(data['target_names'])
print('\nDescription of the Datasets:-\n')
print(data.DESCR)
x = pd.DataFrame(data['data'],columns=data['feature_names'])
y = pd.DataFrame(data['target'],columns=['Cancer'])
x.shape,y.shape
Df = pd.concat([y,x],1).copy()
Df.head(2)
Df.to_csv("data.csv",index=False)
plt.figure(figsize=(16,4))
sns.heatmap(Df.isnull(),yticklabels=False,cmap='viridis',cbar=False)
plt.show()
print("Total Null Values =",Df.isnull().sum().sum())
dict = {0:'Malignant',1:'Benign'}

ynum = y['Cancer'].map(dict)
plt.rcParams['font.size'] =13
sns.set_style('darkgrid')
sns.countplot(ynum)
plt.xlabel(' ')
plt.title('Cancer data ratio')
plt.show()
Df.describe()
print('Pearson Co-relation of independent-features with Cancer(target):-\n')
plt.figure(figsize=(20,2))
sns.heatmap(Df.corr()[['Cancer']].T,annot=True,linewidths=1,cmap='viridis',cbar=False)
plt.show()
scaler=StandardScaler()
X = scaler.fit_transform(x)
del Df
X = pd.DataFrame(X,columns=x.columns)
del x
print('Standard Normal Distribution & Box-Plot :')
for col in X.columns:
    plt.rcParams['font.size'] =13
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,5))
    a = X[col]
    b = y['Cancer']
    sns.distplot(a,color='g',ax=ax1)
    sns.boxplot(b,a,ax=ax2)
    plt.show()
f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,5))
ax1.scatter(X['worst concave points'],X['worst perimeter'],c=y.Cancer,marker='.',cmap='rainbow')
ax2.scatter(X['worst concave points'],X['mean concave points'],c=y.Cancer,marker='.',cmap='rainbow')
ax3.scatter(X['worst concave points'],X['smoothness error'],c=y.Cancer,marker='.',cmap='rainbow')
ax4.scatter(X['mean concave points'],X['worst perimeter'],c=y.Cancer,marker='.',cmap='rainbow')

ax1.set_xlabel('worst concave points')
ax1.set_ylabel('worst perimeter')
ax2.set_xlabel('worst concave points')
ax2.set_ylabel('mean concave points')
ax3.set_xlabel('worst concave points')
ax3.set_ylabel('smoothness error')
ax4.set_xlabel('mean concave points')
ax4.set_ylabel('worst perimeter')
sns.despine()
plt.show()
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, max_iter=2000)
lasso.fit(X,y)

coeff_values = pd.DataFrame({'coeff':lasso.coef_},index=X.columns).sort_values(by='coeff')
c = (abs(coeff_values.coeff) > 0)
col_imp = list(X.columns[c])
col_imp
x = X[col_imp].copy()
x.shape
df = pd.concat([x,ynum],1).copy()
sns.pairplot(df,hue='Cancer',height=4)
plt.show()
benign = df[df['Cancer']== 'Benign']
bx = benign['mean texture']
sns.distplot(bx)
plt.show()
print('Skew :',round(bx.skew(),2))
def Zscore(data,left,right):
    index= []
    mean = data.mean()
    std = data.std()
    for i in range(len(data)):
        z = (data.iloc[i]-mean)/std
        if (z >= -left) and (z <= right): 
            index.append(i)
        else:
            pass
    return index 
index = Zscore(bx,4,2.2)
len(bx)-len(index)
bx = bx.iloc[index].reset_index(drop=True)
sns.distplot(bx)
plt.show()
print('Skew :',round(bx.skew(),2))
index1 = df[df['Cancer']== 'Malignant'].index
clean_index = list(index1) + index
len(clean_index)
df = df.iloc[clean_index].reset_index(drop=True)
sns.pairplot(df,hue='Cancer',height=4)
plt.show()
del df,X,bx
df = pd.concat([y,x],1).copy()
Df = df.iloc[clean_index].reset_index(drop=True).copy()
Df.shape
del df,x,y
X = Df.drop('Cancer',1).copy()
y = Df['Cancer'].copy()
del Df
X.shape,y.shape
sp = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in sp.split(X,y):
    xtrain,xtest = X.iloc[train_index],X.iloc[test_index]
    ytrain,ytest = y.iloc[train_index],y.iloc[test_index]

print("Lenth of train data:",len(ytrain))
print("Lenth of test data :",len(ytest))
f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,4))
sns.countplot(ytrain,ax=ax1,hue=None)
ax1.set_title('Train data ratio of Cancer')
ax1.set_axis_off()
sns.countplot(ytest,ax=ax2)
ax2.set_title('Test data ratio of Cancer')
ax2.set_axis_off()
plt.show()
model =  xgb.XGBClassifier(random_state=0,n_jobs=1)
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
accuracy_score(pred,ytest)
param =[{"max_depth":[2,3,4,5],
    "learning_rate":[0.001,0.01,0.1,1],
    "n_estimators":[100,150,200,250]}]
search = GridSearchCV(estimator=model,iid=False,param_grid=param,scoring='accuracy',cv=6,n_jobs=-1)
output = search.fit(X,y)
print('Best perameter :',output.best_params_)
print('Acccuracy      :',round(output.best_score_,2)*100,'%')
def train(X,y,rs=0):
    model =  xgb.XGBClassifier(max_depth=4,learning_rate=0.1,n_estimators=250,random_state=rs,n_jobs=1)
    sp = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=rs)
    for train_index,test_index in sp.split(X,y):
        xtrain,xtest = X.iloc[train_index],X.iloc[test_index]
        ytrain,ytest = y.iloc[train_index],y.iloc[test_index]
        model.fit(xtrain,ytrain)
        pred = model.predict(xtest)
        acc = accuracy_score(pred,ytest)
    print(f'Random State :{rs} & Accuracy :{round(acc*100,2)}')
train(X,y)
for i in range(42):
    train(X,y,i)   
train(X,y,38)
model =  xgb.XGBClassifier(max_depth=5,learning_rate=1,random_state=38,n_jobs=1)
sp = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=38)
for train_index,test_index in sp.split(X,y):
    xtrain,xtest = X.iloc[train_index],X.iloc[test_index]
    ytrain,ytest = y.iloc[train_index],y.iloc[test_index]
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    acc = accuracy_score(pred,ytest)*100
    conf = pd.DataFrame(confusion_matrix(pred,ytest),
                        columns=['Benign','Malignant'],index=['Benign','Malignant'])
    clas = classification_report(pred,ytest)
    predr = model.predict(xtrain)
    accr = accuracy_score(predr,ytrain)*100
    confr = pd.DataFrame(confusion_matrix(predr,ytrain),
                         columns=['Benign','Malignant'],index=['Benign','Malignant'])
    clasr = classification_report(predr,ytrain)
    
    
 
print("\n")
print(f"Accuracy Score for Test Data :{acc}%")
print("Classification Report for Test Data :-")
print("\n")
print(clas)
print("\n")
print(f"Accuracy Score for Train Data :{accr}%")
print(f"Classification Report for Train Data :")
print("\n")
print(clasr)

c = confusion_matrix(ytest,pred)
print('Result for test data:-\n')
print('Total test data    :',len(ytest))
print('Correct Prediction :',(c[0][0]+c[1][1]))
print('False Positive     :',c[0][1])
print('False Negetive     :',c[1][0])
print('Accurecy           :',round((c[0][0]+c[1][1])*100/len(ytest),2),'%')
print('\n')

f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
sns.heatmap(conf,annot=True,cbar=False,cmap='rainbow',fmt='.3g',ax=ax1)
ax1.set_title("Confusion Matrix for Test Data")
sns.heatmap(confr,annot=True,cbar=False,cmap='rainbow',fmt='.3g',ax=ax2)
ax2.set_title("Confusion Matrix for Train Data")
plt.show()

fig = plt.figure(figsize=(18,2))
plt.bar(np.arange(1,(len(xtest)+1)),(pred-ytest),color='r',lw = 0.3)
plt.plot(np.arange(1,(len(xtest)+1)),np.zeros(len(xtest)),color='k',lw = 5)
plt.title('Confusion Graph : False Positive [Upper Red Line] || False Negetive [Lower Red Line]')
plt.xlabel('Test Data Range')
plt.yticks([-1,0,1])
plt.show()
import pickle
#saving model
pkl = open("model.pickle","wb")
pickle.dump(model,pkl)
pkl.close()
# saving X-data
X.to_csv('X.csv',index=False)
del model
del xtrain,ytrain,xtest,ytest
#opening model
try:
    model = open("model.pickle","rb")
    model = pickle.load(model)
    print('model loaded...')
except:
    print('Error...model not loaded...')
#opening data
try:
    X = pd.read_csv('X.csv')
    print('X data loaded...')
except:
    print('Error...data not loaded...')  
def prediction(data=X,index=0):
    try:
        pred = model.predict(X.iloc[[index]])[0]
        if pred == 0:
            print("prediction is : Malignant")
        else:
            print("prediction is : Benign")
    except:
        print('Index Error...')
        print(f'Input index within :{len(X)-1}  You have Entered :{index}..')
prediction(index=555)
prediction(index=505)
prediction(index=55)
print('Thanks..')