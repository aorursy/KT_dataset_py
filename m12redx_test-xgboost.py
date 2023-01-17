import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  
warnings.simplefilter('ignore', DeprecationWarning)
%matplotlib inline
#数据读取 
df = pd.read_csv('../input/security.csv')
df.columns
#绘制热力图检查数据集是否完整
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='jet')
df.head(10)
df.info()
# 检查每列的值是否唯一
df.nunique()
df['WHOIS_COUNTRY'].unique()
df['WHOIS_STATE_CITY'].unique()
df['WHOIS_REG_YEAR'].unique()
plt.figure(figsize=(16,8))
sns.countplot(data=df, x='WHOIS_REG_YEAR', hue='TIPO')
plt.figure(figsize=(16,8))
sns.countplot(data=df, x='WHOIS_COUNTRY', hue='TIPO')
df.columns
df.head(5)
#独特值在字符串列统计
df[['URL','DOMAIN_NAME','CHARSET', 'SERVER', 'CACHE_CONTROL','WHOIS_COUNTRY','WHOIS_STATE_CITY']].nunique()
#把（Benigna）良性替换为'0'，恶性替换为'1'
def target(tipo):
    if tipo == "Benigna":
        return 0
    else:
        return 1
df['TIPO'] = df['TIPO'].apply(target)
df['TIPO'].head(20)
# 通过绘制热力图我们可以看到和target没有明显的关联性
corr = df.corr()
sns.heatmap(corr, cmap="Greens")
# 我将从字符串列中创建虚拟值，以便能够使用它利用机器学习进行分类
moddf = pd.get_dummies(df, columns=['URL','DOMAIN_NAME','CHARSET', 'SERVER', 'CACHE_CONTROL','WHOIS_COUNTRY','WHOIS_STATE_CITY'])
moddf.columns
# modff数据集分割
from sklearn.cross_validation import train_test_split
X = moddf.drop(axis=1, columns=['TIPO'])
X.head(5)
y = moddf['TIPO']
y.head(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier()
train_t=tree.fit(X_train, y_train)
pred_t= tree.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(classification_report(y_test, pred_t))
print('\n')
print(accuracy_score(y_test, pred_t))
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train, y_train)
pred = logr.predict(X_test)
# Importing metrics modules to evaluate the the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#结果输出
print(classification_report(y_test, pred))
print('\n')
print(accuracy_score(y_test, pred))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# 使用StandardScale 处理变量
scaler = StandardScaler()
sclr_train = X
scaler.fit(sclr_train)
scaled_features = scaler.transform(sclr_train)
df_features = pd.DataFrame(scaled_features,columns=sclr_train.columns)
df_features.head()
# 对处理后的数据进行分割
X_train, X_test, y_train, y_test = train_test_split(scaled_features,y,test_size=0.30, random_state=100)
error_rate = []

for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
# 基于错误率绘图
plt.figure(figsize=(10,4))
plt.plot(range(1,15),error_rate,color='b', linestyle='-', marker='*',
         markerfacecolor='red', markersize=10)
plt.xlabel('K')
plt.ylabel('Error')
# 由图得出 n_neighbours=1 是最低的错误率
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)
# 评估kNN的精准度
print(classification_report(y_test, knnpred))
print('\n')
print(accuracy_score(y_test, knnpred))
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print(classification_report(y_test, xgb_pred))
print('\n')
print(accuracy_score(y_test, xgb_pred))
print("DecisionTree:")
print('\n')
print(classification_report(y_test, pred_t))
print('\n')
confmat = confusion_matrix(y_true = y_test, y_pred =pred_t)
fig, ax = plt.subplots(figsize = (2.5, 2.5))  
ax.matshow(confmat, cmap = plt.cm.Oranges, alpha = 0.3)  
for i in range(confmat.shape[0]):  
    for j in range(confmat.shape[1]):  
        ax.text(x = j, y = i,          #ax.text()在轴上添加文本  
                s = confmat[i, j],   
                va = 'center',   
                ha = 'center')  
plt.xlabel('Predicted label')  
plt.ylabel('True label')  
plt.tight_layout()  
plt.show()  
print(accuracy_score(y_test, pred_t))
print('\n')
print('\n')

print("Logistic Regression:")
print('\n')
print(classification_report(y_test, pred))
print('\n')
confmat = confusion_matrix(y_true = y_test, y_pred =pred)
fig, ax = plt.subplots(figsize = (2.5, 2.5))  
ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)  
for i in range(confmat.shape[0]):  
    for j in range(confmat.shape[1]):  
        ax.text(x = j, y = i,          #ax.text()在轴上添加文本  
                s = confmat[i, j],   
                va = 'center',   
                ha = 'center')  
plt.xlabel('Predicted label')  
plt.ylabel('True label')  
plt.tight_layout()  
plt.show()  
print(accuracy_score(y_test, pred))
print('\n')
print('\n')


print("kNN:")
print('\n')
print(classification_report(y_test, knnpred))
print('\n')
confmat = confusion_matrix(y_true = y_test, y_pred =knnpred) 
fig, ax = plt.subplots(figsize = (2.5, 2.5))  
ax.matshow(confmat, cmap = plt.cm.Reds, alpha = 0.3)  
for i in range(confmat.shape[0]):  
    for j in range(confmat.shape[1]):  
        ax.text(x = j, y = i,          #ax.text()在轴上添加文本  
                s = confmat[i, j],   
                va = 'center',   
                ha = 'center')  
plt.xlabel('Predicted label')  
plt.ylabel('True label')  
plt.tight_layout()  
plt.show()  
print(accuracy_score(y_test, knnpred))
print('\n')
print('\n')

print("xgboost:")
print('\n')
print(classification_report(y_test,xgb_pred))
print('\n')
confmat = confusion_matrix(y_true = y_test, y_pred =xgb_pred)
fig, ax = plt.subplots(figsize = (2.5, 2.5))  
ax.matshow(confmat, cmap = plt.cm.Greens, alpha = 0.3)  
for i in range(confmat.shape[0]):  
    for j in range(confmat.shape[1]):  
        ax.text(x = j, y = i,          #ax.text()在轴上添加文本  
                s = confmat[i, j],   
                va = 'center',   
                ha = 'center')  
plt.xlabel('Predicted label')  
plt.ylabel('True label')  
plt.tight_layout()  
plt.show()  
print(accuracy_score(y_test,xgb_pred))
print('\n')
print('\n')