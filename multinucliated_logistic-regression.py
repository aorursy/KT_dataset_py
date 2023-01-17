import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy.stats import skew

plt.style.use('ggplot')


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox

from sklearn.decomposition import PCA
df_train = pd.read_csv("../input/Dataset_spine.csv",encoding='latin')
list(df_train.columns)
df_train.head()
df_train.info()
df_train.describe()
df_train.nunique()
df_train.info()
df_train.isnull().values.any()
# df_train.Col3 = df_train.Col3.fillna(df_train.mean())
# df_train.Col4 = df_train.Col4.fillna(df_train.mean())
le = LabelEncoder()

le.fit(df_train.Class_att.unique())

df_train.Class_att =le.fit_transform(df_train.Class_att)
sizes = [  len(df_train.Class_att[df_train.Class_att == 0 ]), len(df_train.Class_att[df_train.Class_att == 1 ])]
colors = [ 'lightskyblue', 'lightcoral']
explode = (0.1,0)
patches, texts  , _ = plt.pie(sizes, explode=explode,colors=colors,autopct='%.2f%%',shadow=True, startangle=90)
plt.legend(patches, le.classes_, loc="best")
plt.axis('equal')
plt.title("Lower Back Pain Symptoms Dataset")
plt.tight_layout()
plt.show()
pd.scatter_matrix(df_train,figsize=(16, 16))
plt.show()
plt.scatter(df_train.pelvic_incidence,df_train.pelvic_tilt)
plt.show()
df_train.hist(alpha=0.5, figsize=(16, 10))
plt.show()
X = df_train.loc[:, df_train.columns != 'Class_att']
y = df_train.Class_att
plt.subplots(figsize=(20,15))
sns.heatmap(X.corr(),square=True)
plt.title("co-relation")
plt.show()
plt.subplots(figsize=(20,15))
sns.heatmap(X.cov(),square=True)
plt.title("co-variance")
plt.show()

rfc = RandomForestClassifier(n_jobs=-1)

lr = LogisticRegression(n_jobs=-1)


def feature_importances_RFC(X,y):
    rfc.fit(X,y)
    plt.bar( range(len(rfc.feature_importances_)),rfc.feature_importances_)
    plt.title("feature_importances_RFC")
    plt.xticks(range(len(rfc.feature_importances_)), X.columns)
    plt.xticks(rotation=90)
    plt.show()

y_validation_pred = 0 

def lets_predict(X,y):
    X_train , X_test , y_train, y_test = train_test_split( X , y ,random_state=42)
   
    lr.fit(X_train , y_train)

    y_pred = lr.predict(X_test)
    
    global y_validation_pred 
    y_validation_pred = y_pred
    
   # print("training acc : " , lr.score(X_train, y_train))
   # print("test acc : " , lr.score(y_test,y_pred.round()))
    
    print(accuracy_score(y_test,y_pred.round()))
lets_predict(X,y)
feature_importances_RFC(X,y)
X.skew()
dict_pos = {}
for col in X.columns:
    
    min_value_pos = np.abs(np.min(X[col])) + 1 
    a = []
    
    for i in X[col]:
        a.append( i + min_value_pos)
        
    dict_pos[col] = a   
X_positive_data  =  pd.DataFrame(data=dict_pos)
X_positive_data.skew()
for i in X_positive_data.columns:
    X_positive_data[i] = boxcox(X_positive_data[i])[0]
X_positive_data.hist(alpha=0.5, figsize=(16, 10))
plt.show()
ss = StandardScaler()
std_data  = ss.fit_transform(X_positive_data)
X_STD = pd.DataFrame(data=std_data ,columns=list(X.columns))
lets_predict(X_STD,y)
feature_importances_RFC(X_STD,y)
X_STD.skew()
############ negative skew remove 

# a = []
# for i in X_STD.sacrum_angle:
#     a.append(i**2)

# X_STD.sacrum_angle = a 

# a = []
# for i in df_train.pelvic_radius:
#     a.append(i**2)

# X_STD.pelvic_radius = a 


plt.subplots(figsize=(20,15))
sns.heatmap(X_STD.corr(),square=True)
plt.title("co-relation after box-cox")
plt.show()
pca = PCA(n_components=1)
X_STD.columns
pca_apply_list = ['pelvic_radius' ,'sacral_slope']
new_value = pca.fit_transform(X_STD[pca_apply_list])
new_value.shape
list0fcol = list(X_STD.columns)
for i in pca_apply_list:
    list0fcol.remove(i)
len(list0fcol)
X_PCA_without =  X_STD[list0fcol]
X_pca_with = pd.DataFrame(data=new_value , columns=['pca1'])
new_pca_df = pd.concat([X_PCA_without,X_pca_with] ,axis=1)
new_pca_df.head()
plt.subplots(figsize=(20,15))
sns.heatmap(new_pca_df.corr(),square=True)
plt.title("co-relation after PCA")
plt.show()
lets_predict(new_pca_df,df_train.Class_att)
feature_importances_RFC(new_pca_df,df_train.Class_att)